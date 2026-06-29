"""Walking controller: binds the planner references to the WBC QP.

Per control step:
  1. query plan.reference(t)         -> CoM, swing-foot, support mode
  2. set task targets (CoM, swing foot) and the active contact set
  3. solve the WBC QP                 -> joint torques
  4. apply torques to MuJoCo
The controller is reusable across Stage 4 (flat), 5 (push) and 6 (incline);
only the plan / scene change.
"""
import numpy as np
import mujoco

from src.dynamics.model_terms import ModelTerms
from src.dynamics.contacts import ContactSet
from src.control.wbc_qp import WBCQP
from src.control.tasks import CoMTask, OrientationTask, PostureTask, FootTask
from src.control.gravity_comp import GravityCompensator


class WalkingController:
    def __init__(self, env, params):
        self.env = env
        self.params = params
        m = params["model"]
        lc = m["feet"]["left"]["corners"]
        rc = m["feet"]["right"]["corners"]
        self.terms = ModelTerms(env.model, lc + rc)
        self.contacts = ContactSet(self.terms, lc, rc,
                                   m["feet"]["left"]["site"], m["feet"]["right"]["site"])

        self.base_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, m["base_body"])
        self.left_site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE,
                                           m["feet"]["left"]["site"])
        self.right_site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE,
                                            m["feet"]["right"]["site"])

        tp = params["wbc"]["tasks"]
        self.com_task = CoMTask(self.terms, self.base_id, tp["com"]["kp"],
                                tp["com"]["kd"], tp["com"]["weight"])
        self.k_dcm = tp["com"].get("k_dcm", 3.0)
        self.ori_task = OrientationTask(self.terms, self.base_id, tp["orientation"]["kp"],
                                        tp["orientation"]["kd"], tp["orientation"]["weight"])
        q_nom = env.data.qpos[[env.model.jnt_qposadr[env.model.dof_jntid[d]]
                               for d in self.terms.act_dof]].copy()
        self.pos_task = PostureTask(self.terms, q_nom, tp["posture"]["kp"],
                                    tp["posture"]["kd"], tp["posture"]["weight"])
        sf = tp["swing_foot"]
        self.swing_tasks = {
            "left": FootTask(self.terms, self.left_site, sf["kp"], sf["kd"], sf["weight"],
                             track_orientation=True, kp_ori=sf.get("kp_ori"),
                             kd_ori=sf.get("kd_ori")),
            "right": FootTask(self.terms, self.right_site, sf["kp"], sf["kd"], sf["weight"],
                              track_orientation=True, kp_ori=sf.get("kp_ori"),
                              kd_ori=sf.get("kd_ori")),
        }
        self.foot_sites = {"left": self.left_site, "right": self.right_site}
        # nominal (flat) foot orientation captured at the keyframe
        self.swing_tasks["left"].R_des = env.data.site_xmat[self.left_site].reshape(3, 3).copy()
        self.swing_tasks["right"].R_des = env.data.site_xmat[self.right_site].reshape(3, 3).copy()
        # capture-point foot placement state
        cap = params.get("capture", {})
        self.capture_enabled = cap.get("enabled", False)
        self.capture_gain = cap.get("gain", 1.0)
        self.capture_max = cap.get("max_shift", 0.12)
        self._prev_support = None
        self._foot_offset = np.zeros(2)
        # slope feedforward: forward CoM accel to counter gravity-along-slope (Stage 6)
        self.slope_accel_ff = np.zeros(2)

        self.wbc = WBCQP(self.terms, params)
        self.gc = GravityCompensator(self.terms, mujoco.mj_getTotalmass(env.model),
                                     params["env"]["gravity"], params["wbc"]["reg"]["force"])
        # keep pelvis at its nominal (upright, facing +x) orientation
        self.ori_task.R_des = env.data.xmat[self.base_id].reshape(3, 3).copy()
        self.R_surface = None  # flat (world-z). Stage 6 sets the incline rotation.

    def step(self, plan, t):
        env = self.env
        ref = plan.reference(t)
        # --- CoM command: DCM feedback in xy, position PD in z ---
        # xi = com + com_vel/omega ;  xi_dot_ref = omega (xi_ref - zmp_ref)
        # xi_dot_cmd = xi_dot_ref + k_dcm (xi_ref - xi)
        # p_zmp_cmd  = xi - xi_dot_cmd/omega ;  a_com = omega^2 (com - p_zmp_cmd)
        w = ref["omega"]
        J = self.com_task.jacobian(env.data)
        com = self.com_task.com(env.data)
        com_vel = J @ env.data.qvel
        xi = com[:2] + com_vel[:2] / w
        xi_dot_ref = w * (ref["dcm"] - ref["zmp"])
        xi_dot_cmd = xi_dot_ref + self.k_dcm * (ref["dcm"] - xi)
        p_zmp_cmd = xi - xi_dot_cmd / w
        a_xy = w * w * (com[:2] - p_zmp_cmd) + self.slope_accel_ff
        self.com_task.a_ref = np.array([a_xy[0], a_xy[1], 0.0])
        # xy error handled by a_ref above -> zero the PD error; z tracks height.
        self.com_task.p_ref = np.array([com[0], com[1], ref["com"][2]])
        self.com_task.v_ref = np.array([com_vel[0], com_vel[1], 0.0])
        # contact set
        self.contacts.set_support(ref["support"])
        tasks = [self.com_task, self.ori_task, self.pos_task]
        # swing foot task (only while a foot is airborne)
        if ref["phase"] == "SS" and ref["swing"] is not None:
            # Capture-point step adjustment: shift the swing-foot landing by
            # gain*(xi_measured - xi_ref). Updated CONTINUOUSLY through the SS so a
            # mid-step push is corrected immediately (not one step late). Ramped by
            # phase progress so it starts from the current foot pose (no jump).
            if self.capture_enabled:
                xi_meas = com[:2] + com_vel[:2] / w
                off = self.capture_gain * (xi_meas - ref["dcm"])
                self._foot_offset = np.clip(off, -self.capture_max, self.capture_max)
            sw = self.swing_tasks[ref["swing"]]
            adj = ref["swing_pos"].copy()
            adj[:2] = adj[:2] + ref["progress"] * self._foot_offset   # ramp in, no jump
            sw.p_ref = adj
            sw.v_ref = ref["swing_vel"]
            tasks.append(sw)
        self._prev_support = ref["support"]
        # solve + apply
        tau_fb = self.gc.compute(env.data,
                                 active_site_ids=self.contacts.active_site_ids)[0]
        res = self.wbc.solve(env.data, tasks, self.contacts.stance,
                             R_surface=self.R_surface, fallback_tau=tau_fb)
        env.step(res["tau"])
        return ref, res
