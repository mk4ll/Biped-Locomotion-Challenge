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
from src.estimation.terrain_estimator import ContactTerrainEstimator


class WalkingController:
    def __init__(self, env, params, terrain=None, mcfg=None):
        self.env = env
        self.params = params
        self.terrain = terrain
        m = mcfg if mcfg is not None else params["model"]   # robot model config
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
        # --- Contact-based terrain estimator (Improvement 7) ---
        self._terrain_est = ContactTerrainEstimator()
        if terrain is not None:
            self._terrain_est.set_fallback(terrain)
        self._prev_phase = None    # for SS->DS touchdown detection
        self._prev_swing = None    # which foot was swinging in the last SS phase
        # Approximate standing height above terrain surface (used externally)
        self.z_above = terrain.height(0.0, 0.0) if terrain is not None else 0.0
        # slope feedforward: forward CoM accel to counter gravity-along-slope (Stage 6)
        self.slope_accel_ff = np.zeros(2)
        # low-pass on the CoM-height reference (terrain.height is a step fn on stairs;
        # filtering it avoids a vertical jerk each time the CoM crosses a tread edge)
        self.comz_lp_alpha = params["wbc"]["tasks"]["com"].get("comz_lp_alpha", 1.0)
        self._comz_lp = None

        self.wbc = WBCQP(self.terms, params)
        self.gc = GravityCompensator(self.terms, mujoco.mj_getTotalmass(env.model),
                                     params["env"]["gravity"], params["wbc"]["reg"]["force"])
        # keep pelvis at its nominal (upright, facing +x) orientation
        self.ori_task.R_des = env.data.xmat[self.base_id].reshape(3, 3).copy()
        # Terrain-aware friction-cone / foot orientation. For flat/None -> world-z.
        # For incline -> surface normal (constant). For stairs -> per-tread (flat).
        self.R_surface = (terrain.surface_R(0.0, 0.0) if terrain is not None else None)

        # --- DCM Preview-MPC (optional) ---
        mp = params.get("dcm_mpc", {})
        self.mpc = None
        self._mpc_dt = 1.0 / mp.get("freq_hz", 50.0)
        self._mpc_next_t = 0.0
        self._mpc_last_cop = None
        # MPC is initialised lazily on first step() call (needs omega from the plan).
        self._mpc_params = mp if mp.get("enabled", False) else None

        # --- Arm swing (contralateral hip-shoulder coupling) ---
        as_cfg = params.get("arm_swing", {})
        self.k_arm = as_cfg.get("gain", 0.0) if as_cfg.get("enabled", False) else 0.0
        self._q_nom_base = self.pos_task.q_nom.copy()  # reference nominal (never modified)
        # Map shoulder pitch joint names -> index in pos_task.q_nom
        self._sho_idx = {}   # side -> act_dof index (or None if joint not in model)
        for side, jname in [("left", as_cfg.get("left_shoulder", "left_shoulder_pitch_joint")),
                             ("right", as_cfg.get("right_shoulder", "right_shoulder_pitch_joint"))]:
            idx = self._find_act_idx(env.model, jname)
            if idx is not None:
                self._sho_idx[side] = idx

        # Override elbow nominal to elbow_angle (default 1.57 rad = 90°) when arm swing is on.
        # The keyframe default is 1.28 rad (~73°); 1.57 gives a proper right-angle bend.
        if as_cfg.get("enabled", False):
            elbow_angle = as_cfg.get("elbow_angle", 1.57)
            for side, jname in [("left",  as_cfg.get("left_elbow",  "left_elbow_joint")),
                                 ("right", as_cfg.get("right_elbow", "right_elbow_joint"))]:
                idx = self._find_act_idx(env.model, jname)
                if idx is not None:
                    self._q_nom_base[idx] = elbow_angle
                    self.pos_task.q_nom[idx] = elbow_angle

        # --- Step timing QP (Khadiv et al. 2016) ---
        # The StepTimingQP class (src/planning/step_timing.py) is available and tested.
        # Full per-step timing optimisation requires an event-driven online replanner
        # (the offline fixed plan doesn't update u_nom after actual foot landings).
        # In this controller, enabling step_timing widens the capture-point reachability
        # limit, allowing larger corrective steps under strong lateral pushes.
        st = params.get("step_timing", {})
        if st.get("enabled", False):
            # Wider foot-shift budget + slightly higher gain for better push recovery
            self.capture_max = st.get("max_shift_st", 0.20)
            self.capture_gain = st.get("gain_st", 1.2)

    @property
    def terrain_estimator(self) -> ContactTerrainEstimator:
        """The contact-based terrain estimator (Improvement 7)."""
        return self._terrain_est

    def _find_act_idx(self, model, joint_name):
        """Return the index of joint_name within pos_task.q_nom (act_dof order), or None."""
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            return None
        dof = model.jnt_dofadr[jid]
        act_dof_list = list(self.terms.act_dof)
        return act_dof_list.index(dof) if dof in act_dof_list else None

    def _init_mpc(self, omega):
        from src.planning.dcm_mpc import DCMPreviewMPC, MPCWeights
        mp = self._mpc_params
        wts = MPCWeights(
            w_xi=mp.get("w_xi", 1.0),
            w_p=mp.get("w_p", 4e-2),
            w_dp=mp.get("w_dp", 2e-2),
        )
        foot_half = tuple(mp.get("foot_half", [0.09, 0.04]))
        self.mpc = DCMPreviewMPC(
            omega=omega,
            horizon=mp.get("horizon", 1.2),
            dt=mp.get("mpc_dt", 0.02),
            weights=wts,
            foot_half=foot_half,
            box_margin=mp.get("box_margin", 0.0),
        )

    def _surface_R(self, x):
        if self.terrain is None:
            return None
        return self.terrain.surface_R(x, 0.0)

    def step(self, plan, t):
        env = self.env
        ref = plan.reference(t)
        w = ref["omega"]
        J = self.com_task.jacobian(env.data)
        com = self.com_task.com(env.data)
        com_vel = J @ env.data.qvel
        xi = com[:2] + com_vel[:2] / w   # measured DCM

        # --- Touchdown detection for contact terrain estimator (Improvement 7) ---
        # An SS->DS transition means the swing foot just touched down.
        curr_phase = ref.get("phase", "DS")
        if self._prev_phase == "SS" and curr_phase == "DS":
            if self._prev_swing is not None:
                site_id = self.foot_sites[self._prev_swing]
                fx, fy, fz = env.data.site_xpos[site_id]
                self._terrain_est.record(fx, fy, fz)
                # Online footstep replanning (Improvement 3): update remaining plan
                # with the actual landing position so future steps stay consistent.
                if hasattr(plan, 'replan_touchdown'):
                    actual_pos = env.data.site_xpos[site_id].copy()
                    plan.replan_touchdown(t, self._prev_swing, actual_pos)
        if curr_phase == "SS":
            self._prev_swing = ref.get("swing")
        self._prev_phase = curr_phase

        # --- CoM command: DCM MPC or one-step proportional feedback ---
        if self._mpc_params is not None:
            if self.mpc is None:
                self._init_mpc(w)
            # Throttle MPC solve to mpc_freq_hz; hold CoP between solves.
            if t >= self._mpc_next_t or self._mpc_last_cop is None:
                self._mpc_last_cop = self.mpc.solve(plan, t, xi)
                self._mpc_next_t = t + self._mpc_dt
            p_zmp_cmd = self._mpc_last_cop
        else:
            # One-step proportional law (original): xi_dot_cmd + k*(xi_ref - xi)
            xi_dot_ref = w * (ref["dcm"] - ref["zmp"])
            xi_dot_cmd = xi_dot_ref + self.k_dcm * (ref["dcm"] - xi)
            p_zmp_cmd = xi - xi_dot_cmd / w
        a_xy = w * w * (com[:2] - p_zmp_cmd) + self.slope_accel_ff
        self.com_task.a_ref = np.array([a_xy[0], a_xy[1], 0.0])
        # low-pass the CoM-height reference (smooths the stair step function)
        zref = ref["com"][2]
        if self._comz_lp is None:
            self._comz_lp = zref
        self._comz_lp += self.comz_lp_alpha * (zref - self._comz_lp)
        # xy error handled by a_ref above -> zero the PD error; z tracks height.
        self.com_task.p_ref = np.array([com[0], com[1], self._comz_lp])
        self.com_task.v_ref = np.array([com_vel[0], com_vel[1], 0.0])
        # contact set
        self.contacts.set_support(ref["support"])
        # heading: keep the pelvis yaw tracking the commanded heading (for turning)
        th = ref.get("heading", 0.0)
        if abs(th) > 1e-9:
            c, s = np.cos(th), np.sin(th)
            self.ori_task.R_des = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        # terrain-aware friction cone: orient to the surface under the support feet
        if self.terrain is not None:
            self.R_surface = self._surface_R(com[0])
        tasks = [self.com_task, self.ori_task, self.pos_task]
        # swing foot task (only while a foot is airborne)
        if ref["phase"] == "SS" and ref["swing"] is not None:
            # Capture-point step adjustment: shift landing by gain*(xi_meas-xi_ref).
            # When step_timing is enabled, widen the max reachability limit so the
            # controller can take larger corrective steps under strong lateral pushes.
            if self.capture_enabled:
                # Capture-point step adjustment: shift landing by gain*(xi_meas-xi_ref).
                # Ramped by progress so it starts from the current foot pose (no jump).
                off = self.capture_gain * (xi - ref["dcm"])
                self._foot_offset = np.clip(off, -self.capture_max, self.capture_max)
                adj = ref["swing_pos"].copy()
                adj[:2] = adj[:2] + ref["progress"] * self._foot_offset
            else:
                adj = ref["swing_pos"].copy()
                adj[:2] = adj[:2] + ref["progress"] * self._foot_offset
            # Enforce minimum lateral clearance between feet so the
            # capture-point correction never brings feet dangerously close.
            swing = ref["swing"]
            stance_side = "right" if swing == "left" else "left"
            stance_y = env.data.site_xpos[self.foot_sites[stance_side]][1]
            _MIN_FOOT_SEP = 0.07   # 7 cm minimum lateral gap (G1 foot width ~5 cm)
            if swing == "left":
                adj[1] = max(adj[1], stance_y + _MIN_FOOT_SEP)
            else:
                adj[1] = min(adj[1], stance_y - _MIN_FOOT_SEP)
            sw = self.swing_tasks[ref["swing"]]
            sw.p_ref = adj
            sw.v_ref = ref["swing_vel"]
            # land the swing foot aligned to the terrain it is heading onto
            if self.terrain is not None:
                sw.R_des = self._surface_R(adj[0])
            tasks.append(sw)

            # Arm swing: contralateral hip-shoulder coupling (sinusoidal, phase-based).
            # Right leg swings → left arm forward (+), right arm back (-), and vice versa.
            if self.k_arm > 0 and len(self._sho_idx) == 2:
                s = ref["progress"]
                amp = self.k_arm * np.sin(np.pi * s)
                swing = ref["swing"]
                ipsi = swing         # ipsilateral to swing leg → arm goes back
                contra = "right" if swing == "left" else "left"  # contralateral → arm forward
                if contra in self._sho_idx:
                    self.pos_task.q_nom[self._sho_idx[contra]] = (
                        self._q_nom_base[self._sho_idx[contra]] + amp)
                if ipsi in self._sho_idx:
                    self.pos_task.q_nom[self._sho_idx[ipsi]] = (
                        self._q_nom_base[self._sho_idx[ipsi]] - amp)
        else:
            # DS: restore nominal shoulder angles
            for side, idx in self._sho_idx.items():
                self.pos_task.q_nom[idx] = self._q_nom_base[idx]

        self._prev_support = ref["support"]
        # solve + apply
        tau_fb = self.gc.compute(env.data,
                                 active_site_ids=self.contacts.active_site_ids)[0]
        res = self.wbc.solve(env.data, tasks, self.contacts.stance,
                             R_surface=self.R_surface, fallback_tau=tau_fb)
        env.step(res["tau"])
        return ref, res
