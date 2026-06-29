"""Footstep planning, gait timeline, and DCM (capture-point) trajectory.

The walk is a sequence of phases, each with a piecewise-*affine* ZMP reference:

  * SS (single support): ZMP held at the stance foot centre while the other
    foot swings to its next footstep.
  * DS (double support):  ZMP slides from the current stance foot to the next,
    transferring weight; both feet are in contact.

For an affine ZMP p(t) = p0 + ṗ t, the divergent component of motion
ξ = x_com + ẋ_com/ω has the closed form

    ξ(t) = p(t) + ṗ/ω + (ξ0 - p0 - ṗ/ω) e^{ω t}

so the boundary DCM values are found by a backward recursion from a terminal
DCM placed over the final support (the robot comes to rest balanced).  At run
time the controller queries :meth:`WalkPlan.eval` for the reference DCM, its
rate, the nominal ZMP, the active contacts, and the swing-foot target.

This is terrain-agnostic: footstep heights/normals come from the ``terrain``.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import numpy as np

from .swing import SwingTrajectory

GRAV = 9.81


@dataclasses.dataclass
class Phase:
    kind: str                      # 'DS' or 'SS'
    duration: float
    p0: np.ndarray                 # ZMP at phase start (xy)
    p1: np.ndarray                 # ZMP at phase end   (xy)
    contacts: Tuple[str, ...]      # feet in contact this phase
    swing: Optional[str] = None    # swinging foot (SS only)
    swing_from: Optional[np.ndarray] = None   # (3,)
    swing_to: Optional[np.ndarray] = None     # (3,)
    clearance: float = 0.06        # swing apex clearance
    yaw: float = 0.0               # heading (rad) during this phase
    # xy centres of the feet *in contact* this phase (for the MPC support box)
    contact_centers: Optional[List[np.ndarray]] = None
    # filled by the DCM recursion:
    xi_start: Optional[np.ndarray] = None
    t_start: float = 0.0

    @property
    def pdot(self):
        return (self.p1 - self.p0) / self.duration


class WalkPlan:
    def __init__(self, phases: List[Phase], omega: float, com_height: float):
        self.phases = phases
        self.omega = omega
        self.com_height = com_height
        self._dcm_recursion()
        t = 0.0
        for ph in phases:
            ph.t_start = t
            t += ph.duration
        self.total_time = t

    # ---- planning ------------------------------------------------------
    def _dcm_recursion(self):
        w = self.omega
        last = self.phases[-1]
        xi_end = last.p1.copy()          # rest with DCM over final ZMP
        for ph in reversed(self.phases):
            pdot = ph.pdot
            xi_start = (ph.p0 + pdot / w
                        + (xi_end - ph.p1 - pdot / w) * np.exp(-w * ph.duration))
            ph.xi_start = xi_start
            xi_end = xi_start            # continuity: start of i == end of i-1

    def shift_future(self, ds_idx, dxy):
        """Translate the upcoming transfer-DS landing and all later footsteps by
        dxy (used to commit a reactive step adjustment), then re-solve the DCM.

        The transfer DS keeps its p0 (old stance, unmoved) but its p1 (the just-
        landed foot) and every subsequent phase shift by dxy.
        """
        dxy = np.asarray(dxy, float)
        d3 = np.array([dxy[0], dxy[1], 0.0])
        self.phases[ds_idx].p1 = self.phases[ds_idx].p1 + dxy
        for ph in self.phases[ds_idx + 1:]:
            ph.p0 = ph.p0 + dxy
            ph.p1 = ph.p1 + dxy
            if ph.swing_from is not None:
                ph.swing_from = ph.swing_from + d3
            if ph.swing_to is not None:
                ph.swing_to = ph.swing_to + d3
        # keep the just-landed foot's support centre consistent for the MPC box
        for ph in self.phases[ds_idx:]:
            if ph.contact_centers is not None:
                ph.contact_centers = [c + dxy for c in ph.contact_centers]
        self._dcm_recursion()

    def support_box(self, t: float, foot_half, margin: float = 0.0):
        """Axis-aligned bounds (lo, hi) of the planned support polygon at time t.

        Built from the *planned* foot centres in contact (single foot in SS, the
        hull of both feet in DS) padded by the foot half-extents.  The MPC clamps
        its commanded CoP to this over the whole preview horizon, so it never
        plans a centre-of-pressure the upcoming footstep can't realise.
        """
        i, _ = self._locate(t)
        centers = self.phases[i].contact_centers
        fh = np.asarray(foot_half, float)
        if not centers:                       # defensive fallback
            return np.array([-1e3, -1e3]), np.array([1e3, 1e3])
        los = np.array([c - fh for c in centers])
        his = np.array([c + fh for c in centers])
        return los.min(0) - margin, his.max(0) + margin

    # ---- runtime query -------------------------------------------------
    def _locate(self, t: float):
        if t <= 0:
            return 0, 0.0
        for i, ph in enumerate(self.phases):
            if t < ph.t_start + ph.duration:
                return i, t - ph.t_start
        return len(self.phases) - 1, self.phases[-1].duration

    def eval(self, t: float):
        w = self.omega
        i, tau = self._locate(t)
        ph = self.phases[i]
        pdot = ph.pdot
        p = ph.p0 + pdot * tau
        e = np.exp(w * tau)
        xi = p + pdot / w + (ph.xi_start - ph.p0 - pdot / w) * e
        xi_dot = pdot + w * (ph.xi_start - ph.p0 - pdot / w) * e

        swing = None
        if ph.kind == "SS" and ph.swing is not None:
            traj = SwingTrajectory(ph.swing_from, ph.swing_to, ph.duration,
                                   clearance=ph.clearance)
            pos, vel, acc = traj.sample(tau)
            swing = {"side": ph.swing, "pos": pos, "vel": vel, "acc": acc}

        return {
            "phase": i, "kind": ph.kind, "tau": tau,
            "zmp": p, "xi": xi, "xi_dot": xi_dot,
            "contacts": ph.contacts, "swing": swing, "yaw": ph.yaw,
            "done": t >= self.total_time,
        }


def plan_walk(foot_xy, foot_z, com_height,
              n_steps=8, step_len=0.0, first_swing="right",
              t_ss=0.7, t_ds=0.2, t_ds0=0.8, t_settle=0.8,
              swing_clearance=0.06, terrain=None):
    """Build a :class:`WalkPlan`.

    ``foot_xy`` : dict {'left': (2,), 'right': (2,)} initial foot centres.
    ``step_len == 0`` gives in-place stepping (marching).
    ``t_ds0`` is the (longer) initial weight-shift so the DCM reaches the first
    stance foot from rest before the swing foot lifts.
    """
    w = float(np.sqrt(GRAV / com_height))

    def gh(xy):
        return terrain.height(xy[0], xy[1]) if terrain is not None else foot_z

    pos = {s: np.array([foot_xy[s][0], foot_xy[s][1], gh(foot_xy[s])])
           for s in ("left", "right")}
    other = {"left": "right", "right": "left"}

    phases: List[Phase] = []
    mid = 0.5 * (pos["left"][:2] + pos["right"][:2])
    stance = other[first_swing]

    def centers(contacts):
        return [pos[s][:2].copy() for s in contacts]

    # initial DS: shift ZMP from the centre onto the first stance foot
    phases.append(Phase("DS", t_ds0, mid.copy(), pos[stance][:2].copy(),
                        contacts=("left", "right"),
                        contact_centers=centers(("left", "right"))))

    swing = first_swing
    for k in range(n_steps):
        stance = other[swing]
        stance_xy = pos[stance][:2].copy()
        swing_from = pos[swing].copy()
        # terrain decides the forward landing x (tread-centre snapping on stairs);
        # flat/incline just advance by step_len.  step_len==0 keeps marching.
        if terrain is not None and step_len != 0:
            tx = terrain.footstep_x(stance_xy[0], swing_from[0], step_len)
        else:
            tx = stance_xy[0] + step_len
        target_xy = np.array([tx, pos[swing][1]])
        swing_to = np.array([target_xy[0], target_xy[1], gh(target_xy)])

        # single support: ZMP fixed at stance, other foot swings
        phases.append(Phase("SS", t_ss, stance_xy, stance_xy.copy(),
                            contacts=(stance,), swing=swing,
                            swing_from=swing_from, swing_to=swing_to,
                            clearance=swing_clearance,
                            contact_centers=[stance_xy.copy()]))
        pos[swing] = swing_to

        # double support: transfer ZMP onto the foot that just landed
        last = (k == n_steps - 1)
        nxt = swing_to[:2].copy()
        phases.append(Phase("DS", t_ds, stance_xy.copy(), nxt,
                            contacts=("left", "right"),
                            contact_centers=centers(("left", "right"))))
        swing = other[swing]

    # settle: bring ZMP to the centre of the final stance and come to rest
    mid_f = 0.5 * (pos["left"][:2] + pos["right"][:2])
    phases.append(Phase("DS", t_settle, phases[-1].p1.copy(), mid_f,
                        contacts=("left", "right"),
                        contact_centers=centers(("left", "right"))))

    return WalkPlan(phases, w, com_height)


def _rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def plan_walk_velocity(foot_xy, foot_z, com_height,
                       vx=0.0, vy=0.0, vyaw=0.0, theta0=0.0,
                       n_steps=8, first_swing="right",
                       t_ss=0.7, t_ds=0.2, t_ds0=0.8, t_settle=0.8,
                       swing_clearance=0.06, terrain=None,
                       max_dx=0.18, max_dy=0.10, max_dyaw=0.30):
    """Omnidirectional footstep plan from a body-frame velocity command.

    ``vx, vy`` are forward / left velocities (m/s) and ``vyaw`` the turn rate
    (rad/s), all in the torso frame; ``theta0`` is the current heading.  Each
    step the torso advances ``v * T_cycle`` (rotated into the world) and turns,
    and the swing foot is placed at its body-frame home offset under the new
    torso pose -- so the same DCM/ZMP machinery produces forward, backward,
    sideways and turning gaits (and any blend).  Per-step displacements are
    clamped to kinematic limits.
    """
    w = float(np.sqrt(GRAV / com_height))

    def gh(xy):
        return terrain.height(xy[0], xy[1]) if terrain is not None else foot_z

    other = {"left": "right", "right": "left"}
    pos = {s: np.array([foot_xy[s][0], foot_xy[s][1], gh(foot_xy[s])])
           for s in ("left", "right")}

    # body-frame foot offsets relative to the torso centre at heading theta0.
    # Zero the fore-aft (body-x) component so the planned stance is square: after
    # a diagonal/strafe segment the feet are staggered, and perpetuating that
    # skew destabilises the next segment.  Keep the measured lateral width.
    theta = float(theta0)
    c = 0.5 * (pos["left"][:2] + pos["right"][:2])
    Rin = _rot(-theta)
    off = {}
    for s in ("left", "right"):
        b = Rin @ (pos[s][:2] - c)
        off[s] = np.array([0.0, b[1]])

    T_cycle = t_ss + t_ds
    dx = float(np.clip(vx * T_cycle, -max_dx, max_dx))
    dy = float(np.clip(vy * T_cycle, -max_dy, max_dy))
    dyaw = float(np.clip(vyaw * T_cycle, -max_dyaw, max_dyaw))

    phases: List[Phase] = []
    mid = c.copy()
    stance = other[first_swing]

    def centers(contacts):
        return [pos[s][:2].copy() for s in contacts]

    phases.append(Phase("DS", t_ds0, mid.copy(), pos[stance][:2].copy(),
                        contacts=("left", "right"), yaw=theta,
                        contact_centers=centers(("left", "right"))))

    swing = first_swing
    for k in range(n_steps):
        stance = other[swing]
        stance_xy = pos[stance][:2].copy()
        swing_from = pos[swing].copy()
        # advance + turn the torso, then place the swing foot under it
        theta += dyaw
        c = c + _rot(theta) @ np.array([dx, dy])
        target_xy = c + _rot(theta) @ off[swing]
        swing_to = np.array([target_xy[0], target_xy[1], gh(target_xy)])

        phases.append(Phase("SS", t_ss, stance_xy, stance_xy.copy(),
                            contacts=(stance,), swing=swing,
                            swing_from=swing_from, swing_to=swing_to,
                            clearance=swing_clearance, yaw=theta,
                            contact_centers=[stance_xy.copy()]))
        pos[swing] = swing_to
        phases.append(Phase("DS", t_ds, stance_xy.copy(), swing_to[:2].copy(),
                            contacts=("left", "right"), yaw=theta,
                            contact_centers=centers(("left", "right"))))
        swing = other[swing]

    mid_f = 0.5 * (pos["left"][:2] + pos["right"][:2])
    phases.append(Phase("DS", t_settle, phases[-1].p1.copy(), mid_f,
                        contacts=("left", "right"), yaw=theta,
                        contact_centers=centers(("left", "right"))))

    plan = WalkPlan(phases, w, com_height)
    plan.theta_end = theta
    return plan
