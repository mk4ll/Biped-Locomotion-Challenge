"""Contact-set management and linearized friction-cone constraints.

A 'contact point' is a foot corner site carrying a 3D force f = [fx, fy, fz]
expressed in the LOCAL SURFACE FRAME (t1, t2, n). The friction pyramid is:
    f_n  >= f_min
    |f_t1| <= mu f_n
    |f_t2| <= mu f_n
On flat ground the surface frame == world frame (n = world +z). For an
inclined surface (Stage 6) the same code applies once we rotate each point's
force into the surface frame -- this is why the cone is built per-point with
a rotation R_surface (default identity = flat).
"""
import numpy as np


class ContactSet:
    """Tracks which foot corner sites are currently in contact (stance)."""

    def __init__(self, terms, left_corners, right_corners,
                 left_foot_site, right_foot_site):
        import mujoco
        self.terms = terms
        # Corner sites = force application points (for friction cones / CoP).
        self.left_corner_ids = [terms.site_ids[terms.site_names.index(n)]
                                for n in left_corners]
        self.right_corner_ids = [terms.site_ids[terms.site_names.index(n)]
                                 for n in right_corners]
        # Foot reference site = where the 6D rigid contact constraint is applied.
        self.left_foot_site = mujoco.mj_name2id(terms.model, mujoco.mjtObj.mjOBJ_SITE,
                                                left_foot_site)
        self.right_foot_site = mujoco.mj_name2id(terms.model, mujoco.mjtObj.mjOBJ_SITE,
                                                 right_foot_site)
        self.set_support("double")

    def set_support(self, mode: str):
        """mode in {'double','left','right'}."""
        self.mode = mode
        L = {"site": self.left_foot_site, "corners": self.left_corner_ids}
        R = {"site": self.right_foot_site, "corners": self.right_corner_ids}
        if mode == "double":
            self.stance = [L, R]
        elif mode == "left":
            self.stance = [L]
        elif mode == "right":
            self.stance = [R]
        else:
            raise ValueError(mode)
        # Flattened corner (force) site ids for the current stance.
        self.active_site_ids = [c for foot in self.stance for c in foot["corners"]]
        self.stance_foot_sites = [foot["site"] for foot in self.stance]
        return self.active_site_ids

    @property
    def ncp(self) -> int:
        return len(self.active_site_ids)


def friction_pyramid(ncp: int, mu: float, f_min: float, R_surface=None):
    """Build G, h for the inequality  G f <= h  over stacked forces f (3*ncp,).

    Forces are in world frame; R_surface (3x3, world<-surface) maps a surface
    force into world. Default identity (flat). Per point (surface frame):
        -f_n <= -f_min
        +f_t1 - mu f_n <= 0 ;  -f_t1 - mu f_n <= 0
        +f_t2 - mu f_n <= 0 ;  -f_t2 - mu f_n <= 0
    Expressed on world force f_world via rows acting through R_surface^T.
    Returns (G [5*ncp x 3*ncp], h [5*ncp]).
    """
    if R_surface is None:
        R_surface = np.eye(3)
    # Surface basis as rows: t1, t2, n  (in world coords).
    t1 = R_surface[:, 0]
    t2 = R_surface[:, 1]
    n = R_surface[:, 2]
    # Per-point 5x3 block mapping world force -> constraint LHS.
    block = np.array([
        -n,                 # -f_n
        t1 - mu * n,        # f_t1 - mu f_n
        -t1 - mu * n,       # -f_t1 - mu f_n
        t2 - mu * n,        # f_t2 - mu f_n
        -t2 - mu * n,       # -f_t2 - mu f_n
    ])
    rhs = np.array([-f_min, 0.0, 0.0, 0.0, 0.0])

    G = np.zeros((5 * ncp, 3 * ncp))
    h = np.zeros(5 * ncp)
    for k in range(ncp):
        G[5 * k:5 * k + 5, 3 * k:3 * k + 3] = block
        h[5 * k:5 * k + 5] = rhs
    return G, h
