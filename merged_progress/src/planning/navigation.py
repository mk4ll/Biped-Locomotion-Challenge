"""Go-to-goal path planning around circular obstacles (tables).

A simple, robust artificial-potential-field planner: gradient descent toward the
goal with repulsion from each obstacle plus a tangential 'swirl' term so the path
arcs AROUND a table instead of stalling head-on. Returns a smooth 2-D centre-line
path that the footstep planner (plan_path) follows.
"""
import numpy as np


def plan_path(start, goal, obstacles, robot_radius=0.30,
              step=0.04, max_iters=5000, swirl=0.7, influence=1.0):
    """start/goal: (x,y). obstacles: list of (x,y,radius). Returns Nx2 path.

    Wide influence radius + moderate swirl => the path starts curving early and
    arcs gently around tables (no sharp turns the walking gait can't follow).
    """
    start = np.asarray(start, float)[:2]
    goal = np.asarray(goal, float)[:2]
    obs = [(np.array([ox, oy]), r + robot_radius) for (ox, oy, r) in obstacles]

    p = start.copy()
    path = [p.copy()]
    for _ in range(max_iters):
        to_goal = goal - p
        dgoal = np.linalg.norm(to_goal)
        if dgoal < 0.08:
            break
        f = to_goal / dgoal                      # attraction (unit)
        for c, R in obs:
            d = p - c
            dist = np.linalg.norm(d)
            if dist < R + influence:             # within influence radius
                n = d / (dist + 1e-9)
                strength = (R + influence - dist) / influence
                f = f + 1.4 * strength * n       # repulsion (push away)
                tang = np.array([-n[1], n[0]])   # tangential swirl (arc around)
                if np.dot(tang, to_goal) < 0:
                    tang = -tang
                f = f + swirl * strength * tang
        nf = np.linalg.norm(f)
        if nf < 1e-6:
            break
        p = p + step * f / nf
        path.append(p.copy())
    path.append(goal.copy())
    return _smooth(np.array(path), k=25)


def _smooth(path, k=9):
    """Moving-average smooth (keep endpoints)."""
    if len(path) < k:
        return path
    out = path.copy()
    half = k // 2
    for i in range(half, len(path) - half):
        out[i] = path[i - half:i + half + 1].mean(axis=0)
    return out


def path_curviness(path, step=0.11):
    """Max heading change per STEP-length window (predicts the gait's turn rate)."""
    seg = np.diff(path, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(cum[-1])
    if total < step:
        return 0.0
    ss = np.arange(0.0, total, step)
    head = []
    for s in ss:
        i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(seg) - 1))
        head.append(np.arctan2(seg[i, 1], seg[i, 0]))
    return float(np.abs(np.diff(np.unwrap(head))).max()) if len(head) > 1 else 0.0


def make_slalom(n, x0=0.9, dx=1.6, amp=0.26, radius=(0.18, 0.22), robot_amp=0.30,
                jitter=0.10, seed=None):
    """Generate a weaving SLALOM corridor and its matching planned trajectory.

    Algorithm (see README): tables are placed alternately on the +y / -y side of
    the centre-line, one per 'gate' spaced ``dx`` apart (with random jitter so each
    run differs). The planned trajectory is a smooth sinusoid in OPPOSITE phase to
    the tables -- it dips to -robot_amp exactly where a +amp table sits and to
    +robot_amp where a -amp table sits -- so the robot threads each gate on the
    free side. ``dx`` is chosen so the sinusoid's curvature stays within the gait's
    turn rate (no sharp turns). Returns (tables, goal, path[N,2]).
    """
    rng = np.random.default_rng(seed)
    sign0 = 1 if rng.random() < 0.5 else -1
    tables, centers = [], []
    for i in range(n):
        x = x0 + i * dx + rng.uniform(-jitter, jitter)
        y = sign0 * (amp if i % 2 == 0 else -amp) + rng.uniform(-jitter, jitter)
        r = rng.uniform(*radius)
        tables.append((float(x), float(y), float(r)))
        centers.append((x, y))
    goal = (x0 + (n - 1) * dx + 0.8, 0.0)

    # planned trajectory: sinusoid in opposite phase to the tables, ramped in/out
    xs = np.linspace(0.0, goal[0], 400)
    path = []
    for x in xs:
        # envelope: 0 before the first gate and after the last, full in between
        env = np.clip(min((x - (x0 - dx)) / dx, (goal[0] - x) / dx, 1.0), 0.0, 1.0)
        y = -sign0 * robot_amp * np.cos(np.pi * (x - x0) / dx) * env
        path.append([x, y])
    path = np.array(path)
    path[-1] = goal
    return tables, goal, _smooth(path, k=15)


def random_tables(n, area=(0.7, 3.2, -1.0, 1.0), radius=(0.16, 0.24),
                  start=(0, 0), goal=None, clearance=0.6, seed=None, tries=400):
    """Place n well-spaced tables in the area, clear of start/goal.

    Tables are kept apart (>= 0.9 m centre-to-centre) so no gap is too tight for
    the robot to thread -- the path arcs around rather than squeezing between.
    """
    rng = np.random.default_rng(seed)
    xlo, xhi, ylo, yhi = area
    goal = goal if goal is not None else (xhi + 0.5, 0.0)
    tables = []
    for _ in range(n):
        for _ in range(tries):
            x = rng.uniform(xlo, xhi); y = rng.uniform(ylo, yhi)
            r = rng.uniform(*radius)
            if np.hypot(x - start[0], y - start[1]) < clearance + r:
                continue
            if np.hypot(x - goal[0], y - goal[1]) < clearance + r:
                continue
            if all(np.hypot(x - tx, y - ty) > 0.9 for (tx, ty, tr) in tables):
                tables.append((x, y, r)); break
    return tables, goal
