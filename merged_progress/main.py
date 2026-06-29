"""Interactive task launcher for the merged G1 locomotion project.

Run:
    python main.py

Pick a task by pressing its key (single keypress, no Enter needed):
    1..9, 0, a, b, c  ->  run that task
    v                 ->  toggle the live MuJoCo viewer on/off
    r                 ->  toggle the robot model (G1 <-> Talos)
    ESC or q          ->  exit

Each task runs in an isolated subprocess, so a failure in one never crashes the
menu. Press Ctrl+C during a task to abort it and return to the menu.
"""
import os
import sys
import subprocess

import platform

ROOT = os.path.dirname(os.path.abspath(__file__))
if platform.system() == "Darwin":
    PY = "mjpython"
else:
    PY = sys.executable

# key -> (title, [script+args], supports_viewer, supports_robot, "what you should see")
TASKS = [
    ("1", "Inspect model (DOF, torque actuators, frames)",
     ["scripts/00_inspect_model.py"], False, False,
     "nq=36 nv=35 nu=29, ALL actuators motor/torque, foot/pelvis frames, mass 33.3 kg (G1)."),
    ("2", "Gravity compensation — dynamics sanity (Stage 1)",
     ["scripts/01_gravity_comp.py"], True, False,
     "Robot stands perfectly still by feedforward torque (drift ~1.7 mm). RESULT: PASS."),
    ("3", "Standing balance: weight-shift + single-support (Stage 2)",
     ["scripts/02_stand_balance.py"], True, True,
     "Stands, sways left/right, then balances on ONE foot. QP 100% feasible. RESULT: PASS."),
    ("4", "Offline planner plots — footsteps + DCM (Stage 3)",
     ["scripts/03_plan_walk.py"], False, False,
     "Saves logs/stage3_plan.png (footsteps + CoM/DCM/ZMP, no robot motion). RESULT: PASS."),
    ("5", "Flat walking",
     ["scripts/run_walk.py", "--terrain", "flat"], True, True,
     "Robot walks forward ~0.9-1.0 m on flat ground. RESULT: PASS."),
    ("6", "Incline walking (12 deg uphill)",
     ["scripts/run_walk.py", "--terrain", "incline", "--angle", "12"], True, True,
     "Robot climbs a 12 deg slope, feet land flat. RESULT: PASS (G1 to 16 deg, Talos to ~8 deg)."),
    ("7", "Stairs climbing (6 x 2.5 cm)",
     ["scripts/run_walk.py", "--terrain", "stairs"], True, True,
     "G1 climbs a full staircase tread-by-tread (+0.15 m). (Talos stairs need own tuning.)"),
    ("8", "Omnidirectional — walk a curve (turn while walking)",
     ["scripts/run_omni.py", "--vx", "0.10", "--vyaw", "0.12"], True, True,
     "Robot walks a curved path turning ~42 deg. (Try --vy 0.08 for strafe.) RESULT: PASS."),
    ("9", "Push recovery while walking",
     ["scripts/05_push_recovery.py"], True, True,
     "External shoves hit the pelvis mid-walk; the robot steps to recover. RESULT: PASS."),
    ("0", "Standing-on-incline slip-limit sweep (theory vs experiment)",
     ["scripts/06_walk_incline.py", "--sweep"], False, False,
     "Stands up to 26 deg, slips at 27 deg == arctan(mu). Matches theory (G1)."),
    ("a", "Generate lecture-style plots (path / footsteps / CoM height)",
     ["scripts/plot_walk.py", "--terrain", "stairs"], False, False,
     "Saves logs/plot_walk_stairs.png: footstep placement + CoM height climbing the treads."),
    ("b", "Full evaluation battery (all scenarios -> report)",
     ["scripts/evaluate.py"], False, False,
     "Runs everything headless, writes logs/eval_report.md. All scenarios PASS (G1)."),
    ("c", "FUN: Waiter -- navigate around tables holding a tray + frappe",
     ["scripts/run_navigate.py", "--seed", "1"], True, True,
     "Robot plans a path around random tables, delivers a frappe; tray stays level."),
    ("d", "FUN: Sisyphus -- push a boulder up an incline",
     ["scripts/run_sisyphus.py", "--angle", "5"], True, True,
     "Robot walks up a slope shoving a big boulder (r=0.55m) with its hands/arms (+1.3m uphill)."),
    ("e", "DCM preview-MPC flat walk (anticipates support-polygon limits)",
     ["scripts/run_walk.py", "--terrain", "flat", "--mpc"], True, True,
     "MPC replaces one-step law: receding-horizon QP over ~60 steps. Same gait, better disturbance."),
    ("f", "Flat walk with natural arm swing (contralateral coupling)",
     ["scripts/run_walk.py", "--terrain", "flat", "--arm-swing"], True, True,
     "Shoulder pitch tracks contralateral hip: natural arm swing throughout gait cycle."),
    ("g", "Step timing QP — joint footstep + timing optimisation (Khadiv et al.)",
     ["scripts/run_walk.py", "--terrain", "flat", "--step-timing"], True, True,
     "Step timing QP optimises landing POSITION and TIMING jointly from measured DCM. Better lateral push recovery."),
    ("h", "Online velocity following — change direction mid-gait",
     ["scripts/run_velocity_change.py"], True, True,
     "Sequential plan segments with changing velocity commands: forward→turn→forward→veer. Adaptive navigation demo."),
    ("i", "Hard stairs (standard indoor: 4 cm risers, 20 cm run)",
     ["scripts/run_walk.py", "--terrain", "stairs", "--hard-stairs"], True, True,
     "Climbs steeper stairs than the default 2.5 cm risers. Swing apex 0.10 m, longer SS to clear 4 cm steps."),
    ("j", "SANDBOX -- walk forever (continuous flat walk, viewer required)",
     ["scripts/run_sandbox.py"], True, True,
     "Walks flat ground in an infinite loop. Re-plans each time the fixed plan ends. Ctrl-C or close viewer to stop."),
    ("k", "Turn in place — ~92° yaw rotation (no forward translation)",
     ["scripts/run_turn.py"], True, True,
     "Pure yaw rotation: vx=vy=0, vyaw=0.2 rad/s, 12 steps → ~92° turn. "
     "After 90° the foot support polygon rotates 90°; this is the certified stable limit."),
]
TASK_BY_KEY = {t[0]: t[1:] for t in TASKS}
# Speed preset names — mapped to --speed argument of run_walk.py
# slow ~0.15 m/s | normal ~0.27 m/s | fast ~0.42 m/s
SPEED_PRESETS = {"slow": "slow", "normal": "normal", "fast": "fast"}
SPEED_LABELS  = {"slow": "~0.15 m/s", "normal": "~0.36 m/s", "fast": "~0.47 m/s+MPC"}
SPEED_ORDER = ["normal", "slow", "fast"]
# tasks whose speed preset the [s] toggle controls
SPEED_TASKS = {"5", "6", "e", "f", "g", "j"}


def getkey():
    """Read a single keypress (no Enter). Returns the character; ESC -> '\\x1b'."""
    try:
        import msvcrt
        ch = msvcrt.getwch()
        return ch
    except ImportError:
        import termios, tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch


def menu(viewer, robot, speed):
    print("\n" + "=" * 72)
    print(f"  Unitree {robot.upper()} — Merged Locomotion  (torque WBC + DCM)")
    print("=" * 72)
    for (k, title, _cmd, vw, rb, _see) in TASKS:
        tag = ("  [viewer]" if vw else "") + ("  [robot]" if rb else "")
        print(f"  [{k}]  {title}{tag}")
    print("-" * 72)
    print(f"  [v] viewer: {'ON ' if viewer else 'OFF'}   "
          f"[r] robot: {robot.upper():5s}   "
          f"[s] speed: {speed} ({SPEED_LABELS[speed]})")
    print("  [ESC / q]  exit")
    print("=" * 72)
    print("Press a key...")


def run_task(key, viewer, robot, speed):
    title, cmd, supports_vw, supports_rb, see = TASK_BY_KEY[key]
    args = list(cmd)
    if viewer and supports_vw:
        args.append("--viewer")
    if supports_rb:
        args += ["--robot", robot]
    elif robot != "g1":
        print(f"\n(note: task '{title}' is G1-only; running on G1.)")
    if key in SPEED_TASKS:
        args += ["--speed", SPEED_PRESETS[speed]]
    print("\n" + "-" * 72)
    print(f"RUN [{robot.upper()}]: {title}")
    print(f"WHAT YOU SHOULD SEE: {see}")
    print(f"$ python {' '.join(args)}")
    print("-" * 72)
    try:
        subprocess.run([PY, *[os.path.join(ROOT, args[0])] + args[1:]], cwd=ROOT)
    except KeyboardInterrupt:
        print("\n[aborted -> back to menu]")
    print("\n[done -> press a key for the menu]")
    getkey()


def main():
    viewer = False
    robot = "g1"
    speed = "normal"
    while True:
        menu(viewer, robot, speed)
        ch = getkey()
        if ch in ("\x1b", "q", "Q"):
            print("bye.")
            return
        if ch in ("v", "V"):
            viewer = not viewer
            continue
        if ch in ("r", "R"):
            robot = "talos" if robot == "g1" else "g1"
            continue
        if ch in ("s", "S"):
            speed = SPEED_ORDER[(SPEED_ORDER.index(speed) + 1) % len(SPEED_ORDER)]
            continue
        if ch in TASK_BY_KEY:
            run_task(ch, viewer, robot, speed)
        # any other key: just redraw the menu


if __name__ == "__main__":
    main()
