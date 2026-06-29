"""Render a gallery of GIFs documenting the project -> logs/*.gif.

Runs the standard demo set for both robots so progress is easy to eyeball and
share.  Safe to re-run; each GIF is overwritten.

  python scripts/make_gifs.py            # full gallery
  python scripts/make_gifs.py --robot talos
  python scripts/make_gifs.py --quick    # one walk per robot
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RENDER = os.path.join(HERE, "render_walk.py")


def jobs(robot, quick):
    """(args, label) pairs for render_walk.py."""
    js = [(["--robot", robot], f"{robot} forward walk (flat)")]
    if quick:
        return js
    js += [
        (["--robot", robot, "--mpc"], f"{robot} walk + DCM preview-MPC"),
        (["--robot", robot, "--terrain", "incline", "--angle", "8",
          "--step-len", "0.12"], f"{robot} walk up 8 deg incline"),
        (["--robot", robot, "--march"], f"{robot} march in place"),
    ]
    return js


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", default="both", choices=["g1", "talos", "both"])
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    robots = ["g1", "talos"] if args.robot == "both" else [args.robot]

    todo = [j for r in robots for j in jobs(r, args.quick)]
    print(f"Rendering {len(todo)} GIFs into logs/ ...\n")
    for k, (jargs, label) in enumerate(todo, 1):
        print(f"[{k}/{len(todo)}] {label}")
        r = subprocess.run([sys.executable, RENDER, *jargs],
                           cwd=os.path.dirname(HERE),
                           capture_output=True, text=True)
        out = (r.stdout or "").strip().splitlines()
        print("    " + (out[-1] if out else (r.stderr.strip()[-200:] or "no output")))
    print("\nDone. See logs/*.gif")


if __name__ == "__main__":
    main()
