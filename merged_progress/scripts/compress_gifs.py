"""Downsample the full-res GIFs to small artifact-embeddable previews.

Reads each GIF from logs/gifs/, takes a representative clip, resizes to
240x180, drops to 10fps, quantises to 96 colours, and saves a _preview.gif.
Target: each preview < 3 MB.

  python scripts/compress_gifs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import imageio.v3 as iio
from PIL import Image

GIFS = Path(__file__).resolve().parents[1] / "logs" / "gifs"
OUT_W, OUT_H = 192, 144   # preview resolution
N_COLORS = 64              # GIF palette depth
KEEP_FPS = 8               # preview framerate

# (filename, start_frame, end_frame) — clip the most visually interesting part
# Source GIF is at 20fps; indices below are into saved frames
CLIPS = {
    "flat_walk.gif":        (20,  180),   # ~8s clip: mid-walk
    "incline_12deg.gif":    (20,  180),
    "stairs_25mm.gif":      (40,  220),   # stair climb
    "stairs_40mm_hard.gif": (40,  220),
    "push_recovery_80N.gif":(30,  190),   # includes push moment
    "velocity_change.gif":  (20,  220),   # forward + turn visible
}

# Source GIF fps is 20; keep every N frames to hit KEEP_FPS
SUBSAMPLE = max(1, round(20 / KEEP_FPS))


def compress(name, start, end):
    src = GIFS / name
    if not src.exists():
        print(f"  SKIP (not found): {name}")
        return None

    print(f"  reading {name} ...", end=" ", flush=True)
    try:
        frames_raw = iio.imread(str(src), index=None)   # (N, H, W, 3/4)
    except Exception as e:
        print(f"ERROR: {e}")
        return None

    frames_raw = frames_raw[start:end:SUBSAMPLE]
    print(f"{len(frames_raw)} frames after clip+subsample")

    out_frames = []
    for f in frames_raw:
        if f.shape[2] == 4:
            f = f[..., :3]  # drop alpha
        img = Image.fromarray(f.astype(np.uint8))
        img = img.resize((OUT_W, OUT_H), Image.LANCZOS)
        img = img.quantize(colors=N_COLORS, method=Image.Quantize.MEDIANCUT)
        img = img.convert("RGB")
        out_frames.append(np.array(img, dtype=np.uint8))

    stem = name.replace(".gif", "_preview.gif")
    dst  = GIFS / stem
    iio.imwrite(str(dst), out_frames, extension=".gif", fps=KEEP_FPS, loop=0)
    kb = dst.stat().st_size // 1024
    print(f"    -> {stem}  ({kb} KB)")
    return stem


if __name__ == "__main__":
    print("Compressing GIFs to preview size...")
    for fname, (s, e) in CLIPS.items():
        compress(fname, s, e)
    print("Done.")
