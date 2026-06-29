"""Compress all raw GIFs from capture_all_gifs.py into small previews.

Reads logs/gifs/raw/*.gif, downscales to 240x180, 10 fps, 64 colours, and
writes logs/gifs/preview/*.gif.

  python scripts/compress_all_gifs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import imageio.v3 as iio
from PIL import Image

RAW     = Path(__file__).resolve().parents[1] / "logs" / "gifs" / "raw"
PREVIEW = Path(__file__).resolve().parents[1] / "logs" / "gifs" / "preview"
PREVIEW.mkdir(parents=True, exist_ok=True)

OUT_W, OUT_H = 240, 180
OUT_FPS = 10
N_COLORS = 64
FRAME_STEP = 2      # keep every 2nd raw frame to halve frame count


def compress(src: Path) -> Path:
    raw_frames = iio.imread(str(src), index=None)     # (N, H, W, 3)
    if raw_frames.ndim == 3:
        raw_frames = raw_frames[np.newaxis]            # single frame edge case

    out_frames = []
    for i, f in enumerate(raw_frames):
        if i % FRAME_STEP != 0:
            continue
        img = Image.fromarray(f[:, :, :3] if f.shape[2] == 4 else f)
        img = img.resize((OUT_W, OUT_H), Image.LANCZOS)
        img = img.quantize(N_COLORS, method=Image.Quantize.MEDIANCUT).convert("RGB")
        out_frames.append(np.array(img, dtype=np.uint8))

    if not out_frames:
        return None

    dst = PREVIEW / src.name
    iio.imwrite(str(dst), out_frames, extension=".gif",
                fps=OUT_FPS, loop=0)
    kb_src = src.stat().st_size // 1024
    kb_dst = dst.stat().st_size // 1024
    print(f"  {src.name}: {kb_src} KB → {kb_dst} KB  ({len(out_frames)} frames)")
    return dst


if __name__ == "__main__":
    raws = sorted(RAW.glob("*.gif"))
    if not raws:
        print(f"No raw GIFs found in {RAW}. Run capture_all_gifs.py first.")
        sys.exit(1)

    total_in = total_out = 0
    for src in raws:
        dst = compress(src)
        if dst:
            total_in  += src.stat().st_size
            total_out += dst.stat().st_size

    print(f"\nTotal: {total_in//1024} KB raw → {total_out//1024} KB preview")
    print(f"Preview GIFs in: {PREVIEW}")
