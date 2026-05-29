#
# Generate the Same-Sensor Evaluation (SSE) train/test split as a COLMAP
# `sparse/0/test.txt` file (consumed by scene/dataset_readers.py).
#
# CARLA-NVS layout: 60 images per sequence, 6 source cameras x 10 frames,
# ordered as frames 1-10 -> cam1, 11-20 -> cam3, ..., 51-60 -> cam11.
# SSE holds out 2 frames per camera (default the 4th and 9th within each block),
# giving 12 test / 48 train images per sequence (matches the paper).
#
# Usage:
#   python -m gsnet.make_sse_split --seq_dir input_output/110_base
#   python -m gsnet.make_sse_split --seq_dir input_output/110_base --block 10 --holdout 4 9
#

import argparse
import os


def sse_test_names(num_images=60, block=10, holdout=(4, 9), ext=".png"):
    """Return the list of test image filenames for the SSE split."""
    holdout0 = set(h - 1 for h in holdout)  # 1-indexed within block -> 0-indexed
    names = []
    for n in range(1, num_images + 1):
        if (n - 1) % block in holdout0:
            names.append(f"{n}{ext}")
    return names


def write_split(seq_dir, num_images=60, block=10, holdout=(4, 9), ext=".png"):
    names = sse_test_names(num_images, block, holdout, ext)
    out = os.path.join(seq_dir, "sparse", "0", "test.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"[sse] {out}: {len(names)} test images -> {names}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", required=True, help="sequence COLMAP dir (has sparse/0)")
    ap.add_argument("--num_images", type=int, default=60)
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--holdout", type=int, nargs="+", default=[4, 9])
    ap.add_argument("--ext", default=".png")
    args = ap.parse_args()
    write_split(args.seq_dir, args.num_images, args.block, tuple(args.holdout), args.ext)


if __name__ == "__main__":
    main()
