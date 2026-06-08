#
# Build side-by-side comparison montages of the TEST renders for visual
# inspection: per test view, [GT | sfm | gsnet | mvs] stitched horizontally with
# labels. Reads the renders saved by render.py at
#   <exp_dir>/<clip>/<init>_<densify>/test/ours_<iter>/{renders,gt}/<idx>.png
#
# Usage:
#   python -m gsnet.make_compare --exp_dir runs/nusc_filt/eval \
#       --clips 348_clip_09 332_clip_09 331_clip_09 299_clip_09 325_clip_09 \
#       --densify d15000 --iter 30000 --inits sfm gsnet mvs \
#       --out_dir /mnt/zihanw/nusc_compare
#
import argparse
import os

from PIL import Image, ImageDraw


def load(p):
    return Image.open(p).convert("RGB")


def montage(imgs, labels, pad=4, bar=20):
    h = min(im.height for im in imgs)
    imgs = [im.resize((int(im.width * h / im.height), h)) for im in imgs]
    W = sum(im.width for im in imgs) + pad * (len(imgs) - 1)
    canvas = Image.new("RGB", (W, h + bar), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    x = 0
    for im, lab in zip(imgs, labels):
        canvas.paste(im, (x, bar))
        d.text((x + 3, 4), lab, fill=(220, 0, 0))
        x += im.width + pad
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True, help="e.g. runs/nusc_filt/eval")
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--densify", default="d15000")
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--inits", nargs="+", default=["sfm", "gsnet", "mvs"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_views", type=int, default=0, help="0=all test views per clip")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    def d_of(init, sub):
        return os.path.join(args.exp_dir, "{c}", f"{init}_{args.densify}",
                            "test", f"ours_{args.iter}", sub)

    n = 0
    for c in args.clips:
        gt_dir = d_of(args.inits[0], "gt").format(c=c)
        if not os.path.isdir(gt_dir):
            print(f"[skip] {c}: no {gt_dir}")
            continue
        views = sorted(v for v in os.listdir(gt_dir) if v.lower().endswith((".png", ".jpg")))
        if args.max_views:
            views = views[:args.max_views]
        for v in views:
            imgs, labels = [load(os.path.join(gt_dir, v))], ["GT"]
            for init in args.inits:
                p = os.path.join(d_of(init, "renders").format(c=c), v)
                if os.path.exists(p):
                    imgs.append(load(p)); labels.append(init)
            out = os.path.join(args.out_dir, f"{c}_{args.densify}_{v}")
            montage(imgs, labels).save(out)
            n += 1
    print(f"[compare] wrote {n} montages -> {args.out_dir}")


if __name__ == "__main__":
    main()
