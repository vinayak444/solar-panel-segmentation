# tools/infer_cli.py
import os, argparse, cv2
from core import Segmenter

p = argparse.ArgumentParser()
p.add_argument("--in_dir",  required=True)
p.add_argument("--out_dir", required=True)
p.add_argument("--model",   default="/app/weights/u2net_multiclass.pt")
p.add_argument("--img_size", type=int, default=512)
p.add_argument("--cols", type=int, default=4)
p.add_argument("--rows", type=int, default=5)
p.add_argument("--out_width", type=int, default=1500)
p.add_argument("--pad", type=int, default=20)
args = p.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
seg = Segmenter(args.model, device="cpu", img_size=args.img_size, grid=(args.cols,args.rows))

for name in os.listdir(args.in_dir):
    if not name.lower().endswith((".jpg",".jpeg",".png")): continue
    bgr = cv2.imread(os.path.join(args.in_dir, name))
    if bgr is None: 
        print("skip:", name); continue
    _, fg = seg.predict_mask(bgr)
    rgba  = seg.cutout_rgba(bgr, fg)
    rb, rr, rg = seg.rectify(bgr, fg, out_width=args.out_width, pad=args.pad)

    stem = os.path.splitext(name)[0]
    cv2.imwrite(os.path.join(args.out_dir, f"{stem}_fgmask.png"), fg)
    cv2.imwrite(os.path.join(args.out_dir, f"{stem}_bg_removed.png"), rgba)
    if rr is not None:
        cv2.imwrite(os.path.join(args.out_dir, f"{stem}_rect_rgba.png"), rr)
        cv2.imwrite(os.path.join(args.out_dir, f"{stem}_rect_bgr.png"),  rb)
        cv2.imwrite(os.path.join(args.out_dir, f"{stem}_rect_grid.png"), rg)

print("Done →", args.out_dir)
