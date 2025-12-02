import os, cv2, torch, numpy as np
from model import U2NET  # your U²-Net model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Paths ===
CKPT = "C:/Users/vinni\Downloads\weights_mc-20250922T085355Z-1-001\weights_mc/u2net_multiclass.pt"  # your trained model
IMG_DIR = "C:/Users/vinni/Downloads/images"
OUTDIR = "C:/Users/vinni/Downloads/images"
os.makedirs(OUTDIR, exist_ok=True)


def letterbox(img, size=512):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    rsz = cv2.resize(img, (nw, nh))
    canvas = np.zeros((size, size, 3), np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = rsz
    return canvas, (top, left, nh, nw), (h, w)


@torch.no_grad()
def predict_one(model, path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    x, meta, orig = letterbox(bgr, 512)

    # BGR->RGB without negative strides, then make contiguous before from_numpy
    rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb.transpose(2, 0, 1))  # CHW, contiguous

    x_t = torch.from_numpy(rgb).float().unsqueeze(0) / 255.0
    x_t = x_t.to(DEVICE)

    out = model(x_t)[0]  # [1,3,512,512]
    pred = torch.argmax(out, dim=1)[0].cpu().numpy().astype(np.uint8)  # 0/1/2

    # unletterbox back to original size
    top, left, nh, nw = meta
    H, W = orig
    un = np.zeros((512, 512), np.uint8)
    un[top:top + nh, left:left + nw] = pred[top:top + nh, left:left + nw]
    un = cv2.resize(un, (W, H), interpolation=cv2.INTER_NEAREST)

    # merge classes: foreground = {1,2}
    fg_mask = np.isin(un, (1, 2)).astype(np.uint8) * 255
    return bgr, fg_mask


import cv2, numpy as np

def _order_pts(pts):
    pts = np.asarray(pts, np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl,tr,br,bl], np.float32)

def _largest_cnt(mask):
    m = (mask > 0).astype(np.uint8)
    # Close small gaps, then remove thin regions
    kernel = np.ones((7,7), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)  # remove noise
    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    return max(cnts, key=cv2.contourArea)


def _quad_from_contour(cnt):
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for frac in np.linspace(0.01, 0.08, 8):
        approx = cv2.approxPolyDP(hull, epsilon=frac*peri, closed=True)
        if len(approx) == 4:
            return approx.reshape(-1,2).astype(np.float32)
    rect = cv2.minAreaRect(hull)
    box  = cv2.boxPoints(rect).astype(np.float32)
    return box

def crop_top(image, ratio=0.05):
    H = image.shape[0]
    cut = int(H * ratio)
    return image[cut:, :]


def warp_array_to_grid(image_bgr, fg_mask, cols=4, rows=5, out_width=1200, pad=0):
    cnt = _largest_cnt(fg_mask)
    if cnt is None or cv2.contourArea(cnt) < 200:
        return None, None, None

    src_quad = _quad_from_contour(cnt)
    src_quad = _order_pts(src_quad)

    W = int(out_width + 2*pad)
    H = int((rows/cols)*out_width + 2*pad)

    dst_quad = np.array([[pad, pad],
                         [W-pad-1, pad],
                         [W-pad-1, H-pad-1],
                         [pad, H-pad-1]], np.float32)

    M = cv2.getPerspectiveTransform(src_quad, dst_quad)

    # Warp image and mask
    rect_bgr  = cv2.warpPerspective(image_bgr, M, (W, H), flags=cv2.INTER_LINEAR)
    rect_mask = cv2.warpPerspective(fg_mask,   M, (W, H), flags=cv2.INTER_NEAREST)

    # ---- FIXED PART ----
    rect_rgba = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2BGRA)
    rect_rgba[:,:,3] = (rect_mask > 0).astype(np.uint8) * 255   # enforce transparency
    # --------------------

    # Grid overlay (optional)
    rect_grid = rect_bgr.copy()
    for c in range(1, cols):
        x = int(pad + c*(W-2*pad)/cols)
        cv2.line(rect_grid, (x,pad), (x,H-pad-1), (0,0,0), 2)
    for r in range(1, rows):
        y = int(pad + r*(H-2*pad)/rows)
        cv2.line(rect_grid, (pad,y), (W-pad-1,y), (0,0,0), 2)
    cv2.rectangle(rect_grid, (pad,pad), (W-pad-1,H-pad-1), (0,0,0), 3)
    rect_bgr = crop_top(rect_bgr, ratio=0.02)
    rect_rgba = crop_top(rect_rgba, ratio=0.02)
    rect_grid = crop_top(rect_grid, ratio=0.02)

    return rect_bgr, rect_rgba, rect_grid




def cutout_rgba(img, mask):
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask  # keep panels/arrays visible, background transparent
    return rgba


if __name__ == "__main__":
    net = U2NET(in_ch=3, out_ch=3).to(DEVICE)
    net.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    net.eval()

    for name in os.listdir(IMG_DIR):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(IMG_DIR, name)

        img, mask = predict_one(net, img_path)

    # Save binary mask (foreground merged)
        stem = name.rsplit('.', 1)[0]
        cv2.imwrite(os.path.join(OUTDIR, f"{stem}_fgmask.png"), mask)

    # Save RGBA cutout (background removed)
        rgba = cutout_rgba(img, mask)
        cv2.imwrite(os.path.join(OUTDIR, f"{stem}_bg_removed.png"), rgba)

    # --- NEW: Rectify the foreground (deskew to front view) ---
    # If you know grid: cols=6, rows=3; otherwise omit to preserve aspect
        rect_bgr, rect_rgba, rect_grid = warp_array_to_grid(img, mask, cols=4, rows=5, out_width=1500, pad=20)

        if rect_bgr is not None:
           cv2.imwrite(os.path.join(OUTDIR, f"{stem}_rect_bgr.png"), rect_bgr)
        if rect_rgba is not None:
           cv2.imwrite(os.path.join(OUTDIR, f"{stem}_rect_rgba.png"), rect_rgba)

    print("✓ Background removal + rectification →", OUTDIR)


