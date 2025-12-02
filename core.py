# core.py
import cv2, torch, numpy as np
from U2NET.model.u2net import U2NET

def letterbox(img, size=512):
    h,w = img.shape[:2]
    s = min(size/w, size/h); nw,nh = int(w*s), int(h*s)
    rsz = cv2.resize(img,(nw,nh))
    canvas = np.zeros((size,size,3),np.uint8)
    top,left = (size-nh)//2,(size-nw)//2
    canvas[top:top+nh, left:left+nw] = rsz
    return canvas,(top,left,nh,nw),(h,w)

def _order_pts(pts):
    pts = np.asarray(pts,np.float32)
    s=pts.sum(1); d=np.diff(pts,1).ravel()
    tl,br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr,bl = pts[np.argmin(d)], pts[np.argmax(d)]
    return np.array([tl,tr,br,bl],np.float32)

def _largest_cnt(mask):
    m = (mask>0).astype(np.uint8)
    k = np.ones((7,7),np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k)
    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts,key=cv2.contourArea) if cnts else None

def _quad_from_contour(cnt):
    hull = cv2.convexHull(cnt); peri = cv2.arcLength(hull, True)
    for frac in np.linspace(0.01,0.08,8):
        approx = cv2.approxPolyDP(hull, epsilon=frac*peri, closed=True)
        if len(approx)==4:
            return approx.reshape(-1,2).astype(np.float32)
    rect = cv2.minAreaRect(hull)
    return cv2.boxPoints(rect).astype(np.float32)

class Segmenter:
    def __init__(self, model_path, device="cpu", img_size=512, grid=(4,5)):
        self.device   = device
        self.img_size = img_size
        self.cols, self.rows = grid
        self.model = U2NET(in_ch=3, out_ch=3).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def predict_mask(self, bgr):
        x, meta, orig = letterbox(bgr, self.img_size)
        rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb.transpose(2,0,1))
        xt  = torch.from_numpy(rgb).float().unsqueeze(0)/255.0
        xt  = xt.to(self.device)
        out = self.model(xt)[0]  # [1,3,H,W]
        pred= torch.argmax(out,1)[0].cpu().numpy().astype(np.uint8)  # 0/1/2
        top,left,nh,nw = meta; H,W = orig
        un = np.zeros((self.img_size,self.img_size), np.uint8)
        un[top:top+nh, left:left+nw] = pred[top:top+nh, left:left+nw]
        un = cv2.resize(un,(W,H), interpolation=cv2.INTER_NEAREST)
        fg = np.isin(un,(1,2)).astype(np.uint8)*255
        return un, fg

    @staticmethod
    def cutout_rgba(img, mask):
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        rgba[:,:,3] = mask
        return rgba

    def rectify(self, image_bgr, fg_mask, out_width=1500, pad=20):
        cnt = _largest_cnt(fg_mask)
        if cnt is None or cv2.contourArea(cnt)<200:
            return None, None, None
        src = _order_pts(_quad_from_contour(cnt))
        W   = int(out_width + 2*pad)
        H   = int((self.rows/self.cols)*out_width + 2*pad)
        dst = np.array([[pad,pad],[W-pad-1,pad],[W-pad-1,H-pad-1],[pad,H-pad-1]], np.float32)
        M   = cv2.getPerspectiveTransform(src, dst)
        rect_bgr  = cv2.warpPerspective(image_bgr, M, (W,H), flags=cv2.INTER_LINEAR)
        rect_mask = cv2.warpPerspective(fg_mask,   M, (W,H), flags=cv2.INTER_NEAREST)
        rect_rgba = self.cutout_rgba(rect_bgr, (rect_mask>0).astype(np.uint8)*255)
        rect_grid = rect_bgr.copy()
        for c in range(1, self.cols):
            x = int(pad + c*(W-2*pad)/self.cols); cv2.line(rect_grid, (x,pad),(x,H-pad-1),(0,0,0),2)
        for r in range(1, self.rows):
            y = int(pad + r*(H-2*pad)/self.rows); cv2.line(rect_grid, (pad,y),(W-pad-1,y),(0,0,0),2)
        cv2.rectangle(rect_grid,(pad,pad),(W-pad-1,H-pad-1),(0,0,0),3)
        rect_bgr  = rect_bgr[int(0.02*H):, :]
        rect_rgba = rect_rgba[int(0.02*H):, :]
        rect_grid = rect_grid[int(0.02*H):, :]
        return rect_bgr, rect_rgba, rect_grid
