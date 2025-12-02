import os, cv2, torch, numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm

from model import U2NET  # repo import

# ====== EDIT: your dataset root ======
DATA_ROOT = "F:\Solar_Panel_detection.v6i.coco-segmentation\Solar_Panel_detection.v6i.coco-segmentation"
IMG_SIZE  = 512
BATCH     = 4
EPOCHS    = 30
LR        = 5e-5
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR   = "weights_mc"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Dataset ---
class PanelsMultiDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=512, augment=False):
        self.img_dir, self.mask_dir = img_dir, mask_dir
        self.names = [n for n in os.listdir(img_dir) if n.lower().endswith((".jpg",".jpeg",".png"))]
        self.size = size
        self.augment = augment
    def __len__(self): return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        img = cv2.imread(os.path.join(self.img_dir, name))  # BGR
        h, w = img.shape[:2]
        scale = min(self.size / w, self.size / h)
        nw, nh = int(w * scale), int(h * scale)
        img_res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.size, self.size, 3), np.uint8)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas[top:top + nh, left:left + nw] = img_res

        # ✅ no negative strides
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(np.ascontiguousarray(rgb).transpose(2, 0, 1)).float() / 255.0

        stem = os.path.splitext(name)[0] + ".png"
        mask = cv2.imread(os.path.join(self.mask_dir, stem), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(os.path.join(self.mask_dir, stem))
        mask_res = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        mask_canvas = np.zeros((self.size, self.size), np.uint8)
        mask_canvas[top:top + nh, left:left + nw] = mask_res
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_canvas)).long()

        return img_t, mask_t, name


def get_loaders(root=DATA_ROOT):
    tr = PanelsMultiDataset(os.path.join(root,"train"), os.path.join(root,"train/masks_bin"), IMG_SIZE, augment=True)
    va = PanelsMultiDataset(os.path.join(root,"valid"), os.path.join(root,"valid/masks_bin"), IMG_SIZE, augment=False)
    return DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True), \
           DataLoader(va, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

# --- Build model (3 classes) & load pretrained (skip last layer) ---
def build_model():
    net = U2NET(in_ch=3, out_ch=3).to(DEVICE)  # 3-class
    # load pretrained binary weights
    sd_pre = torch.load("F:\saved_models\saved_models/u2net/u2net.pth", map_location="cpu")
    sd_new = net.state_dict()
    kept = 0
    for k,v in sd_pre.items():
        if k in sd_new and v.shape == sd_new[k].shape:
            sd_new[k] = v; kept += 1
    net.load_state_dict(sd_new)
    print(f"Loaded {kept} layers from pretrained u2net.pth (skipped last classifier).")
    return net

# --- Train ---
def train_epoch(net, loader, criterion, opt):
    net.train(); total=0
    for x,y,_ in tqdm(loader, desc="train", leave=False):
        x,y = x.to(DEVICE), y.to(DEVICE)
        outs = net(x)                   # U2NET returns list of logits; take highest-res
        logits = outs[0] if isinstance(outs,(list,tuple)) else outs  # [B,3,H,W]
        loss = criterion(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()*x.size(0)
    return total/len(loader.dataset)

@torch.no_grad()
def valid_epoch(net, loader, criterion):
    net.eval(); total=0
    for x,y,_ in tqdm(loader, desc="valid", leave=False):
        x,y = x.to(DEVICE), y.to(DEVICE)
        outs = net(x)
        logits = outs[0] if isinstance(outs,(list,tuple)) else outs
        loss = criterion(logits, y)
        total += loss.item()*x.size(0)
    return total/len(loader.dataset)

def main():
    tr, va = get_loaders()
    net = build_model()
    # (Optional) freeze early layers:
    # for n,p in net.named_parameters():
    #     if "outconv" not in n: p.requires_grad = True  # set False to freeze
    crit = nn.CrossEntropyLoss()
    opt  = torch.optim.AdamW(net.parameters(), lr=LR)
    best = 1e9
    for e in range(1, EPOCHS+1):
        tr_loss = train_epoch(net, tr, crit, opt)
        va_loss = valid_epoch(net, va, crit)
        print(f"Epoch {e}: train {tr_loss:.4f} | valid {va_loss:.4f}")
        if va_loss < best:
            best = va_loss
            torch.save(net.state_dict(), os.path.join(OUT_DIR, "u2net_multiclass.pt"))
            print("✓ Saved", os.path.join(OUT_DIR, "u2net_multiclass.pt"))

if __name__ == "__main__":
    main()
