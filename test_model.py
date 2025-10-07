# test_model.py
import os, sys, json, argparse, cv2, torch, numpy as np
import torch.nn.functional as F
from collections import OrderedDict

# ====== IMPORT โมเดลของคุณ ======
from Models.ConvNeXt_UPerNet_DGCN_MTL import ConvNeXt_UPerNet_DGCN_MTL


# ---------- Utils ----------
def safe_load_ckpt(path, map_location="cpu"):
    try:
        return torch.load(path, weights_only=True, map_location=map_location)
    except TypeError:
        return torch.load(path, map_location=map_location)

def load_cfg(cfg_path="cfg.json", dataset="DeepGlobe"):
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    mean_bgr = np.array(eval(cfg["Datasets"][dataset]["mean"]), dtype=np.float32)
    thr = float(cfg["training_settings"].get("road_binary_thresh", 0.5))
    return cfg, mean_bgr, thr

def load_model(weights, device):
    model = ConvNeXt_UPerNet_DGCN_MTL().to(device).eval()
    ckpt = safe_load_ckpt(weights, map_location=device)
    sd = ckpt.get("state_dict", ckpt)
    new_sd = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in sd.items())
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict: missing={list(missing)}, unexpected={list(unexpected)}")
    return model

def list_images(path):
    exts = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")
    if os.path.isdir(path):
        return [os.path.join(path, n) for n in sorted(os.listdir(path)) if n.lower().endswith(exts)]
    return [path] if path.lower().endswith(exts) else []

def preprocess(img_bgr, mean_bgr, size=None):
    # resize NxN หากกำหนด
    if size is not None:
        img_bgr = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    x = img_bgr.astype(np.float32) - mean_bgr.reshape(1,1,3)
    x = torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0)  # [1,3,H,W]
    return img_bgr, x

def infer_logits(model, x, device, fp16=False):
    with torch.inference_mode():
        if device.startswith("cuda") and fp16:
            from torch.cuda.amp import autocast
            with autocast(dtype=torch.float16):
                road_list, _ = model(x.to(device, non_blocking=True))
        else:
            road_list, _ = model(x.to(device, non_blocking=True))
    return road_list[-1]  # logits [1,C,H,W]

def logits_to_prob1_and_mask(logits, target_hw, thr):
    prob1 = F.softmax(logits, dim=1)[:, 1, :, :]                  # road prob
    if prob1.shape[-2:] != target_hw:
        prob1 = F.interpolate(prob1.unsqueeze(1), target_hw, mode="bilinear", align_corners=False).squeeze(1)
    prob1 = prob1.squeeze(0).float().cpu().numpy()                # HxW float
    pred01 = (prob1 >= float(thr)).astype(np.uint8)               # HxW 0/1
    return prob1, pred01

# ---------- Saver: Triptych (Original | Overlay | Road Mask) ----------
# แทนที่ 2 ฟังก์ชันเดิมด้วยเวอร์ชันนี้

def _panel_with_title(bgr, title, bar_h=90, font_scale=1.6, thick=3):
    """เติมแถบหัวเรื่องด้านบน พื้นขาว ตัวอักษรดำกึ่งกลาง"""
    h, w = bgr.shape[:2]
    bar = np.full((bar_h, w, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(title, font, font_scale, thick)
    x = max(10, (w - tw) // 2)
    y = (bar_h + th) // 2
    cv2.putText(bar, title, (x, y), font, font_scale, (0,0,0), thick, cv2.LINE_AA)
    return np.vstack([bar, bgr])

def save_triptych(original_bgr, pred01, out_dir, stem,
                  gap=40, margin=30, target_panel_height=720):
    """
    ทำภาพรวม 3 ช่อง: Original | Predicted Mask Overlay | Road Mask
    พร้อมหัวเรื่องบนพื้นขาว ขนาดเท่ากัน + ช่องว่างระหว่างภาพ
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- เตรียมภาพทั้ง 3 ---
    # 1) Original
    p_orig = original_bgr.copy()

    # 2) Overlay (โทนฟ้าโปร่ง)
    mask = (pred01.astype(np.uint8) * 255)
    overlay_color = np.zeros_like(original_bgr); overlay_color[...,0] = 255
    overlay = cv2.addWeighted(
        original_bgr, 1.0,
        np.where(mask[...,None] > 0, overlay_color, 0).astype(np.uint8),
        0.35, 0
    )
    p_overlay = overlay

    # 3) Road mask (ขาว/ดำ)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    p_mask = mask_bgr

    # --- ใส่หัวข้อ ---
    p1 = _panel_with_title(p_orig,    "Original Image")
    p2 = _panel_with_title(p_overlay, "Predicted Mask Overlay")
    p3 = _panel_with_title(p_mask,    "Road Mask")

    # --- ปรับความสูงให้เท่ากัน ---
    def rh(img, H=target_panel_height):
        if img.shape[0] == H:
            return img
        r = H / img.shape[0]
        return cv2.resize(img, (int(img.shape[1]*r), H), interpolation=cv2.INTER_LINEAR)

    p1, p2, p3 = rh(p1), rh(p2), rh(p3)

    # --- สร้างแคนวาสพื้นขาว + margin + gap ระหว่างพาเนล ---
    H = max(p1.shape[0], p2.shape[0], p3.shape[0])
    W = p1.shape[1] + p2.shape[1] + p3.shape[1] + 2*gap + 2*margin
    canvas = np.full((H + 2*margin, W, 3), 255, dtype=np.uint8)

    x = margin
    canvas[margin:margin+p1.shape[0], x:x+p1.shape[1]] = p1; x += p1.shape[1] + gap
    canvas[margin:margin+p2.shape[0], x:x+p2.shape[1]] = p2; x += p2.shape[1] + gap
    canvas[margin:margin+p3.shape[0], x:x+p3.shape[1]] = p3

    out_grid = os.path.join(out_dir, f"{stem}_triptych.png")
    cv2.imwrite(out_grid, canvas)
    # เก็บไฟล์แยกด้วย เผื่อใช้งานต่อ
    cv2.imwrite(os.path.join(out_dir, f"{stem}_mask.png"), mask)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_overlay.png"), overlay)
    print(f"[OK] saved -> {out_grid}")


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True, help="ไฟล์รูปเดี่ยว หรือ โฟลเดอร์รูป")
    ap.add_argument("--output", default="inference_out")
    ap.add_argument("--cfg", default="cfg.json")
    ap.add_argument("--dataset", default="DeepGlobe")
    ap.add_argument("--size", type=int, default=512, help="resize NxN ก่อน infer (ลด VRAM)")
    ap.add_argument("--fp16", action="store_true", help="ใช้ half precision บน CUDA")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.device.startswith("cuda"):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # เตรียมรายชื่อรูป
    in_path = args.input if os.path.isabs(args.input) else os.path.abspath(args.input)
    imgs = list_images(in_path)
    if not imgs:
        raise FileNotFoundError(f"No images found under: {in_path}")

    print(f"[info] Loading weights from {args.weights}")
    cfg, mean_bgr, thr = load_cfg(args.cfg, args.dataset)
    model = load_model(args.weights, args.device)

    os.makedirs(args.output, exist_ok=True)
    for ip in imgs:
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] cannot read: {ip}")
            continue

        img_r, x = preprocess(img, mean_bgr, size=args.size)
        logits = infer_logits(model, x, args.device, fp16=args.fp16)
        prob1, pred01 = logits_to_prob1_and_mask(logits, img_r.shape[:2], thr)

        stem = os.path.splitext(os.path.basename(ip))[0]
        save_triptych(img_r, pred01, args.output, stem)

        # เคลียร์ VRAM ต่อรูป (กัน fragmentation)
        del x, logits
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

