# debug_sanity.py
import os, json, cv2, torch, numpy as np
from torch.utils import data

# ถ้ารันจากโฟลเดอร์อื่น ให้แก้ BASE_DIR เป็นรากโปรเจกต์ที่มี cfg.json
BASE_DIR = os.path.abspath(".")
os.chdir(BASE_DIR)

# --------- Helper: list files (ใช้ใน DatasetUtility) ----------
def _list_files(folder):
    exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    return sorted(
        [os.path.join(folder, f) for f in os.listdir(folder)
         if f.lower().endswith(exts)]
    )
# --------------------------------------------------------------

# โหลด cfg
with open("cfg.json",'r') as f:
    cfg = json.load(f)

# เลือกชุดข้อมูลที่ต้องการทดสอบ (ให้ตรงกับที่คุณจะเทรนจริง)
DATASET_NAME = "DeepGlobe"          # "DeepGlobe" | "MassachusettsRoads" | "Spacenet"

# เอา mean BGR จาก cfg มาหักออก (จะได้กลับภาพเพื่อพรีวิวได้ถูก)
MEAN_BGR = np.array(eval(cfg["Datasets"][DATASET_NAME]["mean"]), np.float32)

# import DatasetUtility
from Tools import DatasetUtility

# สร้าง dataset + dataloader (batch=1) ฝั่ง train
ds = getattr(DatasetUtility, DATASET_NAME)(cfg, model_name="any",
                                          dataset_name=DATASET_NAME,
                                          loader_type="training_settings")
loader = data.DataLoader(ds, batch_size=1, shuffle=True)

# ดึง 5 ตัวอย่างแบบสุ่มมาตรวจ
N_SAMPLES = min(5, len(ds))

def denorm_bgr(img_chw):
    """รับภาพ torch [C,H,W] ที่ถูกหัก mean แล้ว -> คืน BGR uint8"""
    x = img_chw.numpy().transpose(1,2,0) + MEAN_BGR.reshape(1,1,3)
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x

def to_mask_2d(lbls):
    """
    lbls อาจเป็น:
      - list ของเทนเซอร์ [B,H,W] หลายสเกล -> เอาสเกลสุดท้าย index 0
      - เทนเซอร์เดียว [B,H,W] หรือ [H,W]
    คืนค่า mask 2D uint8 (0/255)
    """
    if isinstance(lbls, (list, tuple)):
        t = lbls[-1]              # tensor [B,H,W]
        if t.ndim == 3: t = t[0]  # [H,W]
        arr = t.detach().cpu().numpy()
    else:
        t = lbls
        if torch.is_tensor(t):
            if t.ndim == 3: t = t[0]
            arr = t.detach().cpu().numpy()
        else:
            # numpy
            arr = lbls
            if arr.ndim == 3: arr = arr[0]
    # ให้เป็น 0/1 แล้วสเกลเป็น 0/255
    arr = (arr > 0).astype(np.uint8) * 255
    return arr

os.makedirs("sanity_debug", exist_ok=True)

for idx, batch in enumerate(loader):
    if idx >= N_SAMPLES: break
    img, lbls, _ = batch  # img: [B,3,H,W], lbls: list of [B,H,W] หรือเหมือนกัน

    # เตรียมภาพ/แมสก์
    bgr = denorm_bgr(img[0])                 # HxWx3 BGR uint8
    mask = to_mask_2d(lbls)                  # HxW uint8 0/255
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # overlay (สีแดงโปร่ง)
    overlay = cv2.addWeighted(bgr, 1.0, np.where(mask_bgr>0, (0,0,255), 0).astype(np.uint8), 0.35, 0)

    # ต่อภาพให้เห็นชัด [ต้นฉบับ | GT | overlay]
    grid = np.concatenate([bgr, mask_bgr, overlay], axis=1)

    # ชื่อไฟล์ภาพต้นฉบับ (สำหรับดีบักชื่อ)
    try:
        pair = ds.imagelabeldict["training_settings"][idx]
        basename_img  = os.path.basename(pair["image"])
        basename_mask = os.path.basename(pair["label"])
    except Exception:
        basename_img, basename_mask = f"sample_{idx}.png", f"mask_{idx}.png"

    out_path = os.path.join("sanity_debug", f"sanity_{idx}_{basename_img}")
    cv2.imwrite(out_path, grid)
    print(f"[OK] saved -> {out_path} | pair: {basename_img} <-> {basename_mask}")

print("Done. เปิดโฟลเดอร์ sanity_debug แล้วดูภาพว่า GT ทาบถูกตำแหน่งหรือไม่.")
