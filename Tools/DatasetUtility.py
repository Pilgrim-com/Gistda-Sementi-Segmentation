# Tools/DatasetUtility.py
import collections
import math
import os
import random
import re
import cv2
import numpy as np
import torch
from torch.utils import data
from pathlib import Path
from skimage.morphology import skeletonize

import Tools.sknw as sknw
import Tools.LineSimplification as LineSimp
import Tools.LineConversion as LineConv
import Tools.LineDataExtraction as LineData


# ---------- helpers ----------
def _list_files(folder, exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
    folder = str(folder)
    paths = []
    for e in exts:
        paths.extend([str(p) for p in Path(folder).glob(f"*{e}")])
    return sorted(paths)

def _stem_no_suffix(path_or_name: str):
    s = Path(path_or_name).stem
    # รองรับ suffix ยอดฮิตของแมสก์
    for suf in ["_mask", "_masks", "_label", "_labels", "_road", "_roads", "_gt", "_binary", "-label", "-mask"]:
        if s.endswith(suf):
            return s[:-len(suf)]
    return s

def _digits_key(name: str):
    # ดึง "ตัวเลขติดกันทั้งหมด" จากชื่อไฟล์ไว้เป็นคีย์ เช่น img_000123.png -> "000123"
    m = re.findall(r"\d+", Path(name).stem)
    return "".join(m) if m else ""

def _resolve_path(base_dir: str, p: str) -> str:
    """
    ถ้า p เป็น absolute และมีจริง -> ใช้ตามนั้น
    ถ้า p เป็น absolute แต่ 'ไม่มีจริง' -> ตีความเป็นพาธใต้โปรเจกต์: <BASE_DIR>/<p ที่ตัด / ทิ้ง>
    ถ้า p เป็น relative -> รวมกับ base_dir
    """
    p = p.strip()
    if os.path.isabs(p):
        if os.path.exists(p):
            return p
        cand = os.path.join(base_dir, p.lstrip(os.sep))
        return os.path.abspath(cand)
    return os.path.abspath(os.path.join(base_dir, p))


def _make_pairs(imgs, lbls, dataset_name):
    """
    จับคู่ภาพ-แมสก์ด้วยหลายกลยุทธ์: stem -> digits -> substring -> order
    คืนค่า: list[{"image": <path>, "label": <path>}]
    """
    # 1) stem หลังตัด suffix
    img_map = {}
    for p in imgs:
        k = _stem_no_suffix(p)
        img_map.setdefault(k, []).append(p)

    lbl_map = {}
    for p in lbls:
        k = _stem_no_suffix(p)
        lbl_map.setdefault(k, []).append(p)

    pairs = []
    stems_inter = sorted(set(img_map.keys()) & set(lbl_map.keys()))
    if len(stems_inter) > 0:
        for k in stems_inter:
            # ถ้ามีหลายไฟล์ต่อ stem เลือกอันแรก (และเตือน)
            if len(img_map[k]) > 1 or len(lbl_map[k]) > 1:
                print(f"[WARN] หลายไฟล์มี stem เดียวกัน: stem={k} imgs={len(img_map[k])}, lbls={len(lbl_map[k])} -> เลือกอันแรก")
            pairs.append({"image": img_map[k][0], "label": lbl_map[k][0]})
        return pairs, "stem"

    # 2) digits key (เลขในชื่อไฟล์)
    img_d = {}
    for p in imgs:
        k = _digits_key(p)
        if k:
            img_d.setdefault(k, []).append(p)
    lbl_d = {}
    for p in lbls:
        k = _digits_key(p)
        if k:
            lbl_d.setdefault(k, []).append(p)

    dkeys = sorted(set(img_d.keys()) & set(lbl_d.keys()))
    if len(dkeys) > 0:
        for k in dkeys:
            if len(img_d[k]) > 1 or len(lbl_d[k]) > 1:
                print(f"[WARN] หลายไฟล์มี digits เดียวกัน: key={k} imgs={len(img_d[k])}, lbls={len(lbl_d[k])} -> เลือกอันแรก")
            pairs.append({"image": img_d[k][0], "label": lbl_d[k][0]})
        if len(pairs) > 0:
            return pairs, "digits"

    # 3) substring matching (ชื่อหนึ่งเป็นส่วนของอีกชื่อหนึ่ง)
    lbl_stems = [(_stem_no_suffix(p), p) for p in lbls]
    made = 0
    for ip in imgs:
        si = _stem_no_suffix(ip)
        cands = [lp for (ls, lp) in lbl_stems if (si in ls) or (ls in si)]
        if len(cands) == 1:
            pairs.append({"image": ip, "label": cands[0]})
            made += 1
    if made > 0:
        return pairs, "substring"

    # 4) สุดท้ายจริง ๆ: จับคู่ตามลำดับ (ต้องจำนวนเท่ากัน)
    if len(imgs) == len(lbls) and len(imgs) > 0:
        print("[WARN] จับคู่ด้วยการ 'เรียงตามลำดับ' — ควรตรวจ overlay ให้ชัวร์ว่าถูกต้อง")
        for a, b in zip(sorted(imgs), sorted(lbls)):
            pairs.append({"image": a, "label": b})
        return pairs, "order"

    # ไม่เจอทางจับคู่
    return [], None


# ---------- base dataset ----------
class DatasetPreprocessor(data.Dataset):
    def __init__(self, cfg, model_name, dataset_name, loader_type):
        cv2.setNumThreads(0)
        self.cfg = cfg
        self.GraphParameters = [eval(self.cfg["Models"]["scales"]),
                                eval(self.cfg["Models"]["smooth"])]
        np.random.seed(self.cfg["GlobalSeed"])
        torch.manual_seed(self.cfg["GlobalSeed"])
        random.seed(self.cfg["GlobalSeed"])

        self.model_name   = model_name
        self.dataset_name = dataset_name
        self.loader_type  = loader_type
        self.augment      = self.cfg[self.loader_type]["augment"]

        # ===== base dir ของโปรเจกต์ (โฟลเดอร์ที่มี Tools/, Models/, cfg.json) =====
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        key_img, key_lbl = ("train_dir","train_label_dir") if self.loader_type == "training_settings" \
                           else ("valid_dir","valid_label_dir")
        img_dir_cfg = self.cfg["Datasets"][dataset_name][key_img]
        lbl_dir_cfg = self.cfg["Datasets"][dataset_name][key_lbl]
        img_dir = _resolve_path(BASE_DIR, img_dir_cfg)
        lbl_dir = _resolve_path(BASE_DIR, lbl_dir_cfg)

        imgs = _list_files(img_dir)
        lbls = _list_files(lbl_dir)

        # จับคู่ด้วยหลายกลยุทธ์
        pairs, method = _make_pairs(imgs, lbls, dataset_name)

        print(f"[DBG] img_dir={img_dir}, lbl_dir={lbl_dir}")
        print(f"[DBG] n_imgs={len(imgs)}, n_lbls={len(lbls)}, n_pairs={len(pairs)}  (method={method})")
        if len(pairs) > 0:
            sp = pairs[0]
            print("[DBG] sample pair:", os.path.basename(sp['image']), "<->", os.path.basename(sp['label']))

        assert len(pairs) > 0, f"ไม่พบคู่ภาพ/แมสก์ใน {img_dir} และ {lbl_dir}"
        self.imagelabeldict = collections.defaultdict(list)
        self.imagelabeldict[self.loader_type] = pairs

        self.dataset_mean_color = np.array(
            eval(self.cfg["Datasets"][dataset_name]["mean"]), dtype=np.float32
        )

        self.crop_size = self.cfg[self.loader_type]["crop_size"]
        if (self.loader_type == "validation_settings") and (dataset_name == "Spacenet"):
            self.crop_size = self.cfg[self.loader_type]["spacenet_crop_size"]

    def __len__(self):
        return len(self.imagelabeldict[self.loader_type])

    def Preprocess(self, index):
        pair = self.imagelabeldict[self.loader_type][index]
        image = cv2.imread(pair["image"], cv2.IMREAD_COLOR)         # BGR [H,W,3]
        label = cv2.imread(pair["label"], cv2.IMREAD_GRAYSCALE)     # [H,W] (0..255)

        if image is None:
            raise FileNotFoundError(f"อ่านรูปไม่ได้: {pair['image']}")
        if label is None:
            raise FileNotFoundError(f"อ่านแมสก์ไม่ได้: {pair['label']}")

        cropsize = self.crop_size

        # ---- train: random crop; val/test: resize square ----
        if self.loader_type == "training_settings":
            H, W = image.shape[:2]
            if (H >= cropsize) and (W >= cropsize):
                y0 = np.random.randint(0, H - cropsize + 1)
                x0 = np.random.randint(0, W - cropsize + 1)
                image = image[y0:y0 + cropsize, x0:x0 + cropsize, :]
                label = label[y0:y0 + cropsize, x0:x0 + cropsize]
        else:
            image = cv2.resize(image, (cropsize, cropsize), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (cropsize, cropsize), interpolation=cv2.INTER_NEAREST)

        # ---- augmentation (sync image & mask) ----
        if self.augment and (self.loader_type == "training_settings"):
            # geometric (sync)
            if random.random() < 0.5:   # flip ซ้าย-ขวา
                image = np.ascontiguousarray(image[:, ::-1, :])
                label = np.ascontiguousarray(label[:, ::-1])
            if random.random() < 0.5:   # flip บน-ล่าง
                image = np.ascontiguousarray(image[::-1, :, :])
                label = np.ascontiguousarray(label[::-1, :])

            # ENHANCED: More aggressive rotation (not just 90° steps)
            if random.random() < 0.7:  # increased probability
                rot_k = random.randint(0, 3)  # 0/90/180/270
                if rot_k:
                    image = np.ascontiguousarray(np.rot90(image, rot_k))
                    label = np.ascontiguousarray(np.rot90(label, rot_k))

            # ENHANCED: Random rotation with small angles (-15 to +15 degrees)
            if random.random() < 0.4:
                angle = random.uniform(-15, 15)
                H, W = image.shape[:2]
                center = (W // 2, H // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                label = cv2.warpAffine(label, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

            # ENHANCED: Random scaling (zoom in/out)
            if random.random() < 0.5:
                scale = random.uniform(0.75, 1.25)
                H, W = image.shape[:2]
                new_H, new_W = int(H * scale), int(W * scale)
                image_scaled = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
                label_scaled = cv2.resize(label, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
                # Center crop or pad to original size
                if scale > 1.0:  # crop
                    y0 = (new_H - H) // 2
                    x0 = (new_W - W) // 2
                    image = image_scaled[y0:y0+H, x0:x0+W]
                    label = label_scaled[y0:y0+H, x0:x0+W]
                else:  # pad
                    pad_h = (H - new_H) // 2
                    pad_w = (W - new_W) // 2
                    image = np.pad(image_scaled, ((pad_h, H-new_H-pad_h), (pad_w, W-new_W-pad_w), (0,0)), mode='reflect')
                    label = np.pad(label_scaled, ((pad_h, H-new_H-pad_h), (pad_w, W-new_W-pad_w)), mode='reflect')

            # photometric (เฉพาะภาพ)
            # ENHANCED: More aggressive brightness/contrast (increased prob)
            if random.random() < 0.7:  # increased from 0.5
                alpha = 1.0 + random.uniform(-0.3, 0.3)   # contrast (increased range)
                beta  = random.uniform(-30, 30)           # brightness (increased range)
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

            # ENHANCED: More aggressive HSV jitter
            if random.random() < 0.5:  # increased from 0.3
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
                hsv[..., 0] = (hsv[..., 0] + random.randint(-15, 15)) % 180  # hue (increased)
                hsv[..., 1] = np.clip(hsv[..., 1] + random.randint(-40, 40), 0, 255)  # saturation (increased)
                hsv[..., 2] = np.clip(hsv[..., 2] + random.randint(-40, 40), 0, 255)  # value (increased)
                image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # ENHANCED: Gaussian blur (increased prob)
            if random.random() < 0.3:  # increased from 0.2
                k = random.choice([3, 5, 7])  # added kernel size 7
                image = cv2.GaussianBlur(image, (k, k), 0)

            # ENHANCED: Gaussian noise
            if random.random() < 0.3:
                noise = np.random.normal(0, random.uniform(5, 15), image.shape).astype(np.float32)
                image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            # Random erasing (สี่เหลี่ยมเล็ก ๆ) เติมด้วย mean color
            if random.random() < 0.25:
                H, W = image.shape[:2]
                er_h = max(8, int(H * random.uniform(0.03, 0.08)))
                er_w = max(8, int(W * random.uniform(0.03, 0.08)))
                y0 = random.randint(0, max(0, H - er_h))
                x0 = random.randint(0, max(0, W - er_w))
                mean_bgr = self.dataset_mean_color.astype(np.uint8)
                image[y0:y0+er_h, x0:x0+er_w, :] = mean_bgr

            # ENHANCED: Elastic deformation (helpful for roads)
            if random.random() < 0.3:
                H, W = image.shape[:2]
                alpha = random.uniform(25, 35)  # strength of deformation
                sigma = random.uniform(3, 5)     # smoothness
                random_state = np.random.RandomState(None)

                dx = cv2.GaussianBlur((random_state.rand(H, W) * 2 - 1), (0, 0), sigma) * alpha
                dy = cv2.GaussianBlur((random_state.rand(H, W) * 2 - 1), (0, 0), sigma) * alpha

                x, y = np.meshgrid(np.arange(W), np.arange(H))
                map_x = np.float32(x + dx)
                map_y = np.float32(y + dy)

                image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                label = cv2.remap(label, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

            # ENHANCED: Grid distortion (useful for satellite imagery perspective variations)
            if random.random() < 0.25:
                H, W = image.shape[:2]
                num_steps = 5
                distort_limit = random.uniform(0.2, 0.3)

                x_step = W // num_steps
                y_step = H // num_steps

                xx = np.zeros((num_steps + 1, num_steps + 1), dtype=np.float32)
                yy = np.zeros((num_steps + 1, num_steps + 1), dtype=np.float32)

                for i in range(num_steps + 1):
                    for j in range(num_steps + 1):
                        xx[i, j] = j * x_step + random.uniform(-distort_limit * x_step, distort_limit * x_step)
                        yy[i, j] = i * y_step + random.uniform(-distort_limit * y_step, distort_limit * y_step)

                xx = np.clip(xx, 0, W - 1)
                yy = np.clip(yy, 0, H - 1)

                map_x = cv2.resize(xx, (W, H), interpolation=cv2.INTER_LINEAR)
                map_y = cv2.resize(yy, (W, H), interpolation=cv2.INTER_LINEAR)

                image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                label = cv2.remap(label, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

            # ENHANCED: Random shadows (simulate clouds/time of day variations)
            if random.random() < 0.2:
                H, W = image.shape[:2]
                # Create random shadow mask
                num_shadows = random.randint(1, 3)
                shadow_mask = np.ones((H, W), dtype=np.float32)

                for _ in range(num_shadows):
                    # Random ellipse shadow
                    center = (random.randint(0, W), random.randint(0, H))
                    axes = (random.randint(W//6, W//3), random.randint(H//6, H//3))
                    angle = random.randint(0, 180)
                    cv2.ellipse(shadow_mask, center, axes, angle, 0, 360,
                               random.uniform(0.4, 0.7), -1)

                # Apply shadow to image
                shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
                image = (image * shadow_mask[:, :, np.newaxis]).astype(np.uint8)

        # ---- binarize mask -> 0/1 ----
        label = (label > 127).astype(np.uint8)

        # ---- normalize (BGR mean จาก cfg) ----
        image = image.astype(np.float32)
        image -= self.dataset_mean_color
        image = torch.from_numpy(image.transpose(2, 0, 1))  # [C,H,W] float32

        return image, label  # label: np.uint8 [H,W] (0/1)

    def CalculateAnglesFromVectorMap(self, keypoints, height, width):
        # คืนเป็น torch.LongTensor ของ index มุม (0..36)
        _, scaled_orientation_angles = LineData.getVectorMapsAngles(
            (height, width), keypoints, theta=10, bin_size=10
        )
        return torch.from_numpy(scaled_orientation_angles)


# ---------- datasets ----------
class DeepGlobe(DatasetPreprocessor):
    def __init__(self, cfg, model_name, dataset_name, loader_type):
        super().__init__(cfg, model_name, "DeepGlobe", loader_type)

    def __getitem__(self, index):
        image, label = self.Preprocess(index)              # image: [C,H,W] torch.float32, label: [H,W] uint8 (0/1)
        scaled_labels, scaled_orient = [], []

        _, H, W = image.shape
        for i, scale in enumerate(self.GraphParameters[0]):
            h = int(math.ceil(H / (scale * 1.0)))
            w = int(math.ceil(W / (scale * 1.0)))

            if scale != 1:
                # cv2.resize: (width, height)
                lbl = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                lbl = label.copy()

            lbl01 = (lbl > 0).astype(np.uint8)          # 0/1
            scaled_labels.append(torch.from_numpy(lbl01))  # tensor [h,w] uint8

            # สร้างกราฟจาก skeleton แล้วคำนวณ bin ของมุม
            skel = skeletonize(lbl01.astype(bool)).astype(np.uint16)
            graph = sknw.build_sknw(skel, multi=True)
            road_segments = []
            for (u, v) in graph.edges():
                for _, dat in graph[u][v].items():
                    pts = dat["pts"]
                    seg = np.row_stack([graph.nodes[u]["o"], pts, graph.nodes[v]["o"]])
                    seg = LineSimp.Ramer_Douglas_Peucker(seg.tolist(), self.GraphParameters[1][i])
                    road_segments.append(seg)
            keypoints = LineConv.Graph_to_Keypoints(road_segments)
            scaled_orient.append(self.CalculateAnglesFromVectorMap(keypoints, h, w))  # torch.long [h,w]

        return image, scaled_labels, scaled_orient


class MassachusettsRoads(DeepGlobe):
    def __init__(self, cfg, model_name, dataset_name, loader_type):
        super().__init__(cfg, model_name, "MassachusettsRoads", loader_type)


class Spacenet(DatasetPreprocessor):
    def __init__(self, cfg, model_name, dataset_name, loader_type):
        super().__init__(cfg, model_name, "Spacenet", loader_type)
        self.threshold = self.cfg["training_settings"]["road_binary_thresh"]

    def __getitem__(self, index):
        image, label = self.Preprocess(index)
        scaled_labels, scaled_orient = [], []

        _, H, W = image.shape
        for i, scale in enumerate(self.GraphParameters[0]):
            h = int(math.ceil(H / (scale * 1.0)))
            w = int(math.ceil(W / (scale * 1.0)))

            if scale != 1:
                lbl = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                lbl = label.copy()

            lbl01 = (lbl > 0).astype(np.uint8)
            scaled_labels.append(torch.from_numpy(lbl01))

            skel = skeletonize(lbl01.astype(bool)).astype(np.uint16)
            graph = sknw.build_sknw(skel, multi=True)
            road_segments = []
            for (u, v) in graph.edges():
                for _, dat in graph[u][v].items():
                    pts = dat["pts"]
                    seg = np.row_stack([graph.nodes[u]["o"], pts, graph.nodes[v]["o"]])
                    seg = LineSimp.Ramer_Douglas_Peucker(seg.tolist(), self.GraphParameters[1][i])
                    road_segments.append(seg)
            keypoints = LineConv.Graph_to_Keypoints(road_segments)
            scaled_orient.append(self.CalculateAnglesFromVectorMap(keypoints, h, w))

        return image, scaled_labels, scaled_orient