# -*- coding: utf-8 -*-
import os
import sys
import math
import time
import json
import numpy as np
from osgeo import gdal
import random
import argparse
from Models import ConvNeXt_UPerNet_DGCN_MTL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm
from Tools import DatasetUtility
from Tools import Losses
from Tools import util
from Tools import viz_util
tqdm.monitor_interval = 0
import cv2
from skimage import io
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---------------------- Hyper params ----------------------
ANGLE_LAMBDA = 0.2      # Weight for orientation loss compared to road loss
ROAD_CLASS_INDEX = 1    # Index of road class in logits (usually = 1)
# ----------------------------------------------------------

def _shape_match_argmax(logits, target_hw):
    """argmax แล้วปรับขนาดให้เท่ากับ target_hw (H,W)"""
    pred = logits.argmax(1)  # [N,Hp,Wp]
    if pred.shape[-2:] != target_hw:
        pred = F.interpolate(pred.float().unsqueeze(1),
                             target_hw, mode='nearest').squeeze(1).long()
    return pred

def _shape_match_prob1(logits, target_hw):
    """softmax channel=1 (road prob) แล้วปรับขนาดให้เท่ากับ target_hw"""
    prob1 = F.softmax(logits, dim=1)[:, ROAD_CLASS_INDEX, :, :]
    if prob1.shape[-2:] != target_hw:
        prob1 = F.interpolate(prob1.unsqueeze(1), target_hw,
                              mode='bilinear', align_corners=False).squeeze(1)
    return prob1

def train_model(model, nGPUs, cfg, train_loss_file, valid_loss_file, train_loss_angle_file, valid_loss_angle_file,
                train_loader, valid_loader, Optimizer, segmentation_loss, orientation_loss,
                LR_scheduler, Epoch, resume_at_epoch, n_RoadClasses, n_OrientClasses):

    model.train()
    train_loss_road = 0.0
    train_loss_angle = 0.0
    hist = torch.zeros((n_RoadClasses, n_RoadClasses)).cuda()
    hist_angles = torch.zeros((n_OrientClasses, n_OrientClasses)).cuda()

    # grad accumulation ให้สอดคล้องกับ iteration_frequency
    accum_steps = max(1, int(cfg["training_settings"].get("iteration_frequency", 1)))
    Optimizer.zero_grad(set_to_none=True)

    for i, ImageLabelData in enumerate(train_loader, 0):
        imageBGRcropped, scaled_target_road_label, scaled_target_orientation_class = ImageLabelData

        # inputs
        imageBGRcropped = imageBGRcropped.float().cuda(non_blocking=True)

        # GT: ถนนเป็นคลาส 0/1 (long), มุมเป็น index (long)
        scaled_target_road_label = [(_label > 0).long().cuda(non_blocking=True) for _label in scaled_target_road_label]
        scaled_target_orientation_class = [_label.long().cuda(non_blocking=True) for _label in scaled_target_orientation_class]

        # forward
        pred_road_list, pred_orient_list = model(imageBGRcropped)

        # Auto-resize if mismatch (Fix for LinkNet/others outputting 511x511 instead of 512x512)
        # Also handles Deep Supervision mismatch (Model outputs > Dataset labels)
        for k in range(len(pred_road_list)):
            # Use the last available label if we don't have enough scales
            label_idx = k if k < len(scaled_target_road_label) else -1
            target_shape = scaled_target_road_label[label_idx].shape[-2:]
            
            if pred_road_list[k].shape[-2:] != target_shape:
                pred_road_list[k] = F.interpolate(pred_road_list[k], size=target_shape, mode='bilinear', align_corners=False)

        for k in range(len(pred_orient_list)):
            label_idx = k if k < len(scaled_target_orientation_class) else -1
            target_shape = scaled_target_orientation_class[label_idx].shape[-2:]
            
            if pred_orient_list[k].shape[-2:] != target_shape:
                pred_orient_list[k] = F.interpolate(pred_orient_list[k], size=target_shape, mode='bilinear', align_corners=False) 

        # ----- Road CE/Combined Loss (ทุกสเกล) -----
        road_loss = 0
        for r in range(len(pred_road_list)):
            label_idx = r if r < len(scaled_target_road_label) else -1
            road_loss += segmentation_loss(pred_road_list[r], scaled_target_road_label[label_idx])

        # ----- Orientation CE Loss (ทุกสเกล) -----
        angle_loss = 0
        for r in range(len(pred_orient_list)):
            label_idx = r if r < len(scaled_target_orientation_class) else -1
            angle_loss += orientation_loss(pred_orient_list[r], scaled_target_orientation_class[label_idx])

        total_loss = road_loss + ANGLE_LAMBDA * angle_loss
        (total_loss / accum_steps).backward()

        if ((i + 1) % accum_steps == 0) or (i == len(train_loader) - 1):
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            Optimizer.step()
            Optimizer.zero_grad(set_to_none=True)

        train_loss_road += road_loss.item()
        train_loss_angle += angle_loss.item()

        # ---------- Metrics บนสเกลสุดท้าย ----------
        logits_road = pred_road_list[-1]
        logits_angle = pred_orient_list[-1]

        target_road = scaled_target_road_label[-1]            # [N,H,W]
        target_angle = scaled_target_orientation_class[-1]    # [N,H,W]
        H, W = target_road.shape[-2], target_road.shape[-1]

        pred_road_arg = _shape_match_argmax(logits_road, (H, W))
        pred_angle_arg = _shape_match_argmax(logits_angle, (H, W))

        hist += util.fast_hist_torch(pred_road_arg.view(pred_road_arg.size(0), -1),target_road.view(target_road.size(0), -1), n_RoadClasses)
        hist_angles += util.fast_hist_torch(pred_angle_arg.view(pred_angle_arg.size(0), -1),target_angle.view(target_angle.size(0), -1), n_OrientClasses)

        # ----- progress bar (Reduce frequency to every 50 steps) -----
        if (i + 1) % 50 == 0 or (i + 1) == len(train_loader):
            # Move hist to CPU only for logging
            curr_hist = hist.cpu().numpy()
            curr_hist_angles = hist_angles.cpu().numpy()

            p_accu, miou, road_iou, fwacc = util.performMetrics(
                train_loss_file, valid_loss_file, Epoch, curr_hist,
                train_loss_road/(i+1), train_loss_angle/(i+1),
                is_train=True, write=True
            )
            p_accu_angle, miou_angle, fwacc_angle = util.performAngleMetrics(
                train_loss_angle_file, valid_loss_angle_file, Epoch, curr_hist_angles,
                is_train=True, write=True
            )
            viz_util.progress_bar(
                i, len(train_loader),
                "Loss: %.6f | VecLoss: %.6f | road miou: %.4f%%(%.4f%%) | angle miou: %.4f%% "
                % (train_loss_road/(i+1), train_loss_angle/(i+1), miou, road_iou, miou_angle)
            )

        # ----- debug: สัดส่วนถนนใน GT/ทำนาย -----
        # if i % 100 == 0:
        #     gt_pos = (target_road == ROAD_CLASS_INDEX).float().mean().item()
        #     pd_pos = (pred_road_arg == ROAD_CLASS_INDEX).float().mean().item()
        #     print(f"[debug] gt_road%={gt_pos:.4f}, pred_road%={pd_pos:.4f}")

        # cleanup refs
        del (pred_road_list, pred_orient_list,
             logits_road, logits_angle,
             pred_road_arg, pred_angle_arg,
             target_road, target_angle,
             imageBGRcropped,
             scaled_target_road_label,
             scaled_target_orientation_class)

    util.performMetrics(train_loss_file, valid_loss_file,
                        Epoch, hist.cpu().numpy(),
                        train_loss_road / len(train_loader),
                        train_loss_angle / len(train_loader),
                        write=True)
    util.performAngleMetrics(train_loss_angle_file, valid_loss_angle_file, Epoch, hist_angles.cpu().numpy(), write=True)


def validate_model(ExperimentDirectory, model, dataset_name, nGPUs, cfg, train_loss_file, valid_loss_file, train_loss_angle_file, valid_loss_angle_file,
                   train_loader, valid_loader, Optimizer, segmentation_loss, orientation_loss,
                   LR_scheduler, Epoch, n_RoadClasses, n_OrientClasses):
    global best_accuracy
    global best_miou
    model.eval()
    valid_loss_road = 0.0
    valid_loss_angle = 0.0
    hist = torch.zeros((n_RoadClasses, n_RoadClasses)).cuda()
    hist_angles = torch.zeros((n_OrientClasses, n_OrientClasses)).cuda()

    with torch.no_grad():
        for i, ImageLabelData in enumerate(valid_loader, 0):
            imageBGR, scaled_target_road_label, scaled_target_orientation_class = ImageLabelData

            imageBGR = imageBGR.float().cuda(non_blocking=True)

            scaled_target_road_label = [(_label > 0).long().cuda(non_blocking=True) for _label in scaled_target_road_label]
            scaled_target_orientation_class = [_label.long().cuda(non_blocking=True) for _label in scaled_target_orientation_class]

            pred_road_list, pred_orient_list = model(imageBGR)

            # Auto-resize prediction to match GT size (Generalized for all datasets/models)
            # This fixes inconsistencies like LinkNet outputting 511 vs 512
            # Also handles Deep Supervision mismatch (Model outputs > Dataset labels)
            for k in range(len(pred_road_list)):
                label_idx = k if k < len(scaled_target_road_label) else -1
                tgt_sz = scaled_target_road_label[label_idx].shape[-2:]
                
                if pred_road_list[k].shape[-2:] != tgt_sz:
                    pred_road_list[k] = F.interpolate(pred_road_list[k], size=tgt_sz, mode='bilinear', align_corners=False)
                                                                                                                                    
            for k in range(len(pred_orient_list)):
                label_idx = k if k < len(scaled_target_orientation_class) else -1
                tgt_sz = scaled_target_orientation_class[label_idx].shape[-2:]
                
                if pred_orient_list[k].shape[-2:] != tgt_sz:
                    pred_orient_list[k] = F.interpolate(pred_orient_list[k], size=tgt_sz, mode='bilinear', align_corners=False)

            # loss รวมทุกสเกล
            road_loss = 0
            for r in range(len(pred_road_list)):
                label_idx = r if r < len(scaled_target_road_label) else -1
                road_loss += segmentation_loss(pred_road_list[r], scaled_target_road_label[label_idx])

            angle_loss = 0
            for r in range(len(pred_orient_list)):
                label_idx = r if r < len(scaled_target_orientation_class) else -1
                angle_loss += orientation_loss(pred_orient_list[r], scaled_target_orientation_class[label_idx])

            valid_loss_road += road_loss.item()
            valid_loss_angle += angle_loss.item()

            # metrics
            logits_road = pred_road_list[-1]
            logits_angle = pred_orient_list[-1]

            target_road  = scaled_target_road_label[-1]   # [N,H,W]
            target_angle = scaled_target_orientation_class[-1]
            H, W = target_road.shape[-2], target_road.shape[-1]

            pred_road_arg = _shape_match_argmax(logits_road, (H, W))
            pred_angle_arg = _shape_match_argmax(logits_angle, (H, W))
            prob1 = _shape_match_prob1(logits_road, (H, W))

            hist += util.fast_hist_torch(pred_road_arg.view(pred_road_arg.size(0), -1),target_road.view(target_road.size(0), -1), n_RoadClasses)
            hist_angles += util.fast_hist_torch(pred_angle_arg.view(pred_angle_arg.size(0), -1),target_angle.view(target_angle.size(0), -1), n_OrientClasses)

            # ----- progress bar (Reduce frequency to every 10 steps for validation) -----
            if (i + 1) % 10 == 0 or (i + 1) == len(valid_loader):
                # Move hist to CPU only for logging
                curr_hist = hist.cpu().numpy()
                curr_hist_angles = hist_angles.cpu().numpy()

                p_accu, miou, road_iou, fwacc = util.performMetrics(
                    train_loss_file, valid_loss_file, Epoch, curr_hist,
                    valid_loss_road/(i+1), valid_loss_angle/(i+1), is_train=False, write=True)
                p_accu_angle, miou_angle, fwacc_angle = util.performAngleMetrics(
                    train_loss_angle_file, valid_loss_angle_file, Epoch, curr_hist_angles, is_train=False, write=True)

                viz_util.progress_bar(i, len(valid_loader),
                    "Loss: %.6f | VecLoss: %.6f | road miou: %.4f%%(%.4f%%) | angle miou: %.4f%% "
                    % (valid_loss_road/(i+1), valid_loss_angle/(i+1), miou, road_iou, miou_angle))

            # save preview
            if i % 10 == 0 or i == len(valid_loader) - 1:
                images_path = "{}/images/".format(ExperimentDirectory)
                util.ensure_dir(images_path)
                util.savePredictedProb(
                    imageBGR.data.cpu(),
                    target_road.cpu(),
                    pred_road_arg.cpu(),
                    prob1.data.cpu(),
                    pred_angle_arg.cpu(),  # แสดงช่องเพิ่มถ้ามี
                    os.path.join(images_path, "validate_pair_{}_{}.png".format(Epoch, i)),
                    norm_type="Mean")

            del (pred_road_list, pred_orient_list,
                 logits_road, logits_angle, pred_road_arg, pred_angle_arg, prob1,
                 target_road, target_angle, imageBGR,
                 scaled_target_road_label, scaled_target_orientation_class)

    accuracy, miou, road_iou, fwacc = util.performMetrics(
        train_loss_file, valid_loss_file, Epoch, hist.cpu().numpy(),
        valid_loss_road / len(valid_loader), valid_loss_angle / len(valid_loader),
        is_train=False, write=True)
    util.performAngleMetrics(train_loss_angle_file, valid_loss_angle_file, Epoch, hist_angles.cpu().numpy(), is_train=False, write=True)

    global best_accuracy, best_miou, epochs_without_improvement, early_stop_patience
    if miou > best_miou:
        best_accuracy = accuracy
        best_miou = miou
        epochs_without_improvement = 0
        util.save_checkpoint(Epoch, valid_loss_road / len(valid_loader), model, Optimizer, best_accuracy, best_miou, cfg, ExperimentDirectory)
    else:
        epochs_without_improvement += 1

    return valid_loss_road / len(valid_loader), accuracy, miou

def main():

    nGPUs = torch.cuda.device_count()
    with open("cfg.json", 'r') as f:
        cfg = json.load(f)
    Seed = cfg["GlobalSeed"]
    Epochs = cfg["training_settings"]["epochs"]

    # Models
    # Dynamic loading: expects the file "Models/MyModel.py" to contain a class "MyModel"
    import importlib
    ModelFolderList = os.listdir(cfg["Models"]["base_dir"])
    ModelNames = [os.path.splitext(x)[0] for x in ModelFolderList if x.endswith(".py") and not x.startswith("__")]
    ChosenModel = {}
    for model_name in ModelNames:
        try:
            module = importlib.import_module(f"Models.{model_name}")
            # Try to get the class with the same name as the file
            if hasattr(module, model_name):
                ChosenModel[model_name] = getattr(module, model_name)
            else:
                # Fallback or specific mapping (if needed in future)
                print(f"[Warning] Class '{model_name}' not found in Models/{model_name}.py. Skipping.")
        except Exception as e:
            print(f"[Error] Failed to import Models.{model_name}: {e}")


    # Datasets
    DatasetNames = os.listdir(cfg["Datasets"]["base_dir"])
    DatasetClassNames = list(map('.'.join, zip(["DatasetUtility"]*len(DatasetNames), DatasetNames)))
    DatasetList = [eval(y) for y in DatasetClassNames]
    ChosenDataset = dict(zip(DatasetNames, DatasetList))

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default='all', const='all', type=str, nargs='?', choices=ModelNames, help=ModelNames)
    parser.add_argument("-d", "--dataset", default='all', const='all', type=str, nargs='?', choices=DatasetNames, help=DatasetNames)
    parser.add_argument("-e", "--experiment", required=True, type=str, help="Experiment Name")
    parser.add_argument("-r", "--resume", required=False, type=str, default=None, help="Most recent checkpoint (.pt) location (default: None)")
    parser.add_argument("-rd", "--resumedataset", required=False, type=str, default=None, help="Most recent checkpoint (.pt) location (default: None)")
    args = parser.parse_args()

    ExperimentDirectory = os.path.join(cfg["training_settings"]["results_directory"], args.experiment)
    if not os.path.exists(ExperimentDirectory):
        os.makedirs(ExperimentDirectory)

    model = ChosenModel[args.model]()
    dataset = ChosenDataset[args.dataset]

    if nGPUs == 0:
        print("torch.cuda can not find any GPUs on this device. Aborting...")
        sys.exit()
    elif nGPUs == 1:
        model.cuda()
    else:
        model = nn.DataParallel(model).cuda()

    # Use AdamW optimizer for better convergence (especially with transformers/modern architectures)
    Optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["optimizer_settings"]["learning_rate"],
                            weight_decay=cfg["optimizer_settings"]["learning_rate_decay"],
                            betas=(0.9, 0.999),
                            eps=1e-8)

    # ---------- ฟังก์ชันช่วยโหลด checkpoint ให้ปลอดภัย/ยืดหยุ่น ----------
    def _safe_load_ckpt(path):
        # PyTorch รุ่นใหม่: แนะนำ weights_only=True (ลดความเสี่ยง pickle)
        try:
            return torch.load(path, weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _load_state_dict_flex(model, state_dict, strict=False):
        # รองรับทั้ง dict เป็น state_dict ตรง ๆ หรือห่อด้วย key 'state_dict'
        sd = state_dict.get("state_dict", state_dict)
        from collections import OrderedDict
        new_sd = OrderedDict()
        for k, v in sd.items():
            nk = k[7:] if k.startswith("module.") else k  # ตัด prefix 'module.' จาก DataParallel ถ้ามี
            new_sd[nk] = v
        missing, unexpected = model.load_state_dict(new_sd, strict=strict)
        if len(missing) > 0 or len(unexpected) > 0:
            print(f"[warn] load_state_dict: missing={missing}, unexpected={unexpected}")

    # ----------------------------- โหลด checkpoint ------------------------------
    if args.resume is not None:
        # โหมด resume: ปกติจะมาจาก run โค้ดเดียวกัน → พยายามโหลด optimizer ด้วย
        checkpoint = _safe_load_ckpt(args.resume)
        _load_state_dict_flex(model, checkpoint, strict=False)
        try:
            Optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception as e:
            print(f"[warn] failed to load optimizer from --resume ({e}); keep current optimizer.")
        resume_at_epoch = int(checkpoint.get("epoch", 0)) + 1
        epoch_with_best_miou = float(checkpoint.get("miou", 0.0))
    elif args.resumedataset is not None:
        # โหมด resumedataset: ใช้น้ำหนักจากโมเดลอื่น/รอบเก่า → โหลด "เฉพาะน้ำหนักโมเดล"
        checkpoint = _safe_load_ckpt(args.resumedataset)
        _load_state_dict_flex(model, checkpoint, strict=False)
        resume_at_epoch = 1
        epoch_with_best_miou = float(checkpoint.get("miou", 0.0))
        print("[info] loaded model weights only from --resumedataset; optimizer re-initialized.")
    else:
        resume_at_epoch = 1
        np.random.seed(Seed)
        torch.manual_seed(Seed)
        torch.cuda.manual_seed_all(Seed)
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                v = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                nn.init.normal_(module.weight.data, 0.0, math.sqrt(2.0 / v))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
    # ---------------------------------------------------------------------------

 # --- Scheduler ที่ยืดหยุ่น: ใช้ MultiStepLR ถ้ามี milestones ใน cfg; ไม่งั้น fallback -> Cosine + warmup ---
    warmup_epochs = int(cfg["optimizer_settings"].get("warmup_epochs", 0))
    try:
        milestones = eval(cfg["optimizer_settings"]["learning_rate_drop_at_epoch"])
        gamma = float(cfg["optimizer_settings"]["learning_rate_step"])
        if not isinstance(milestones, (list, tuple)):
            raise ValueError("milestones must be list/tuple")
        LR_scheduler = MultiStepLR(Optimizer, milestones=milestones, gamma=gamma)
        print(f"[info] Using MultiStepLR: milestones={milestones}, gamma={gamma}")
    except (KeyError, NameError, ValueError) as e:
        tmax = max(1, int(cfg["training_settings"]["epochs"]) - int(warmup_epochs))
        LR_scheduler = CosineAnnealingLR(Optimizer, T_max=tmax)
        print(f"[info] Using CosineAnnealingLR with warmup_epochs={warmup_epochs} (fallback due to: {e})")

    train_loader = data.DataLoader(dataset(cfg, args.model, args.dataset, "training_settings"),
                                   batch_size=cfg["training_settings"]["batch_size"],
                                   num_workers=4,
                                   shuffle=True,
                                   pin_memory=True)

    valid_loader = data.DataLoader(dataset(cfg, args.model, args.dataset, "validation_settings"),
                                   batch_size=cfg["validation_settings"]["batch_size"],
                                   num_workers=4,
                                   shuffle=False,
                                   pin_memory=True)

    n_RoadClasses = cfg["training_settings"]["roadclass"]
    n_OrientClasses = cfg["training_settings"]["orientationclass"]

    # ----------- Losses -----------
    # Road segmentation: Use CombinedLoss (Focal + Boundary-Aware) for better road IoU
    seg_w = torch.ones(n_RoadClasses, dtype=torch.float32).cuda()
    if n_RoadClasses >= 2 and ROAD_CLASS_INDEX < n_RoadClasses:
        seg_w[ROAD_CLASS_INDEX] = 15.0  # Increased weight for road class (was 10.0)

    # Use CombinedLoss for road segmentation (Focal + Boundary-Aware)
    use_combined_loss = cfg.get("use_combined_loss", True)
    if use_combined_loss:
        print("[info] Using CombinedLoss (Focal + Boundary-Aware) for road segmentation")
        segmentation_loss = Losses.CombinedLoss(
            focal_weight=1.0,
            boundary_weight=0.5,
            alpha=seg_w,
            gamma=2.0,
            ignore_index=255
        ).cuda()
    else:
        print("[info] Using standard CrossEntropyLoss for road segmentation")
        segmentation_loss = nn.CrossEntropyLoss(weight=seg_w, ignore_index=255).cuda()

    # Orientation (class index per pixel)
    orientation_weights = torch.ones(n_OrientClasses).cuda()
    orientation_loss = Losses.CrossEntropyLossImage(weight=orientation_weights, ignore_index=255).cuda()
    # ------------------------------

    ExperimentDirectory = os.path.join(cfg["training_settings"]["results_directory"], args.experiment)
    if not os.path.exists(ExperimentDirectory):
        os.makedirs(ExperimentDirectory)

    train_file = "{}/{}_train_loss.txt".format(ExperimentDirectory, args.dataset)
    valid_file = "{}/{}_valid_loss.txt".format(ExperimentDirectory, args.dataset)
    train_loss_file = open(train_file, "w")
    valid_loss_file = open(valid_file, "w")

    train_file_angle = "{}/{}_train_angle_loss.txt".format(ExperimentDirectory, args.dataset)
    valid_file_angle = "{}/{}_valid_angle_loss.txt".format(ExperimentDirectory, args.dataset)
    train_loss_angle_file = open(train_file_angle, "w")
    valid_loss_angle_file = open(valid_file_angle, "w")

    for Epoch in range(resume_at_epoch, Epochs + 1):
        Epoch_Start_Time = time.perf_counter()
        print("\nTraining Epoch: %d" % Epoch)
        train_model(model, nGPUs, cfg, train_loss_file, valid_loss_file, train_loss_angle_file, valid_loss_angle_file,
                    train_loader, valid_loader, Optimizer, segmentation_loss, orientation_loss,
                    LR_scheduler, Epoch, resume_at_epoch, n_RoadClasses, n_OrientClasses)
        # ---- Warmup (ถ้ากำหนด) + step scheduler ----
        if warmup_epochs > 0 and Epoch <= warmup_epochs:
            # scale LR แบบเส้นตรงช่วง warmup
            base_lrs = [pg.get("initial_lr", pg["lr"]) for pg in Optimizer.param_groups]
            scale = float(Epoch) / max(1, warmup_epochs)
            for pg, base_lr in zip(Optimizer.param_groups, base_lrs):
                pg["lr"] = base_lr * scale
        else:
            LR_scheduler.step()

        if (Epoch % cfg["validation_settings"]["evaluation_frequency"] == 0) or (Epoch > 80):
            print("\nTesting Epoch: %d" % Epoch)
            val_loss, current_accuracy, current_miou = validate_model(ExperimentDirectory, model, args.dataset, nGPUs, cfg, train_loss_file, valid_loss_file, train_loss_angle_file, valid_loss_angle_file,
                                      train_loader, valid_loader, Optimizer, segmentation_loss, orientation_loss,
                                      LR_scheduler, Epoch, n_RoadClasses, n_OrientClasses)

            # Save periodic checkpoint every 5 epochs
            if Epoch % 5 == 0:
                util.save_periodic_checkpoint(Epoch, val_loss, model, Optimizer, current_accuracy, current_miou, cfg, ExperimentDirectory)

            # Early stopping check
            if epochs_without_improvement >= early_stop_patience:
                print(f"\n[Early Stopping] No improvement for {early_stop_patience} validation checks. Best mIoU: {best_miou:.4f}")
                break

        Epoch_End_Time = time.perf_counter()
        print("Time Elapsed for Epoch : {1}".format(Epoch, Epoch_End_Time - Epoch_Start_Time))

if __name__=="__main__":
    best_accuracy = 0
    best_miou = 0
    epochs_without_improvement = 0
    early_stop_patience = 5  # Stop if no improvement for 5 validation checks (20 epochs at eval_freq=4)
    main()
