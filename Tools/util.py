"""
# Reference:
# https://github.com/anilbatra2185/road_connectivity
"""

"""
# Reference:
# https://github.com/anilbatra2185/road_connectivity
"""

import argparse
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from skimage.morphology import skeletonize


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def setSeed(config):
    if config["seed"] is None:
        manualSeed = np.random.randint(1, 10000)
    else:
        manualSeed = config["seed"]
    print("Random Seed: ", manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)


def getParllelNetworkStateDict(state_dict):
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def to_variable(tensor, volatile=False, requires_grad=True):
    # Note: Variable is deprecated in modern PyTorch; keep for backward-compat
    return Variable(tensor.float().cuda(), requires_grad=requires_grad)


def weights_init(model, manual_seed=7):
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    random.seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def weights_normal_init(model, manual_seed=7):
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    random.seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


# ==============
# METRIC HELPERS
# ==============

def fast_hist(a, b, n):
    """Compute confusion matrix for labels a (pred) vs b (gt).
    Both a and b can be any shape; will be flattened. Values outside [0, n-1]
    on the prediction side are ignored. Ground-truth is assumed already valid or masked.
    Returns an (n, n) matrix with rows = GT classes, cols = Pred classes or vice versa?
    NOTE: In many implementations, row is GT and col is Pred. Here we follow the
    original code's encoding: bincount(n * a + b). That makes 'a' act as row-index
    and 'b' as column-index. Keep consistent across the project.
    """
    a = np.asarray(a).astype(int).ravel()
    b = np.asarray(b).astype(int).ravel()
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k] + b[k], minlength=n ** 2).reshape(n, n)


def _safe_metrics_from_hist(hist: np.ndarray):
    """Compute metrics safely (avoid divide-by-zero, ignore absent classes).

    Returns:
        pixel_acc, mean_acc, mean_iou, per_class_iou, fwavacc
    All as float (per_class_iou is 1D ndarray).
    """
    hist = hist.astype(np.float64)
    total = hist.sum()
    diag = np.diag(hist)
    row = hist.sum(axis=1)  # GT per class (following project convention)
    col = hist.sum(axis=0)  # Pred per class

    if total <= 0:
        return 0.0, 0.0, 0.0, np.zeros_like(diag), 0.0

    with np.errstate(divide='ignore', invalid='ignore'):
        pixel_acc = diag.sum() / total

        per_class_acc = np.divide(diag, row, out=np.zeros_like(diag), where=row > 0)
        m_acc = per_class_acc[row > 0].mean() if (row > 0).any() else 0.0

        denom_iou = row + col - diag
        per_class_iou = np.divide(diag, denom_iou, out=np.zeros_like(diag), where=denom_iou > 0)
        m_iou = per_class_iou[denom_iou > 0].mean() if (denom_iou > 0).any() else 0.0

        freq = np.divide(row, total, out=np.zeros_like(row), where=total > 0)
        fwavacc = (freq * per_class_iou).sum()

    return float(pixel_acc), float(m_acc), float(m_iou), per_class_iou, float(fwavacc)


def performAngleMetrics(train_loss_angle_file, val_loss_angle_file, epoch, hist, is_train=True, write=False):
    pixel_accuracy, mean_accuracy, mean_iou, per_class_iou, fwavacc = _safe_metrics_from_hist(hist)

    if write and is_train:
        train_loss_angle_file.write(
            "[%d], Pixel Accuracy:%.3f, Mean Accuracy:%.3f, Mean IoU:%.3f, Freq.Weighted Accuray:%.3f  \n"
            % (epoch, 100 * pixel_accuracy, 100 * mean_accuracy, 100 * mean_iou, 100 * fwavacc)
        )
    elif write and not is_train:
        val_loss_angle_file.write(
            "[%d], Pixel Accuracy:%.3f, Mean Accuracy:%.3f, Mean IoU:%.3f, Freq.Weighted Accuray:%.3f  \n"
            % (epoch, 100 * pixel_accuracy, 100 * mean_accuracy, 100 * mean_iou, 100 * fwavacc)
        )

    return 100 * pixel_accuracy, 100 * mean_iou, 100 * fwavacc


def performMetrics(train_loss_file, val_loss_file, epoch, hist, loss, loss_vec, is_train=True, write=False):
    pixel_accuracy, mean_accuracy, mean_iou, per_class_iou, fwavacc = _safe_metrics_from_hist(hist)

    # ป้องกัน index error กรณีจำนวนคลาส < 2
    cls0 = per_class_iou[0] if len(per_class_iou) > 0 else 0.0
    cls1 = per_class_iou[1] if len(per_class_iou) > 1 else 0.0

    if write and is_train:
        train_loss_file.write(
            "[%d], Loss:%.5f, Loss(VecMap):%.5f, Pixel Accuracy:%.3f, Mean Accuracy:%.3f, Mean IoU:%.3f, Class IoU:[%.5f/%.5f], Freq.Weighted Accuray:%.3f  \n"
            % (
                epoch,
                loss,
                loss_vec,
                100 * pixel_accuracy,
                100 * mean_accuracy,
                100 * mean_iou,
                cls0,
                cls1,
                100 * fwavacc,
            )
        )
    elif write and not is_train:
        val_loss_file.write(
            "[%d], Loss:%.5f, Loss(VecMap):%.5f, Pixel Accuracy:%.3f, Mean Accuracy:%.3f, Mean IoU:%.3f, Class IoU:[%.5f/%.5f], Freq.Weighted Accuray:%.3f  \n"
            % (
                epoch,
                loss,
                loss_vec,
                100 * pixel_accuracy,
                100 * mean_accuracy,
                100 * mean_iou,
                cls0,
                cls1,
                100 * fwavacc,
            )
        )

    return 100 * pixel_accuracy, 100 * mean_iou, 100 * cls1, 100 * fwavacc


# =============
# I/O UTILITIES
# =============

def save_checkpoint(epoch, loss, model, optimizer, best_accuracy, best_miou, config, experiment_dir):
    if torch.cuda.device_count() > 1:
        arch = type(model.module).__name__
    else:
        arch = type(model).__name__
    state = {
        "arch": arch,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "pixel_accuracy": best_accuracy,
        "miou": best_miou,
        "config": config,
    }
    filename = os.path.join(experiment_dir, "checkpoint-epoch{:03d}-loss-{:.4f}.pth.tar".format(epoch, loss))
    torch.save(state, filename)
    model_best_path = os.path.join(experiment_dir, "model_best.pth.tar")
    if os.path.exists(model_best_path):
        os.remove(model_best_path)
    os.rename(filename, model_best_path)
    print("Saving current best: {} ...".format("model_best.pth.tar"))

def save_periodic_checkpoint(epoch, loss, model, optimizer, accuracy, miou, config, experiment_dir):
    """Save checkpoint every N epochs (periodic backup)"""
    if torch.cuda.device_count() > 1:
        arch = type(model.module).__name__
    else:
        arch = type(model).__name__
    state = {
        "arch": arch,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "pixel_accuracy": accuracy,
        "miou": miou,
        "config": config,
    }
    filename = os.path.join(experiment_dir, "checkpoint_epoch_{:03d}.pth.tar".format(epoch))
    torch.save(state, filename)
    print("Saved periodic checkpoint: checkpoint_epoch_{:03d}.pth.tar (miou: {:.2f}%)".format(epoch, miou))


def savePredictedProb(real, gt, predicted, predicted_prob, pred_affinity=None, image_name="", norm_type="Mean"):
    b, c, h, w = real.size()
    grid = []
    mean_bgr = np.array([70.95016901, 71.16398124, 71.30953645])
    deviation_bgr = np.array([34.00087859, 35.18201658, 36.40463264])

    for idx in range(b):
        real_ = np.asarray(real[idx].numpy().transpose(1, 2, 0), dtype=np.float32)
        if norm_type == "Mean":
            real_ = real_ + mean_bgr
        elif norm_type == "Std":
            real_ = (real_ * deviation_bgr) + mean_bgr
        real_ = np.asarray(real_, dtype=np.uint8)

        gt_ = gt[idx].numpy() * 255.0
        gt_ = np.asarray(gt_, dtype=np.uint8)
        gt_ = np.stack((gt_,) * 3).transpose(1, 2, 0)

        predicted_ = (predicted[idx]).numpy() * 255.0
        predicted_ = np.asarray(predicted_, dtype=np.uint8)
        predicted_ = np.stack((predicted_,) * 3).transpose(1, 2, 0)

        predicted_prob_ = (predicted_prob[idx]).numpy() * 255.0
        predicted_prob_ = np.asarray(predicted_prob_, dtype=np.uint8)
        predicted_prob_ = cv2.applyColorMap(predicted_prob_, cv2.COLORMAP_JET)

        if pred_affinity is not None:
            hsv = np.zeros_like(real_)
            hsv[..., 1] = 255
            affinity_ = pred_affinity[idx].numpy()
            mag = np.copy(affinity_)
            mag[mag < 36] = 1
            mag[mag >= 36] = 0
            affinity_[affinity_ == 36] = 0
            hsv[..., 0] = affinity_ * 10 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            affinity_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            pair = np.concatenate((real_, gt_, predicted_, predicted_prob_, affinity_bgr), axis=1)
        else:
            pair = np.concatenate((real_, gt_, predicted_, predicted_prob_), axis=1)
        grid.append(pair)

    if pred_affinity is not None:
        cv2.imwrite(image_name, np.array(grid).reshape(b * h, 5 * w, 3))
    else:
        cv2.imwrite(image_name, np.array(grid).reshape(b * h, 4 * w, 3))


def savePredictedProbStiched(real, gt, predicted, predicted_prob, pred_affinity=None, image_name="", norm_type="Mean"):
    b, c, h, w = real.size()
    grid = []

    real_tiles = []
    gt_tiles = []
    pred_tiles = []
    pred_prob_tiles = []

    mean_bgr = np.array([70.95016901, 71.16398124, 71.30953645])
    deviation_bgr = np.array([34.00087859, 35.18201658, 36.40463264])

    for idx in range(b):
        real_ = np.asarray(real[idx].numpy().transpose(1, 2, 0), dtype=np.float32)
        if norm_type == "Mean":
            real_ = real_ + mean_bgr
        elif norm_type == "Std":
            real_ = (real_ * deviation_bgr) + mean_bgr

        real_ = np.asarray(real_, dtype=np.uint8)
        gt_ = gt[idx].numpy() * 255.0
        gt_ = np.asarray(gt_, dtype=np.uint8)
        gt_ = np.stack((gt_,) * 3).transpose(1, 2, 0)

        predicted_ = (predicted[idx]).numpy() * 255.0
        predicted_ = np.asarray(predicted_, dtype=np.uint8)
        predicted_ = np.stack((predicted_,) * 3).transpose(1, 2, 0)

        predicted_prob_ = (predicted_prob[idx]).numpy() * 255.0
        predicted_prob_ = np.asarray(predicted_prob_, dtype=np.uint8)
        predicted_prob_ = cv2.applyColorMap(predicted_prob_, cv2.COLORMAP_JET)

        real_tiles.append(real_)
        gt_tiles.append(gt_)
        pred_tiles.append(predicted_)
        pred_prob_tiles.append(predicted_prob_)

    firstrow = np.concatenate((real_tiles[0], real_tiles[1], gt_tiles[0], gt_tiles[1]), axis=1)
    secondrow = np.concatenate((real_tiles[2], real_tiles[3], gt_tiles[2], gt_tiles[3]), axis=1)
    thirdrow = np.concatenate((pred_tiles[0], pred_tiles[1], pred_prob_tiles[0], pred_prob_tiles[1]), axis=1)
    fourthrow = np.concatenate((pred_tiles[2], pred_tiles[3], pred_prob_tiles[2], pred_prob_tiles[3]), axis=1)

    grid.append(firstrow)
    grid.append(secondrow)
    grid.append(thirdrow)
    grid.append(fourthrow)

    cv2.imwrite(image_name, np.array(grid).reshape(b * h, 4 * w, 3))


def get_relaxed_precision(a, b, buffer):
    tp = 0
    indices = np.where(a == 1)
    for ind in range(len(indices[0])):
        tp += (np.sum(b[indices[0][ind] - buffer : indices[0][ind] + buffer + 1, indices[1][ind] - buffer : indices[1][ind] + buffer + 1]) > 0).astype(int)
    return tp


# Globals for multi-row stitching
mr_real_tiles = []
mr_gt_tiles = []
mr_pred_tiles = []
mr_pred_prob_tiles = []


def savePredictedProbStichedMR(i_stage, real, gt, predicted, predicted_prob, pred_affinity=None, image_name="", norm_type="Mean"):
    b, c, h, w = real.size()
    mr_grid = []

    mean_bgr = np.array([70.95016901, 71.16398124, 71.30953645])
    deviation_bgr = np.array([34.00087859, 35.18201658, 36.40463264])

    for idx in range(b):
        real_ = np.asarray(real[idx].numpy().transpose(1, 2, 0), dtype=np.float32)
        if norm_type == "Mean":
            real_ = real_ + mean_bgr
        elif norm_type == "Std":
            real_ = (real_ * deviation_bgr) + mean_bgr

        real_ = np.asarray(real_, dtype=np.uint8)
        gt_ = gt[idx].numpy() * 255.0
        gt_ = np.asarray(gt_, dtype=np.uint8)
        gt_ = np.stack((gt_,) * 3).transpose(1, 2, 0)

        predicted_ = (predicted[idx]).numpy() * 255.0
        predicted_ = np.asarray(predicted_, dtype=np.uint8)
        predicted_ = np.stack((predicted_,) * 3).transpose(1, 2, 0)

        predicted_prob_ = (predicted_prob[idx]).numpy() * 255.0
        predicted_prob_ = np.asarray(predicted_prob_, dtype=np.uint8)
        predicted_prob_ = cv2.applyColorMap(predicted_prob_, cv2.COLORMAP_JET)

        mr_real_tiles.append(real_)
        mr_gt_tiles.append(gt_)
        mr_pred_tiles.append(predicted_)
        mr_pred_prob_tiles.append(predicted_prob_)

    if (i_stage > 1) and ((i_stage + 1) % 3 == 0):
        firstrow = np.concatenate((mr_real_tiles[0], mr_real_tiles[1], mr_real_tiles[2], mr_gt_tiles[0], mr_gt_tiles[1], mr_gt_tiles[2]), axis=1)
        secondrow = np.concatenate((mr_real_tiles[3], mr_real_tiles[4], mr_real_tiles[5], mr_gt_tiles[3], mr_gt_tiles[4], mr_gt_tiles[5]), axis=1)
        thirdrow = np.concatenate((mr_real_tiles[6], mr_real_tiles[7], mr_real_tiles[8], mr_gt_tiles[6], mr_gt_tiles[7], mr_gt_tiles[8]), axis=1)
        fourthrow = np.concatenate((mr_pred_tiles[0], mr_pred_tiles[1], mr_pred_tiles[2], mr_pred_prob_tiles[0], mr_pred_prob_tiles[1], mr_pred_prob_tiles[2]), axis=1)
        fifthrow = np.concatenate((mr_pred_tiles[3], mr_pred_tiles[4], mr_pred_tiles[5], mr_pred_prob_tiles[3], mr_pred_prob_tiles[4], mr_pred_prob_tiles[5]), axis=1)
        sixthrow = np.concatenate((mr_pred_tiles[6], mr_pred_tiles[7], mr_pred_tiles[8], mr_pred_prob_tiles[6], mr_pred_prob_tiles[7], mr_pred_prob_tiles[8]), axis=1)

        mr_grid.append(firstrow)
        mr_grid.append(secondrow)
        mr_grid.append(thirdrow)
        mr_grid.append(fourthrow)
        mr_grid.append(fifthrow)
        mr_grid.append(sixthrow)

        cv2.imwrite(image_name, np.array(mr_grid).reshape(6 * h, 6 * w, 3))

        mr_real_tiles.clear()
        mr_gt_tiles.clear()
        mr_pred_tiles.clear()
        mr_pred_prob_tiles.clear()


def relaxed_f1(pred, gt, buffer=3):
    """Compute relaxed F1 on skeletonized predictions/GT with pixel tolerance buffer.

    Returns:
        rprecision_tp, rrecall_tp, pred_positive, gt_positive
    """
    rprecision_tp, rrecall_tp, pred_positive, gt_positive = 0, 0, 0, 0
    for b in range(pred.shape[0]):
        pred_sk = skeletonize(pred[b])
        gt_sk = skeletonize(gt[b])
        rprecision_tp += get_relaxed_precision(pred_sk, gt_sk, buffer)
        rrecall_tp += get_relaxed_precision(gt_sk, pred_sk, buffer)
        pred_positive += len(np.where(pred_sk == 1)[0])
        gt_positive += len(np.where(gt_sk == 1)[0])

    return rprecision_tp, rrecall_tp, pred_positive, gt_positive
