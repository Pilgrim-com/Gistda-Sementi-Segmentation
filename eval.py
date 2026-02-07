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
from torch.optim.lr_scheduler import MultiStepLR
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
# import matplotlib.pyplot as plt
# import albumentations as alb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def evaluate_model(ExperimentDirectory, model, dataset_name, nGPUs, cfg, train_loss_file, valid_loss_file, train_loss_angle_file, valid_loss_angle_file, 
                   valid_loader, Optimizer, segmentation_loss, orientation_loss, 
                   LR_scheduler, Epoch, n_RoadClasses, n_OrientClasses):
    global best_accuracy
    global best_miou
    model.eval()
    valid_loss_road = 0
    valid_loss_angle = 0
    hist = np.zeros((n_RoadClasses, n_RoadClasses))
    hist_angles = np.zeros((n_OrientClasses, n_OrientClasses))
    
    crop_size = cfg["validation_settings"]["crop_size"]
    if dataset_name == "Spacenet":
       crop_size = cfg["validation_settings"]["spacenet_crop_size"]
    
    relaxed_precision_tp = 0.0
    relaxed_recall_tp = 0.0
    predicted_positive = 0.0
    groundtruth_positive = 0.0
    
    with torch.no_grad():
        for i, ImageLabelData in enumerate(valid_loader, 0):
            imageBGR, scaled_target_road_label, scaled_target_orientation_class = ImageLabelData

            # inputs
            imageBGR = imageBGR.float().cuda()
            scaled_target_road_label = [_label.cuda() for _label in scaled_target_road_label]
            scaled_target_orientation_class = [_label.cuda() for _label in scaled_target_orientation_class]

            # Forward pass
            outputs = model(imageBGR)
            
            # Handle Deep Supervision (Model returns [list_road, list_angle]) or Single Output
            if isinstance(outputs, (list, tuple)):
                predicted_road_list = outputs[0]
                predicted_orient_list = outputs[1]
                
                # Check if road output is list (Deep Supervision) or tensor
                if isinstance(predicted_road_list, (list, tuple)):
                    predicted_road = predicted_road_list
                else:
                    predicted_road = [predicted_road_list]

                # Check if orientation output is list or tensor
                if isinstance(predicted_orient_list, (list, tuple)):
                    predicted_orientation_class = predicted_orient_list
                else:
                    predicted_orientation_class = [predicted_orient_list]
            else:
                # Old model style (single output)
                predicted_road = [outputs]
                predicted_orientation_class = [None]
            

            # Always interpolate to match target size (Standard Segmentation Evaluation)
            # Use the last (finest) label if we don't have enough multi-scale labels
            for ipr,resize_predicted_road in enumerate(predicted_road):
                target_idx = ipr if ipr < len(scaled_target_road_label) else -1
                target_shape = scaled_target_road_label[target_idx].shape[-2:]
                
                if resize_predicted_road.shape[-2:] != target_shape:
                    predicted_road[ipr] = F.interpolate(resize_predicted_road, size=target_shape, mode='bilinear', align_corners=False)
            
            for ipoc,resize_predicted_orientation_class in enumerate(predicted_orientation_class):
                target_idx = ipoc if ipoc < len(scaled_target_orientation_class) else -1
                target_shape = scaled_target_orientation_class[target_idx].shape[-2:]
                
                if resize_predicted_orientation_class.shape[-2:] != target_shape:
                    predicted_orientation_class[ipoc] = F.interpolate(resize_predicted_orientation_class, size=target_shape, mode='bilinear', align_corners=False)
            
            # Loss Calculation with safe indices
            road_loss = 0
            for r in range(len(predicted_road)):
                target_idx = r if r < len(scaled_target_road_label) else -1
                road_loss += segmentation_loss(predicted_road[r], scaled_target_road_label[target_idx])
                
            angle_loss = 0
            for r in range(len(predicted_orientation_class)):
                target_idx = r if r < len(scaled_target_orientation_class) else -1
                angle_loss += orientation_loss(predicted_orientation_class[r], scaled_target_orientation_class[target_idx])
                
            valid_loss_road += road_loss.item()
            valid_loss_angle += angle_loss.item()

            predicted_road = predicted_road[-1]
            predicted_orientation_class = predicted_orientation_class[-1]
            
            _, predicted_road_ = torch.max(predicted_road, 1)
            _, predicted_angle_ = torch.max(predicted_orientation_class, 1)
            
            target_road = scaled_target_road_label[-1].view(-1, crop_size, crop_size).long()
            target_angle = scaled_target_orientation_class[-1].view(-1, crop_size, crop_size).long()
            
            hist += util.fast_hist(predicted_road_.view(predicted_road_.size(0), -1).cpu().numpy(),
                                   target_road.view(target_road.size(0), -1).cpu().numpy(), n_RoadClasses)
                                   
            hist_angles += util.fast_hist(predicted_angle_.view(predicted_angle_.size(0), -1).cpu().numpy(),
                                          target_angle.view(target_angle.size(0), -1).cpu().numpy(),n_OrientClasses)
            
            p_accu, miou, road_iou, fwacc = util.performMetrics(train_loss_file, valid_loss_file, 
                                                                Epoch, hist, 
                                                                valid_loss_road / (i + 1), valid_loss_angle / (i + 1), is_train=False, write=True)
            
            if i % 1 == 0 or i == len(valid_loader) - 1:
                images_path = "{}/images_eval/".format(ExperimentDirectory)
                util.ensure_dir(images_path)
                if dataset_name == "MassachusettsRoads":
                    util.savePredictedProbStichedMR(
                        i,
                        imageBGR.data.cpu(),
                        scaled_target_road_label[-1].cpu(),
                        predicted_road_.cpu(),
                        F.softmax(predicted_road, dim=1).data.cpu()[:, 1, :, :],
                        predicted_angle_.cpu(),
                        os.path.join(images_path, "validate_pair_{}_{}.png".format(Epoch, i)),
                        norm_type="Mean")
                else:
                    util.savePredictedProbStiched(
                        imageBGR.data.cpu(),
                        scaled_target_road_label[-1].cpu(),
                        predicted_road_.cpu(),
                        F.softmax(predicted_road, dim=1).data.cpu()[:, 1, :, :],
                        predicted_angle_.cpu(),
                        os.path.join(images_path, "validate_pair_{}_{}.png".format(Epoch, i)),
                        norm_type="Mean")

            relaxed_precision_tp_in, relaxed_recall_tp_in, predicted_positive_in, groundtruth_positive_in = util.relaxed_f1(predicted_road_.cpu().numpy(), scaled_target_road_label[-1].cpu().numpy(), buffer=4)
            relaxed_precision_tp = relaxed_precision_tp + relaxed_precision_tp_in
            relaxed_recall_tp = relaxed_recall_tp + relaxed_recall_tp_in
            predicted_positive = predicted_positive  + predicted_positive_in
            groundtruth_positive = groundtruth_positive + groundtruth_positive_in
            
            precision = relaxed_precision_tp/(groundtruth_positive + 1e-12)
            recall = relaxed_recall_tp/(groundtruth_positive + 1e-12)
            f1measure = 2*precision*recall/(precision + recall + 1e-12)
            iou = precision*recall/(precision+recall-(precision*recall) + 1e-12)
            
            print("[Testing {}/{}] precision={}, recall={}, F1={}, IoU-r={}".format(i, len(valid_loader), precision, recall, f1measure, iou))

            del (predicted_road,
                predicted_orientation_class,
                predicted_road_, predicted_angle_,
                target_road, target_angle,
                imageBGR,
                scaled_target_road_label,
                scaled_target_orientation_class)

        accuracy, miou, road_iou, fwacc = util.performMetrics(train_loss_file, valid_loss_file,
                                                              Epoch, hist,
                                                              valid_loss_road / len(valid_loader), valid_loss_angle / len(valid_loader), 
                                                              is_train=False, write=True)
                                                              
        print("[FINAL] precision={}, recall={}, F1={}, IoU-r={}, road-iou={}".format(precision, recall, f1measure, iou, road_iou))

    return valid_loss_road / len(valid_loader)

def evaluate():

    nGPUs = torch.cuda.device_count()
    with open("cfg.json", 'r') as f:
        cfg = json.load(f)
    Seed = cfg["GlobalSeed"]
    Epochs = cfg["training_settings"]["epochs"]\
    
    # Models
    # Dynamic loading using importlib
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
                # Fallback: try to find any class in the module? Or just print warning
                # Providing a way to match previous specific naming if needed, but usually ClassName == FileName
                print(f"[Warning] Class '{model_name}' not found in Models/{model_name}.py. Skipping.")
        except Exception as e:
            print(f"[Error] Failed to import Models.{model_name}: {e}")
    
    # Datasets
    DatasetNames = os.listdir(cfg["Datasets"]["base_dir"])
    # Filter out non-directories or hidden files if necessary, but original code just did listdir.
    # Original logic assumes these names match classes in DatasetUtility.
    DatasetClassNames = list(map('.'.join, zip(["DatasetUtility"]*len(DatasetNames), DatasetNames)))
    # Use robust eval or getattr? 
    # Since DatasetUtility IS imported at top, eval("DatasetUtility.DeepGlobe") works.
    # To be safer/cleaner without eval: getattr(DatasetUtility, name)
    DatasetList = []
    for name in DatasetNames:
        try:
            DatasetList.append(getattr(DatasetUtility, name))
        except AttributeError:
            print(f"[Warning] Dataset class '{name}' not found in DatasetUtility. Skipping.")
            DatasetList.append(None) # Handle gracefully? Original would crash.
            
    # Filter out Nones to avoid crash later
    ValidNames = [n for n, d in zip(DatasetNames, DatasetList) if d is not None]
    ValidLists = [d for d in DatasetList if d is not None]
    ChosenDataset = dict(zip(ValidNames, ValidLists))
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default='all', const='all', type=str, nargs='?', choices = list(ChosenModel.keys()), help = "Model Name")
    parser.add_argument("-d", "--dataset", default='all', const='all', type=str, nargs='?', choices = ValidNames, help = "Dataset Name")
    parser.add_argument("-e", "--experiment", required=True, type=str, help = "Experiment Name")
    parser.add_argument("-r", "--resume", required=False, type=str, default=None, help="Most recent checkpoint (.pt) location (default: None)")
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
    
    Optimizer = optim.SGD(model.parameters(), 
                              lr = cfg["optimizer_settings"]["learning_rate"], 
                              momentum=0.9, 
                              weight_decay = cfg["optimizer_settings"]["learning_rate_decay"])

    if args.resume is not None:
        try:
            checkpoint = torch.load(args.resume, weights_only=False)
        except TypeError: # For older PyTorch versions that doesn't support weights_only arg
            checkpoint = torch.load(args.resume)
            
        model.load_state_dict(checkpoint["state_dict"])
        Optimizer.load_state_dict(checkpoint["optimizer"])
        resume_at_epoch = checkpoint["epoch"] + 1
        epoch_with_best_miou = checkpoint["miou"]
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
                
    LR_scheduler = MultiStepLR(Optimizer, 
                               milestones = eval(cfg["optimizer_settings"]["learning_rate_drop_at_epoch"]), 
                               gamma = cfg["optimizer_settings"]["learning_rate_step"])
    
    valid_loader = data.DataLoader(dataset(cfg, args.model, args.dataset, "validation_settings"), 
                                   batch_size = cfg["validation_settings"]["batch_size"], 
                                   num_workers=4, 
                                   shuffle=False, 
                                   pin_memory=False)
                                   
    n_RoadClasses = cfg["training_settings"]["roadclass"]
    n_OrientClasses = cfg["training_settings"]["orientationclass"]
    segmentation_weights = torch.ones(n_RoadClasses).cuda()
    orientation_weights = torch.ones(n_OrientClasses).cuda()

    segmentation_loss = Losses.mIoULoss(weight = segmentation_weights, n_classes = n_RoadClasses).cuda()
    orientation_loss = Losses.CrossEntropyLossImage(weight=orientation_weights, ignore_index=255).cuda() 
    
    train_file = "{}/{}_train_loss.txt".format(ExperimentDirectory, args.dataset)
    valid_file = "{}/{}_valid_loss.txt".format(ExperimentDirectory, args.dataset)
    train_loss_file = open(train_file, "w")
    valid_loss_file = open(valid_file, "w")
    
    train_file_angle = "{}/{}_train_angle_loss.txt".format(ExperimentDirectory, args.dataset)
    valid_file_angle = "{}/{}_valid_angle_loss.txt".format(ExperimentDirectory, args.dataset)
    train_loss_angle_file = open(train_file_angle, "w")
    valid_loss_angle_file = open(valid_file_angle, "w")
    
    for Epoch in range(0, 1):
        start_time = time.perf_counter()
        print("\nTesting Epoch: %d" % Epoch)
        val_loss = evaluate_model(ExperimentDirectory, model, args.dataset, nGPUs, cfg, train_loss_file, valid_loss_file, train_loss_angle_file, valid_loss_angle_file, 
                                  valid_loader, Optimizer, segmentation_loss, orientation_loss, 
                                  LR_scheduler, Epoch, n_RoadClasses, n_OrientClasses)
        end_time = time.perf_counter()
        print("Time Elapsed for epoch => {1}".format(Epoch, end_time - start_time))
    
if __name__=="__main__":
    best_accuracy = 0
    best_miou = 0
    evaluate()