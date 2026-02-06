#!/bin/bash

# Make script exit if any command fails
set -e

echo "Starting evaluation for LinkNet..."
python eval.py -m LinkNet -d DeepGlobe -e Train_LinkNet -r Experiments/Train_LinkNet/model_best.pth.tar

echo "Starting evaluation for PSPNet..."
python eval.py -m PSPNet -d DeepGlobe -e Train_PSPNet -r Experiments/Train_PSPNet/model_best.pth.tar

echo "Starting evaluation for SegNet..."
python eval.py -m SegNet -d DeepGlobe -e Train_SegNet -r Experiments/Train_SegNet/model_best.pth.tar

echo "Starting evaluation for UNet..."
python eval.py -m UNet -d DeepGlobe -e Train_UNet -r Experiments/Train_UNet/model_best.pth.tar

echo "Starting evaluation for UNetPlusPlus..."
python eval.py -m UNetPlusPlus -d DeepGlobe -e Train_UNetPlusPlus -r Experiments/Train_UNetPlusPlus/model_best.pth.tar

echo "Starting evaluation for ConvNeXt_UPerNet_DGCN_MTL..."
python eval.py -m ConvNeXt_UPerNet_DGCN_MTL -d DeepGlobe -e ConvNeXt_UPerNet_DGCN_MTL -r Experiments/ConvNeXt_UPerNet_DGCN_MTL/model_best.pth.tar

echo "All evaluations completed successfully!"
