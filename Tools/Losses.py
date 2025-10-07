import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#Segmentations
class mIoULoss(nn.Module):
    def __init__(self, weight=None, n_classes=2):
        super().__init__()
        self.classes = n_classes
        self.epsilon = 1e-8
        self.weights = weight * weight

    def forward(self, inputs, target):
        N = inputs.size()[0]
        ntar, htar, wtar = target.size()
        target_oneHot = torch.zeros(ntar, self.classes, htar, wtar, dtype = torch.long).cuda()
        target_oneHot = target_oneHot.scatter_(1, target.long().view(ntar, 1, htar, wtar), 1)
        inputs = F.softmax(inputs, dim=1)
        inter = inputs * target_oneHot
        inter = inter.view(N, self.classes, -1).sum(2)
        union = inputs + target_oneHot - (inputs * target_oneHot)
        union = union.view(N, self.classes, -1).sum(2)
        loss = (self.weights * inter) / (self.weights * union + self.epsilon)
        
        return -torch.mean(loss)

# Orientations # https://discuss.pytorch.org/t/nllloss-vs-crossentropyloss/92777
class CrossEntropyLossImage(nn.Module): #https://discuss.pytorch.org/t/multi-class-cross-entropy-loss-function-implementation-in-pytorch/19077/12
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super().__init__()
        self.CE_loss = nn.CrossEntropyLoss(weight = weight, ignore_index = ignore_index, reduction = reduction)

    def forward(self, inputs, targets):
        return self.CE_loss(inputs, targets.long().cuda())


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Paper: https://arxiv.org/abs/1708.02002
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # class weights
        self.gamma = gamma  # focusing parameter (default 2.0)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C, H, W] logits
            targets: [N, H, W] long tensor with class indices
        """
        targets = targets.long()
        C = inputs.size(1)

        # Get probabilities
        p = F.softmax(inputs, dim=1)

        # Flatten for easier computation
        inputs_flat = inputs.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [N*H*W, C]
        targets_flat = targets.view(-1)  # [N*H*W]
        p_flat = p.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [N*H*W, C]

        # Create mask for valid pixels (not ignore_index)
        valid_mask = targets_flat != self.ignore_index

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        # Apply mask
        inputs_valid = inputs_flat[valid_mask]
        targets_valid = targets_flat[valid_mask]
        p_valid = p_flat[valid_mask]

        # Get the probability of the true class
        p_t = p_valid[torch.arange(p_valid.size(0)), targets_valid]

        # Calculate focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Calculate cross entropy
        ce_loss = F.cross_entropy(inputs_valid, targets_valid, reduction='none')

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Apply alpha balancing if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets_valid]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BoundaryAwareLoss(nn.Module):
    """
    Boundary-Aware Loss that emphasizes pixels near road boundaries
    Helps with thin linear structures like roads
    """
    def __init__(self, weight=None, ignore_index=255, boundary_weight=2.0, kernel_size=5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')
        self.boundary_weight = boundary_weight
        self.kernel_size = kernel_size

    def get_boundary_mask(self, targets):
        """
        Extract boundary pixels using morphological operations
        Args:
            targets: [N, H, W] class labels
        Returns:
            boundary_mask: [N, H, W] with 1 at boundaries, 0 elsewhere
        """
        N, H, W = targets.shape
        boundary_mask = torch.zeros_like(targets, dtype=torch.float32)

        # Convert to numpy for morphological operations
        for i in range(N):
            target_np = targets[i].cpu().numpy().astype(np.uint8)

            # Skip if all pixels are same class
            if len(np.unique(target_np)) == 1:
                continue

            # Create binary mask for road class (assuming road class is 1)
            road_mask = (target_np == 1).astype(np.uint8)

            # Dilate and erode to find boundaries
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))
            dilated = cv2.dilate(road_mask, kernel, iterations=1)
            eroded = cv2.erode(road_mask, kernel, iterations=1)

            # Boundary is the difference between dilated and eroded
            boundary = dilated - eroded
            boundary_mask[i] = torch.from_numpy(boundary).float()

        return boundary_mask.cuda()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C, H, W] logits
            targets: [N, H, W] long tensor with class indices
        """
        targets = targets.long()

        # Get pixel-wise cross entropy loss
        ce = self.ce_loss(inputs, targets)  # [N, H, W]

        # Get boundary mask
        boundary_mask = self.get_boundary_mask(targets)  # [N, H, W]

        # Apply higher weight to boundary pixels
        weights = 1.0 + boundary_mask * (self.boundary_weight - 1.0)

        # Weight the loss
        weighted_loss = ce * weights

        # Average over valid pixels
        valid_mask = targets != 255
        if valid_mask.sum() > 0:
            loss = (weighted_loss * valid_mask.float()).sum() / valid_mask.float().sum()
        else:
            loss = weighted_loss.mean()

        return loss


class CombinedLoss(nn.Module):
    """
    Combines multiple loss functions with configurable weights
    """
    def __init__(self, focal_weight=1.0, boundary_weight=0.5,
                 alpha=None, gamma=2.0, ignore_index=255):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
        self.boundary_loss = BoundaryAwareLoss(weight=alpha, ignore_index=ignore_index,
                                               boundary_weight=2.0, kernel_size=5)
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight

    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        boundary = self.boundary_loss(inputs, targets)
        return self.focal_weight * focal + self.boundary_weight * boundary