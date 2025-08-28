import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):
    def __init__(self, alpha=1.0, threshold=0.1, weight_mode='linear'):
        super().__init__()
        self.alpha = alpha
        self.threshold = threshold
        self.weight_mode = weight_mode

        # Standard Sobel operators (normalized to [-1, 1])
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32) / 4.0
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32) / 4.0
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def get_edge(self, x):
        # Convert to grayscale using luminance-preserving weights (ITU-R BT.601)
        if x.size(1) == 3:
            gray = 0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2]
        else:
            gray = x[:, 0]
        gray = gray.unsqueeze(1)

        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, pred, target):
        """
        Compute weighted edge-aware L1 loss between predicted and target images.

        Args:
            pred (torch.Tensor): Predicted image (B, C, H, W)
            target (torch.Tensor): Ground truth image (B, C, H, W)

        Returns:
            torch.Tensor: Scalar weighted edge loss
        """
        pred_edge = self.get_edge(pred)
        target_edge = self.get_edge(target)
        l1_loss = torch.abs(pred_edge - target_edge)

        # Build weight map based on target edge strength
        if self.weight_mode == 'linear':
            weight_map = target_edge
        elif self.weight_mode == 'quadratic':
            weight_map = target_edge ** 2
        elif self.weight_mode == 'binary':
            thresh = self.threshold * target_edge.max() # Dynamic threshold
            weight_map = (target_edge > thresh).float()
        else:
            weight_map = torch.ones_like(target_edge)

        # Normalize weight map to maintain expected value of 1
        if self.weight_mode != 'uniform':
            weight_map = weight_map / (weight_map.mean() + 1e-8)

        loss = (l1_loss * weight_map).mean()
        return self.alpha * loss