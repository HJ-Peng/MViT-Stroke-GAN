import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

        # Use groups=3 for channel-wise convolution: each channel is processed independently
        self.filter_x = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
        self.filter_y = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)

        # Single-channel Sobel kernels
        sobel_kernel_x = torch.tensor([[[[-1.0, 0.0, 1.0],
                                         [-2.0, 0.0, 2.0],
                                         [-1.0, 0.0, 1.0]]]])  # [1,1,3,3]
        sobel_kernel_y = torch.tensor([[[[-1.0, -2.0, -1.0],
                                         [0.0, 0.0, 0.0],
                                         [1.0, 2.0, 1.0]]]])  # [1,1,3,3]

        # Expand to 3 output channels, one for each input channel
        # shape: [1,1,3,3] -> [3,1,3,3]
        sobel_kernel_x = sobel_kernel_x.repeat(3, 1, 1, 1)  # 注意：repeat(3, 1, 1, 1)
        sobel_kernel_y = sobel_kernel_y.repeat(3, 1, 1, 1)

        self.filter_x.weight = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.filter_y.weight = nn.Parameter(sobel_kernel_y, requires_grad=False)

    def forward(self, pred, target):
        """
        Compute gradient L1 loss between predicted and target images.

        Args:
            pred (torch.Tensor): Predicted image (B, 3, H, W)
            target (torch.Tensor): Target image (B, 3, H, W)+

        Returns:
            torch.Tensor: Scalar loss value (sum of L1 losses in x and y directions)
        """
        pred_grad_x = self.filter_x(pred)
        pred_grad_y = self.filter_y(pred)
        target_grad_x = self.filter_x(target)
        target_grad_y = self.filter_y(target)

        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)

        return loss_x + loss_y