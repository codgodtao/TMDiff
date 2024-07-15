#################################################################################
#                                 MOE_Conv                                 #
#################################################################################
import torch
from torch import autograd, nn as nn



def get_device(x):
    gpu_idx = x.get_device()
    return f"cuda:{gpu_idx}" if gpu_idx >= 0 else "cpu"

import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.router = nn.Linear(in_channels,out_channels)

    def forward(self,x):
        return self.router(x)
    

class MoEConv(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(MoEConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.router = Router(512,out_channels)

    def forward(self, x,y):
        # Assuming router function returns a tensor of shape (batch_size, n_expert)
        self.scores = self.router(y)  # Shape: (batch_size, n_expert)

        # Perform the convolution operation
        out = super(MoEConv, self).forward(x)  # Shape: (batch_size, out_channels * n_expert, D, H, W)

        # Apply scores to select channels
        scores_expanded = self.scores.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # Shape: (batch_size, C, 1, 1, 1)
        out_selected = (out * scores_expanded).sum(dim=1)  # Shape: (batch_size, out_channels, D, H, W)

        return out_selected


# Example usage
in_channels = 3
out_channels = 32
kernel_size = 3
x = torch.randn(8, in_channels, 32, 32, 32)  # Example input tensor

moe_conv = MoEConv(in_channels, out_channels, kernel_size)
y = torch.randn(8,512)
output = moe_conv(x,y)
print(output.shape)  # Should be (8, out_channels, 30, 30, 30) if stride=1 and padding=1


import torch
import torch.nn as nn

class Router(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.router = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.router(x)

class MoEConv(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, top_k=5):
        super(MoEConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.router = Router(512, out_channels)
        self.top_k = top_k

    def forward(self, x, y):
        # Assuming router function returns a tensor of shape (batch_size, n_expert)
        self.scores = self.router(y)  # Shape: (batch_size, n_expert)

        # Get the top-K scores and their indices
        topk_scores, topk_indices = torch.topk(self.scores, self.top_k, dim=1)

        # Create a mask with 1s at the top-K indices and 0s elsewhere
        mask = torch.zeros_like(self.scores)
        mask.scatter_(1, topk_indices, 1)

        # Perform the convolution operation
        out = super(MoEConv, self).forward(x)  # Shape: (batch_size, out_channels * n_expert, D, H, W)

        # Apply the mask to the scores
        scores_expanded = mask.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # Shape: (batch_size, C, 1, 1, 1)
        out_selected = (out * scores_expanded).sum(dim=1)  # Shape: (batch_size, out_channels, D, H, W)

        return out_selected

# Example usage
in_channels = 3
out_channels = 16
kernel_size = 3
top_k = 5

model = MoEConv(in_channels, out_channels, kernel_size, top_k=top_k)
x = torch.randn(1, in_channels, 10, 10, 10)  # Example input tensor
y = torch.randn(1, 512)  # Example input for the router

output = model(x, y)
print(output.shape)  # Should be (1, out_channels, D, H, W)
