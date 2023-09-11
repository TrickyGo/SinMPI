import torch
from torch import nn 
import torch.nn.functional as F 

class GaussianFilter(nn.Module):
    def __init__(self, ksize=5, sigma=None):
        super(GaussianFilter, self).__init__()
        # initialize guassian kernel
        if sigma is None:
            sigma = 0.3 * ((ksize-1) / 2.0 - 1) + 0.8
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(ksize)
        x_grid = x_coord.repeat(ksize).view(ksize, ksize)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate the 2-dimensional gaussian kernel
        center = ksize // 2
        weight = torch.exp(-torch.sum((xy_grid - center)**2., dim=-1) / (2*sigma**2))
        # Make sure sum of values in gaussian kernel equals 1.
        weight /= torch.sum(weight)
        self.gaussian_weight = weight

    def forward(self, x):
        return self.filter(x)

class TrainableFilter(nn.Module):
    def __init__(self, ksize=5, sigma_space=None, sigma_density=1):
        super(TrainableFilter, self).__init__()
        # initialization
        if sigma_space is None:
            self.sigma_space = 0.3 * ((ksize-1) * 0.5 - 1) + 0.8
        else:
            self.sigma_space = sigma_space
        if sigma_density is None:
            self.sigma_density = self.sigma_space
        else:
            self.sigma_density = sigma_density

        self.pad = (ksize-1) // 2
        self.ksize = ksize
        # get the spatial gaussian weight
        self.weight_space = GaussianFilter(ksize=self.ksize, sigma=self.sigma_space).gaussian_weight.cuda()
        # # create gaussian filter as convolutional layer
        self.weight_space = torch.nn.Parameter(self.weight_space)

    def forward(self, x):
        # Extracts sliding local patches from a batched input tensor.
        x_pad = F.pad(x, pad=[self.pad, self.pad, self.pad, self.pad], mode='reflect')
        x_patches = x_pad.unfold(2, self.ksize, 1).unfold(3, self.ksize, 1)
        patch_dim = x_patches.dim()

        # Calculate the 2-dimensional gaussian kernel
        diff_density = x_patches - x.unsqueeze(-1).unsqueeze(-1)
        weight_density = torch.exp(-(diff_density ** 2) / (2 * self.sigma_density ** 2))
        # Normalization
        # weight_density /= weight_density.sum(dim=(-1, -2), keepdim=True)
        weight_density = weight_density / weight_density.sum(dim=(-1, -2), keepdim=True)
        # print(weight_density.shape)

        # Keep same shape with weight_density
        weight_space_dim = (patch_dim - 2) * (1, ) + (self.ksize, self.ksize)
        weight_space = self.weight_space.view(*weight_space_dim).expand_as(weight_density)

        # get the final kernel weight
        weight = weight_density * weight_space
        weight_sum = weight.sum(dim=(-1, -2))
        x = (weight * x_patches).sum(dim=(-1, -2)) / weight_sum
        return x
