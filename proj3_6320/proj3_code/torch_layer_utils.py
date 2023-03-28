#useful util functions implemented with pytorch

import torch
from torch import nn
import numpy as np
from math import floor
"""
Image gradients are needed for both SIFT and the Harris Corner Detector, so we
implement the necessary code only once, here.
"""


class ImageGradientsLayer(torch.nn.Module):
    """
    ImageGradientsLayer: Compute image gradients Ix & Iy. This can be
    approximated by convolving with Sobel filter.
    """
    def __init__(self):
        super().__init__()

        # Create convolutional layer
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3,
            bias=False, padding=(1,1), padding_mode='zeros')

        # Instead of learning weight parameters, here we set the filter to be
        # Sobel filter
        self.conv2d.weight = get_sobel_xy_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of ImageGradientsLayer. We'll test with a
        single-channel image, and 1 image at a time (batch size = 1).

        Args:
        -   x: input tensor of size (num_image, channel, height, width)

        Returns:
        -   output: output of HarrisNet network, (num_image, 2, height, width)
            tensor for Ix and Iy, respectively.
        """
        return self.conv2d(x)


def get_gaussian_kernel(ksize=7, sigma=5) -> torch.nn.Parameter:
    """
    Generate a Gaussian kernel to be used in HarrisNet for calculating a second moment matrix
    (SecondMomentMatrixLayer). You can call this function to get the 2D gaussian filter.

    Since you already implement this in Proj2 we won't be grading this part again, but it is
    important that you get the correct value here in order to pass the unit tests for HarrisNet.

    This might be useful:
    1) We suggest using the outer product trick, it's faster and simpler. And you are less likely to mess up the value.
    2) Make sure the value sum to 1
    3) Some useful torch functions:
    - torch.mm https://pytorch.org/docs/stable/torch.html#torch.mm
    - torch.t https://pytorch.org/docs/stable/torch.html#torch.t
    4) Similar to get_sobel_xy_parameters, you should return the filter in torch.nn.Parameter.


    Args:
    -   ksize: kernel size
    -   sigma: kernel standard deviation

    Returns:
    -   kernel: torch.nn.Parameter of size [ksize, ksize]
    """

    ############################
    ### TODO: YOUR CODE HERE ###
    norm_mu = int(ksize / 2)
    gaussian_filter = np.zeros(ksize)
    for i in range(ksize):
        exponent = -(((i - norm_mu) ** 2)) /(2 * (sigma ** 2))
        gaussian_filter[i] = np.exp(exponent)

    gaussian_filter = np.expand_dims(gaussian_filter, axis=0) / gaussian_filter.sum()
    gaussian_filter = np.matmul(np.transpose(gaussian_filter), gaussian_filter)
    gaussian_filter = np.reshape(gaussian_filter,(1,1,ksize,ksize)).astype(np.float32)

    kernel = nn.Parameter(torch.from_numpy(gaussian_filter))

# #     raise NotImplementedError('`get_gaussian_kernel` need to be '
# #         + 'implemented')

#     ### END OF STUDENT CODE ####
#     ############################
    

    return kernel


def get_sobel_xy_parameters() -> torch.nn.Parameter:
    """
    Populate the conv layer weights for the Sobel layer (image gradient
    approximation).

    There should be two sets of filters: each should have size (1 x 3 x 3)
    for 1 channel, 3 pixels in height, 3 pixels in width. When combined along
    the batch dimension, this conv layer should have size (2 x 1 x 3 x 3), with
    the Sobel_x filter first, and the Sobel_y filter second.

    Args:
    -   None
    Returns:
    -   kernel: Torch parameter representing (2, 1, 3, 3) conv filters
    """

    ############################
    ### TODO: YOUR CODE HERE ###
    sobel_x = np.array(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]]).astype(np.float32)
    sobel_y = np.array(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]]).astype(np.float32)
    sobel_filter = np.concatenate(
        [sobel_x.reshape(1,1,3,3),
         sobel_y.reshape(1,1,3,3)], axis=0)
    kernel = torch.nn.Parameter(torch.from_numpy(sobel_filter))

#     raise NotImplementedError('`get_sobel_xy_parameters` need to be '
#         + 'implemented')

    ### END OF STUDENT CODE ####
    ############################

    return kernel
