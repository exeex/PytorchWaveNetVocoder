import logging
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn


def encode_mu_law(x, mu=256):
    """PERFORM MU-LAW ENCODING.

    Args:
        x (ndarray): Audio signal with the range from -1 to 1.
        mu (int): Quantized level.

    Returns:
        ndarray: Quantized audio signal with the range from 0 to mu - 1.

    """
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5).astype(np.int64)


def decode_mu_law(y, mu=256):
    """PERFORM MU-LAW DECODING.

    Args:
        x (ndarray): Quantized audio signal with the range from 0 to mu - 1.
        mu (int): Quantized level.

    Returns:
        ndarray: Audio signal with the range from -1 to 1.

    """
    mu = mu - 1
    fx = (y - 0.5) / mu * 2 - 1
    x = np.sign(fx) / mu * ((1 + mu) ** np.abs(fx) - 1)
    return x


def initialize(m):
    """INITILIZE CONV WITH XAVIER.

    Arg:
        m (torch.nn.Module): Pytorch nn module instance.

    """
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        try:
            nn.init.constant_(m.bias, 0.0)
        except AttributeError:
            pass

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class OneHot(nn.Module):
    """CONVERT TO ONE-HOT VECTOR.

    Args:
        depth (int): Dimension of one-hot vector

    """

    def __init__(self, depth):
        super(OneHot, self).__init__()
        self.depth = depth

    def forward(self, x):
        """FORWARD CALCULATION.

        Arg:
            x (LongTensor): Long tensor variable with the shape (B, T).

        Returns:
            Tensor: Float tensor variable with the shape (B, depth, T).

        """
        x = x % self.depth
        x = torch.unsqueeze(x, 2)
        x_onehot = x.new_zeros(x.size(0), x.size(1), self.depth).float()

        return x_onehot.scatter_(2, x, 1)


class CausalConv1d(nn.Module):
    """1D DILATED CAUSAL CONVOLUTION."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        """FORWARD CALCULATION.

        Args:
            x (Tensor): Float tensor variable with the shape  (B, C, T).

        Returns:
            Tensor: Float tensor variable with the shape (B, C, T).

        """
        x = self.conv(x)
        if self.padding != 0:
            x = x[:, :, :-self.padding]
        return x


class UpSampling(nn.Module):
    """UPSAMPLING LAYER WITH DECONVOLUTION.

    Args:
        upsampling_factor (int): Upsampling factor.

    """

    def __init__(self, upsampling_factor, bias=True):
        super(UpSampling, self).__init__()
        self.upsampling_factor = upsampling_factor
        self.bias = bias
        self.conv = nn.ConvTranspose2d(1, 1,
                                       kernel_size=(1, self.upsampling_factor),
                                       stride=(1, self.upsampling_factor),
                                       bias=self.bias)

    def forward(self, x):
        """FORWARD CALCULATION.

        Args:
            x (Tensor): Float tensor variable with the shape (B, C, T).

        Returns:
            Tensor: Float tensor variable with the shape (B, C, T'),
                where T' = T * upsampling_factor.

        """
        x = x.unsqueeze(1)  # B x 1 x C x T
        x = self.conv(x)  # B x 1 x C x T'
        return x.squeeze(1)