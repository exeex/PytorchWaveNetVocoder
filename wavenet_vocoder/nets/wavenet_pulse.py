# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

from .wavenet_utils import CausalConv1d, UpSampling, OneHot
from .wavenet import WaveNet


class WaveNetPulse(WaveNet):

    def __init__(self, n_quantize=256, n_aux=28, n_p=4, n_resch=512, n_skipch=256,
                 dilation_depth=10, dilation_repeat=3, kernel_size=2, upsampling_factor=0):

        super(WaveNetPulse, self).__init__(n_quantize, n_aux, n_resch, n_skipch,
                                           dilation_depth, dilation_repeat, kernel_size, upsampling_factor)

        self.n_p = n_p

        # for residual blocks
        self.p_1x1_sigmoid = nn.ModuleList()
        self.p_1x1_tanh = nn.ModuleList()
        for _ in self.dilations:
            self.p_1x1_sigmoid += [nn.Conv1d(self.n_p, self.n_resch, 1)]
            self.p_1x1_tanh += [nn.Conv1d(self.n_p, self.n_resch, 1)]

    def forward(self, x, h, p=None):
        """FORWARD CALCULATION.

        Args:
            x (Tensor): Long tensor variable with the shape (B, T).
            p (Tensor): Float tensor variable with the shape (B, C, T).
            h (Tensor): Float tensor variable with the shape (B, n_aux, T),

        Returns:
            Tensor: Float tensor variable with the shape (B, T, n_quantize).

        """
        # preprocess
        output = self._preprocess(x)  # dilated conv x
        if self.upsampling_factor > 0:
            h = self.upsampling(h)

        # residual block
        skip_connections = []
        for l in range(len(self.dilations)):
            output, skip = self._residual_forward(output, h,
                                                  self.dil_sigmoid[l], self.dil_tanh[l],
                                                  self.aux_1x1_sigmoid[l], self.aux_1x1_tanh[l],
                                                  self.skip_1x1[l], self.res_1x1[l],
                                                  p, self.p_1x1_sigmoid[l], self.p_1x1_tanh[l], )
            skip_connections.append(skip)

        # skip-connection part
        output = sum(skip_connections)
        output = self._postprocess(output)

        return output

    def _residual_forward(self, x, h,
                          dil_sigmoid, dil_tanh,
                          aux_1x1_sigmoid, aux_1x1_tanh,
                          skip_1x1, res_1x1,
                          p=None, p_1x1_sigmoid=None, p_1x1_tanh=None):
        """

        Visualization of tensor connection:
         ________________________(skip_1x1)______________________________[skip:resch]
        |                                                      |
       [x]_____(dil)___________(+)___________(tanh)____        |
            |__(dil)____________|____                  |       |
                                |    |                 |       |
       [h]_____(1x1)____________|    |                 |       |
            |__(1x1)____________|____|                (x)___(res_1x1)_____[output:qch]
                                |    |                 |
       [p]_____(1x1)____________|    |                 |
            |__(1x1)________________(+)______(sigm)____|


        """
        output_sigmoid = dil_sigmoid(x)
        aux_output_sigmoid = aux_1x1_sigmoid(h)
        p_output_sigmoid = p_1x1_sigmoid(p)

        output_tanh = dil_tanh(x)
        aux_output_tanh = aux_1x1_tanh(h)
        p_output_tanh = p_1x1_tanh(p)

        # print(output_sigmoid.shape, aux_output_sigmoid.shape, p_output_sigmoid.shape)
        output = torch.sigmoid(output_sigmoid + aux_output_sigmoid + p_output_sigmoid) * \
                 torch.tanh(output_tanh + aux_output_tanh + p_output_tanh)
        skip = skip_1x1(output)
        output = res_1x1(output)
        output = output + x
        return output, skip
