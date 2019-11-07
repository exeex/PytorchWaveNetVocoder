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
        # output = output + x[:, :, -1:]  # B x C x 1
        return output, skip

    def _generate_residual_forward(self, x, h,
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
        # output = output + x
        output = output + x[:, :, -1:]  # B x C x 1
        return output, skip

    def batch_fast_generate(self, x, h, n_samples_list, *args, intervals=None, mode="sampling"):
        """GENERATE WAVEFORM WITH FAST ALGORITHM IN BATCH MODE.

        Args:
            x (tensor): Long tensor variable with the shape (B, T).
            h (tensor): Float tensor variable with the shape (B, n_aux, max(n_samples_list) + T).
            n_samples_list (list): List of number of samples to be generated (B,).
            intervals (int): Log interval.
            mode (str): "sampling" or "argmax".

        Returns:
            list: List of ndarray which is generated quantized wavenform.

        """

        p = args[0]

        # get min max length
        max_n_samples = max(n_samples_list)
        min_n_samples = min(n_samples_list)
        min_idx = np.argmin(n_samples_list)

        # upsampling
        if self.upsampling_factor > 0:
            h = self.upsampling(h)

        # padding if the length less than
        n_pad = self.receptive_field - x.size(1)
        if n_pad > 0:
            x = F.pad(x, (n_pad, 0), "constant", self.n_quantize // 2)
            p = F.pad(p, (n_pad, 0), "constant", self.n_quantize // 2)
            h = F.pad(h, (n_pad, 0), "replicate")

        # prepare buffer
        output = self._preprocess(x)
        h_ = h[:, :, :x.size(1)]
        output_buffer = []
        buffer_size = []
        for l, d in enumerate(self.dilations):
            output, _ = self._residual_forward(output, h_,
                                               self.dil_sigmoid[l], self.dil_tanh[l],
                                               self.aux_1x1_sigmoid[l], self.aux_1x1_tanh[l],
                                               self.skip_1x1[l], self.res_1x1[l],
                                               p, self.p_1x1_sigmoid[l], self.p_1x1_tanh[l])
            if d == 2 ** (self.dilation_depth - 1):
                buffer_size.append(self.kernel_size - 1)
            else:
                buffer_size.append(d * 2 * (self.kernel_size - 1))
            output_buffer.append(output[:, :, -buffer_size[l] - 1: -1])

        # generate
        samples = x  # B x T
        end_samples = []
        start = time.time()
        for i in range(max_n_samples):
            output = samples[:, -self.kernel_size * 2 + 1:]
            output = self._preprocess(output)  # B x C x T
            h_ = h[:, :, samples.size(-1) - 1].contiguous().unsqueeze(-1)  # B x C x 1
            output_buffer_next = []
            skip_connections = []
            for l, d in enumerate(self.dilations):
                output, skip = self._generate_residual_forward(
                    output, h_, self.dil_sigmoid[l], self.dil_tanh[l],
                    self.aux_1x1_sigmoid[l], self.aux_1x1_tanh[l],
                    self.skip_1x1[l], self.res_1x1[l])
                output = torch.cat([output_buffer[l], output], dim=2)
                output_buffer_next.append(output[:, :, -buffer_size[l]:])
                skip_connections.append(skip)

            # update buffer
            output_buffer = output_buffer_next

            # get predicted sample
            output = sum(skip_connections)
            output = self._postprocess(output)[:, -1]  # B x n_quantize
            if mode == "sampling":
                posterior = F.softmax(output, dim=-1)
                dist = torch.distributions.Categorical(posterior)
                sample = dist.sample()  # B
            elif mode == "argmax":
                sample = output.argmax(-1)  # B
            else:
                logging.error("mode should be sampling or argmax")
                sys.exit(1)
            samples = torch.cat([samples, sample.view(-1, 1)], dim=1)

            # show progress
            if intervals is not None and (i + 1) % intervals == 0:
                logging.info("%d/%d estimated time = %.3f sec (%.3f sec / sample)" % (
                    i + 1, max_n_samples,
                    (max_n_samples - i - 1) * ((time.time() - start) / intervals),
                    (time.time() - start) / intervals))
                start = time.time()

            # check length
            if (i + 1) == min_n_samples:
                while True:
                    # get finished sample
                    end_samples += [samples[min_idx, -min_n_samples:].cpu().numpy()]
                    # get index of unfinished samples
                    idx_list = [idx for idx in range(len(n_samples_list)) if idx != min_idx]
                    if len(idx_list) == 0:
                        # break when all of samples are finished
                        break
                    else:
                        # remove finished sample
                        samples = samples[idx_list]
                        h = h[idx_list]
                        output_buffer = [out_[idx_list] for out_ in output_buffer]
                        del n_samples_list[min_idx]
                        # update min length
                        prev_min_n_samples = min_n_samples
                        min_n_samples = min(n_samples_list)
                        min_idx = np.argmin(n_samples_list)

                    # break when there is no same length samples
                    if min_n_samples != prev_min_n_samples:
                        break

        return end_samples
