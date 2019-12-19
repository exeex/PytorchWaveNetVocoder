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


class Residual1d(nn.Module):
    def __init__(self, *modules):
        super(Residual1d, self).__init__()
        self.seq = nn.Sequential(*modules)
        self.weight = nn.Parameter(torch.ones(1), requires_grad=True)
        # TODO: try bias true

    def forward(self, x):
        skip = x
        x = self.seq(x)
        y = x * self.weight.expand_as(x) + skip
        return y


class PulseConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, bias=True):
        super(PulseConv1d, self).__init__(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size, bias=bias)
        self.__p_size = kernel_size

        # TODO: try bias true

    def forward(self, x):
        y = super(PulseConv1d, self).forward(x)
        # y = y[:, :, : -self.__p_size - 1]
        y = y[:, :, self.__p_size + 1:]
        return y


class UpSamplingSmooth(nn.Module):
    """UPSAMPLING LAYER WITH DECONVOLUTION.

    Args:
        upsampling_factor (int): Upsampling factor.

    """

    def __init__(self, upsampling_factor, p_ch=50, kernel_size=None):
        super(UpSamplingSmooth, self).__init__()
        self.upsampling_factor = upsampling_factor
        # self.upsample_layer = nn.Upsample(scale_factor=self.upsampling_factor)
        self.upsample_layer = UpSampling(upsampling_factor)

        # convs = []
        # convs.append(Residual1d(PulseConv1d(p_ch, p_ch, kernel_size=10),
        #                         PulseConv1d(p_ch, p_ch, kernel_size=10),
        #                         nn.BatchNorm1d(p_ch),
        #                         nn.LeakyReLU()))
        # convs.append(Residual1d(PulseConv1d(p_ch, p_ch, kernel_size=10),
        #                         PulseConv1d(p_ch, p_ch, kernel_size=10),
        #                         nn.BatchNorm1d(p_ch),
        #                         nn.LeakyReLU()))
        #
        # self.convs = nn.Sequential(*convs)

        if not kernel_size:
            self.kernel_size = upsampling_factor
        else:
            self.kernel_size = kernel_size

        # box blur
        self.smooth_kernel = nn.Parameter(torch.ones(1, 1, self.kernel_size) / self.kernel_size,
                                          requires_grad=False)

    def forward(self, x):
        """FORWARD CALCULATION.

        Args:
            x (Tensor): Float tensor variable with the shape (B, C, T).

        Returns:
            Tensor: Float tensor variable with the shape (B, C, T'),
                where T' = T * upsampling_factor.

        """
        x = self.upsample_layer(x)  # B x C x T'
        # x = self.convs(x)
        x = F.conv1d(x, self.smooth_kernel.expand([x.shape[1], -1, -1]), groups=x.shape[1],
                     padding=self.kernel_size)
        # return x[:, :, : -self.upsampling_factor - 1]
        return x[:, :, self.kernel_size + 1:]


### test code ###
# conv1d = PulseConv1d(1, 1, 3)
# x = torch.FloatTensor([0,0,1,0,0,0,0,0,1,0,0,0,0])
# x = x.unsqueeze(0).unsqueeze(0)
# print(x.shape)
# y = conv1d(x)
# print(x.shape, y.shape)
#
# import matplotlib.pyplot as plt
#
# plt.plot(x[0,0,:])
# plt.show()
# plt.plot(y[0,0,:].detach().numpy())
# plt.show()


class WaveNetPulse(WaveNet):

    def __init__(self, n_quantize=256, n_aux=28, n_p=4, n_resch=512, n_skipch=256,
                 dilation_depth=10, dilation_repeat=3, kernel_size=2, upsampling_factor=0):

        super(WaveNetPulse, self).__init__(n_quantize, n_aux, n_resch, n_skipch,
                                           dilation_depth, dilation_repeat, kernel_size, upsampling_factor)

        logging.info("Now you are using Wavenet PULSE version!!!")
        self.n_p = n_p  # 12

        mcep_ch = 25
        p_ch = mcep_ch * 2
        self.p_conv = []
        self.p_conv.append(PulseConv1d(self.n_p, p_ch, kernel_size=24))
        # self.p_conv.append(Residual1d(PulseConv1d(p_ch, p_ch, kernel_size=24),
        #                               PulseConv1d(p_ch, p_ch, kernel_size=24),
        #                               nn.BatchNorm1d(p_ch),
        #                               nn.LeakyReLU()))
        self.p_conv = nn.Sequential(*self.p_conv)

        self.mcep_norm = nn.Sequential(nn.BatchNorm1d(p_ch), nn.LeakyReLU(p_ch))  # try?

        self.upsampling_mcep = nn.Sequential(nn.Conv1d(mcep_ch, p_ch, 1), UpSamplingSmooth(upsampling_factor))

        # for residual blocks
        # self.p_1x1_sigmoid = nn.ModuleList()
        # self.p_1x1_tanh = nn.ModuleList()
        self.p_dil_sigmoid = nn.ModuleList()
        self.p_dil_tanh = nn.ModuleList()
        for d in self.dilations:
            # self.p_1x1_sigmoid += [nn.Conv1d(p_ch, self.n_resch, 1)]
            # self.p_1x1_tanh += [nn.Conv1d(p_ch, self.n_resch, 1)]
            self.p_dil_sigmoid += [CausalConv1d(p_ch, self.n_resch, self.kernel_size, d)]
            self.p_dil_tanh += [CausalConv1d(p_ch, self.n_resch, self.kernel_size, d)]

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

        mcep = h[:, 1:-1, :]  # extract mcep
        # h : (vuv[1]+mcep[25]+ap_code[1])
        h = torch.cat([h[:, 0:1, :], h[:, -1:, :]], axis=1)  # extract vuv ap_code

        if self.upsampling_factor > 0:
            h = self.upsampling(h)
            # print(self.upsampling_factor, h.shape)
            mcep = self.upsampling_mcep(mcep)
            # print(h.shape)

        # p = p.unsqueeze(1)
        # residual block
        p = self.p_conv(p)
        skip_connections = []
        for l in range(len(self.dilations)):
            output, skip = self._residual_forward(output, h,
                                                  self.dil_sigmoid[l], self.dil_tanh[l],
                                                  self.aux_1x1_sigmoid[l], self.aux_1x1_tanh[l],
                                                  self.skip_1x1[l], self.res_1x1[l],
                                                  p, self.p_dil_sigmoid[l], self.p_dil_tanh[l], mcep)
            skip_connections.append(skip)

        # skip-connection part
        output = sum(skip_connections)
        output = self._postprocess(output)

        return output

    def _residual_forward(self, x, h,
                          dil_sigmoid, dil_tanh,
                          aux_1x1_sigmoid, aux_1x1_tanh,
                          skip_1x1, res_1x1,
                          p=None, p_dil_sigmoid=None, p_dil_tanh=None, mcep=None):
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
       [p]______(*)______(1x1)__|    |                 |
       [mcep]____|   |___(1x1)______(+)______(sigm)____|


        """
        mcep = self.mcep_norm(mcep)  # TODO: check mcep mean, max, min?
        p = p * torch.relu(mcep)

        output_sigmoid = dil_sigmoid(x)
        aux_output_sigmoid = aux_1x1_sigmoid(h)
        # p_output_sigmoid = p_1x1_sigmoid(p)

        output_tanh = dil_tanh(x)
        # aux_output_tanh = aux_1x1_tanh(h)
        p_output_tanh = p_dil_tanh(p)

        # print(output_sigmoid.shape, aux_output_sigmoid.shape, p_output_sigmoid.shape)
        output = torch.sigmoid(output_sigmoid + aux_output_sigmoid) * \
                 torch.tanh(output_tanh + p_output_tanh)

        # output = torch.sigmoid(output_sigmoid + aux_output_sigmoid + p_output_sigmoid) * \
        #          torch.tanh(output_tanh + aux_output_tanh + p_output_tanh)

        skip = skip_1x1(output)
        output = res_1x1(output)
        # output = output + x
        output = output + x[:, :, -output.shape[2]:]  # B x C x T_output
        return output, skip

    def _generate_residual_forward(self, x, h,
                                   dil_sigmoid, dil_tanh,
                                   aux_1x1_sigmoid, aux_1x1_tanh,
                                   skip_1x1, res_1x1,
                                   p=None, p_1x1_sigmoid=None, p_1x1_tanh=None):

        return self._residual_forward(x, h,
                                      dil_sigmoid, dil_tanh,
                                      aux_1x1_sigmoid, aux_1x1_tanh,
                                      skip_1x1, res_1x1,
                                      p, p_1x1_sigmoid, p_1x1_tanh)

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
            p = F.pad(p, (n_pad, 0), "constant", 0)
            h = F.pad(h, (n_pad, 0), "replicate")

        p = self.p_conv(p)

        # prepare buffer
        output = self._preprocess(x)
        h_ = h[:, :, :x.size(1)]
        p_ = p[:, :, :x.size(1)]
        output_buffer = []
        buffer_size = []
        for l, d in enumerate(self.dilations):
            output, _ = self._residual_forward(output, h_,
                                               self.dil_sigmoid[l], self.dil_tanh[l],
                                               self.aux_1x1_sigmoid[l], self.aux_1x1_tanh[l],
                                               self.skip_1x1[l], self.res_1x1[l],
                                               p_, self.p_dil_sigmoid[l], self.p_dil_tanh[l])
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
            # _p = p[:, -self.kernel_size * 2 + 1:]
            output = self._preprocess(output)  # B x C x T
            h_ = h[:, :, samples.size(-1) - 1].contiguous().unsqueeze(-1)  # B x C x 1
            p_ = p[:, :, samples.size(-1) - 1].contiguous().unsqueeze(-1)  # B x C x 1
            output_buffer_next = []
            skip_connections = []
            for l, d in enumerate(self.dilations):
                # a = output.shape
                output, skip = self._generate_residual_forward(output, h_,
                                                               self.dil_sigmoid[l], self.dil_tanh[l],
                                                               self.aux_1x1_sigmoid[l], self.aux_1x1_tanh[l],
                                                               self.skip_1x1[l], self.res_1x1[l],
                                                               p_, self.p_1x1_sigmoid[l], self.p_1x1_tanh[l])
                # print(output.shape, skip.shape)

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
                        p = p[idx_list]
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

    def fast_generate(self, x, h, n_samples, *args, intervals=None, mode="sampling"):
        """GENERATE WAVEFORM WITH FAST ALGORITHM.

        Args:
            x (tensor): Long tensor variable with the shape  (1, T).
            h (tensor): Float tensor variable with the shape  (1, n_aux, n_samples + T).
            n_samples (int): Number of samples to be generated.
            intervals (int): Log interval.
            mode (str): "sampling" or "argmax".

        Returns:
            ndarray: Generated quantized wavenform (n_samples,).

        References:
            Fast Wavenet Generation Algorithm: https://arxiv.org/abs/1611.09482

        """
        # upsampling
        if self.upsampling_factor > 0:
            h = self.upsampling(h)

        # padding if the length less than
        n_pad = self.receptive_field - x.size(1)
        if n_pad > 0:
            x = F.pad(x, (n_pad, 0), "constant", self.n_quantize // 2)
            h = F.pad(h, (n_pad, 0), "replicate")

        # prepare buffer
        output = self._preprocess(x)
        h_ = h[:, :, :x.size(1)]
        output_buffer = []
        buffer_size = []
        for l, d in enumerate(self.dilations):
            output, _ = self._residual_forward(
                output, h_, self.dil_sigmoid[l], self.dil_tanh[l],
                self.aux_1x1_sigmoid[l], self.aux_1x1_tanh[l],
                self.skip_1x1[l], self.res_1x1[l])
            if d == 2 ** (self.dilation_depth - 1):
                buffer_size.append(self.kernel_size - 1)
            else:
                buffer_size.append(d * 2 * (self.kernel_size - 1))
            output_buffer.append(output[:, :, -buffer_size[l] - 1: -1])

        # generate
        samples = x[0]
        start = time.time()
        for i in range(n_samples):
            output = samples[-self.kernel_size * 2 + 1:].unsqueeze(0)
            output = self._preprocess(output)
            h_ = h[:, :, samples.size(0) - 1].contiguous().view(1, self.n_aux, 1)
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
            output = self._postprocess(output)[0]
            if mode == "sampling":
                posterior = F.softmax(output[-1], dim=0)
                dist = torch.distributions.Categorical(posterior)
                sample = dist.sample().unsqueeze(0)
            elif mode == "argmax":
                sample = output.argmax(-1)
            else:
                logging.error("mode should be sampling or argmax")
                sys.exit(1)
            samples = torch.cat([samples, sample], dim=0)

            # show progress
            if intervals is not None and (i + 1) % intervals == 0:
                logging.info("%d/%d estimated time = %.3f sec (%.3f sec / sample)" % (
                    i + 1, n_samples,
                    (n_samples - i - 1) * ((time.time() - start) / intervals),
                    (time.time() - start) / intervals))
                start = time.time()

        return samples[-n_samples:].cpu().numpy()
