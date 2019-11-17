#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import math
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torch.multiprocessing as mp

from sklearn.preprocessing import StandardScaler
from torchvision import transforms

from wavenet_vocoder.nets.wavenet_utils import decode_mu_law
from wavenet_vocoder.nets.wavenet_utils import encode_mu_law
from wavenet_vocoder.nets import WaveNet
from wavenet_vocoder.nets.wavenet_pulse import WaveNetPulse
from wavenet_vocoder.utils import extend_time
from wavenet_vocoder.utils import find_files
from wavenet_vocoder.utils import read_hdf5
from wavenet_vocoder.utils import read_txt
from wavenet_vocoder.utils import shape_hdf5


def pad_list(batch_list, pad_value=0.0):
    """PAD VALUE.

    Args:
        batch_list (list): List of batch, where the shape of i-th batch (T_i, C).
        pad_value (float): Value to pad.

    Returns:
        ndarray: Padded batch with the shape (B, T_max, C).

    """
    batch_size = len(batch_list)
    maxlen = max([batch.shape[0] for batch in batch_list])
    n_feats = batch_list[0].shape[-1]
    batch_pad = np.zeros((batch_size, maxlen, n_feats))
    for idx, batch in enumerate(batch_list):
        batch_pad[idx, :batch.shape[0]] = batch

    return batch_pad


def pad_along_axis(array: np.ndarray, target_length, axis=0):
    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)

    if pad_size < 0:
        return array.take(indices=range(target_length), axis=axis)

    npad = [(0, 0) for _ in range(axis_nb)]
    npad[axis] = (0, pad_size)

    b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)

    return b


def decode_generator(feat_list,
                     batch_size=32,
                     feature_type="world",
                     wav_transform=None,
                     feat_transform=None,
                     upsampling_factor=80,
                     use_upsampling_layer=True,
                     use_speaker_code=False,
                     use_pulse=True):
    """GENERATE DECODING BATCH.

    Args:
        feat_list (list): List of feature files.
        batch_size (int): Batch size in decoding.
        feature_type (str): Feature type.
        wav_transform (func): Preprocessing function for waveform.
        feat_transform (func): Preprocessing function for aux feats.
        upsampling_factor (int): Upsampling factor.
        use_upsampling_layer (bool): Whether to use upsampling layer.
        use_speaker_code (bool): Whether to use speaker code>

    Returns:
        generator: Generator instance.

    """
    # ---------------------------
    # sample-by-sample generation
    # ---------------------------
    if batch_size == 1:
        for featfile in feat_list:
            x = np.zeros((1))
            h = read_hdf5(featfile, "/" + feature_type)
            if not use_upsampling_layer:
                h = extend_time(h, upsampling_factor)
            if use_speaker_code:
                sc = read_hdf5(featfile, "/speaker_code")
                sc = np.tile(sc, [h.shape[0], 1])
                h = np.concatenate([h, sc], axis=1)

            # perform pre-processing
            if wav_transform is not None:
                x = wav_transform(x)
            if feat_transform is not None:
                h = feat_transform(h)

            if use_pulse:
                h = np.concatenate([h[:, 0:1], h[:, 2:]], axis=1)  # remove cont_f0_lpf

            # convert to torch variable
            x = torch.from_numpy(x).long()
            h = torch.from_numpy(h).float()
            x = x.unsqueeze(0)  # 1 => 1 x 1
            h = h.transpose(0, 1).unsqueeze(0)  # T x C => 1 x C x T

            # send to cuda
            if torch.cuda.is_available():
                x = x.cuda()
                h = h.cuda()

            # get target length and file id
            if not use_upsampling_layer:
                n_samples = h.size(2) - 1
            else:
                n_samples = h.size(2) * upsampling_factor - 1
            feat_id = os.path.basename(featfile).replace(".h5", "")

            yield feat_id, (x, h, n_samples)

    # ----------------
    # batch generation
    # ----------------
    else:
        # sort with the feature length
        shape_list = [shape_hdf5(f, "/" + feature_type)[0] for f in feat_list]

        idx = np.argsort(shape_list)
        feat_list = [feat_list[i] for i in idx]

        # divide into batch list
        n_batch = math.ceil(len(feat_list) / batch_size)
        batch_lists = np.array_split(feat_list, n_batch)
        batch_lists = [f.tolist() for f in batch_lists]

        for batch_list in batch_lists:
            batch_x = []
            batch_h = []
            batch_p = []
            n_samples_list = []
            feat_ids = []
            for featfile in batch_list:
                # make seed waveform and load aux feature
                x = np.zeros((1))
                h = read_hdf5(featfile, "/" + feature_type)
                p = read_hdf5(featfile, "/" + 'world_pulse')

                if not use_upsampling_layer:
                    h = extend_time(h, upsampling_factor)
                if use_speaker_code:
                    sc = read_hdf5(featfile, "/speaker_code")
                    sc = np.tile(sc, [h.shape[0], 1])
                    h = np.concatenate([h, sc], axis=1)

                # perform pre-processing
                if wav_transform is not None:
                    x = wav_transform(x)
                if feat_transform is not None:
                    h = feat_transform(h)

                if use_pulse:
                    h = np.concatenate([h[:, 0:1], h[:, 2:]], axis=1)  # remove cont_f0_lpf
                # append to list
                batch_x += [x]
                batch_h += [h]
                batch_p += [p]
                if not use_upsampling_layer:
                    n_samples_list += [h.shape[0] - 1]
                else:
                    n_samples_list += [h.shape[0] * upsampling_factor - 1]
                feat_ids += [os.path.basename(featfile).replace(".h5", "")]

            # convert list to ndarray
            batch_x = np.stack(batch_x, axis=0)

            len_p_max = max([len(p) for p in batch_p])
            batch_p = [pad_along_axis(p, len_p_max, axis=0) for p, n_sample in zip(batch_p, n_samples_list)]
            batch_p = np.stack(batch_p)
            batch_h = pad_list(batch_h)

            # convert to torch variable
            batch_x = torch.from_numpy(batch_x).long()  # B, 1
            batch_p = torch.from_numpy(batch_p).float().unsqueeze(1)  # B, C=1, T
            batch_h = torch.from_numpy(batch_h).float().transpose(1, 2)  # B, C, T(Frame)

            print(batch_x.shape, batch_p.shape, batch_h.shape)

            # send to cuda
            if torch.cuda.is_available():
                batch_x = batch_x.cuda()
                batch_h = batch_h.cuda()
                batch_p = batch_p.cuda()

            yield feat_ids, (batch_x, batch_h, batch_p, n_samples_list)

# pulse

"""
--checkpoint /home/cswu/research/PytorchWaveNetVocoder/pulse_repeat1_re/checkpoint-200000.pkl
--feats /home/cswu/research/PytorchWaveNetVocoder/egs/arctic/sd/hdf5/ev_slt 
--outdir eva_out_pulse
--stats /home/cswu/research/PytorchWaveNetVocoder/egs/arctic/sd/data/tr_slt/stats.h5 
--config /home/cswu/research/PytorchWaveNetVocoder/pulse_repeat1_re/model.conf
--use_pulse
"""

# no pulse

"""
--checkpoint /home/cswu/research/PytorchWaveNetVocoder/no_pulse_repeat1/checkpoint-200000.pkl
--config /home/cswu/research/PytorchWaveNetVocoder/no_pulse_repeat1/model.conf
--outdir eva_out_no_pulse
--feats /home/cswu/research/PytorchWaveNetVocoder/egs/arctic/sd/hdf5/ev_slt 
--stats /home/cswu/research/PytorchWaveNetVocoder/egs/arctic/sd/data/tr_slt/stats.h5 
"""


def parse_args():
    parser = argparse.ArgumentParser()
    # decode setting
    parser.add_argument("--feats", required=True,
                        type=str, help="list or directory of aux feat files")
    parser.add_argument("--checkpoint", required=True,
                        type=str, help="model file")
    parser.add_argument("--outdir", required=True,
                        type=str, help="directory to save generated samples")
    parser.add_argument("--stats", default=None,
                        type=str, help="hdf5 file including statistics")
    parser.add_argument("--config", default=None,
                        type=str, help="configure file")
    parser.add_argument("--fs", default=16000,
                        type=int, help="sampling rate")
    parser.add_argument("--batch_size", default=32,
                        type=int, help="number of batch size in decoding")
    parser.add_argument("--n_gpus", default=1,
                        type=int, help="number of gpus")
    parser.add_argument("--use_pulse", default=False, action='store_true', help="using pulse signal")

    # other setting
    parser.add_argument("--intervals", default=1000,
                        type=int, help="log interval")
    parser.add_argument("--seed", default=1,
                        type=int, help="seed number")
    parser.add_argument("--verbose", default=1,
                        type=int, help="log level")
    args = parser.parse_args()

    return args


def main(args):
    """RUN DECODING."""

    # set log level
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')
    else:
        logging.basicConfig(level=logging.WARNING,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')
        logging.warning("logging is disabled.")

    # show arguments
    for key, value in vars(args).items():
        logging.info("%s = %s" % (key, str(value)))

    # check arguments
    if args.stats is None:
        args.stats = os.path.dirname(args.checkpoint) + "/stats.h5"
    if args.config is None:
        args.config = os.path.dirname(args.checkpoint) + "/model.conf"
    if not os.path.exists(args.stats):
        raise FileNotFoundError("statistics file is missing (%s)." % (args.stats))
    if not os.path.exists(args.config):
        raise FileNotFoundError("config file is missing (%s)." % (args.config))

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # fix seed
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # fix slow computation of dilated convargs.feats
    # https://github.com/pytorch/pytorch/issues/15054#issuecomment-450191923
    torch.backends.cudnn.benchmark = True

    # load config
    config = torch.load(args.config)

    # get file list
    if os.path.isdir(args.feats):
        feat_list = sorted(find_files(args.feats, "*.h5"))
    elif os.path.isfile(args.feats):
        feat_list = read_txt(args.feats)
    else:
        logging.error("--feats should be directory or list.")
        sys.exit(1)

    # prepare the file list for parallel decoding
    feat_lists = np.array_split(feat_list, args.n_gpus)
    feat_lists = [f_list.tolist() for f_list in feat_lists]

    # define transform
    scaler = StandardScaler()
    scaler.mean_ = read_hdf5(args.stats, "/" + config.feature_type + "/mean")
    scaler.scale_ = read_hdf5(args.stats, "/" + config.feature_type + "/scale")
    wav_transform = transforms.Compose([
        lambda x: encode_mu_law(x, config.n_quantize)])
    feat_transform = transforms.Compose([
        lambda x: scaler.transform(x)])

    # define gpu decode function
    def gpu_decode(feat_list, gpu):
        # set default gpu and do not track gradient
        torch.cuda.set_device(gpu)
        torch.set_grad_enabled(False)

        # define model and load parameters
        if config.use_upsampling_layer:
            upsampling_factor = config.upsampling_factor
        else:
            upsampling_factor = 0

        if args.use_pulse:
            _WaveNet = WaveNetPulse
        else:
            _WaveNet = WaveNet
            config.n_aux = 28

        model = _WaveNet(
            n_quantize=config.n_quantize,
            n_aux=config.n_aux,
            n_resch=config.n_resch,
            n_skipch=config.n_skipch,
            dilation_depth=config.dilation_depth,
            dilation_repeat=config.dilation_repeat,
            kernel_size=config.kernel_size,
            upsampling_factor=upsampling_factor)

        model.load_state_dict(torch.load(args.checkpoint, map_location=lambda storage, loc: storage)["model"])
        model.eval()
        model.cuda()
        print(args.use_pulse)
        # define generator
        generator = decode_generator(
            feat_list,
            batch_size=args.batch_size,
            feature_type=config.feature_type,
            wav_transform=wav_transform,
            feat_transform=feat_transform,
            upsampling_factor=config.upsampling_factor,
            use_upsampling_layer=config.use_upsampling_layer,
            use_speaker_code=config.use_speaker_code,
            use_pulse=args.use_pulse)

        # decode
        if args.batch_size > 1:
            for feat_ids, (batch_x, batch_h, batch_p, n_samples_list) in generator:
                logging.info("decoding start")
                samples_list = model.batch_fast_generate(
                    batch_x, batch_h, n_samples_list, batch_p, intervals=args.intervals)
                for feat_id, samples in zip(feat_ids, samples_list):
                    wav = decode_mu_law(samples, config.n_quantize)
                    sf.write(args.outdir + "/" + feat_id + ".wav", wav, args.fs, "PCM_16")
                    logging.info("wrote %s.wav in %s." % (feat_id, args.outdir))
        else:
            raise NotImplementedError

    # parallel decode
    processes = []
    for gpu, feat_list in enumerate(feat_lists):
        p = mp.Process(target=gpu_decode, args=(feat_list, gpu,))
        p.start()
        processes.append(p)

    # wait for all process
    for p in processes:
        p.join()


if __name__ == "__main__":
    args = parse_args()

    # data_folder = '/home/cswu/research/PytorchWaveNetVocoder/egs/arctic/sd/wav_hpf/tr_slt'
    #
    # filenames = os.listdir(data_folder)
    # # filenames = sorted(find_files(args.waveforms, "*.wav", use_dir_name=False))
    # print(filenames)
    #
    # data_folder = '/home/cswu/research/PytorchWaveNetVocoder/egs/arctic/sd/wav_hpf/tr_slt'
    # args.hdf5dir = 'test'
    # args.wavdir = data_folder

    main(args)

    # wav_list = [os.path.join(data_folder, filename) for filename in filenames]
    # wav_list = wav_list[:2]
    # world_feature_extract(wav_list, args)
