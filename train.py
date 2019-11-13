#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os
import sys
import time

from dateutil.relativedelta import relativedelta
from distutils.util import strtobool

import numpy as np
import six
import torch

from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision import transforms

from wavenet_vocoder.nets.wavenet_utils import encode_mu_law
from wavenet_vocoder.nets.wavenet_utils import initialize
from wavenet_vocoder.nets import WaveNet
from wavenet_vocoder.nets.wavenet_pulse import WaveNetPulse
from wavenet_vocoder.utils import find_files
from wavenet_vocoder.utils import read_txt
from dataset import train_generator
from wavenet_vocoder.utils import read_hdf5
from tensorboardX import SummaryWriter

waveforms = "data/tr_slt/wav_nwf.scp"
feats = "data/tr_slt/feats.scp"
stats = "data/tr_slt/stats.h5"
expdir = "/home/cswu/research/PytorchWaveNetVocoder/pulse_repeat3_re"
# resume = "/home/cswu/research/PytorchWaveNetVocoder/pulse_repeat3/checkpoint-200000.pkl"
resume = None
os.chdir('egs/arctic/sd')

writer = SummaryWriter(os.path.join(expdir, 'runs'))


class WaveNetTrainer:

    def __init__(self):
        args = self.parse_arg()
        self.args = args
        self.save_arg()
        # set log level
        if args.verbose == 1:
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

        # make experimental directory
        if not os.path.exists(args.expdir):
            os.makedirs(args.expdir)

        # fix seed
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.model = self.build_model()
        self.dataloader = self.get_dataloader()

        # define optimizer and loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay)
        self.criterion = nn.CrossEntropyLoss()

        # check gpu and then send to gpu
        if torch.cuda.is_available():
            self.model.cuda()
            self.criterion.cuda()
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.cuda()
        else:
            logging.error("gpu is not available. please check the setting.")
            sys.exit(1)

    def parse_arg(self):
        parser = argparse.ArgumentParser()
        # path setting
        parser.add_argument("--waveforms", default=waveforms, type=str, help="directory or list of wav files")
        parser.add_argument("--feats", default=feats, type=str, help="directory or list of aux feat files")
        parser.add_argument("--stats", default=stats, type=str, help="hdf5 file including statistics")
        parser.add_argument("--expdir", default=expdir, type=str, help="directory to save the model")
        parser.add_argument("--feature_type", default="world", choices=["world", "melspc"],
                            type=str,
                            help="feature type")
        # network structure setting
        parser.add_argument("--n_quantize", default=256, type=int, help="number of quantization")
        parser.add_argument("--n_aux", default=28, type=int, help="number of dimension of aux feats")
        parser.add_argument("--n_resch", default=512, type=int, help="number of channels of residual output")
        parser.add_argument("--n_skipch", default=256, type=int, help="number of channels of skip output")
        parser.add_argument("--dilation_depth", default=10, type=int, help="depth of dilation")
        parser.add_argument("--dilation_repeat", default=3, type=int, help="number of repeating of dilation")
        parser.add_argument("--kernel_size", default=2, type=int, help="kernel size of dilated causal convolution")
        parser.add_argument("--upsampling_factor", default=80, type=int, help="upsampling factor of aux features")
        parser.add_argument("--use_upsampling_layer", default=True, type=strtobool, help="flag to use upsampling layer")
        parser.add_argument("--use_speaker_code", default=False, type=strtobool, help="flag to use speaker code")
        parser.add_argument("--use_pulse", default=False, action='store_true', help="using pulse signal")

        # network training setting
        parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay coefficient")
        parser.add_argument("--batch_length", default=20000, type=int,
                            help="batch length (if set 0, utterance batch will be used)")
        parser.add_argument("--batch_size", default=1, type=int,
                            help="batch size (if use utterance batch, batch_size will be 1.")
        parser.add_argument("--iters", default=200000, type=int, help="number of iterations")
        # other setting
        parser.add_argument("--checkpoint_interval", default=10000, type=int, help="how frequent saving model")
        parser.add_argument("--intervals", default=100, type=int, help="log interval")
        parser.add_argument("--seed", default=1, type=int, help="seed number")
        parser.add_argument("--resume", default=resume, nargs="?", type=str, help="model path to restart training")
        parser.add_argument("--n_gpus", default=1, type=int, help="number of gpus")
        parser.add_argument("--verbose", default=1, type=int, help="log level")
        return parser.parse_args()

    def save_arg(self):
        # save args as conf
        os.makedirs(self.args.expdir, exist_ok=True)
        torch.save(self.args, self.args.expdir + "/model.conf")

    def build_model(self):
        # fix slow computation of dilated conv
        # https://github.com/pytorch/pytorch/issues/15054#issuecomment-450191923
        torch.backends.cudnn.benchmark = True

        # define network
        if self.args.use_upsampling_layer:
            upsampling_factor = self.args.upsampling_factor
        else:
            upsampling_factor = 0

        if self.args.use_pulse:
            _Wavenet = WaveNetPulse
            assert self.args.n_aux == 27
        else:
            _Wavenet = WaveNet
            assert self.args.n_aux == 28

        model = _Wavenet(
            n_quantize=self.args.n_quantize,
            n_aux=self.args.n_aux,
            n_resch=self.args.n_resch,
            n_skipch=self.args.n_skipch,
            dilation_depth=self.args.dilation_depth,
            dilation_repeat=self.args.dilation_repeat,
            kernel_size=self.args.kernel_size,
            upsampling_factor=upsampling_factor)
        logging.info(model)
        model.apply(initialize)
        model.train()

        if self.args.n_gpus > 1:
            device_ids = range(self.args.n_gpus)
            model = torch.nn.DataParallel(model, device_ids)
            model.receptive_field = model.module.receptive_field
            if self.args.n_gpus > self.args.batch_size:
                logging.warning("batch size is less than number of gpus.")

        return model

    def get_dataloader(self):
        args = self.args
        # define transforms
        scaler = StandardScaler()
        scaler.mean_ = read_hdf5(args.stats, "/" + args.feature_type + "/mean")
        scaler.scale_ = read_hdf5(args.stats, "/" + args.feature_type + "/scale")
        wav_transform = transforms.Compose([
            lambda x: encode_mu_law(x, args.n_quantize)])
        feat_transform = transforms.Compose([
            lambda x: scaler.transform(x)])

        # define generator
        if os.path.isdir(args.waveforms):
            filenames = sorted(find_files(args.waveforms, "*.wav", use_dir_name=False))
            wav_list = [args.waveforms + "/" + filename for filename in filenames]
            feat_list = [args.feats + "/" + filename.replace(".wav", ".h5") for filename in filenames]

        elif os.path.isfile(args.waveforms):
            wav_list = read_txt(args.waveforms)
            feat_list = read_txt(args.feats)
        else:
            logging.error("--waveforms should be directory or list.")
            sys.exit(1)
        assert len(wav_list) == len(feat_list)
        logging.info("number of training data = %d." % len(wav_list))
        generator = train_generator(
            wav_list, feat_list,
            receptive_field=self.model.receptive_field,
            batch_length=args.batch_length,
            batch_size=args.batch_size,
            feature_type=args.feature_type,
            wav_transform=wav_transform,
            feat_transform=feat_transform,
            shuffle=True,
            upsampling_factor=args.upsampling_factor,
            use_upsampling_layer=args.use_upsampling_layer,
            use_speaker_code=args.use_speaker_code,
            use_pulse=args.use_pulse)

        # charge minibatch in queue
        while not generator.queue.full():
            time.sleep(0.1)

        return generator

    def load_parameter(self):
        # resume model and optimizer
        if self.args.resume is not None and len(self.args.resume) != 0:
            checkpoint = torch.load(self.args.resume, map_location=lambda storage, loc: storage)
            iterations = checkpoint["iterations"]
            if self.args.n_gpus > 1:
                self.model.module.load_state_dict(checkpoint["model"])
            else:
                self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info("restored from %d-iter checkpoint." % iterations)

            return iterations
        else:
            return 0

    def get_loss(self, y_pred, y_target):

        return self.criterion(
            y_pred[:, self.model.receptive_field:].contiguous().view(-1, self.args.n_quantize),
            y_target[:, self.model.receptive_field:].contiguous().view(-1))

    def train_loop(self, start_iteration):

        # train
        loss = 0
        total = 0
        for i in six.moves.range(start_iteration, self.args.iters):
            start = time.time()
            (x, h, p), y_target = self.dataloader.next()
            y_pred = self.model(x, h, p)
            batch_loss = self.get_loss(y_pred, y_target)
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            loss += batch_loss.item()
            total += time.time() - start
            logging.debug("batch loss = %.3f (%.3f sec / batch)" % (
                batch_loss.item(), time.time() - start))

            # report progress
            if (i + 1) % self.args.intervals == 0:
                logging.info(f"(iter:%d) average loss = %.6f (%.3f sec / batch)" % (
                    i + 1, loss / self.args.intervals, total / self.args.intervals))
                writer.add_scalar('data/loss', loss / self.args.intervals, i + 1)

                logging.info("estimated required time = "
                             "{0.days:02}:{0.hours:02}:{0.minutes:02}:{0.seconds:02}"
                    .format(relativedelta(
                    seconds=int((self.args.iters - (i + 1)) * (total / self.args.intervals)))))
                loss = 0
                total = 0

            # save intermidiate model
            if (i + 1) % self.args.checkpoint_interval == 0:
                if self.args.n_gpus > 1:
                    save_checkpoint(self.args.expdir, self.model.module, self.optimizer, i + 1)
                else:
                    save_checkpoint(self.args.expdir, self.model, self.optimizer, i + 1)

        # save final model
        if self.args.n_gpus > 1:
            torch.save({"model": self.model.module.state_dict()}, self.args.expdir + "/checkpoint-final.pkl")
        else:
            torch.save({"model": self.model.state_dict()}, self.args.expdir + "/checkpoint-final.pkl")
        logging.info("final checkpoint created.")


def save_checkpoint(checkpoint_dir, model, optimizer, iterations):
    """SAVE CHECKPOINT.

    Args:
        checkpoint_dir (str): Directory to save checkpoint.
        model (torch.nn.Module): Pytorch model instance.
        optimizer (torch.optim.optimizer): Pytorch optimizer instance.
        iterations (int): Number of current iterations.

    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iterations": iterations}
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(checkpoint, checkpoint_dir + "/checkpoint-%d.pkl" % iterations)
    logging.info("%d-iter checkpoint created." % iterations)


if __name__ == "__main__":
    trainer = WaveNetTrainer()
    load = False
    if load:
        start_iterations = trainer.load_parameter()
        start_iterations = 0
    else:
        start_iterations = 0

    trainer.train_loop(start_iterations)

    loader = trainer.get_dataloader()
