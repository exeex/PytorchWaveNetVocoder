import numpy as np
from wavenet_vocoder.utils import background
from wavenet_vocoder.utils import extend_time
import logging
import soundfile as sf
from wavenet_vocoder.utils import read_hdf5
import torch
from scipy.io import wavfile as wf


def p_trans_binary(p):
    p = (p > 0)
    p = p.astype(np.float32)
    return p


def p_trans_binary_multi_channel(p, fn=12):
    p_chans = [p[:, np.newaxis] == 2 ^ n for n in range(fn)]
    p = np.concatenate(p_chans, axis=1)
    p = p.astype(np.float32)
    return p


def validate_length(x, y, upsampling_factor=None):
    """VALIDATE LENGTH.

    Args:
        x (ndarray): ndarray with x.shape[0] = len_x.
        y (ndarray): ndarray with y.shape[0] = len_y.
        upsampling_factor (int): Upsampling factor.

    Returns:
        ndarray: Length adjusted x with same length y.
        ndarray: Length adjusted y with same length x.

    """
    if upsampling_factor is None:
        if x.shape[0] < y.shape[0]:
            y = y[:x.shape[0]]
        if x.shape[0] > y.shape[0]:
            x = x[:y.shape[0]]
        assert len(x) == len(y)
    else:
        if x.shape[0] > y.shape[0] * upsampling_factor:
            x = x[:y.shape[0] * upsampling_factor]
        if x.shape[0] < y.shape[0] * upsampling_factor:
            mod_y = y.shape[0] * upsampling_factor - x.shape[0]
            mod_y_frame = mod_y // upsampling_factor + 1
            y = y[:-mod_y_frame]
            x = x[:y.shape[0] * upsampling_factor]
        assert len(x) == len(y) * upsampling_factor

    return x, y


@background(max_prefetch=16)
def train_generator(wav_list, feat_list, receptive_field,
                    batch_length=None,
                    batch_size=1,
                    feature_type="world",
                    wav_transform=None,
                    feat_transform=None,
                    pulse_transform=p_trans_binary_multi_channel,
                    shuffle=True,
                    upsampling_factor=80,
                    use_upsampling_layer=True,
                    use_speaker_code=False,
                    use_pulse=True):
    """GENERATE TRAINING BATCH.

    Args:
        wav_list (list): List of wav files.
        feat_list (list): List of feat files.
        receptive_field (int): Size of receptive filed.
        batch_length (int): Batch length (if set None, utterance batch will be used.).
        batch_size (int): Batch size (if batch_length = None, batch_size will be 1.).
        feature_type (str): Auxiliary feature type.
        wav_transform (func): Preprocessing function for waveform.
        feat_transform (func): Preprocessing function for aux feats.
        shuffle (bool): Whether to shuffle the file list.
        upsampling_factor (int): Upsampling factor.
        use_upsampling_layer (bool): Whether to use upsampling layer.
        use_speaker_code (bool): Whether to use speaker code.
        use_pulse (bool): use pulse signal

    Returns:
        generator: Generator instance.

    """
    # shuffle list
    if shuffle:
        n_files = len(wav_list)
        idx = np.random.permutation(n_files)
        wav_list = [wav_list[i] for i in idx]
        feat_list = [feat_list[i] for i in idx]

    # check batch_length
    if batch_length is not None and use_upsampling_layer:
        batch_mod = (receptive_field + batch_length) % upsampling_factor
        logging.warning("batch length is decreased due to upsampling (%d -> %d)" % (
            batch_length, batch_length - batch_mod))
        batch_length -= batch_mod

    # show warning
    if batch_length is None and batch_size > 1:
        logging.warning("in utterance batch mode, batchsize will be 1.")

    while True:
        batch_x, batch_p, batch_h, batch_t = [], [], [], []
        # process over all of files
        for wavfile, featfile in zip(wav_list, feat_list):
            # load waveform and aux feature
            # x, fs = sf.read(wavfile, dtype=np.float32)
            fs, data = wf.read(wavfile)
            # print(data.shape)
            x = data.astype(np.float) / 32768
            h = read_hdf5(featfile, "/" + feature_type)
            p = read_hdf5(featfile, "/" + 'world_pulse')
            # p
            if pulse_transform:
                p = pulse_transform(p)

            if not use_upsampling_layer:
                h = extend_time(h, upsampling_factor)

            if use_speaker_code:
                sc = read_hdf5(featfile, "/speaker_code")
                sc = np.tile(sc, [h.shape[0], 1])
                h = np.concatenate([h, sc], axis=1)

            # check both lengths are same
            logging.debug("before x length = %d" % x.shape[0])
            logging.debug("before h length = %d" % h.shape[0])
            if use_upsampling_layer:
                x, h = validate_length(x, h, upsampling_factor)
            else:
                x, h = validate_length(x, h)
            logging.debug("after x length = %d" % x.shape[0])
            logging.debug("after h length = %d" % h.shape[0])

            # ---------------------------------------
            # use mini batch without upsampling layer
            # ---------------------------------------
            if batch_length is not None and not use_upsampling_layer:
                raise NotImplementedError

            # ------------------------------------
            # use mini batch with upsampling layer <-------This TODO
            # ------------------------------------
            elif batch_length is not None and use_upsampling_layer:
                # make buffer array
                if "x_buffer" not in locals():
                    x_buffer = np.empty((0), dtype=np.float32)
                    # p_buffer = np.empty((0), dtype=np.float32)
                    p_buffer = np.empty((0, p.shape[1]), dtype=np.float32)
                    h_buffer = np.empty((0, h.shape[1]), dtype=np.float32)
                x_buffer = np.concatenate([x_buffer, x], axis=0)
                p_buffer = np.concatenate([p_buffer, p], axis=0)
                h_buffer = np.concatenate([h_buffer, h], axis=0)

                while len(h_buffer) > (receptive_field + batch_length) // upsampling_factor:
                    # set batch size
                    h_bs = (receptive_field + batch_length) // upsampling_factor
                    x_bs = h_bs * upsampling_factor + 1
                    p_bs = h_bs * upsampling_factor + 1

                    # get pieces
                    h_ = h_buffer[:h_bs]
                    x_ = x_buffer[:x_bs]
                    p_ = p_buffer[:p_bs]

                    # perform pre-processing
                    if wav_transform is not None:
                        x_ = wav_transform(x_)
                    if feat_transform is not None:
                        h_ = feat_transform(h_)

                    if use_pulse:
                        h_ = np.concatenate([h_[:, 0:1], h_[:, 2:]], axis=1)  # remove cont_f0_lpf

                    # convert to torch variable
                    x_ = torch.from_numpy(x_).long()
                    p_ = torch.from_numpy(p_).float()
                    h_ = torch.from_numpy(h_).float()

                    # remove the last and first sample for training
                    batch_h += [h_.transpose(0, 1)]  # (D x T)
                    batch_x += [x_[:-1]]  # (T)
                    batch_p += [p_[:-1].transpose(0, 1)]  # (C x T)
                    batch_t += [x_[1:]]  # (T)

                    # set shift size
                    h_ss = batch_length // upsampling_factor
                    x_ss = h_ss * upsampling_factor
                    p_ss = h_ss * upsampling_factor

                    # update buffer
                    h_buffer = h_buffer[h_ss:]
                    x_buffer = x_buffer[x_ss:]
                    p_buffer = p_buffer[p_ss:]

                    # return mini batch
                    if len(batch_x) == batch_size:
                        batch_x = torch.stack(batch_x)
                        batch_p = torch.stack(batch_p)
                        batch_h = torch.stack(batch_h)
                        batch_t = torch.stack(batch_t)

                        # send to cuda
                        if torch.cuda.is_available():
                            batch_x = batch_x.cuda()
                            batch_p = batch_p.cuda()
                            batch_h = batch_h.cuda()
                            batch_t = batch_t.cuda()

                        yield (batch_x, batch_h, batch_p), batch_t

                        batch_x, batch_h, batch_p, batch_t, = [], [], [], []

            # --------------------------------------------
            # use utterance batch without upsampling layer
            # --------------------------------------------
            elif batch_length is None and not use_upsampling_layer:
                raise NotImplementedError

            # -----------------------------------------
            # use utterance batch with upsampling layer
            # -----------------------------------------
            else:
                raise NotImplementedError

        # re-shuffle
        if shuffle:
            idx = np.random.permutation(n_files)
            wav_list = [wav_list[i] for i in idx]
            feat_list = [feat_list[i] for i in idx]


if __name__ == '__main__':
    from wavenet_vocoder.utils import read_txt
    from torchvision import transforms
    from wavenet_vocoder.nets.wavenet_utils import encode_mu_law
    import os

    # TODO not work
    os.chdir('egs/arctic/sdp')
    wav_list_test = read_txt("data/ev_slt/wav_hpf.scp")
    feat_list_test = read_txt("data/ev_slt/feats.scp")
    waveforms = "data/tr_slt/wav_hpf.scp"
    feats = "data/tr_slt/feats.scp"
    stats = "data/tr_slt/stats.h5"
    expdir = "exp/pulse_repeat1_1130"

    n_quantize = 256
    wav_transform = transforms.Compose([
        lambda x: encode_mu_law(x, n_quantize)])
    # feat_transform = transforms.Compose([
    #     lambda x: scaler.transform(x)])

    generator = train_generator(
        wav_list_test, wav_list_test,
        receptive_field=512,
        batch_length=19900,
        batch_size=2,
        feature_type='world',
        wav_transform=wav_transform,
        feat_transform=None,
        shuffle=True,
        upsampling_factor=160,
        use_upsampling_layer=True,
        use_speaker_code=False,
        use_pulse=True)

    a = generator.next()

    print(a)
