from vocoder_eva.eval import eval_MCD, eval_rmse_f0, eval_snr
import os
from glob import glob
from pathlib import Path
import librosa


def shift_semi_tone_f0_pulse(f0, shift=2):
    freq_scale = 2 ** (shift / 12)
    # print(f0.max())
    f0[:] = f0[:] * freq_scale
    # print(f0.max())
    return f0


class EvaDataset:

    def __init__(self, raw_folder, syn_folder):
        self.r_files = {Path(y).stem: Path(y) for x in os.walk(raw_folder) for y in glob(os.path.join(x[0], '*.wav'))}
        self.s_files = {Path(y).stem: Path(y) for x in os.walk(syn_folder) for y in glob(os.path.join(x[0], '*.wav'))}

        assert len(self.r_files) == len(self.r_files)

        self.file_names = set(self.r_files.keys()).union(set(self.s_files.keys()))
        self.file_names = list(self.file_names)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        r_file = self.r_files[file_name]
        s_file = self.s_files[file_name]

        aud_r, sr_r = librosa.load(r_file, sr=None)
        aud_s, sr_s = librosa.load(s_file, sr=None)

        assert sr_r == sr_s

        return aud_r, aud_s, sr_r


def evaluate_f0(dataset: EvaDataset, shift):
    f0_rmse_list = []
    vuv_precision_list = []

    for aud_r, aud_s, sr in dataset:
        f0_rmse_mean, vuv_accuracy, vuv_precision = eval_rmse_f0(aud_r, aud_s, sr, method='dio', tone_shift=shift)
        # print(f0_rmse_mean, vuv_precision)
        f0_rmse_list.append(f0_rmse_mean)
        vuv_precision_list.append(vuv_precision)

    avg_f0_rmse = sum(f0_rmse_list) / len(dataset)
    avg_vuv_precision = sum(vuv_precision_list) / len(dataset)

    print('avg_f0_rmse:', avg_f0_rmse, 'avg_vuv_precision:', avg_vuv_precision)


if __name__ == '__main__':
    # r: raw, s: synthesised
    r_folder = '/home/cswu/research/PytorchWaveNetVocoder/egs/arctic/sdp/wav_hpf/ev_slt'
    s_folder_1 = '/home/cswu/research/PytorchWaveNetVocoder/out_shifts/eva_out_pulse-1'
    s_folder_2 = '/home/cswu/research/PytorchWaveNetVocoder/out_shifts/eva_out_pulse3'

    d1 = EvaDataset(r_folder, s_folder_1)
    d2 = EvaDataset(r_folder, s_folder_2)

    print('## case : -1 ##')
    evaluate_f0(d1, -1)
    print('## case : 3 ##')
    evaluate_f0(d2, 3)
