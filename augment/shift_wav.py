import pyworld as pw
import soundfile as sf

from glob import glob
import os
from pathlib import Path
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor


def shift_wav(in_file, shift=0):
    in_file = Path(in_file)
    x, fs = sf.read(str(in_file))
    _f0, t = pw.dio(x, fs)  # raw pitch extractor
    f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
    sp = pw.cheaptrick(x, f0, t, fs)  # extract smoothed spectrogram
    ap = pw.d4c(x, f0, t, fs)  # extract aperiodicity

    shift_scale = 2 ** (shift / 12)

    y = pw.synthesize(f0 * shift_scale, sp, ap, fs)

    if shift >= 0:
        out_file = in_file.parent / f"{in_file.stem}+{shift}.wav"
    else:
        out_file = in_file.parent / f"{in_file.stem}-{-shift}.wav"

    sf.write(str(out_file), y, fs)


class WavDataset:
    def __init__(self, data_folder):
        self.files = [y for x in os.walk(data_folder) for y in glob(os.path.join(x[0], '*.wav'))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        return Path(self.files[item])


if __name__ == '__main__':
    from functools import partial

    folder = "downloads"
    d = WavDataset(folder)

    shift_n1 = partial(shift_wav, shift=-1)
    shift_p1 = partial(shift_wav, shift=1)

    with ProcessPoolExecutor() as executor:
        for _ in tqdm(executor.map(shift_n1, d.files)):
            pass

    with ProcessPoolExecutor() as executor:
        for _ in tqdm(executor.map(shift_p1, d.files)):
            pass

    # for file in tqdm(d):
    #     shift_wav(file, out_file, shift=1)
    #     out_file = file.parent / f"{file.stem}-1.wav"
    #     shift_wav(file, out_file, shift=-1)
