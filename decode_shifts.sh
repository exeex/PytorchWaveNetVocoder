

# bash decode_shifts.sh 2> >(tee log1203.txt)

checkpoint=/home/cswu/research/PytorchWaveNetVocoder/augment/exp/pulse_repeat1_1228/checkpoint-150000.pkl
config=/home/cswu/research/PytorchWaveNetVocoder/augment/exp/pulse_repeat1_1228/model.conf
outdir=out_shifts/eva_out_pulse1228
feats=/home/cswu/research/PytorchWaveNetVocoder/augment/hdf5/ev_slt
stats=/home/cswu/research/PytorchWaveNetVocoder/augment/data/tr_slt/stats.h5


for f0_shift in {-1..0..1};
do
  echo "shift : $f0_shift"
  echo "out_dir: $outdir$f0_shift"
  python decode.py --checkpoint $checkpoint --config $config --outdir $outdir$f0_shift --feats $feats --stats $stats --f0_shift $f0_shift --use_pulse
done
