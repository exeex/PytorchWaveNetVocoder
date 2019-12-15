

# bash decode_shifts.sh 2> >(tee log1203.txt)

checkpoint=/home/cswu/research/PytorchWaveNetVocoder/egs/arctic/sdp/exp/pulse_repeat1_1203/checkpoint-160000.pkl
config=/home/cswu/research/PytorchWaveNetVocoder/egs/arctic/sdp/exp/pulse_repeat1_1203/model.conf
outdir=out_shifts/eva_out_pulse1214
feats=/home/cswu/research/PytorchWaveNetVocoder/egs/arctic/sdp/hdf5/ev_slt
stats=/home/cswu/research/PytorchWaveNetVocoder/egs/arctic/sdp/data/tr_slt/stats.h5


for f0_shift in {-1..0..1};
do
  echo "shift : $f0_shift"
  echo "out_dir: $outdir$f0_shift"
  python decode.py --checkpoint $checkpoint --config $config --outdir $outdir$f0_shift --feats $feats --stats $stats --f0_shift $f0_shift --use_pulse
done
