

# bash decode_shifts.sh 2> >(tee log.txt)

checkpoint=/home/cswu/research/PytorchWaveNetVocoder/egs/arctic/sdp/exp/pulse_repeat1_1130/checkpoint-130000.pkl
config=/home/cswu/research/PytorchWaveNetVocoder/egs/arctic/sdp/exp/pulse_repeat1_1130/model.conf
outdir=out_shifts/eva_out_pulse
feats=/home/cswu/research/PytorchWaveNetVocoder/egs/arctic/sdp/hdf5/ev_slt
stats=/home/cswu/research/PytorchWaveNetVocoder/egs/arctic/sdp/data/tr_slt/stats.h5


for f0_shift in {-5..5..2};
do
  echo "shift : $f0_shift"
  echo "out_dir: $outdir$f0_shift"
  python decode.py --checkpoint $checkpoint --config $config --outdir $outdir$f0_shift --feats $feats --stats $stats --f0_shift $f0_shift --use_pulse
done
