


checkpoint=/home/slee/PytorchWaveNetVocoder/egs/arctic/sd/exp/tr_arctic_16k_sd_world_slt_nq256_na28_nrc512_nsc256_ks2_dp10_dr3_lr1e-4_wd0.0_bl20000_bs1_ns_up/checkpoint-200000.pkl
config=/home/slee/PytorchWaveNetVocoder/egs/arctic/sd/exp/tr_arctic_16k_sd_world_slt_nq256_na28_nrc512_nsc256_ks2_dp10_dr3_lr1e-4_wd0.0_bl20000_bs1_ns_up/model.conf
outdir=/home/slee/PytorchWaveNetVocoder/output/semi_tone_shift
feats=/home/slee/PytorchWaveNetVocoder/egs/arctic/sd/hdf5/ev_slt
stats=/home/slee/PytorchWaveNetVocoder/egs/arctic/sd/exp/tr_arctic_16k_sd_world_slt_nq256_na28_nrc512_nsc256_ks2_dp10_dr3_lr1e-4_wd0.0_bl20000_bs1_ns_up/stats.h5

for f0_shift in {-5..5..2};
do
  echo "shift : $f0_shift"
  echo "out_dir: $outdir$f0_shift"
  python decode.py --checkpoint $checkpoint --config $config --outdir $outdir$f0_shift --feats $feats --stats $stats --f0_shift $f0_shift
done
