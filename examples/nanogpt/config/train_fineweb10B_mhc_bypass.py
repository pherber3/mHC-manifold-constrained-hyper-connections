# FineWeb10B with mHC + bypass stream (5 streams: 1 bypass + 4 mixing)
# ~20M param GPT-2 style model
#
# The bypass stream (stream 0) skips H_res entirely, while streams 1-4
# mix normally via a 4x4 doubly stochastic H_res matrix.
#
# Usage:
#   python train.py config/train_fineweb10B_mhc_bypass.py
#   torchrun --standalone --nproc_per_node=4 train.py config/train_fineweb10B_mhc_bypass.py

exec(open("config/train_fineweb10B_mhc_48l.py").read())

out_dir = "out-fineweb10B-mhc-bypass"
wandb_run_name = "mhc-bypass"

# 5 streams: 1 bypass (stream 0) + 4 mixing (streams 1-4)
hc_num_streams = 5
mhc_bypass_stream = True

# Use tau=1.0 to enable gradient flow for H_res learning
sinkhorn_tau = 1.0
