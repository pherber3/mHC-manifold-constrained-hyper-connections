# FineWeb10B with mHC residual-only mode (4 streams)
# Disables H_post branch injection to isolate H_res path
# Used to test whether H_pre/H_post injections compensate for Markov mixing
#
# Usage:
#   python train.py config/train_fineweb10B_mhc_resonly.py
#   torchrun --standalone --nproc_per_node=4 train.py config/train_fineweb10B_mhc_resonly.py

exec(open("config/train_fineweb10B_mhc.py").read())

out_dir = "out-fineweb10B-mhc-resonly"
wandb_run_name = "mhc-resonly"
mhc_residual_only = True  # Disable H_post - residual path only
