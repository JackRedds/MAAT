# win_size = kernel_length * sample_rate = 2048 samples here, and dense
# self-attention memory/compute scales as batch_size * n_heads * win_size^2,
# so batch_size is kept small relative to other datasets to fit in GPU memory.
# Lower --kernel_length (shrinks win_size quadratically) or --batch_size
# further if you still hit an OOM; raise --batch_size if you have headroom.
python main.py --anormly_ratio 0.85 --num_epochs 1  --batch_size 16  --mode train --dataset GW  --data_path dataset/GW \
  --ifos H1 L1 --sample_rate 2048 --kernel_length 1 --fduration 1 --psd_length 8 --batches_per_epoch 200
# python main.py --anormly_ratio 0.85 --num_epochs 10  --batch_size 16  --mode test   --dataset GW  --data_path dataset/GW \
#   --ifos H1 L1 --sample_rate 2048 --kernel_length 1 --fduration 1 --psd_length 8 --batches_per_epoch 200
