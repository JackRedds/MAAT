python main.py --anormly_ratio 0.85 --num_epochs 1  --batch_size 256  --mode train --dataset GW  --data_path dataset/GW \
  --ifos H1 L1 --sample_rate 2048 --kernel_length 1 --fduration 1 --psd_length 8 --batches_per_epoch 200
python main.py --anormly_ratio 0.85 --num_epochs 10  --batch_size 256  --mode test   --dataset GW  --data_path dataset/GW \
  --ifos H1 L1 --sample_rate 2048 --kernel_length 1 --fduration 1 --psd_length 8 --batches_per_epoch 200
