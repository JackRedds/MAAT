import os
import argparse

from torch.backends import cudnn
from utils.utils import *

from solver import Solver, GWSolver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver_cls = GWSolver if config.dataset == 'GW' else Solver
    solver = solver_cls(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=4.00)
    parser.add_argument('--gpu_index', type=int, default=0, help='Index of the GPU to use')
    parser.add_argument('--multi_gpu', type=str2bool, default=True, help='Enable multi-GPU training')
    # GWSolver-only options (used when --dataset GW). Note --win_size above is
    # ignored for GW: the model's window length instead falls out of
    # --sample_rate/--kernel_length (see GWSolver's docstring).
    parser.add_argument('--ifos', type=str, nargs='+', default=['H1', 'L1'],
                        help="Interferometers to use as input channels, e.g. --ifos H1 L1")
    parser.add_argument('--sample_rate', type=float, default=2048.0,
                        help='Strain sample rate in Hz')
    parser.add_argument('--kernel_length', type=float, default=1.0,
                        help='Length, in seconds, of the whitened window seen by the model')
    parser.add_argument('--fduration', type=float, default=1.0,
                        help="Whitening filter impulse response length, in seconds "
                             "(fduration / 2 seconds are cropped from each edge of the whitened window)")
    parser.add_argument('--psd_length', type=float, default=8.0,
                        help='Length, in seconds, of data used to estimate the background PSD')
    parser.add_argument('--fftlength', type=float, default=None,
                        help='FFT length, in seconds, used for PSD estimation '
                             '(defaults to kernel_length + fduration)')
    parser.add_argument('--highpass', type=float, default=None,
                        help='Highpass cutoff frequency in Hz applied during whitening')
    parser.add_argument('--batches_per_epoch', type=int, default=200,
                        help='Number of training batches sampled per epoch')
    parser.add_argument('--val_batches_per_epoch', type=int, default=None,
                        help='Number of validation batches sampled per epoch (default 25); '
                             'the final test/threshold pass is always exhaustive')
    parser.add_argument('--val_fraction', type=float, default=0.1,
                        help='Fraction of GW training segment files held out for validation')
    parser.add_argument('--glitch_file', type=str, default=None,
                        help='Optional CSV (gps_time column) or .npy of known glitch GPS times, '
                             'used to label held-out GW test segments for evaluation')
    parser.add_argument('--glitch_half_width', type=float, default=0.1,
                        help='Seconds around each glitch time labeled anomalous')
    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
