"""
Gravitational-wave background strain loading for glitch (instrumental
anomaly) detection, built directly on the data primitives used by
ml4gw/amplfi rather than a hand-rolled windowing scheme:

- ``ml4gw.dataloading.Hdf5TimeSeriesDataset`` / ``InMemoryDataset`` sample
  fixed-length multichannel windows from HDF5 background segment files,
  exactly as amplfi's ``AmplfiDataset`` does for its train/val/test loaders
  (see ``amplfi.train.data.datasets.base.AmplfiDataset``).
- ``ml4gw.transforms.SpectralDensity`` + ``ml4gw.transforms.Whiten`` are
  used to estimate a background PSD from a leading ``psd_length`` seconds
  of each window and whiten the rest, the same two-step preprocessing
  amplfi performs via its ``PsdEstimator``/``Whiten`` pair inside
  ``AmplfiDataset.inject`` (see ``amplfi.train.augmentations.PsdEstimator``,
  reproduced here so MAAT doesn't need to depend on the ``amplfi`` package
  itself, only the public ``ml4gw`` library).

Segment files follow the same on-disk layout amplfi expects::

    <data_path>/train/background/*.h5   (or *.hdf5)
    <data_path>/test/background/*.h5

with each file holding one contiguous strain array per interferometer,
keyed by IFO name (e.g. "H1", "L1"), sample spacing in the dataset's ``dx``
attribute and GPS start time in ``x0`` (as written by gwpy's
``TimeSeries.write``) -- see ``AmplfiDataset.get_train_val_fnames``,
``get_test_fnames`` and ``train_val_split`` for the file-discovery and
file-level train/validation split this mirrors.
"""
import os

import h5py
import numpy as np
import pandas as pd
import torch
from ml4gw.dataloading import Hdf5TimeSeriesDataset, InMemoryDataset
from ml4gw.transforms import Whiten
from ml4gw.transforms.spectral import SpectralDensity


class PsdEstimator(torch.nn.Module):
    """
    Reproduces ``amplfi.train.augmentations.PsdEstimator``: splits a
    timeseries into a leading ``length``-second segment used to estimate a
    background PSD and a trailing segment, returning the latter alongside
    the PSD of the former.
    """

    def __init__(self, length, sample_rate, fftlength, overlap=None, average="median", fast=True):
        super().__init__()
        self.size = int(length * sample_rate)
        self.spectral_density = SpectralDensity(sample_rate, fftlength, overlap, average, fast=fast)

    def forward(self, X):
        splits = [X.size(-1) - self.size, self.size]
        background, X = torch.split(X, splits, dim=-1)
        self.spectral_density.to(device=background.device)
        psd = self.spectral_density(background.double())
        return X, psd


def list_segment_files(dirpath):
    """Sorted (by GPS start time) list of ``background-<gpsstart>-<duration>.h5`` files."""
    if not os.path.isdir(dirpath):
        return []

    def gps_start(fname):
        try:
            return float(fname.split(".")[0].split("-")[-2])
        except (IndexError, ValueError):
            return fname

    files = sorted(
        (f for f in os.listdir(dirpath) if f.endswith(".h5") or f.endswith(".hdf5")),
        key=gps_start,
    )
    return [os.path.join(dirpath, f) for f in files]


def train_val_split(train_files, val_fraction):
    """Hold out a trailing fraction of segment files for validation, like `AmplfiDataset.train_val_split`."""
    n_val = max(1, round(len(train_files) * val_fraction)) if len(train_files) > 1 else 0
    fit_files = train_files[:-n_val] if n_val else train_files
    val_files = train_files[-n_val:] if n_val else train_files
    return fit_files, val_files


def load_glitch_times(glitch_file):
    """Load known glitch GPS times from a CSV (`gps_time` column) or `.npy` array."""
    if glitch_file is None:
        return None
    if not os.path.exists(glitch_file):
        print(f"Warning: glitch_file '{glitch_file}' not found, test segments will be treated as unlabeled")
        return None
    if glitch_file.endswith(".csv"):
        return pd.read_csv(glitch_file)["gps_time"].to_numpy()
    return np.load(glitch_file)


class GWStrainDataset(torch.utils.data.IterableDataset):
    """
    Samples multichannel background strain windows from HDF5 segment files
    and applies amplfi's PSD-estimation + whitening preprocessing, yielding
    batches ready for MAAT's reconstruction/association-discrepancy loss.

    A raw window of ``kernel_length + fduration + psd_length`` seconds is
    pulled per sample. The leading ``psd_length`` seconds estimate a PSD
    (`PsdEstimator`); the trailing ``kernel_length + fduration`` seconds are
    then whitened against it (`ml4gw.transforms.Whiten`), which crops
    ``fduration / 2`` seconds from each edge to remove filter settle-in.
    What remains is ``kernel_length`` seconds of whitened strain per
    interferometer -- the window MAAT actually trains/tests on.

    Args:
        fnames: HDF5 segment files to sample from.
        ifos: interferometer channel names, e.g. ``["H1", "L1"]``.
        sample_rate: strain sample rate in Hz.
        kernel_length: length, in seconds, of the whitened window seen by the model.
        fduration: whitening filter impulse response length, in seconds.
        psd_length: length, in seconds, of data used to estimate the PSD.
        batch_size: number of windows per batch.
        mode: ``"train"`` samples random windows across all of `fnames` via
            `Hdf5TimeSeriesDataset` (mirrors `AmplfiDataset.train_dataloader`).
            ``"val"``/``"test"`` load `fnames` into memory and sample random,
            possibly-overlapping windows for cheap per-epoch validation.
            ``"thre"`` loads `fnames` into memory and walks it deterministically
            with non-overlapping raw windows for exhaustive final evaluation.
        batches_per_epoch: required for ``"train"``; used as-is for
            ``"val"``/``"test"`` (default 25 if left `None`); ignored for
            ``"thre"``, which is always exhaustive.
        highpass, fftlength: passed through to the PSD/whitening transforms;
            `fftlength` defaults to `kernel_length + fduration`.
        glitch_times: known glitch GPS times (see `load_glitch_times`), used
            only for ``"val"``/``"test"``/``"thre"`` label construction --
            training remains unsupervised regardless.
        glitch_half_width: seconds around each glitch time labeled anomalous.
    """

    def __init__(self, fnames, ifos, sample_rate, kernel_length, fduration, psd_length,
                 batch_size, mode="train", batches_per_epoch=None, highpass=None,
                 fftlength=None, glitch_times=None, glitch_half_width=0.1, average="median"):
        super().__init__()
        if not fnames:
            raise FileNotFoundError(f"No HDF5 segment files provided for mode '{mode}'")

        self.mode = mode
        self.ifos = list(ifos)
        self.sample_rate = sample_rate

        window_length = kernel_length + fduration
        sample_length = window_length + psd_length
        self.kernel_size = int(sample_length * sample_rate)
        self.win_size = int(kernel_length * sample_rate)
        window_samples = int(window_length * sample_rate)
        crop = (window_samples - self.win_size) // 2
        # offset, within each raw kernel_size window, of the kept/whitened segment:
        # the leading (kernel_size - window_samples) samples are consumed by PSD
        # estimation, then `crop` more are consumed by the whitening filter settle-in
        self._offset = (self.kernel_size - window_samples) + crop

        fftlength = fftlength or window_length
        self.psd_estimator = PsdEstimator(
            window_length, sample_rate, fftlength, average=average, fast=highpass is not None
        )
        self.whitener = Whiten(fduration, sample_rate, highpass)

        if mode == "train":
            self._backend = Hdf5TimeSeriesDataset(
                fnames, channels=self.ifos, kernel_size=self.kernel_size,
                batch_size=batch_size, batches_per_epoch=batches_per_epoch, coincident=True,
            )
        else:
            X, y = self._load(fnames, glitch_times, glitch_half_width)
            exhaustive = mode == "thre"
            self._backend = InMemoryDataset(
                X, kernel_size=self.kernel_size, y=y,
                batch_size=batch_size,
                stride=self.kernel_size if exhaustive else 1,
                batches_per_epoch=None if exhaustive else (batches_per_epoch or 25),
                coincident=True,
                shuffle=not exhaustive,
            )

        print(f"GWStrainDataset[{mode}]: {len(fnames)} file(s), "
              f"{self.kernel_size} raw samples/window -> {self.win_size} whitened samples/window, "
              f"{len(self)} batch(es)/epoch")

    def _load(self, fnames, glitch_times, glitch_half_width):
        segments, labels = [], []
        for fname in fnames:
            with h5py.File(fname, "r") as f:
                dset0 = f[self.ifos[0]]
                n = dset0.shape[0]
                dx = dset0.attrs["dx"]
                gps_start = float(dset0.attrs.get("x0", 0.0))
                segments.append(np.stack([np.asarray(f[ifo][:], dtype=np.float32) for ifo in self.ifos], axis=0))
            lab = np.zeros(n, dtype=np.float32)
            if glitch_times is not None:
                gps_end = gps_start + n * dx
                half = int(round(glitch_half_width / dx))
                for t in glitch_times[(glitch_times >= gps_start) & (glitch_times < gps_end)]:
                    center = int(round((t - gps_start) / dx))
                    lab[max(0, center - half): min(n, center + half + 1)] = 1.0
            labels.append(lab)
        X = torch.tensor(np.concatenate(segments, axis=1), dtype=torch.float32)
        y = torch.tensor(np.concatenate(labels, axis=0), dtype=torch.float32)
        return X, y

    def __len__(self):
        return len(self._backend)

    def __iter__(self):
        for batch in self._backend:
            if self.mode == "train":
                X, y = batch, None
            else:
                X, y = batch

            X, psd = self.psd_estimator(X)
            X = self.whitener(X, psd)                     # (batch, channels, win_size)
            X = X.permute(0, 2, 1).contiguous().float()    # (batch, win_size, channels)

            if y is None:
                y = torch.zeros(X.shape[0], X.shape[1], 1)
            else:
                y = y[:, self._offset: self._offset + self.win_size].unsqueeze(-1)
            yield X, y


def get_gw_loader(data_path, batch_size, mode, ifos, sample_rate, kernel_length, fduration, psd_length,
                   val_fraction=0.1, batches_per_epoch=200, val_batches_per_epoch=None,
                   highpass=None, fftlength=None, glitch_file=None, glitch_half_width=0.1):
    """
    Build the `GWStrainDataset` for `mode` in {"train", "val", "test", "thre"},
    following the same file discovery / train-val split as
    `AmplfiDataset.get_train_val_fnames`, `get_test_fnames` and
    `train_val_split`.
    """
    train_files = list_segment_files(os.path.join(data_path, "train", "background"))
    test_files = list_segment_files(os.path.join(data_path, "test", "background"))
    if not train_files:
        raise FileNotFoundError(f"No *.h5/*.hdf5 segment files found under {data_path}/train/background")
    if not test_files:
        raise FileNotFoundError(f"No *.h5/*.hdf5 segment files found under {data_path}/test/background")
    fit_files, val_files = train_val_split(train_files, val_fraction)

    fnames = {"train": fit_files, "val": val_files, "test": test_files, "thre": test_files}[mode]
    glitch_times = load_glitch_times(glitch_file) if mode in ("test", "thre") else None

    return GWStrainDataset(
        fnames, ifos, sample_rate, kernel_length, fduration, psd_length,
        batch_size=batch_size, mode=mode,
        batches_per_epoch=batches_per_epoch if mode == "train" else val_batches_per_epoch,
        highpass=highpass, fftlength=fftlength,
        glitch_times=glitch_times, glitch_half_width=glitch_half_width,
    )
