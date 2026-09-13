"""
Radar noise model  (the "+ noise" branch of Fig. 2c).

RadSimReal adds noise to the convolved tensor. In a real radar tensor the noise
floor sits ~80 dB below the PSF peak (paper, Sec. 3). The variance can be
*measured* from an empty region of a real tensor, or set by a target
signal-to-noise ratio for standalone runs.

We model complex thermal noise whose magnitude follows a Rayleigh distribution
(the modulus of circular complex Gaussian), which is the standard model for the
magnitude of an FMCW radar range-azimuth-doppler cell.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class NoiseModel:
    sigma: float = None            # per-quadrature std of complex noise (linear)
    seed: int = None

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    @classmethod
    def from_snr_db(cls, signal_peak: float, snr_db: float, seed: int = None):
        """
        Set noise so that (peak / noise_floor) ~= snr_db.
        Noise floor here is the mean Rayleigh magnitude = sigma * sqrt(pi/2).
        """
        floor = signal_peak / (10 ** (snr_db / 20.0))
        sigma = floor / np.sqrt(np.pi / 2.0)
        return cls(sigma=sigma, seed=seed)

    @classmethod
    def from_tensor_region(cls, tensor: np.ndarray, region=None, seed: int = None):
        """
        Measure noise std from an (assumed target-free) region of a real tensor.
        `region` is a boolean mask or None (use the lowest-10% magnitude cells).
        """
        t = np.asarray(tensor, dtype=np.float64)
        if region is None:
            thr = np.quantile(t, 0.10)
            vals = t[t <= thr]
        else:
            vals = t[region]
        # for Rayleigh magnitude, sigma = mean / sqrt(pi/2)
        sigma = float(vals.mean() / np.sqrt(np.pi / 2.0)) if vals.size else 0.0
        return cls(sigma=sigma, seed=seed)

    def add(self, tensor: np.ndarray) -> np.ndarray:
        """Add Rayleigh-magnitude noise to a linear-magnitude tensor."""
        if not self.sigma or self.sigma <= 0:
            return tensor
        re = self._rng.normal(0.0, self.sigma, size=tensor.shape)
        im = self._rng.normal(0.0, self.sigma, size=tensor.shape)
        noise_mag = np.sqrt(re * re + im * im)
        # incoherent addition of signal magnitude and noise magnitude
        return np.sqrt(tensor ** 2 + noise_mag ** 2)
