"""
Point Spread Function (PSF)  -- the core of RadSimReal.

Paper claim (Sec. 3): a conventional physical radar simulation produces a
radar tensor that equals the *superposition of the radar's PSF centred on each
reflection point*, plus noise. Therefore the tensor can be produced by

        tensor = (reflection_point_field)  *  PSF      (3D convolution)
                 + noise

The PSF is a multi-dimensional kernel over (range, azimuth, doppler). Its shape
is fixed by the radar hardware + signal processing, but -- crucially -- it can
be *measured* from the radar tensor of a single narrow object (a pole or a
corner reflector) without knowing any hardware details. That is exactly what
lets RadSimReal skip proprietary radar specs.

This module provides:
  - PSF                : container (kernel + its axis metadata)
  - truncate_psf       : keep the smallest box holding `energy_fraction` (paper: 99%)
  - synthetic_psf      : analytic PSF (wide azimuth spread, narrow range/doppler)
                         for standalone runs with no measured PSF available
  - psf_from_tensor    : *measure* a PSF from a real radar tensor of a point target
                         (the procedure the paper uses; details in its supp. mat.)
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PSF:
    kernel: np.ndarray            # (kr, ka, kd) non-negative, peak-centred
    range_res: float              # metres per range bin  (must match RadarConfig)
    azimuth_res_deg: float        # degrees per azimuth bin
    doppler_res: float            # m/s per doppler bin

    def __post_init__(self):
        self.kernel = np.asarray(self.kernel, dtype=np.float64)
        assert self.kernel.ndim == 3, "PSF kernel must be 3D (range, azimuth, doppler)"

    @property
    def shape(self):
        return self.kernel.shape

    @property
    def energy(self) -> float:
        return float(self.kernel.sum())

    def normalized(self) -> "PSF":
        e = self.energy
        k = self.kernel / e if e > 0 else self.kernel
        return PSF(k, self.range_res, self.azimuth_res_deg, self.doppler_res)

    def peak_index(self):
        return np.unravel_index(np.argmax(self.kernel), self.kernel.shape)


def truncate_psf(psf: PSF, energy_fraction: float = 0.99) -> PSF:
    """
    Shrink the PSF to the smallest centred box that still contains
    `energy_fraction` of its total energy (paper uses 99%, giving a >1000x
    volume reduction -> the dominant speed-up over conventional simulation).

    Strategy: grow a symmetric half-width along each axis from the peak until
    the enclosed energy fraction is reached, per-axis, taking the marginal
    energy profile along each dimension.
    """
    k = psf.kernel
    total = k.sum()
    if total <= 0:
        return psf

    peak = np.unravel_index(np.argmax(k), k.shape)
    half = []
    for ax in range(3):
        # marginal energy profile along this axis (sum over the other two)
        other = tuple(a for a in range(3) if a != ax)
        prof = k.sum(axis=other)
        c = peak[ax]
        n = prof.shape[0]
        acc = prof[c]
        hw = 0
        target = energy_fraction * prof.sum()
        while acc < target and (c - hw - 1 >= 0 or c + hw + 1 < n):
            hw += 1
            lo = c - hw
            hi = c + hw
            add = 0.0
            if lo >= 0:
                add += prof[lo]
            if hi < n:
                add += prof[hi]
            acc += add
        half.append(hw)

    r0, r1 = max(peak[0] - half[0], 0), min(peak[0] + half[0] + 1, k.shape[0])
    a0, a1 = max(peak[1] - half[1], 0), min(peak[1] + half[1] + 1, k.shape[1])
    d0, d1 = max(peak[2] - half[2], 0), min(peak[2] + half[2] + 1, k.shape[2])
    sub = k[r0:r1, a0:a1, d0:d1].copy()
    return PSF(sub, psf.range_res, psf.azimuth_res_deg, psf.doppler_res)


def synthetic_psf(
    cfg,
    range_sigma_bins: float = 1.2,
    azimuth_sigma_bins: float = 6.0,
    doppler_sigma_bins: float = 1.0,
    size_sigmas: float = 4.0,
    sidelobe_db: float = -25.0,
) -> PSF:
    """
    Analytic PSF for standalone / no-measured-PSF runs.

    Matches the qualitative shape described in the paper (Fig. 3a): a *wide*
    spread in azimuth (coarse angular resolution) but *narrow* spread in range
    and Doppler (high range/Doppler resolution). A modest sinc-like sidelobe
    ring is added in azimuth to imitate array beam sidelobes.

    Resolutions are taken from `cfg` so the PSF grid is consistent with the
    tensor grid.
    """
    def axis_kernel(sigma, sidelobe=False):
        half = max(int(np.ceil(size_sigmas * sigma)), 1)
        x = np.arange(-half, half + 1, dtype=np.float64)
        g = np.exp(-0.5 * (x / sigma) ** 2)
        if sidelobe:
            # add attenuated sinc sidelobes to mimic a beam pattern
            with np.errstate(divide="ignore", invalid="ignore"):
                s = np.sinc(x / (2.5 * sigma))
            s = np.abs(s)
            lin = 10 ** (sidelobe_db / 20.0)
            g = g + lin * s
        return g

    gr = axis_kernel(range_sigma_bins)
    ga = axis_kernel(azimuth_sigma_bins, sidelobe=True)
    gd = axis_kernel(doppler_sigma_bins)

    kernel = gr[:, None, None] * ga[None, :, None] * gd[None, None, :]
    kernel = np.maximum(kernel, 0.0)
    return PSF(
        kernel=kernel,
        range_res=cfg.range_res,
        azimuth_res_deg=cfg.azimuth_res_deg,
        doppler_res=cfg.doppler_res,
    )


def psf_from_tensor(
    tensor: np.ndarray,
    cfg,
    peak_index: tuple = None,
    box_half: tuple = (6, 40, 4),
    floor_db: float = -60.0,
) -> PSF:
    """
    MEASURE a PSF from a real radar tensor that contains a single narrow point
    target (pole / corner reflector). This is the procedure the paper relies on
    to obtain a radar's PSF WITHOUT any hardware knowledge.

    Steps:
      1. locate the target (global max) unless peak_index is given.
      2. crop a box around it.
      3. floor low-energy cells (noise) to zero using floor_db below peak.
      4. return the cropped, centred kernel.

    Parameters
    ----------
    tensor    : (R, A, D) linear-magnitude radar tensor of a point target.
    cfg       : RadarConfig (for resolution metadata).
    box_half  : half-widths (range, azimuth, doppler) of the crop box.
    floor_db  : cells below (peak + floor_db) are zeroed as noise.
    """
    tensor = np.asarray(tensor, dtype=np.float64)
    if peak_index is None:
        peak_index = np.unravel_index(np.argmax(tensor), tensor.shape)
    pr, pa, pd = peak_index
    hr, ha, hd = box_half

    r0, r1 = max(pr - hr, 0), min(pr + hr + 1, tensor.shape[0])
    a0, a1 = max(pa - ha, 0), min(pa + ha + 1, tensor.shape[1])
    d0, d1 = max(pd - hd, 0), min(pd + hd + 1, tensor.shape[2])
    sub = tensor[r0:r1, a0:a1, d0:d1].copy()

    peak = sub.max()
    if peak > 0:
        lin_floor = peak * (10 ** (floor_db / 20.0))
        sub[sub < lin_floor] = 0.0
    return PSF(
        kernel=sub,
        range_res=cfg.range_res,
        azimuth_res_deg=cfg.azimuth_res_deg,
        doppler_res=cfg.doppler_res,
    )
