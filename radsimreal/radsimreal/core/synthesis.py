"""
Tensor synthesis engine  (Fig. 2c: Convolution of Reflection Points & PSF).

Given reflection points (with reflectivity) and a measured/synthetic PSF, build
the radar tensor by:

    1. RASTERIZE the reflection points into a sparse 3D field on the
       (range x azimuth x doppler) grid, accumulating reflectivity per cell.
    2. CONVOLVE that field with the (truncated) PSF.
    3. ADD noise.

Because the reflection-point field is extremely sparse and the truncated PSF is
tiny (paper: >1000x smaller than the full tensor), we do NOT run a dense FFT
convolution. Instead we do a *scatter-add* of the PSF kernel at each occupied
cell -- i.e. sparse convolution -- which is what makes RadSimReal fast.

A dense FFT path is also provided (convolve_fft) for validation / equivalence
checking against the sparse path.
"""
from __future__ import annotations

import numpy as np

from .radar_config import RadarConfig
from .reflection_points import ReflectionPoints
from ..psf.psf import PSF


def rasterize(points: ReflectionPoints, cfg: RadarConfig) -> np.ndarray:
    """
    Accumulate point reflectivity into a dense (R, A, D) grid.
    Points outside the grid are dropped. Multiple points in a cell sum.
    """
    R = points.range
    A = points.azimuth_deg
    V = points.velocity

    mask = cfg.in_bounds(R, A, V)
    R, A, V, W = R[mask], A[mask], V[mask], points.reflectivity[mask]

    rb = cfg.range_to_bin(R)
    ab = cfg.azimuth_to_bin(A)
    db = cfg.doppler_to_bin(V)

    field = np.zeros(cfg.tensor_shape, dtype=np.float64)
    np.add.at(field, (rb, ab, db), W)
    return field


def convolve_sparse(field: np.ndarray, psf: PSF) -> np.ndarray:
    """
    Sparse 3D convolution by scatter-adding the PSF kernel at each nonzero cell.
    Equivalent to scipy.ndimage convolution but only touches occupied voxels,
    which is the RadSimReal speed advantage.
    """
    k = psf.kernel
    kr, ka, kd = k.shape
    cr, ca, cd = kr // 2, ka // 2, kd // 2   # kernel centre

    out = np.zeros_like(field)
    nz = np.argwhere(field > 0)
    if nz.size == 0:
        return out

    Rn, An, Dn = field.shape
    for (r, a, d) in nz:
        w = field[r, a, d]
        # destination window in the output tensor
        r0, r1 = r - cr, r - cr + kr
        a0, a1 = a - ca, a - ca + ka
        d0, d1 = d - cd, d - cd + kd
        # clip to bounds and correspondingly clip the kernel
        kr0 = max(0, -r0); ka0 = max(0, -a0); kd0 = max(0, -d0)
        kr1 = kr - max(0, r1 - Rn); ka1 = ka - max(0, a1 - An); kd1 = kd - max(0, d1 - Dn)
        r0c, a0c, d0c = max(r0, 0), max(a0, 0), max(d0, 0)
        r1c, a1c, d1c = min(r1, Rn), min(a1, An), min(d1, Dn)
        if r0c >= r1c or a0c >= a1c or d0c >= d1c:
            continue
        out[r0c:r1c, a0c:a1c, d0c:d1c] += w * k[kr0:kr1, ka0:ka1, kd0:kd1]
    return out


def convolve_fft(field: np.ndarray, psf: PSF) -> np.ndarray:
    """
    Dense FFT convolution -- used to validate the sparse path (equivalence
    check from the paper's Fig. 3). Requires scipy.
    """
    from scipy.signal import fftconvolve

    out = fftconvolve(field, psf.kernel, mode="same")
    return np.maximum(out, 0.0)
