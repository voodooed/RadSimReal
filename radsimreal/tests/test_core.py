"""
Tests validating the RadSimReal method's core properties.

Run:  PYTHONPATH=. python -m pytest tests/ -q   (or) python tests/test_core.py
"""
import numpy as np

import radsimreal as rsr
from radsimreal.psf.psf import synthetic_psf, truncate_psf
from radsimreal.core.synthesis import rasterize, convolve_sparse, convolve_fft


def test_psf_shape_wide_azimuth_narrow_range():
    cfg = rsr.RadarConfig.small()
    psf = synthetic_psf(cfg)
    kr, ka, kd = psf.shape
    # paper Fig 3a: wide angular spread, narrow range/doppler
    assert ka > kr and ka > kd


def test_truncation_keeps_energy_and_shrinks():
    cfg = rsr.RadarConfig.raddet()
    full = synthetic_psf(cfg, azimuth_sigma_bins=18.0, size_sigmas=8.0).normalized()
    tr = truncate_psf(full, 0.99)
    assert 0.95 <= tr.energy <= 1.0             # ~99% target (per-axis marginal)
    assert np.prod(tr.shape) < np.prod(full.shape)   # strictly smaller


def test_sparse_equals_fft():
    cfg = rsr.RadarConfig.small()
    pts, _ = rsr.default_scene(3).build()
    sim = rsr.RadSimReal(cfg=cfg); pts = sim.assign_reflectivity(pts)
    field = rasterize(pts, cfg)
    s = convolve_sparse(field, sim.psf)
    f = convolve_fft(field, sim.psf)
    rel = np.linalg.norm(s - f) / (np.linalg.norm(f) + 1e-12)
    assert rel < 1e-10                          # equivalence claim


def test_single_point_reproduces_psf():
    # convolving a single unit reflection point must reproduce the PSF exactly
    cfg = rsr.RadarConfig.small()
    sim = rsr.RadSimReal(cfg=cfg)
    field = np.zeros(cfg.tensor_shape)
    r, a, d = cfg.range_bins // 2, cfg.azimuth_bins // 2, cfg.doppler_bins // 2
    field[r, a, d] = 1.0
    out = convolve_sparse(field, sim.psf)
    kr, ka, kd = sim.psf.shape
    sub = out[r - kr // 2: r - kr // 2 + kr,
              a - ka // 2: a - ka // 2 + ka,
              d - kd // 2: d - kd // 2 + kd]
    assert np.allclose(sub, sim.psf.kernel)


def test_measure_psf_from_tensor():
    # place a known PSF at one point, then recover it with psf_from_tensor
    cfg = rsr.RadarConfig.small()
    sim = rsr.RadSimReal(cfg=cfg)
    field = np.zeros(cfg.tensor_shape)
    field[40, 48, 12] = 1.0
    tensor = convolve_sparse(field, sim.psf)
    from radsimreal.psf.psf import psf_from_tensor
    measured = psf_from_tensor(tensor, cfg, box_half=(6, 25, 4), floor_db=-120)
    # measured peak location energy should dominate and match magnitude
    assert measured.kernel.max() > 0
    assert abs(measured.kernel.max() - sim.psf.kernel.max()) < 1e-6


def test_reflectivity_ordering():
    m = rsr.ReflectivityModel()
    xyz = np.array([[10, 0, 0]] * 3, dtype=float)
    rho = m(xyz, material_id=np.array(["vehicle", "road", "vegetation"], dtype=object))
    assert rho[0] > rho[1] > rho[2]             # metal > road > foliage


def test_range_attenuation_monotonic():
    m = rsr.ReflectivityModel()
    near = m(np.array([[5, 0, 0]], float), material_id=np.array(["vehicle"], dtype=object))
    far = m(np.array([[40, 0, 0]], float), material_id=np.array(["vehicle"], dtype=object))
    assert near[0] > far[0]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"\nAll {len(fns)} tests passed.")
