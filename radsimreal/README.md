# RadSimReal (unofficial implementation)

A from-scratch implementation of **RadSimReal: Bridging the Gap Between
Synthetic and Real Data in Radar Object Detection With Simulation**
(Bialer & Haitman, CVPR 2024). The authors did not release code; this codebase
reconstructs the method exactly as described in the paper and wraps it so it can
be driven **standalone** (no simulator), from **CARLA**, or **SimDaaS** (Unreal Engine) simulator.

> This is a research reimplementation. It follows
> the paper's described method; absolute reflectivity/noise scales are heuristic
> (as the paper notes, only *relative* structure matters before PSF convolution
> and normalisation).

---

## 1. The method in one paragraph

A conventional physical radar simulation produces a range-azimuth-Doppler tensor
that is mathematically **the sum of the radar's Point Spread Function (PSF)
centred on every reflection point, plus noise**. RadSimReal exploits this: rather
than modelling proprietary radar hardware and signal processing, it (1) generates
reflection points from a 3D scene, (2) assigns each an RF reflectivity from
physical formulas (material, orientation, range), (3) **convolves the sparse
reflection-point field with a *measured* PSF**, (4) adds noise, and (5) collapses
Doppler by a max to get the radar image. The PSF is measured from a single narrow
target (pole / corner reflector), so **no radar design details are needed**, and
truncating it to 99% of its energy shrinks it by >1000x, giving a large speed-up.

```
 3D scene ──ray-trace──▶ reflection points ──assign RF reflectivity──▶ rasterize
    (env sim)                                                             │
                                                              convolve with PSF
                                                                          │
                                                                     + noise
                                                                          │
                                                             max over Doppler
                                                                          │
                                                                 range-azimuth image
```

---

## 2. What's in the codebase

```
radsimreal/
  core/
    radar_config.py       RadarConfig: tensor grid (range x azimuth x doppler),
                          resolutions, coord<->bin maps. Presets: .raddet(), .small().
    reflection_points.py  ReflectionPoints: xyz + reflectivity + velocity + material;
                          derives range/azimuth/elevation on demand.
    noise.py              NoiseModel: Rayleigh-magnitude noise; variance from a
                          target SNR (from_snr_db) or measured from a real tensor
                          region (from_tensor_region).
    synthesis.py          rasterize() sparse point field; convolve_sparse()
                          (scatter-add, the fast path) and convolve_fft()
                          (dense, for equivalence checking).
    image.py              tensor_to_image() (max over Doppler) and to_db().
  psf/
    psf.py                PSF container; synthetic_psf() (analytic, wide-azimuth /
                          narrow-range,doppler); truncate_psf() (99% energy box);
                          psf_from_tensor() (MEASURE a PSF from a point-target tensor).
  environment/
    reflectivity.py       ReflectivityModel: rho = |Gamma(material)|^2 * g(theta) *
                          (R_ref/R)^4 * A. MATERIAL_TABLE + CARLA_TAG_TO_MATERIAL.
    synthetic_scene.py    Standalone scene generator (cars/poles/walls/pedestrians)
                          with ground-truth range-azimuth boxes. No simulator needed.
  adapters/
    base.py               EnvironmentAdapter interface + world->radar transforms,
                          radial-velocity projection, Euler->rotation helpers.
    carla_adapter.py      CarlaRadarAdapter: semantic-lidar -> reflection points
                          (materials from semantic tags, Doppler from actor velocities).
                          Also an offline replay helper for saved arrays.
    simdaas_adapter.py    SimDaaSAdapter: Unreal export -> reflection points, with a
                          documented per-frame data contract and .npz/.json loaders.
  pipeline/
    simulator.py          RadSimReal orchestrator: assign_reflectivity(),
                          simulate_tensor(), simulate_image().
    dataset.py            generate_synthetic_dataset(): many frames -> images
                          (+ optional tensors) + annotations.json.
  cli/main.py             `radsimreal demo | dataset | simdaas`.
examples/
  demo_standalone.py      Runs the whole pipeline with no simulator; saves a figure.
  demo_carla.py           Live CARLA example (needs a CARLA server + `carla`).
  demo_simdaas.py         Writes a fake UE frame, loads it, simulates (no UE needed).
tests/test_core.py        Validates: PSF shape, 99% truncation, sparse==FFT,
                          single-point==PSF, PSF measurement, reflectivity ordering.
```

---

## 3. Install & run

```bash
pip install -e .          # installs numpy/scipy/matplotlib and the `radsimreal` CLI

# Standalone smoke test (no simulator required):
python examples/demo_standalone.py          # -> radsimreal_demo.png
python tests/test_core.py                    # -> All 7 tests passed
radsimreal dataset --out data/ --frames 200  # -> images + annotations
```

Minimal API use:

```python
import radsimreal as rsr

cfg = rsr.RadarConfig.raddet()                 # or .small() for quick CPU runs
scene = rsr.default_scene(seed=1)
points, gt_boxes = scene.build()

sim = rsr.RadSimReal(cfg=cfg)                  # synthetic PSF, sparse convolution
points = sim.assign_reflectivity(points)       # material+geometry -> RF reflectivity
res = sim.simulate_image(points)               # res.tensor (R,A,D), res.image (R,A)
```

Using a **measured** PSF (the paper's real workflow):

```python
from radsimreal.psf.psf import psf_from_tensor
psf = psf_from_tensor(point_target_tensor, cfg)   # from a pole/corner-reflector scan
sim = rsr.RadSimReal(cfg=cfg, psf=psf)
```

Measured noise:

```python
sim.noise = rsr.NoiseModel.from_tensor_region(real_empty_tensor)   # or from_snr_db(...)
```

See `docs/USAGE.md` for CARLA and SimDaaS integration details.

---

## 4. Fidelity checks (from `tests/`)

| Paper property | Test | Result |
|---|---|---|
| PSF is wide in azimuth, narrow in range/Doppler (Fig. 3a) | `test_psf_shape_*` | pass |
| Truncated PSF keeps ~99% energy, strictly smaller (Sec. 3) | `test_truncation_*` | pass |
| Sparse convolution == dense FFT (equivalence, Fig. 3) | `test_sparse_equals_fft` | rel. err ~1e-16 |
| A single unit point reproduces the PSF exactly | `test_single_point_*` | pass |
| A PSF can be measured back from a point-target tensor | `test_measure_psf_*` | pass |
| Reflectivity ordering metal>road>foliage, near>far | `test_reflectivity_*` | pass |

A sparse+truncated convolution ran ~**78x faster** than a dense FFT with the full
PSF on the RADDet grid in local benchmarking (paper reports ~1000x at their tensor
sizes).
