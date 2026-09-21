# Usage & Integration Guide

This guide covers (A) the shared concepts, (B) CARLA integration, and
(C) SimDaaS / Unreal integration. The RadSimReal core is identical across all
three — only the *reflection-point source* changes.

---

## A. Shared concepts

### The radar frame
All reflection points must end up in the **radar frame**:

```
x = forward (boresight),  y = left,  z = up,  origin at the radar.   (right-handed)
```

Range = ‖(x,y,z)‖, azimuth = atan2(y, x) in degrees, Doppler = radial velocity
(positive = approaching). Adapters are responsible for getting native simulator
coordinates into this frame.

### The tensor grid
`RadarConfig` defines the discrete (range × azimuth × Doppler) grid the tensor
lives on. Match it to the radar you are emulating:

```python
from radsimreal import RadarConfig
cfg = RadarConfig.raddet()          # TI AWR1843 (paper): 0-50 m, ±85°, ±13 m/s
# or custom:
cfg = RadarConfig(range_max=64, range_bins=256, azimuth_min_deg=-60,
                  azimuth_max_deg=60, azimuth_bins=192, doppler_bins=64)
```

### The PSF — the one thing you must get right for realism
Three options, in increasing fidelity:

1. **Synthetic PSF (default).** `synthetic_psf(cfg)` — wide in azimuth, narrow in
   range/Doppler. Good enough to validate the pipeline and for ablations.
2. **Measured PSF (paper's method).** Record a radar tensor of a single narrow
   target (a pole or corner reflector), then:
   ```python
   from radsimreal.psf.psf import psf_from_tensor
   psf = psf_from_tensor(point_target_tensor, cfg)
   sim = RadSimReal(cfg=cfg, psf=psf)
   ```
   This is what makes RadSimReal radar-agnostic: the PSF encodes all the hardware
   + signal-processing behaviour without you knowing any of it.
3. **Per-radar PSF library.** Measure one PSF per radar type once, cache it, and
   reuse it for unlimited synthetic frames.

### Reflectivity
`sim.assign_reflectivity(points)` fills each point's RF reflectivity from its
material (via `MATERIAL_TABLE`), surface orientation (if normals are provided),
and range (R⁻⁴). Override materials by editing `MATERIAL_TABLE` /
`CARLA_TAG_TO_MATERIAL` / `SIMDAAS_MATERIAL_MAP`, or bypass entirely by supplying
`reflectivity` directly on the points.

### Noise
`NoiseModel.from_snr_db(peak, snr_db)` for standalone runs, or
`NoiseModel.from_tensor_region(real_tensor)` to match a real radar's measured
noise floor.

---

## B. CARLA integration

### Requirements
- A running CARLA server (e.g. `./CarlaUE4.sh`).
- The `carla` Python package matching your server version.
- Synchronous mode strongly recommended for reproducible frames.

### How it works
`CarlaRadarAdapter` spawns a **semantic LiDAR** (`sensor.lidar.ray_cast_semantic`)
at the radar pose. Semantic LiDAR returns, per hit: 3D point, semantic tag,
cosine of incidence, and actor id. That gives:
- **geometry** → reflection points,
- **material** → from the semantic tag (`CARLA_TAG_TO_MATERIAL`),
- **Doppler** → from the hit actor's velocity, projected on the line of sight
  (ego velocity subtracted if the radar is attached to the ego).

The semantic LiDAR ray-casting *is* the "ray tracing from radar to objects" of
Fig. 2a; RadSimReal then does the radar-image formation of Fig. 2c.

### Live example

```bash
python examples/demo_carla.py --host localhost --port 2000 --out carla.png
```

Or in your own loop:

```python
import carla, radsimreal as rsr
from radsimreal.adapters.carla_adapter import CarlaRadarAdapter

world = carla.Client("localhost", 2000).get_world()
radar_tf = carla.Transform(carla.Location(x=0, y=0, z=0.5))
adapter = CarlaRadarAdapter(world, transform=radar_tf, attach_to=ego_vehicle).spawn()

sim = rsr.RadSimReal(cfg=rsr.RadarConfig.raddet())
world.tick()                                   # advance one frame (sync mode)
points = adapter.get_reflection_points()
points = sim.assign_reflectivity(points)
res = sim.simulate_image(points)               # res.image is your radar image
adapter.destroy()
```

### Offline replay
If you dump semantic-lidar frames to disk as structured arrays with fields
`x,y,z,cos,idx,tag`, replay them without a server:

```python
from radsimreal.adapters.carla_adapter import reflection_points_from_semantic_array
points = reflection_points_from_semantic_array(np.load("frame.npy"))
```

### Tuning tips
- Increase `points_per_second` and `channels` for denser surfaces (more faithful
  spreading functions) at the cost of speed.
- Densify vertical coverage with `upper_fov` / `lower_fov` around the objects of
  interest.
- Map any CARLA tags your build uses to materials by editing
  `CARLA_TAG_TO_MATERIAL` in `environment/reflectivity.py`.

---

## C. SimDaaS (Unreal Engine) integration

Because SimDaaS is your own engine, RadSimReal talks to it through an explicit,
minimal **data contract**. Your Unreal side produces reflection points per radar
frame; the Python side consumes them via `SimDaaSAdapter`.

### The per-frame data contract

Emit, for each radar frame, the following (numpy `.npz` or `.json`):

| Key | Shape | Required | Meaning |
|---|---|---|---|
| `points_world` | (N,3) | ✔ | Surface hit points, UE **world** frame |
| `radar_location` | (3,) | ✔ | Radar origin, UE world frame |
| `radar_rotation` | (3,) | ✔ | (roll, pitch, yaw) degrees, UE convention |
| `material` | (N,) | ○ | Per-point material name/id (mapped by `SIMDAAS_MATERIAL_MAP`) |
| `normals_world` | (N,3) | ○ | Per-point surface normal (improves orientation gain) |
| `velocities_world` | (N,3) | ○ | Per-point velocity (enables Doppler) |
| `reflectivity` | (N,) | ○ | Precomputed linear reflectivity (overrides `material`) |

**Units:** UE default is **centimetres, left-handed (x fwd, y right, z up)**. The
adapter converts to the radar frame (metres, y-left) automatically when you set
`units="cm"`. If your export is already radar-frame/metric, construct the adapter
with `already_radar_frame=True`.

The reference writer shows the exact format:

```python
from radsimreal.adapters.simdaas_adapter import write_simdaas_frame
write_simdaas_frame("frame.npz",
    points_world=P,                 # (N,3) cm, UE world
    radar_location=(0,0,60),        # cm
    radar_rotation=(0,0,0),         # deg
    material=material_names,        # (N,) e.g. "M_Car"
    velocities_world=V)             # (N,3) cm/s, UE world
```

### How to produce `points_world` in Unreal

Any of these work; pick what fits your SimDaaS build:

1. **Line traces / sphere traces** from the radar pose across a spherical grid
   covering the radar FOV — closest to Fig. 2a's ray tracing. Record `ImpactPoint`,
   `ImpactNormal`, the hit component's **Physical Material**, and the hit actor's
   velocity.
2. **Scene Capture 2D** with depth + world-normal + a material-id G-buffer, read
   back and unprojected to world points (dense, GPU-friendly).
3. **Niagara / point-cloud export** if you already generate a LiDAR-like cloud;
   attach material and velocity per point.

Tag physical materials consistently (e.g. `M_Car`, `M_Pole`, `M_Building`) and add
them to `SIMDAAS_MATERIAL_MAP` in `adapters/simdaas_adapter.py`.

### Consuming a frame

```bash
# no live UE needed — writes a fake UE frame then simulates:
python examples/demo_simdaas.py                       # -> simdaas_radsimreal.png
# or point the CLI at a real exported frame:
radsimreal simdaas --frame frame.npz --units cm --out sim.png
```

Programmatically:

```python
import radsimreal as rsr
from radsimreal.adapters.simdaas_adapter import SimDaaSAdapter

adapter = SimDaaSAdapter(units="cm")
points = adapter.load_frame("frame.npz")              # world->radar handled here
sim = rsr.RadSimReal(cfg=rsr.RadarConfig.raddet())
points = sim.assign_reflectivity(points)              # uses normals if you exported them
res = sim.simulate_image(points)
```

### Live (socket) integration sketch
If you drive RadSimReal from the same process/socket that holds SimDaaS data,
skip disk entirely and call `get_reflection_points(...)` with in-memory arrays:

```python
points = adapter.get_reflection_points(
    points_world=P, radar_pose={"location": loc, "rotation": (r,p,y)},
    material=mats, normals_world=normals, velocities_world=vels)
```

---

## D. Building a training dataset

```python
from radsimreal.pipeline.dataset import generate_synthetic_dataset
from radsimreal import RadarConfig
generate_synthetic_dataset("data/", n_frames=10000, cfg=RadarConfig.raddet(),
                           snr_db=30, save_tensors=False)
```

Output:
```
data/images/000000.npy ...      # (R,A) radar images (float32)
data/tensors/000000.npy ...     # (R,A,D) if save_tensors=True
data/annotations.json           # per-frame range-azimuth boxes
data/meta.json                  # radar config + settings
```

To generate from **CARLA/SimDaaS** frames instead of synthetic scenes, loop your
adapter over frames and write `res.image` + your boxes with the same layout — the
annotation format is `{"frame": "000000", "boxes": [{"class","r_min","r_max",
"az_min_deg","az_max_deg"}, ...]}`.

These images/boxes are the exact inputs the paper feeds to U-Net / RADDet /
Probabilistic detectors; train those on the synthetic set and test on real radar
to reproduce the paper's sim-to-real analysis.

---

## E. Reproducing the paper's key experiments

1. **PSF equivalence (Fig. 3).** `convolve_sparse` vs `convolve_fft` — see
   `tests/test_sparse_equals_fft` (relative error ~1e-16).
2. **99% truncation & speed (Sec. 3.2).** `truncate_psf(psf, 0.99)`; benchmark
   sparse-truncated vs dense-full convolution.
3. **Sim-to-real gap (Sec. 4).** Generate a RadSimReal dataset for a radar,
   train a detector, evaluate on the matching real dataset (RADDet/CARRADA/CRUW).
   Use a **measured** PSF from that radar for best fidelity.
