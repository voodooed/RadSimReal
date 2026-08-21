# RadSimReal Integration Guide
### How to connect CARLA, SimDaaS, or any other simulator to the RadSimReal pipeline

This document is self-contained: it explains the integration architecture once,
then gives a full, copy-pasteable walkthrough for **CARLA**, for **SimDaaS**
(your in-house Unreal simulator), and for **any other simulator** you might
plug in later. It assumes you have the `radsimreal` package installed
(`pip install -e .` from the package root).

---

## 0. The one idea that makes integration simple

RadSimReal's physics core (`RadSimReal` in `radsimreal/pipeline/simulator.py`)
does not know or care where reflection points came from. It only consumes one
object:

```python
ReflectionPoints(
    xyz,          # (N,3) float, RADAR FRAME, metres
    reflectivity, # (N,)  float, linear RF reflectivity (can be all-ones; filled in later)
    velocity,     # (N,)  float, radial velocity m/s, +ve = approaching
    material_id,  # (N,)  optional, str or int labels used to look up reflectivity
)
```

**Every integration — CARLA, SimDaaS, or a future simulator — is just a function
that produces this object for one radar frame.** That function is called an
*adapter*. Once you have it, the rest of the pipeline is identical:

```python
sim = RadSimReal(cfg=radar_config)              # same object regardless of source
points = adapter.get_reflection_points(...)      # <-- the only simulator-specific part
points = sim.assign_reflectivity(points)          # physics: material+orientation+range
result = sim.simulate_image(points)               # PSF convolution + noise + Doppler max
# result.image  -> (range, azimuth) radar image
# result.tensor -> (range, azimuth, doppler) radar tensor
```

### The radar frame — get this right first, everything else follows
```
x = forward (boresight direction the radar points)
y = left
z = up
origin = the radar's own position
right-handed, metres
```
range = ‖xyz‖, azimuth = atan2(y, x) in degrees, elevation = derived similarly.
**The single most common integration bug is a coordinate-frame or units mismatch
between your simulator and this convention** — Sections 1 and 2 show exactly how
each adapter converts into it, and Section 4 gives a checklist to debug it if
your radar image looks wrong (targets at the wrong azimuth sign, wrong range, or
empty images).

### Three ways to feed points to the pipeline, by increasing effort/fidelity
| Level | What you provide | Effort | Fidelity |
|---|---|---|---|
| 1 | Just `xyz` (material/velocity defaulted) | minutes | geometry only, no material contrast, no Doppler |
| 2 | `xyz` + `material` (+ optionally `normals`) | ~1 hour | realistic reflectivity contrast (cars bright, foliage dim) |
| 3 | Level 2 + `velocities` + a **measured PSF** for your radar | ~1 day | full paper-fidelity: realistic image, correct Doppler, radar-specific PSF |

Start at level 1 to validate the pipeline is wired correctly, then move up.

---

## 1. CARLA integration

### 1.1 What CARLA gives you, and why we use semantic LiDAR

CARLA doesn't have a "give me raw reflection points" API, but its
**semantic LiDAR** sensor (`sensor.lidar.ray_cast_semantic`) is functionally
identical to the "ray-trace from radar to objects" step in the paper's
environment-simulation stage: it fires rays from a sensor pose across a
spherical scan pattern and returns, per hit:

- 3D impact point (sensor frame)
- `cos_inc_angle` — cosine of the incidence angle (useful for orientation)
- `object_idx` — the id of the actor that was hit (lets us look up its velocity)
- `object_tag` — CARLA's semantic segmentation class (car, pole, building, ...)

This is exactly the geometry + material + (indirectly) orientation information
RadSimReal's reflectivity model needs. `CarlaRadarAdapter` wraps this sensor.

### 1.2 Prerequisites

- A running CARLA server: `./CarlaUE4.sh` (Linux) or `CarlaUE4.exe` (Windows),
  any recent 0.9.x release.
- The matching `carla` Python wheel installed in the **same** Python environment
  as `radsimreal`:
  ```bash
  pip install carla==<your-server-version>
  ```
  (Check `python -c "import carla; print(carla.__file__)"` works before continuing.)
- `radsimreal` installed: `pip install -e .` from the package root.

The adapter module (`radsimreal/adapters/carla_adapter.py`) imports `carla`
**lazily inside its methods**, so `import radsimreal` never fails even on a
machine without CARLA installed — you only need the `carla` package on the
machine that actually talks to the server.

### 1.3 Step-by-step: minimal working integration

```python
import carla
import radsimreal as rsr
from radsimreal.adapters.carla_adapter import CarlaRadarAdapter

# --- 1. connect ---
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()

# --- 2. use synchronous mode (strongly recommended) ---
# In async mode, sensor callbacks can arrive out of step with your control
# code, so "the latest frame" may not be the frame you think it is.
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05        # 20 Hz simulation step
world.apply_settings(settings)

# --- 3. define the radar pose in the WORLD frame ---
# carla.Transform uses CARLA's own convention (left-handed, cm... actually
# CARLA Python API already gives you metres, but is still left-handed:
# x forward, y RIGHT, z up). The adapter converts this to our right-handed,
# y-left radar frame internally -- you do NOT need to flip anything yourself.
radar_transform = carla.Transform(
    carla.Location(x=2.5, y=0.0, z=0.5),   # 2.5 m ahead of vehicle origin, 0.5 m up
    carla.Rotation(pitch=0, yaw=0, roll=0) # facing forward
)

# --- 4. (optional) attach to a vehicle so the radar moves with it ---
# If you skip attach_to, the radar is a static world sensor at `radar_transform`.
blueprint_lib = world.get_blueprint_library()
vehicle_bp = blueprint_lib.filter("vehicle.tesla.model3")[0]
spawn_point = world.get_map().get_spawn_points()[0]
ego = world.spawn_actor(vehicle_bp, spawn_point)
ego.set_autopilot(True)

# --- 5. spawn the RadSimReal CARLA adapter's sensor ---
adapter = CarlaRadarAdapter(
    world,
    transform=radar_transform,   # pose RELATIVE to attach_to if attach_to is set
    attach_to=ego,                # None => static world-frame radar
    lidar_range=80.0,             # metres, should exceed your RadarConfig.range_max
    points_per_second=1_500_000,  # raise for denser point clouds
    channels=64,                  # vertical rings; raise for finer elevation coverage
    upper_fov=15.0, lower_fov=-25.0,
)
adapter.spawn()

# --- 6. build the RadSimReal simulator once ---
cfg = rsr.RadarConfig.raddet()          # match your target radar's grid
sim = rsr.RadSimReal(cfg=cfg)           # add psf=... here once you have a measured PSF

# --- 7. main loop ---
try:
    for frame_idx in range(200):
        world.tick()                                    # advance simulation by one step
        points = adapter.get_reflection_points()         # <-- CARLA -> ReflectionPoints
        points = sim.assign_reflectivity(points)          # material/orientation/range physics
        result = sim.simulate_image(points)                # PSF convolution + noise
        radar_image = result.image                          # (range, azimuth) — your output
        # ... save radar_image, run a detector on it, log it, etc.
finally:
    adapter.destroy()
    settings.synchronous_mode = False
    world.apply_settings(settings)
    ego.destroy()
```

Run the equivalent ready-made script:
```bash
python examples/demo_carla.py --host localhost --port 2000 --out carla.png
```

### 1.4 What each adapter argument controls

| Argument | Effect | Tuning guidance |
|---|---|---|
| `transform` | Radar pose (world frame, or relative to `attach_to`) | Match your real radar's mounting position on the vehicle |
| `attach_to` | `None` (static) or a `carla.Actor` (moving with it) | Set to the ego vehicle for a forward-facing automotive radar |
| `lidar_range` | Max ray distance (m) | Set ≥ `RadarConfig.range_max` or you'll silently lose far targets |
| `points_per_second` | Ray density | Increase for smoother reflectivity fields on large surfaces (walls); costs CARLA-side performance |
| `channels` / `upper_fov` / `lower_fov` | Vertical scan coverage | Widen if your radar has a wide elevation FOV; narrow + increase channels for a flat/2D radar model |
| `rotation_frequency` | Scan rate | 20 Hz default matches most CARLA ticking rates; keep it ≥ your fixed_delta_seconds⁻¹ |

### 1.5 Doppler / velocity handling

`get_reflection_points(compute_doppler=True)` looks up each hit actor's world
velocity via `world.get_actors()`, subtracts the ego's own velocity if the radar
is `attach_to`-attached, and projects the relative velocity onto the
point's line-of-sight (`radial_velocity` in `adapters/base.py`). Static
background (roads, buildings) correctly comes back with zero Doppler.

If you don't need Doppler (e.g. you only care about the range-azimuth image),
call `adapter.get_reflection_points(compute_doppler=False)` to skip the actor
velocity lookup and speed things up slightly.

### 1.6 Mapping CARLA's semantic tags to materials

`CARLA_TAG_TO_MATERIAL` in `radsimreal/environment/reflectivity.py` maps CARLA's
integer semantic tags (its CityScapes-style palette) to reflectivity material
keys (`"vehicle"`, `"pole"`, `"building"`, ...). If your CARLA build/town uses
custom semantic tags, or you add custom static meshes with a specific tag,
extend this dict:

```python
from radsimreal.environment.reflectivity import CARLA_TAG_TO_MATERIAL
CARLA_TAG_TO_MATERIAL[40] = "guardrail"   # example: a custom tag id
```

### 1.7 Offline replay (no live server)

If you've already logged semantic-LiDAR data to disk (e.g. during a data
collection run), you can replay it without CARLA running:

```python
import numpy as np
from radsimreal.adapters.carla_adapter import reflection_points_from_semantic_array

# arr must be a structured array with fields x,y,z,cos,idx,tag
# (this is exactly the dtype CARLA's semantic LiDAR raw_data unpacks to —
#  see CarlaRadarAdapter._on_data for the exact dtype if you're saving it yourself)
arr = np.load("logged_frame.npy")
points = reflection_points_from_semantic_array(arr)
points = sim.assign_reflectivity(points)
result = sim.simulate_image(points)
```

### 1.8 CARLA troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: No CARLA data yet` | Called `get_reflection_points()` before any `world.tick()` after `spawn()` | Tick at least once before reading; in async mode, sleep briefly instead |
| Radar image is empty / all noise | `lidar_range` < `RadarConfig.range_max`, or radar pose is inside geometry | Increase `lidar_range`; check `transform` isn't embedded in the vehicle mesh |
| Targets at mirrored azimuth | You manually flipped a coordinate on top of the adapter's own conversion | Don't do any coordinate conversion yourself — pass `carla.Transform` as-is |
| Doppler looks wrong/huge | Velocities came from a custom actor spawner that doesn't set physics velocity | Ensure actors use CARLA's physics/autopilot so `get_velocity()` is meaningful |
| Frame-to-frame inconsistent results | Running in asynchronous mode | Switch to synchronous mode as shown in 1.3 |

---

## 2. SimDaaS (Unreal Engine) integration

### 2.1 Why SimDaaS needs an explicit contract

Unlike CARLA, SimDaaS is your own engine — there's no shared library to import.
So instead of a live-connection adapter, `SimDaaSAdapter` defines a **small,
explicit data contract**: your Unreal side just has to produce a handful of
arrays per radar frame, in a documented format, and hand them to the adapter
either as **in-memory arrays** (if you embed Python) or as a **file on disk**
(`.npz`/`.json`) that Python reads. Everything downstream is identical to CARLA.

### 2.2 The data contract (memorize this table)

Per radar frame, provide:

| Key | Shape | Required? | Meaning |
|---|---|---|---|
| `points_world` | `(N, 3)` | **required** | Surface hit points, in **Unreal world space** |
| `radar_location` | `(3,)` | **required** | Radar's world-space origin |
| `radar_rotation` | `(3,)` | **required** | `(roll, pitch, yaw)` in degrees, UE convention |
| `material` | `(N,)` | optional | Per-point material name/id (e.g. `"M_Car"`) |
| `normals_world` | `(N, 3)` | optional | Per-point surface normal (improves orientation realism) |
| `velocities_world` | `(N, 3)` | optional | Per-point velocity (enables Doppler) |
| `reflectivity` | `(N,)` | optional | Precomputed reflectivity — **overrides** `material` if given |

**Units and handedness:** Unreal's default is **centimetres, left-handed**
(x forward, y **right**, z up). `SimDaaSAdapter(units="cm")` — the default —
converts cm→m and flips the y axis for you automatically. If your export
pipeline already outputs metric, right-handed, radar-frame data (e.g. you did
the conversion engine-side), construct it with `already_radar_frame=True` and
skip conversion entirely.

### 2.3 Step 1 — get reflection points out of Unreal

Pick whichever fits your SimDaaS build; all three feed the same contract.

**(a) Line traces from the radar pose** (simplest, good for a first integration):
scan a spherical grid of rays from the radar's location/rotation, record each
hit's impact point, normal, physical-material name, and the hit actor's
velocity. A full reference implementation (Unreal-Python and C++ pseudocode) is
in `docs/UNREAL_EXPORTER.md` — copy `scan_radar_frame()` / `export_frame()`
directly.

**(b) GPU scene capture** (recommended for dense scenes / production use):
render depth + world-normal + a material-id buffer from a Scene Capture
Component at the radar pose, read the buffers back, and unproject pixels to
world-space points using the depth buffer and camera intrinsics. This gives you
tens of thousands of points per frame far more cheaply than per-ray traces.

**(c) An existing point-cloud/LiDAR export**, if SimDaaS already has one:
just attach a material label and velocity to each point you already emit.

Whichever you choose, the *output* is always the same: arrays matching the
table in 2.2.

### 2.4 Step 2 — write (or hand off) a frame

If Unreal and Python are **separate processes**, write one file per frame:

```python
# runs wherever your export pipeline lives (could be a small script Unreal
# calls out to, or a step in your own SimDaaS tooling)
from radsimreal.adapters.simdaas_adapter import write_simdaas_frame

write_simdaas_frame(
    "frames/000123.npz",
    points_world=P,              # (N,3) cm, UE world frame — straight from your exporter
    radar_location=(0, 0, 60),   # cm
    radar_rotation=(0, 0, 0),    # degrees
    material=material_names,     # (N,) e.g. array(["M_Car", "M_Pole", ...])
    velocities_world=V,          # (N,3) cm/s
    normals_world=Nrm,           # (N,3) optional
)
```

If Unreal and Python are **in the same process** (e.g. an embedded-Python
plugin, or a socket bridge that hands you numpy arrays directly), skip the file
and call the adapter directly with in-memory arrays — see 2.5.

### 2.5 Step 3 — consume the frame in RadSimReal

**From disk:**
```python
import radsimreal as rsr
from radsimreal.adapters.simdaas_adapter import SimDaaSAdapter

adapter = SimDaaSAdapter(units="cm")
points = adapter.load_frame("frames/000123.npz")   # does the full UE->radar conversion

cfg = rsr.RadarConfig.raddet()
sim = rsr.RadSimReal(cfg=cfg)
points = sim.assign_reflectivity(points)             # uses normals_world if you provided them
result = sim.simulate_image(points)
radar_image = result.image
```

**In-memory / live (no file):**
```python
points = adapter.get_reflection_points(
    points_world=P,
    radar_pose={"location": (0, 0, 60), "rotation": (0, 0, 0)},
    material=material_names,
    normals_world=Nrm,
    velocities_world=V,
)
points = sim.assign_reflectivity(points)
result = sim.simulate_image(points)
```

Or from the command line, for a quick check of one exported frame:
```bash
radsimreal simdaas --frame frames/000123.npz --units cm --out sim.png
```

Run the fully self-contained example (writes a fake UE frame, then simulates —
useful to sanity-check the adapter before you have real SimDaaS output):
```bash
python examples/demo_simdaas.py       # -> simdaas_radsimreal.png
```

### 2.6 Mapping SimDaaS material names to reflectivity

`SIMDAAS_MATERIAL_MAP` in `radsimreal/adapters/simdaas_adapter.py` maps your
Unreal physical-material names to the reflectivity table's keys:

```python
SIMDAAS_MATERIAL_MAP = {
    "M_Car": "vehicle", "M_Pole": "pole", "M_Building": "building",
    "M_Road": "road", "M_Foliage": "vegetation", ...
}
```
Add an entry for every physical material you tag in Unreal:
```python
from radsimreal.adapters.simdaas_adapter import SIMDAAS_MATERIAL_MAP
SIMDAAS_MATERIAL_MAP["M_GuardrailSteel"] = "guardrail"
SIMDAAS_MATERIAL_MAP["M_ConcreteBarrier"] = "concrete"
```
Any material name not in the map falls back to `"unknown"` (a middling
reflectivity) rather than erroring, so an incomplete map degrades gracefully —
but you should fill it in for anything that matters to your scenes.

### 2.7 Live/streaming integration pattern (recommended for closed-loop use)

For a training loop or closed-loop simulation where SimDaaS and Python run
concurrently, avoid per-frame disk I/O with a small socket bridge:

```
[Unreal / SimDaaS process]                     [Python / RadSimReal process]
   scan_radar_frame()                             socket / shared-memory server
        │  serialize P, material, V, normals              │
        └───────────────── send ─────────────────────────►│
                                                    adapter.get_reflection_points(...)
                                                    sim.assign_reflectivity(...)
                                                    sim.simulate_image(...)
                                                            │
        ◄──────────────── (optional) send image back ──────┘
```

Any transport works (raw TCP socket + numpy `.tobytes()`, ZeroMQ, shared
memory, gRPC) — the contract is unchanged: whatever arrives on the Python side
just needs to become `points_world`, `radar_location`, `radar_rotation`, and
the optional arrays, then passed to `adapter.get_reflection_points(...)`.

### 2.8 SimDaaS troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Everything ends up at the same range/azimuth | `radar_location`/`radar_rotation` not subtracted (`already_radar_frame=True` set incorrectly) | Set `already_radar_frame=False` unless you did the world→radar transform yourself |
| Targets mirrored left/right vs. expectation | Double-flipped y (you flipped it in your exporter AND the adapter flips it again) | Only flip once — let the adapter handle handedness for raw UE-world data |
| Doppler values in the hundreds of m/s | Velocities passed in cm/s but adapter told `units="m"`, or vice versa | Keep `velocities_world` in the **same** units as `points_world`; the adapter scales both together based on `units` |
| Radar image nearly empty | Points fall outside `RadarConfig`'s range/azimuth bounds (check `cfg.range_max`, `azimuth_min/max_deg`) after conversion | Print `points.range.min/max()` and `points.azimuth_deg.min/max()` after `load_frame()`/`get_reflection_points()` and compare to `cfg` |
| Materials all read as `"unknown"` | Material names from Unreal don't match keys in `SIMDAAS_MATERIAL_MAP` | Print `np.unique(material_names)` from your exporter and add missing entries to the map |

---

## 3. Integrating "any other simulator"

Because the contract is just `ReflectionPoints`, adding a new simulator is
writing one small adapter class. Use this as a template:

```python
from radsimreal.adapters.base import EnvironmentAdapter, transform_world_to_radar, radial_velocity
from radsimreal.core.reflection_points import ReflectionPoints
import numpy as np

class MySimulatorAdapter(EnvironmentAdapter):
    def __init__(self, units="m", already_radar_frame=False):
        self.units = units
        self.already_radar_frame = already_radar_frame

    def get_reflection_points(self, raw_frame, radar_pose=None) -> ReflectionPoints:
        # 1. pull xyz points + (optional) material/velocity/normals out of your
        #    simulator's native frame representation
        xyz_native = raw_frame["points"]        # whatever your sim calls it
        material   = raw_frame.get("material")
        vel_native = raw_frame.get("velocity")

        # 2. convert into the RADAR FRAME (x fwd, y left, z up, metres),
        #    unless already_radar_frame=True. Use transform_world_to_radar()
        #    and radial_velocity() from adapters.base as building blocks --
        #    they only need: a world position, a world->radar rotation matrix,
        #    and (optionally) world-frame per-point velocities.
        if self.already_radar_frame:
            xyz = xyz_native
        else:
            loc, R = radar_pose["location"], radar_pose["rotation_matrix"]
            xyz = transform_world_to_radar(xyz_native, loc, R)

        velocity = (radial_velocity(xyz, vel_native)
                    if vel_native is not None else np.zeros(len(xyz)))

        return ReflectionPoints(xyz=xyz, reflectivity=np.ones(len(xyz)),
                                velocity=velocity, material_id=material)
```

Checklist for any new adapter:
1. **Confirm your simulator's native handedness/units** and convert to
   right-handed metres with `x` forward, `y` left, `z` up.
2. **Subtract the radar's own pose** (position + rotation) so points end up
   *relative to the radar*, not the world origin.
3. **Map your simulator's material/semantic labels** into the
   `MATERIAL_TABLE` keys in `radsimreal/environment/reflectivity.py` (or pass
   precomputed `reflectivity` directly and skip the map).
4. **Project velocities onto the line-of-sight** with `radial_velocity()` if
   you want Doppler; otherwise leave it as zeros.
5. Feed the result through the same `sim.assign_reflectivity()` →
   `sim.simulate_image()` calls used by every other adapter — nothing else
   changes.

---

## 4. Common integration checklist (any simulator)

Run through this after wiring a new source, before trusting the output:

1. **Point count sanity check.** `len(points)` should be in the hundreds to
   low-thousands for a typical urban scene. Zero points means your ray-trace/
   scan produced nothing (check FOV, range, and that the radar isn't embedded
   in geometry).
2. **Range/azimuth sanity check.**
   ```python
   print(points.range.min(), points.range.max())
   print(points.azimuth_deg.min(), points.azimuth_deg.max())
   ```
   These should roughly match where you expect targets to be, and stay within
   `cfg.range_max` / `cfg.azimuth_min_deg..azimuth_max_deg` — points outside
   the grid are silently dropped by `rasterize()`.
3. **Material sanity check.** `set(points.material_id)` (or `np.unique(...)`)
   should show real material keys, not all `"unknown"` — if it does, your
   label-mapping dict is missing entries (Sections 1.6 / 2.6).
4. **Visual sanity check.** Plot `points.xyz[:,1]` vs `points.xyz[:,0]`
   (a bird's-eye view) before simulating — this catches coordinate/handedness
   bugs immediately, since object shapes will look obviously wrong (mirrored,
   rotated 90°, or collapsed to a line) if the frame conversion is off. See
   `examples/demo_standalone.py` or `examples/demo_simdaas.py` for the exact
   plotting pattern.
5. **Only then look at the radar image.** If steps 1-4 pass and the image
   still looks wrong, the issue is in `RadarConfig` (grid too small/large for
   your scene) or the PSF (try the default `synthetic_psf` first before a
   measured one, to isolate the problem).

---

## 5. Where to go next

- **Measured PSF** (recommended once geometry is validated): capture your
  target radar's tensor of a single pole/corner reflector and call
  `radsimreal.psf.psf.psf_from_tensor(...)`; pass it as `RadSimReal(psf=...)`.
  This is what makes RadSimReal reproduce a *specific* radar's behaviour rather
  than a generic approximation.
- **Dataset generation** for training a detector: loop your adapter over many
  frames and mirror `radsimreal/pipeline/dataset.py`'s output layout
  (`images/`, optional `tensors/`, `annotations.json`, `meta.json`).
- **`docs/USAGE.md`** for the broader API reference (radar config, noise
  model, reflectivity model) and **`docs/UNREAL_EXPORTER.md`** for the full
  Unreal-side exporter code referenced in Section 2.3.
