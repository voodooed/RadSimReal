# SimDaaS (Unreal) — reference exporter

This is a **reference sketch** for the Unreal side of the SimDaaS integration. It
is *not* run by the Python codebase; it documents what your engine must emit so
`SimDaaSAdapter` can consume it. Two options are shown: an Unreal **Python**
(remote-control / editor-python) exporter, and the shape of an equivalent C++
line-trace loop.

The contract (see `docs/USAGE.md` §C): per radar frame, write one `.npz`/`.json`
with `points_world` (N,3, UE cm), `radar_location`, `radar_rotation`, and
optionally `material`, `normals_world`, `velocities_world`, `reflectivity`.

---

## Option 1 — Unreal Python (spherical line-trace scan)

```python
# Runs inside Unreal's Python (unreal module available).
# Emits one RadSimReal SimDaaS frame per call.
import unreal
import numpy as np

def scan_radar_frame(radar_loc, radar_rot, world,
                     az_range=(-60, 60), az_step=0.3,
                     el_range=(-15, 15), el_step=0.5,
                     max_range_cm=6400.0):
    pts, mats, vels, norms = [], [], [], []
    origin = unreal.Vector(*radar_loc)

    for az in np.arange(az_range[0], az_range[1], az_step):
        for el in np.arange(el_range[0], el_range[1], el_step):
            # build a ray direction in the radar's local frame, then rotate
            rot = unreal.Rotator(el, az, 0.0)  # pitch, yaw, roll (deg)
            dir_local = rot.get_forward_vector()
            world_rot = unreal.Rotator(*radar_rot)
            direction = world_rot.rotate_vector(dir_local)
            end = origin + direction * max_range_cm

            hit = unreal.SystemLibrary.line_trace_single(
                world, origin, end,
                unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,
                False, [], unreal.DrawDebugTrace.NONE, True)
            if not hit:
                continue
            res = hit.hit_result  # unreal.HitResult
            ip = res.impact_point
            nrm = res.impact_normal
            comp = res.get_editor_property('component')
            actor = res.get_editor_property('hit_actor')

            pts.append([ip.x, ip.y, ip.z])
            norms.append([nrm.x, nrm.y, nrm.z])

            # material name from the hit component's physical material
            phys = None
            try:
                phys = comp.get_material(0).get_name()
            except Exception:
                pass
            mats.append(phys or "M_Unknown")

            # per-hit velocity from the actor (cm/s in UE world)
            v = actor.get_velocity() if actor else unreal.Vector(0, 0, 0)
            vels.append([v.x, v.y, v.z])

    return (np.array(pts, dtype=np.float32),
            np.array(mats),
            np.array(vels, dtype=np.float32),
            np.array(norms, dtype=np.float32))


def export_frame(path, radar_loc, radar_rot, world):
    P, M, V, N = scan_radar_frame(radar_loc, radar_rot, world)
    np.savez_compressed(
        path,
        points_world=P,
        radar_location=np.array(radar_loc, dtype=np.float32),
        radar_rotation=np.array(radar_rot, dtype=np.float32),
        material=M,
        velocities_world=V,
        normals_world=N,
    )
    # path is now consumable by:  SimDaaSAdapter(units="cm").load_frame(path)
```

Map the material names you emit (`M_Car`, `M_Pole`, ...) to reflectivity keys in
`radsimreal/adapters/simdaas_adapter.py::SIMDAAS_MATERIAL_MAP`.

---

## Option 2 — C++ line-trace loop (shape only)

```cpp
// Pseudocode for the equivalent C++ export inside a SimDaaS actor/subsystem.
TArray<FVector> Points, Normals, Velocities;
TArray<FString> Materials;

const FVector Origin = RadarComponent->GetComponentLocation();
const FRotator RadarRot = RadarComponent->GetComponentRotation();

for (float Az = AzMin; Az < AzMax; Az += AzStep)
for (float El = ElMin; El < ElMax; El += ElStep)
{
    FRotator RayLocal(El, Az, 0.f);
    FVector Dir = RadarRot.RotateVector(RayLocal.Vector());
    FVector End = Origin + Dir * MaxRangeCm;

    FHitResult Hit;
    FCollisionQueryParams Params(SCENE_QUERY_STAT(RadarScan), true);
    if (GetWorld()->LineTraceSingleByChannel(Hit, Origin, End, ECC_Visibility, Params))
    {
        Points.Add(Hit.ImpactPoint);
        Normals.Add(Hit.ImpactNormal);
        Materials.Add(Hit.PhysMaterial.IsValid()
                        ? Hit.PhysMaterial->GetName() : TEXT("M_Unknown"));
        Velocities.Add(Hit.GetActor() ? Hit.GetActor()->GetVelocity()
                                      : FVector::ZeroVector);
    }
}
// Serialise Points/Normals/Velocities/Materials + radar pose to .npz/.json
// (e.g. via a small JSON writer or a socket to the Python side).
```

---

## Performance notes

- Line-trace scanning is simple but O(rays). For dense scenes prefer a
  **GPU scene-capture** path (depth + world-normal + material-id render targets),
  read the buffers back once, and unproject to world points — this yields tens of
  thousands of points per frame cheaply.
- The Python side is fast regardless: rasterisation + **sparse** PSF convolution
  only touches occupied voxels, so cost scales with the number of reflection
  points, not the tensor volume.
- Measure your radar's PSF once (a pole/corner-reflector capture), cache it, and
  pass it via `RadSimReal(cfg=cfg, psf=measured_psf)` for realistic output.
