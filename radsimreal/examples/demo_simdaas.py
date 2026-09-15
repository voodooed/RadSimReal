"""
SimDaaS (Unreal) example -- no live UE needed.

Part 1 writes a synthetic "SimDaaS export" frame in the exact format your Unreal
side should emit (UE world frame, centimetres, left-handed). Part 2 loads it
through the SimDaaS adapter and runs RadSimReal.

    python examples/demo_simdaas.py

To wire up the REAL SimDaaS: on the Unreal side, for each radar frame, collect
surface hit points (line traces / a scene-capture depth+normal readback / a
Niagara point export), tag each with its material, and write one .npz per frame
using the same keys shown in `write_simdaas_frame`. Then point this script (or
`python -m radsimreal.cli.main simdaas --frame yourframe.npz`) at it.
"""
import os
import tempfile
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import radsimreal as rsr
from radsimreal.adapters.simdaas_adapter import write_simdaas_frame, SimDaaSAdapter
from radsimreal.core.image import to_db


def make_fake_ue_frame(path, seed=0):
    """Emit a plausible UE-world-frame export (cm, left-handed, y right)."""
    rng = np.random.default_rng(seed)
    parts_p, parts_m, parts_v = [], [], []

    # car ahead-right, approaching (UE +x forward, +y right)
    car = (rng.random((600, 3)) - 0.5) * np.array([450, 180, 150]) + np.array([2000, 500, 60])
    parts_p.append(car); parts_m += ["M_Car"] * 600
    parts_v.append(np.tile([-500, 0, 0], (600, 1)))

    # pole ahead-left
    z = np.linspace(0, 300, 120)
    pole = np.stack([np.full_like(z, 1200), np.full_like(z, -200), z], axis=1)
    parts_p.append(pole); parts_m += ["M_Pole"] * 120
    parts_v.append(np.zeros((120, 3)))

    # building facade far ahead
    y = (rng.random(700) - 0.5) * 3000 + 0
    zb = rng.random(700) * 300
    wall = np.stack([np.full(700, 3800), y, zb], axis=1)
    parts_p.append(wall); parts_m += ["M_Building"] * 700
    parts_v.append(np.zeros((700, 3)))

    P = np.concatenate(parts_p, 0)
    M = np.array(parts_m)
    V = np.concatenate(parts_v, 0)
    write_simdaas_frame(path, points_world=P, radar_location=(0, 0, 60),
                        radar_rotation=(0, 0, 0), material=M, velocities_world=V)


def main(out="simdaas_radsimreal.png"):
    tmp = tempfile.mkdtemp()
    frame = os.path.join(tmp, "frame.npz")
    make_fake_ue_frame(frame)

    adapter = SimDaaSAdapter(units="cm")
    points = adapter.load_frame(frame)

    cfg = rsr.RadarConfig.raddet()
    sim = rsr.RadSimReal(cfg=cfg)
    points = sim.assign_reflectivity(points)
    clean = sim.simulate_image(points)
    sim.noise = rsr.NoiseModel.from_snr_db(max(clean.image.max(), 1e-9), 32, 0)
    res = sim.simulate_image(points)

    extent = [cfg.azimuth_min_deg, cfg.azimuth_max_deg, cfg.range_max, cfg.range_min]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].scatter(points.xyz[:, 1], points.xyz[:, 0], s=2,
                  c=np.log10(points.reflectivity + 1e-9), cmap="viridis")
    ax[0].set_aspect("equal"); ax[0].set_title("SimDaaS reflection points (radar frame)")
    ax[0].set_xlabel("y (left) [m]"); ax[0].set_ylabel("x (forward) [m]"); ax[0].grid(alpha=0.3)
    im = ax[1].imshow(to_db(res.image, floor_db=-60), aspect="auto", extent=extent,
                      cmap="jet", vmin=-60, vmax=0)
    ax[1].set_xlabel("Azimuth [deg]"); ax[1].set_ylabel("Range [m]")
    ax[1].set_title("SimDaaS -> RadSimReal image (dB)")
    fig.colorbar(im, ax=ax[1], label="[dB]")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
