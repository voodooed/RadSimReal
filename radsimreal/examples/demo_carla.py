"""
CARLA online example (requires a running CARLA server + the `carla` package).

Spawns a semantic-lidar at a chosen radar pose, ticks the world, converts the
returns into reflection points, and runs RadSimReal to produce a radar image.

    python examples/demo_carla.py --host localhost --port 2000

Notes
-----
* Use a CARLA version whose Python API matches your server.
* Run in synchronous mode for reproducible per-tick frames.
* Attach to the ego vehicle (attach_to=ego) if you want a moving radar; the
  adapter subtracts ego velocity so Doppler is relative to the radar.
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--out", default="carla_radsimreal.png")
    ap.add_argument("--ticks", type=int, default=5)
    args = ap.parse_args()

    import carla
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import radsimreal as rsr
    from radsimreal.adapters.carla_adapter import CarlaRadarAdapter
    from radsimreal.core.image import to_db

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    # synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # radar pose: 0.5 m above ground, at world origin (edit as needed)
    radar_tf = carla.Transform(carla.Location(x=0, y=0, z=0.5))
    adapter = CarlaRadarAdapter(world, transform=radar_tf, lidar_range=80.0).spawn()

    cfg = rsr.RadarConfig.raddet()
    sim = rsr.RadSimReal(cfg=cfg)

    res = None
    try:
        for _ in range(args.ticks):
            world.tick()
        points = adapter.get_reflection_points()
        points = sim.assign_reflectivity(points)
        clean = sim.simulate_image(points)
        sim.noise = rsr.NoiseModel.from_snr_db(max(clean.image.max(), 1e-9), 30, 0)
        res = sim.simulate_image(points)
    finally:
        adapter.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)

    extent = [cfg.azimuth_min_deg, cfg.azimuth_max_deg, cfg.range_max, cfg.range_min]
    plt.figure(figsize=(6, 5))
    plt.imshow(to_db(res.image, floor_db=-60), aspect="auto", extent=extent,
               cmap="jet", vmin=-60, vmax=0)
    plt.xlabel("Azimuth [deg]"); plt.ylabel("Range [m]"); plt.colorbar(label="[dB]")
    plt.title("CARLA -> RadSimReal"); plt.tight_layout(); plt.savefig(args.out, dpi=130)
    print("saved", args.out)


if __name__ == "__main__":
    main()
