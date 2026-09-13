"""
RadSimReal command-line interface.

    python -m radsimreal.cli.main demo   --out demo.png
    python -m radsimreal.cli.main dataset --out data/ --frames 200 --tensors
    python -m radsimreal.cli.main simdaas --frame frame.npz --out sim_img.png
"""
from __future__ import annotations

import argparse
import numpy as np

from ..core.radar_config import RadarConfig
from ..pipeline.simulator import RadSimReal, NoiseModel
from ..pipeline.dataset import generate_synthetic_dataset
from ..core.image import to_db


def _radar(name):
    return {"raddet": RadarConfig.raddet, "small": RadarConfig.small}.get(
        name, RadarConfig.raddet
    )()


def cmd_demo(args):
    from examples.demo_standalone import main as demo_main  # type: ignore
    demo_main(out_path=args.out)


def cmd_dataset(args):
    generate_synthetic_dataset(
        out_dir=args.out, n_frames=args.frames, cfg=_radar(args.radar),
        snr_db=args.snr, save_tensors=args.tensors, seed=args.seed,
    )
    print("dataset written to", args.out)


def cmd_simdaas(args):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..adapters.simdaas_adapter import SimDaaSAdapter

    cfg = _radar(args.radar)
    adapter = SimDaaSAdapter(units=args.units)
    points = adapter.load_frame(args.frame)
    sim = RadSimReal(cfg=cfg)
    points = sim.assign_reflectivity(points)
    clean = sim.simulate_image(points)
    sim.noise = NoiseModel.from_snr_db(max(clean.image.max(), 1e-9), args.snr, seed=0)
    res = sim.simulate_image(points)

    extent = [cfg.azimuth_min_deg, cfg.azimuth_max_deg, cfg.range_max, cfg.range_min]
    plt.figure(figsize=(6, 5))
    plt.imshow(to_db(res.image, floor_db=-60), aspect="auto", extent=extent,
               cmap="jet", vmin=-60, vmax=0)
    plt.xlabel("Azimuth [deg]"); plt.ylabel("Range [m]"); plt.colorbar(label="[dB]")
    plt.title("SimDaaS -> RadSimReal"); plt.tight_layout(); plt.savefig(args.out, dpi=130)
    print("saved", args.out)


def main():
    p = argparse.ArgumentParser(prog="radsimreal")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo"); d.add_argument("--out", default="radsimreal_demo.png")
    d.set_defaults(func=cmd_demo)

    ds = sub.add_parser("dataset")
    ds.add_argument("--out", required=True); ds.add_argument("--frames", type=int, default=100)
    ds.add_argument("--radar", default="raddet"); ds.add_argument("--snr", type=float, default=30)
    ds.add_argument("--tensors", action="store_true"); ds.add_argument("--seed", type=int, default=0)
    ds.set_defaults(func=cmd_dataset)

    sd = sub.add_parser("simdaas")
    sd.add_argument("--frame", required=True); sd.add_argument("--out", default="simdaas_img.png")
    sd.add_argument("--radar", default="raddet"); sd.add_argument("--units", default="cm")
    sd.add_argument("--snr", type=float, default=30)
    sd.set_defaults(func=cmd_simdaas)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
