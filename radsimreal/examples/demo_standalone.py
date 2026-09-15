"""
Standalone demo: run the full RadSimReal pipeline with NO external simulator
and save a figure (reflection points -> tensor -> range-azimuth image).

    python examples/demo_standalone.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import radsimreal as rsr
from radsimreal.core.image import to_db


def main(out_path="radsimreal_demo.png", seed=1):
    cfg = rsr.RadarConfig.raddet()

    # 1. environment sim (standalone synthetic scene)
    scene = rsr.default_scene(seed=seed)
    points, boxes = scene.build()

    # 2. build simulator (synthetic PSF, sparse convolution)
    sim = rsr.RadSimReal(cfg=cfg, convolution="sparse")

    # 3. assign physical RF reflectivity (material + geometry + range)
    points = sim.assign_reflectivity(points)

    # 4. noise from a target SNR (measured-variance path also available)
    clean = sim.simulate_image(points)
    sim.noise = rsr.NoiseModel.from_snr_db(clean.image.max(), snr_db=35, seed=seed)

    # 5. simulate
    res = sim.simulate_image(points)

    img_db = to_db(res.image, floor_db=-60.0)

    # ---- plot ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    # reflection points (bird's-eye)
    ax[0].scatter(points.xyz[:, 1], points.xyz[:, 0], s=2,
                  c=np.log10(points.reflectivity + 1e-9), cmap="viridis")
    ax[0].set_xlabel("y (left) [m]"); ax[0].set_ylabel("x (forward) [m]")
    ax[0].set_title("Reflection points (log reflectivity)")
    ax[0].set_aspect("equal"); ax[0].grid(alpha=0.3)

    # radar image (range-azimuth)
    extent = [cfg.azimuth_min_deg, cfg.azimuth_max_deg, cfg.range_max, cfg.range_min]
    im = ax[1].imshow(img_db, aspect="auto", extent=extent, cmap="jet",
                      vmin=-60, vmax=0)
    for b in boxes:
        ax[1].add_patch(plt.Rectangle(
            (b.az_min_deg, b.r_min), b.az_max_deg - b.az_min_deg,
            b.r_max - b.r_min, fill=False, edgecolor="white", lw=1.5))
    ax[1].set_xlabel("Azimuth [deg]"); ax[1].set_ylabel("Range [m]")
    ax[1].set_title("RadSimReal image (dB) + GT boxes")
    fig.colorbar(im, ax=ax[1], label="[dB]")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print("saved", out_path)
    print("tensor", res.tensor.shape, "image", res.image.shape,
          "psf(truncated)", sim.psf.shape)


if __name__ == "__main__":
    main()
