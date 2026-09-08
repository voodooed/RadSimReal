"""
Dataset generation.

Runs the RadSimReal pipeline over many frames and writes a training-ready
dataset: per-frame radar image (and optionally the full tensor) plus 2D
range-azimuth bounding-box annotations -- the ingredients the paper feeds to
U-Net / RADDet / Probabilistic detectors.

Sources:
    - "synthetic"  : randomised standalone scenes (no external simulator)
    - a frame directory of SimDaaS .npz/.json files
    - a directory of saved CARLA semantic-lidar arrays

Output layout:
    out/
      images/000000.npy ...        (R, A) float radar images
      tensors/000000.npy ...       (R, A, D) optional
      annotations.json             list of {frame, boxes:[{class,r_min,...}]}
      meta.json                    radar config + run settings
"""
from __future__ import annotations

import json
import os
import numpy as np

from ..core.radar_config import RadarConfig
from ..pipeline.simulator import RadSimReal, NoiseModel
from ..environment.synthetic_scene import SceneObject, SyntheticScene


def _random_scene(rng: np.random.Generator, seed: int) -> SyntheticScene:
    s = SyntheticScene(seed=seed)
    n_cars = rng.integers(1, 4)
    for _ in range(int(n_cars)):
        x = rng.uniform(8, 45); y = rng.uniform(-12, 12)
        s.add(SceneObject("vehicle", center=(x, y, 0.5), size=(4.5, 1.8, 1.5),
                          velocity=rng.uniform(-8, 8), n_points=500, material="vehicle"))
    for _ in range(int(rng.integers(0, 3))):
        x = rng.uniform(6, 40); y = rng.uniform(-14, 14)
        s.add(SceneObject("pole", center=(x, y, 1.5), size=(0.1, 0.1, 3.0),
                          n_points=100, material="pole"))
    if rng.random() < 0.7:
        s.add(SceneObject("wall", center=(rng.uniform(30, 48), 0, 1.5),
                          size=(0.3, 30.0, 3.0), n_points=700, material="building"))
    for _ in range(int(rng.integers(0, 2))):
        x = rng.uniform(8, 25); y = rng.uniform(-8, 8)
        s.add(SceneObject("pedestrian", center=(x, y, 0.9), size=(0.5, 0.5, 1.7),
                          velocity=rng.uniform(-2, 2), n_points=120, material="pedestrian"))
    return s


def generate_synthetic_dataset(
    out_dir: str,
    n_frames: int = 100,
    cfg: RadarConfig = None,
    snr_db: float = 30.0,
    save_tensors: bool = False,
    seed: int = 0,
):
    cfg = cfg or RadarConfig.raddet()
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    if save_tensors:
        os.makedirs(os.path.join(out_dir, "tensors"), exist_ok=True)

    sim = RadSimReal(cfg=cfg, convolution="sparse")
    rng = np.random.default_rng(seed)
    ann = []

    for i in range(n_frames):
        scene = _random_scene(rng, seed=seed + i)
        points, boxes = scene.build()
        points = sim.assign_reflectivity(points)

        clean = sim.simulate_image(points)
        sim.noise = NoiseModel.from_snr_db(
            max(clean.image.max(), 1e-9), snr_db=snr_db, seed=seed + i
        )
        res = sim.simulate_image(points)

        fid = f"{i:06d}"
        np.save(os.path.join(out_dir, "images", fid + ".npy"), res.image.astype(np.float32))
        if save_tensors:
            np.save(os.path.join(out_dir, "tensors", fid + ".npy"), res.tensor.astype(np.float32))
        ann.append({"frame": fid, "boxes": [b.to_dict() for b in boxes]})

    with open(os.path.join(out_dir, "annotations.json"), "w") as f:
        json.dump(ann, f, indent=2)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({
            "radar": cfg.name,
            "tensor_shape": list(cfg.tensor_shape),
            "range_max": cfg.range_max,
            "azimuth_range_deg": [cfg.azimuth_min_deg, cfg.azimuth_max_deg],
            "snr_db": snr_db, "n_frames": n_frames,
        }, f, indent=2)
    return out_dir
