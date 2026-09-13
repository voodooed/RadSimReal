"""
Reflection-point container.

A "reflection point" in RadSimReal is the output of the *environment
simulation* stage (Fig. 2a): a 3D point that reflected an RF ray back to the
radar, tagged with its RF reflectivity (radar cross-section contribution) and,
optionally, a radial velocity for the Doppler axis.

The environment sim can come from three sources, all producing this same
struct:
  - Standalone synthetic scene generator (no external simulator)   -> environment.synthetic_scene
  - CARLA ray-cast / semantic lidar                                 -> adapters.carla_adapter
  - SimDaaS (Unreal) export                                         -> adapters.simdaas_adapter

Coordinates are in the RADAR frame:
  x forward (range direction of boresight), y left, z up  (right-handed).
We store cartesian xyz and derive (range, azimuth, doppler) on demand so any
adapter only has to fill xyz + reflectivity + velocity.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ReflectionPoints:
    xyz: np.ndarray                 # (N, 3) metres, radar frame
    reflectivity: np.ndarray        # (N,) linear RF reflectivity (>= 0)
    velocity: np.ndarray = None     # (N,) radial velocity m/s (+ = approaching). None -> zeros
    material_id: np.ndarray = None  # (N,) optional int material label (provenance only)

    def __post_init__(self):
        self.xyz = np.asarray(self.xyz, dtype=np.float64).reshape(-1, 3)
        n = self.xyz.shape[0]
        self.reflectivity = np.asarray(self.reflectivity, dtype=np.float64).reshape(n)
        if self.velocity is None:
            self.velocity = np.zeros(n, dtype=np.float64)
        else:
            self.velocity = np.asarray(self.velocity, dtype=np.float64).reshape(n)
        if self.material_id is not None:
            self.material_id = np.asarray(self.material_id).reshape(n)

    def __len__(self):
        return self.xyz.shape[0]

    # ---- polar coordinates in the radar frame ----
    @property
    def range(self) -> np.ndarray:
        return np.linalg.norm(self.xyz, axis=1)

    @property
    def azimuth_deg(self) -> np.ndarray:
        # azimuth measured from +x (boresight), positive toward +y (left)
        return np.degrees(np.arctan2(self.xyz[:, 1], self.xyz[:, 0]))

    @property
    def elevation_deg(self) -> np.ndarray:
        xy = np.linalg.norm(self.xyz[:, :2], axis=1)
        return np.degrees(np.arctan2(self.xyz[:, 2], xy))

    # ---- editing helpers ----
    def filter(self, mask: np.ndarray) -> "ReflectionPoints":
        return ReflectionPoints(
            xyz=self.xyz[mask],
            reflectivity=self.reflectivity[mask],
            velocity=self.velocity[mask],
            material_id=None if self.material_id is None else self.material_id[mask],
        )

    @staticmethod
    def concat(items: list["ReflectionPoints"]) -> "ReflectionPoints":
        items = [it for it in items if len(it) > 0]
        if not items:
            return ReflectionPoints(np.zeros((0, 3)), np.zeros((0,)))
        has_mat = all(it.material_id is not None for it in items)
        return ReflectionPoints(
            xyz=np.concatenate([it.xyz for it in items], axis=0),
            reflectivity=np.concatenate([it.reflectivity for it in items]),
            velocity=np.concatenate([it.velocity for it in items]),
            material_id=(
                np.concatenate([it.material_id for it in items]) if has_mat else None
            ),
        )
