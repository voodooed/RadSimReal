"""
Standalone synthetic scene generator (NO external simulator required).

Produces ReflectionPoints + ground-truth 2D bounding boxes so the whole
RadSimReal pipeline can be exercised on any machine. This stands in for the
"Scene Generation + Ray Tracing" block of Fig. 2a when neither CARLA nor
SimDaaS is available.

Scenes are composed of simple primitives that emit dense surface reflection
points:
    - Car        : a box shell (metal, high reflectivity, moving)
    - Pole       : a thin vertical line (strong point scatterer)
    - Wall       : a planar strip (building facade)
    - Pedestrian : a small low-RCS blob

Each object also yields a ground-truth range-azimuth bounding box, matching the
2D box annotations used by RADDet/CARRADA in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..core.reflection_points import ReflectionPoints


@dataclass
class GTBox:
    """Ground-truth object annotation in range-azimuth space."""
    cls: str
    r_min: float
    r_max: float
    az_min_deg: float
    az_max_deg: float

    def to_dict(self):
        return {
            "class": self.cls,
            "r_min": self.r_min, "r_max": self.r_max,
            "az_min_deg": self.az_min_deg, "az_max_deg": self.az_max_deg,
        }


def _surface_points_box(center, size, n, rng):
    """Dense points on the 6 faces of an axis-aligned box (object frame)."""
    cx, cy, cz = center
    sx, sy, sz = size
    pts = []
    per_face = max(n // 6, 1)
    faces = [
        ((sx/2, 0, 0), (0, sy, sz)), ((-sx/2, 0, 0), (0, sy, sz)),
        ((0, sy/2, 0), (sx, 0, sz)), ((0, -sy/2, 0), (sx, 0, sz)),
        ((0, 0, sz/2), (sx, sy, 0)), ((0, 0, -sz/2), (sx, sy, 0)),
    ]
    for off, ext in faces:
        u = (rng.random(per_face) - 0.5) * (ext[0] if ext[0] else 0)
        v = (rng.random(per_face) - 0.5) * (ext[1] if ext[1] else 0)
        w = (rng.random(per_face) - 0.5) * (ext[2] if ext[2] else 0)
        fp = np.stack([
            np.full(per_face, off[0]) + (u if ext[0] else 0),
            np.full(per_face, off[1]) + (u if not ext[0] and ext[1] else v),
            np.full(per_face, off[2]) + (w if ext[2] else 0),
        ], axis=1)
        pts.append(fp)
    P = np.concatenate(pts, axis=0)
    return P + np.array([cx, cy, cz])


@dataclass
class SceneObject:
    cls: str
    center: tuple            # (x, y, z) in radar frame (metres)
    size: tuple              # (sx, sy, sz)
    velocity: float = 0.0    # radial m/s
    n_points: int = 300
    material: str = "unknown"

    def sample(self, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray, GTBox]:
        if self.cls == "pole":
            # thin vertical line
            z = np.linspace(-self.size[2] / 2, self.size[2] / 2, self.n_points)
            P = np.stack([
                np.full_like(z, self.center[0]),
                np.full_like(z, self.center[1]),
                z + self.center[2],
            ], axis=1)
        elif self.cls == "wall":
            # planar strip in x-z, extended along y
            y = (rng.random(self.n_points) - 0.5) * self.size[1] + self.center[1]
            z = (rng.random(self.n_points) - 0.5) * self.size[2] + self.center[2]
            P = np.stack([
                np.full(self.n_points, self.center[0]), y, z], axis=1)
        else:
            P = _surface_points_box(self.center, self.size, self.n_points, rng)

        mat = np.array([self.material] * P.shape[0], dtype=object)
        vel = np.full(P.shape[0], self.velocity)

        # ground-truth range-azimuth box from the point extent
        rng_ = np.linalg.norm(P, axis=1)
        az = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
        box = GTBox(self.cls, rng_.min(), rng_.max(), az.min(), az.max())
        return P, mat, vel, box


@dataclass
class SyntheticScene:
    objects: list = field(default_factory=list)
    seed: int = 0

    def add(self, obj: SceneObject):
        self.objects.append(obj)
        return self

    def build(self) -> tuple[ReflectionPoints, list[GTBox]]:
        rng = np.random.default_rng(self.seed)
        allP, allM, allV, boxes = [], [], [], []
        for obj in self.objects:
            P, M, V, box = obj.sample(rng)
            allP.append(P); allM.append(M); allV.append(V); boxes.append(box)
        if not allP:
            return ReflectionPoints(np.zeros((0, 3)), np.zeros((0,))), []
        P = np.concatenate(allP, axis=0)
        M = np.concatenate(allM, axis=0)
        V = np.concatenate(allV, axis=0)
        # reflectivity is assigned later by the ReflectivityModel; seed with ones
        pts = ReflectionPoints(
            xyz=P, reflectivity=np.ones(P.shape[0]), velocity=V, material_id=M
        )
        return pts, boxes


def default_scene(seed: int = 0) -> SyntheticScene:
    """A representative urban snippet: two cars, a pole, a wall, a pedestrian."""
    s = SyntheticScene(seed=seed)
    s.add(SceneObject("vehicle", center=(18, -4, 0.5), size=(4.5, 1.8, 1.5),
                      velocity=-6.0, n_points=600, material="vehicle"))
    s.add(SceneObject("vehicle", center=(28, 6, 0.5), size=(4.5, 1.8, 1.5),
                      velocity=4.0, n_points=600, material="vehicle"))
    s.add(SceneObject("pole", center=(12, 2, 1.5), size=(0.1, 0.1, 3.0),
                      velocity=0.0, n_points=120, material="pole"))
    s.add(SceneObject("wall", center=(35, 0, 1.5), size=(0.3, 30.0, 3.0),
                      velocity=0.0, n_points=800, material="building"))
    s.add(SceneObject("pedestrian", center=(15, -1, 0.9), size=(0.5, 0.5, 1.7),
                      velocity=-1.2, n_points=150, material="pedestrian"))
    return s
