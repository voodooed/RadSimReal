"""
SimDaaS (Unreal Engine) adapter.

SimDaaS is your in-house Unreal-based simulator. RadSimReal only needs
reflection points in the radar frame, so this adapter defines a **simple,
explicit data contract** that your Unreal side must fill, plus loaders for the
two most convenient transport formats:

    1. In-process numpy arrays  (if you drive RadSimReal from Python that already
       holds the UE data, e.g. via an embedded Python or a socket bridge).
    2. On-disk frames           (UE writes .npz / .json per tick; Python reads).

DATA CONTRACT  (per radar frame)
--------------------------------
Required:
    points_world : (N,3) float   surface hit points, UE WORLD frame, centimetres
                                  OR metres (set `units`).
    radar_pose   : dict with
                     location : (3,) float  radar origin in UE world frame
                     rotation : (3,) float  (roll, pitch, yaw) degrees, UE convention
Optional (improve fidelity):
    material     : (N,) int or str   per-point material label/key
                                     (mapped by ReflectivityModel / SIMDAAS_MATERIAL_MAP)
    normals_world: (N,3) float       per-point surface normal, UE world frame
    velocities_world : (N,3) float   per-point velocity, UE world frame (m/s)
    reflectivity : (N,) float        precomputed linear reflectivity (overrides material)

COORDINATE HANDLING
-------------------
Unreal is LEFT-HANDED, X forward, Y right, Z up, and defaults to centimetres.
RadSimReal's radar frame is RIGHT-HANDED, x forward, y LEFT, z up, metres.
This adapter converts UE -> radar by:  y -> -y, and cm -> m if units='cm'.
It then transforms world -> radar using the radar pose.

You can override any of this if your SimDaaS export already emits radar-frame,
right-handed, metric points (set `already_radar_frame=True`).
"""
from __future__ import annotations

import json
import os
import numpy as np

from .base import (
    EnvironmentAdapter,
    transform_world_to_radar,
    radial_velocity,
    euler_to_rotation,
)
from ..core.reflection_points import ReflectionPoints


# Map SimDaaS material names to the RF reflectivity table keys.
# Extend this to match the physical materials you tag in Unreal.
SIMDAAS_MATERIAL_MAP = {
    "M_Car": "vehicle", "M_Metal": "metal", "M_Vehicle": "vehicle",
    "M_Pole": "pole", "M_Sign": "sign", "M_Guardrail": "guardrail",
    "M_Building": "building", "M_Concrete": "concrete", "M_Wall": "wall",
    "M_Road": "road", "M_Asphalt": "road", "M_Sidewalk": "sidewalk",
    "M_Ground": "ground", "M_Foliage": "vegetation", "M_Tree": "vegetation",
    "M_Pedestrian": "pedestrian", "M_Glass": "glass", "M_Water": "water",
}


def _ue_to_radarframe_units(arr_xyz, units):
    scale = 0.01 if units == "cm" else 1.0
    a = np.asarray(arr_xyz, dtype=np.float64).reshape(-1, 3) * scale
    # left-handed (y right) -> right-handed (y left)
    a[:, 1] = -a[:, 1]
    return a


class SimDaaSAdapter(EnvironmentAdapter):
    def __init__(self, units: str = "cm", already_radar_frame: bool = False):
        """
        units : 'cm' (UE default) or 'm'.
        already_radar_frame : if True, skip UE->radar conversion and world->radar
                              transform (points are already in the radar frame).
        """
        assert units in ("cm", "m")
        self.units = units
        self.already_radar_frame = already_radar_frame

    # ---- main conversion ----
    def get_reflection_points(
        self,
        points_world: np.ndarray,
        radar_pose: dict = None,
        material=None,
        normals_world: np.ndarray = None,
        velocities_world: np.ndarray = None,
        reflectivity: np.ndarray = None,
    ) -> ReflectionPoints:
        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)

        if self.already_radar_frame:
            xyz = pts.copy()
            normals_r = None if normals_world is None else np.asarray(normals_world).reshape(-1, 3)
            vel_r = None if velocities_world is None else np.asarray(velocities_world).reshape(-1, 3)
        else:
            # UE world -> radar-handed metric world
            xyz_w = _ue_to_radarframe_units(pts, self.units)

            # radar pose
            if radar_pose is None:
                loc = np.zeros(3); R = np.eye(3)
            else:
                loc = _ue_to_radarframe_units(
                    np.asarray(radar_pose["location"]).reshape(1, 3), self.units
                )[0]
                roll, pitch, yaw = radar_pose.get("rotation", (0, 0, 0))
                # UE yaw is left-handed about Z; y-flip already applied, so negate yaw & roll
                R = euler_to_rotation(-roll, pitch, -yaw)

            xyz = transform_world_to_radar(xyz_w, loc, R)

            normals_r = None
            if normals_world is not None:
                nw = _ue_to_radarframe_units(normals_world, "m")  # normals are directions
                # normals were unit; y already flipped; only rotate (no translation)
                normals_r = nw @ R.T
                normals_r /= np.maximum(np.linalg.norm(normals_r, axis=1, keepdims=True), 1e-9)

            vel_r = None
            if velocities_world is not None:
                vw = np.asarray(velocities_world, dtype=np.float64).reshape(-1, 3).copy()
                # velocities share the same length units as points (cm or m)
                if self.units == "cm":
                    vw *= 0.01
                vw[:, 1] = -vw[:, 1]           # handedness
                vel_r = vw @ R.T

        # doppler
        if vel_r is not None:
            doppler = radial_velocity(xyz, vel_r)
        else:
            doppler = np.zeros(xyz.shape[0])

        # materials
        material_id = None
        if reflectivity is None and material is not None:
            material_id = self._map_materials(material)

        refl_seed = (
            np.ones(xyz.shape[0]) if reflectivity is None
            else np.asarray(reflectivity, dtype=np.float64).reshape(-1)
        )

        pts_out = ReflectionPoints(
            xyz=xyz, reflectivity=refl_seed, velocity=doppler, material_id=material_id
        )
        # stash normals for reflectivity assignment (not part of the struct)
        pts_out._normals = normals_r
        return pts_out

    def _map_materials(self, material):
        material = np.asarray(material).reshape(-1)
        if material.dtype.kind in ("U", "S", "O"):
            return np.array(
                [SIMDAAS_MATERIAL_MAP.get(str(m), "unknown") for m in material],
                dtype=object,
            )
        return material  # assume already-mapped ints

    # ---- on-disk frame loader ----
    def load_frame(self, path: str) -> ReflectionPoints:
        """
        Load one SimDaaS frame from .npz or .json.

        .npz keys : points_world (req), and optionally material, normals_world,
                    velocities_world, reflectivity, plus radar_location,
                    radar_rotation.
        .json     : same keys (arrays as nested lists).
        """
        if path.endswith(".npz"):
            d = np.load(path, allow_pickle=True)
            get = lambda k: d[k] if k in d.files else None
        elif path.endswith(".json"):
            with open(path) as f:
                raw = json.load(f)
            get = lambda k: (np.array(raw[k]) if k in raw else None)
        else:
            raise ValueError("SimDaaS frame must be .npz or .json")

        radar_pose = None
        loc = get("radar_location"); rot = get("radar_rotation")
        if loc is not None:
            radar_pose = {
                "location": np.asarray(loc).reshape(3),
                "rotation": tuple(np.asarray(rot).reshape(3)) if rot is not None else (0, 0, 0),
            }
        return self.get_reflection_points(
            points_world=get("points_world"),
            radar_pose=radar_pose,
            material=get("material"),
            normals_world=get("normals_world"),
            velocities_world=get("velocities_world"),
            reflectivity=get("reflectivity"),
        )


def write_simdaas_frame(
    path: str,
    points_world,
    radar_location=(0, 0, 0),
    radar_rotation=(0, 0, 0),
    material=None,
    normals_world=None,
    velocities_world=None,
    reflectivity=None,
):
    """Reference writer (what the Unreal/SimDaaS side should emit per frame)."""
    d = {"points_world": np.asarray(points_world),
         "radar_location": np.asarray(radar_location),
         "radar_rotation": np.asarray(radar_rotation)}
    if material is not None: d["material"] = np.asarray(material)
    if normals_world is not None: d["normals_world"] = np.asarray(normals_world)
    if velocities_world is not None: d["velocities_world"] = np.asarray(velocities_world)
    if reflectivity is not None: d["reflectivity"] = np.asarray(reflectivity)
    np.savez_compressed(path, **d)
