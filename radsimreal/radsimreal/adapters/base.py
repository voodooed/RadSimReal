"""
Environment-source adapter interface.

Every simulator backend (CARLA, SimDaaS/Unreal, or a custom one) must convert
its native geometry into `ReflectionPoints` in the RADAR frame. That is the
ONLY contract RadSimReal needs -- the PSF-convolution core is identical
regardless of source.

Radar frame convention (right-handed):
    x forward (boresight), y left, z up, origin at the radar.

An adapter is responsible for:
    1. obtaining 3D reflection points that lie on object surfaces
       (ray-cast hits, semantic-lidar returns, or a dense mesh sample),
    2. transforming them into the radar frame,
    3. attaching a material label (for RF reflectivity) and, if available,
       a radial velocity (for the Doppler axis) and a surface normal.

Adapters SHOULD NOT compute reflectivity themselves; they hand material labels
to RadSimReal.assign_reflectivity(), keeping the physics in one place.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from ..core.reflection_points import ReflectionPoints


class EnvironmentAdapter(ABC):
    @abstractmethod
    def get_reflection_points(self, *args, **kwargs) -> ReflectionPoints:
        """Return reflection points in the radar frame for one frame/tick."""
        raise NotImplementedError


# ---- shared geometry helpers used by concrete adapters ----

def transform_world_to_radar(
    points_world: np.ndarray,
    radar_position: np.ndarray,
    radar_rotation_matrix: np.ndarray,
) -> np.ndarray:
    """
    points_world : (N,3) in the simulator's world frame.
    radar_position : (3,) radar origin in world frame.
    radar_rotation_matrix : (3,3) world->radar rotation (rows are radar axes).
    """
    p = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    rel = p - np.asarray(radar_position, dtype=np.float64).reshape(1, 3)
    return rel @ np.asarray(radar_rotation_matrix, dtype=np.float64).T


def radial_velocity(
    points_radar: np.ndarray,
    point_velocities_radar: np.ndarray,
) -> np.ndarray:
    """
    Project per-point velocity onto the radar line-of-sight.
    Positive = approaching the radar.
    """
    p = np.asarray(points_radar, dtype=np.float64).reshape(-1, 3)
    v = np.asarray(point_velocities_radar, dtype=np.float64).reshape(-1, 3)
    r = np.linalg.norm(p, axis=1, keepdims=True)
    los = p / np.maximum(r, 1e-9)
    return -np.sum(v * los, axis=1)   # minus: closing range -> positive doppler


def euler_to_rotation(roll_deg, pitch_deg, yaw_deg) -> np.ndarray:
    """Build a world->body rotation matrix from ZYX Euler angles (degrees)."""
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    # body axes expressed in world = Rz@Ry@Rx ; world->body is its transpose
    R_body_from_world = (Rz @ Ry @ Rx).T
    return R_body_from_world
