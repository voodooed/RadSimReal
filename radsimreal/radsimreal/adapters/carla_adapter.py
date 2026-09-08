"""
CARLA adapter.

Converts CARLA sensor output into RadSimReal `ReflectionPoints`. Following the
paper (Fig. 2a uses CARLA for scene generation + ray tracing), we obtain dense
surface reflection points from CARLA's **semantic LiDAR** sensor, which returns,
per hit: 3D position, the object's semantic tag, the surface normal (cosine),
and the actor id. That gives us everything the RF-reflectivity model needs
(material via tag, orientation via normal) plus enough to compute Doppler from
actor velocities.

This module is written so it can be imported and inspected WITHOUT the `carla`
package installed. The actual CARLA calls only execute inside methods, so unit
tests / dry runs work anywhere. Install CARLA (matching your server version) to
run it live.

Typical use
-----------
    import carla
    from radsimreal import RadSimReal, RadarConfig
    from radsimreal.adapters.carla_adapter import CarlaRadarAdapter

    client = carla.Client("localhost", 2000); world = client.get_world()
    adapter = CarlaRadarAdapter(world, transform=radar_transform)
    adapter.spawn()                       # spawns a semantic-lidar at the radar pose
    points = adapter.get_reflection_points()   # one frame
    sim = RadSimReal(cfg=RadarConfig.raddet())
    points = sim.assign_reflectivity(points)
    res = sim.simulate_image(points)
"""
from __future__ import annotations

import numpy as np

from .base import EnvironmentAdapter, radial_velocity
from ..core.reflection_points import ReflectionPoints
from ..environment.reflectivity import CARLA_TAG_TO_MATERIAL


class CarlaRadarAdapter(EnvironmentAdapter):
    def __init__(
        self,
        world,
        transform=None,
        lidar_range: float = 100.0,
        rotation_frequency: float = 20.0,
        points_per_second: int = 2_000_000,
        upper_fov: float = 15.0,
        lower_fov: float = -25.0,
        channels: int = 64,
        attach_to=None,
    ):
        """
        world      : carla.World
        transform  : carla.Transform for the radar/lidar pose (defaults to origin)
        attach_to  : optional carla.Actor (e.g. ego vehicle) to attach the sensor
        """
        self.world = world
        self.transform = transform
        self.attach_to = attach_to
        self.lidar_range = lidar_range
        self.sensor = None
        self._latest = None
        self._blueprint_attrs = dict(
            range=str(lidar_range),
            rotation_frequency=str(rotation_frequency),
            points_per_second=str(points_per_second),
            upper_fov=str(upper_fov),
            lower_fov=str(lower_fov),
            channels=str(channels),
        )

    # ---- lifecycle ----
    def spawn(self):
        import carla  # local import so module loads without carla installed

        bp_lib = self.world.get_blueprint_library()
        bp = bp_lib.find("sensor.lidar.ray_cast_semantic")
        for k, v in self._blueprint_attrs.items():
            if bp.has_attribute(k):
                bp.set_attribute(k, v)
        tf = self.transform or carla.Transform()
        self.sensor = self.world.spawn_actor(bp, tf, attach_to=self.attach_to)
        self.sensor.listen(self._on_data)
        return self

    def _on_data(self, data):
        # semantic lidar: structured array with x,y,z, cos_inc_angle, object_idx, object_tag
        arr = np.frombuffer(data.raw_data, dtype=np.dtype([
            ("x", np.float32), ("y", np.float32), ("z", np.float32),
            ("cos", np.float32), ("idx", np.uint32), ("tag", np.uint32),
        ]))
        self._latest = arr

    def destroy(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None

    # ---- conversion ----
    def _actor_velocity_map(self):
        """actor_id -> velocity vector in world frame (m/s)."""
        vmap = {}
        for actor in self.world.get_actors():
            try:
                v = actor.get_velocity()
                vmap[actor.id] = np.array([v.x, v.y, v.z], dtype=np.float64)
            except Exception:
                pass
        return vmap

    def get_reflection_points(self, compute_doppler: bool = True) -> ReflectionPoints:
        """
        Convert the latest semantic-lidar frame to reflection points in the
        RADAR frame. The semantic lidar already returns points in the SENSOR
        frame (x forward, y right, z up in UE/CARLA left-handed coords), so we
        flip y to get our right-handed radar frame (x forward, y left, z up).
        """
        if self._latest is None:
            raise RuntimeError("No CARLA data yet. Call spawn() and tick the world.")
        arr = self._latest

        # CARLA sensor frame is left-handed (y right). Convert to radar frame.
        xyz = np.stack([arr["x"], -arr["y"], arr["z"]], axis=1).astype(np.float64)

        # materials from semantic tags
        tags = arr["tag"].astype(np.int64)
        material_id = tags  # ints; ReflectivityModel maps via CARLA_TAG_TO_MATERIAL

        # doppler from actor velocities (rigid-body approximation per hit)
        if compute_doppler:
            vmap = self._actor_velocity_map()
            vel_world = np.zeros_like(xyz)
            idx = arr["idx"].astype(np.int64)
            # note: sensor-frame velocity == world velocity minus ego; for a
            # static radar we can use world velocities projected on LOS. For an
            # attached radar, subtract the ego velocity first.
            ego_v = np.zeros(3)
            if self.attach_to is not None:
                v = self.attach_to.get_velocity()
                ego_v = np.array([v.x, -v.y, v.z])
            for uid in np.unique(idx):
                vw = vmap.get(int(uid))
                if vw is None:
                    continue
                vr = np.array([vw[0], -vw[1], vw[2]]) - ego_v
                vel_world[idx == uid] = vr
            doppler = radial_velocity(xyz, vel_world)
        else:
            doppler = np.zeros(xyz.shape[0])

        # provide surface normal proxy via cos angle where useful:
        # semantic lidar gives cos of incidence; we pass reflectivity=1 and let
        # the ReflectivityModel handle material+range. If you want to use the
        # cos directly, multiply it into an area term below.
        refl_seed = np.ones(xyz.shape[0])

        return ReflectionPoints(
            xyz=xyz, reflectivity=refl_seed, velocity=doppler, material_id=material_id
        )


def reflection_points_from_semantic_array(
    arr: np.ndarray, flip_y: bool = True
) -> ReflectionPoints:
    """
    Offline helper: build ReflectionPoints from a saved semantic-lidar numpy
    structured array (fields x,y,z,cos,idx,tag). Useful for replaying recorded
    CARLA data without a live server.
    """
    y = -arr["y"] if flip_y else arr["y"]
    xyz = np.stack([arr["x"], y, arr["z"]], axis=1).astype(np.float64)
    return ReflectionPoints(
        xyz=xyz,
        reflectivity=np.ones(xyz.shape[0]),
        velocity=np.zeros(xyz.shape[0]),
        material_id=arr["tag"].astype(np.int64),
    )
