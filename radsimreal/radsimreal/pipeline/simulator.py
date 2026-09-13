"""
RadSimReal simulator -- top-level orchestration.

Ties the whole method together (Fig. 2a + 2c):

    reflection points ---> [reflectivity assignment] ---> rasterize
                       ---> convolve with PSF ---> add noise ---> tensor
                       ---> reduce over doppler ---> range-azimuth image

Two entry points:
    - simulate_tensor(points)  -> full 3D radar tensor
    - simulate_image(points)   -> range-azimuth image (+ tensor)

The simulator is *source-agnostic*: `points` may come from the standalone
synthetic scene generator, the CARLA adapter, or the SimDaaS adapter. All three
produce ReflectionPoints in the radar frame.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..core.radar_config import RadarConfig
from ..core.reflection_points import ReflectionPoints
from ..core.noise import NoiseModel
from ..core.synthesis import rasterize, convolve_sparse, convolve_fft
from ..core.image import tensor_to_image
from ..psf.psf import PSF, truncate_psf, synthetic_psf
from ..environment.reflectivity import ReflectivityModel


@dataclass
class SimResult:
    tensor: np.ndarray            # (R, A, D)
    image: np.ndarray            # (R, A)
    field: np.ndarray            # sparse reflection-point field (R, A, D)
    cfg: RadarConfig


class RadSimReal:
    def __init__(
        self,
        cfg: RadarConfig = None,
        psf: PSF = None,
        noise: NoiseModel = None,
        reflectivity_model: ReflectivityModel = None,
        truncate_energy: float = 0.99,
        convolution: str = "sparse",      # 'sparse' | 'fft'
        image_reduction: str = "max",
    ):
        self.cfg = cfg or RadarConfig.raddet()
        # PSF: use provided, else synthesize one consistent with the grid
        raw_psf = psf or synthetic_psf(self.cfg)
        # normalise then truncate to keep 99% energy (paper's key speed step)
        self.psf = truncate_psf(raw_psf.normalized(), truncate_energy)
        self.noise = noise
        self.reflectivity_model = reflectivity_model or ReflectivityModel()
        self.convolution = convolution
        self.image_reduction = image_reduction

    # ---- optional: (re)assign reflectivity from geometry+material ----
    def assign_reflectivity(
        self, points: ReflectionPoints, normals=None
    ) -> ReflectionPoints:
        # adapters may attach surface normals as a side-channel attribute
        if normals is None:
            normals = getattr(points, "_normals", None)
        rho = self.reflectivity_model(
            xyz=points.xyz,
            normals=normals,
            material_id=points.material_id,
        )
        return ReflectionPoints(
            xyz=points.xyz,
            reflectivity=rho,
            velocity=points.velocity,
            material_id=points.material_id,
        )

    def _convolve(self, field):
        if self.convolution == "sparse":
            return convolve_sparse(field, self.psf)
        if self.convolution == "fft":
            return convolve_fft(field, self.psf)
        raise ValueError(self.convolution)

    def simulate_tensor(self, points: ReflectionPoints) -> SimResult:
        field = rasterize(points, self.cfg)
        tensor = self._convolve(field)

        if self.noise is not None:
            tensor = self.noise.add(tensor)

        image = tensor_to_image(tensor, self.image_reduction)
        return SimResult(tensor=tensor, image=image, field=field, cfg=self.cfg)

    def simulate_image(self, points: ReflectionPoints) -> SimResult:
        # identical pipeline; kept as an explicit name for clarity
        return self.simulate_tensor(points)
