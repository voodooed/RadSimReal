"""RadSimReal: PSF-convolution radar simulation (Bialer & Haitman, CVPR 2024)."""
from .core.radar_config import RadarConfig
from .core.reflection_points import ReflectionPoints
from .core.noise import NoiseModel
from .core.image import tensor_to_image, to_db
from .psf.psf import PSF, synthetic_psf, truncate_psf, psf_from_tensor
from .environment.reflectivity import ReflectivityModel
from .environment.synthetic_scene import SyntheticScene, SceneObject, default_scene, GTBox
from .pipeline.simulator import RadSimReal, SimResult

__version__ = "0.1.0"
__all__ = [
    "RadarConfig", "ReflectionPoints", "NoiseModel", "tensor_to_image", "to_db",
    "PSF", "synthetic_psf", "truncate_psf", "psf_from_tensor",
    "ReflectivityModel", "SyntheticScene", "SceneObject", "default_scene", "GTBox",
    "RadSimReal", "SimResult",
]
