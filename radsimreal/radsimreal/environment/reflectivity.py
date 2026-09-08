"""
RF reflectivity assignment  (Fig. 2a, "Assign Reflection Intensity").

The paper assigns each reflection point an RF reflectivity using physical
formulas [Mahafza & Elsherbeni, "MATLAB Simulations for Radar Systems Design"]
that account for:
    - surface material   (reflection coefficient of the material)
    - orientation        (angle between surface normal and the radar line-of-sight)
    - distance           (range attenuation of the radar equation)

This module implements a compact, physically-motivated version of that step.
It is deliberately parameterised by a material table so that CARLA semantic
labels or SimDaaS material IDs can be mapped onto RF reflectivity directly.

Model
-----
The received linear reflectivity of a facet is modelled as

    rho = |Gamma(material)|^2  *  g(theta_inc)  *  (R_ref / R)^4  *  A_eff

where
    Gamma(material)  : Fresnel-like normal-incidence reflection coefficient
                       (a per-material scalar in [0, 1], stored in MATERIAL_TABLE)
    g(theta_inc)     : orientation gain, a specular+diffuse lobe in the incidence
                       angle theta_inc between the surface normal and the radar ray
    (R_ref/R)^4      : two-way radar-equation range attenuation (normalised at R_ref)
    A_eff            : effective illuminated area per point (constant unless supplied)

The exponents / constants are consolidated so that a corner reflector at the
reference range returns ~1.0. Absolute scale is irrelevant to RadSimReal (the
final image is convolved with a measured PSF and then normalised), so only the
*relative* reflectivity across materials / angles / ranges matters.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# Normal-incidence reflection coefficient |Gamma| per material class.
# Metals ~1, concrete/road moderate, vegetation low, glass moderate-high.
# Values are heuristic but ordered to match automotive-radar intuition.
MATERIAL_TABLE = {
    "unknown":     0.30,
    "vehicle":     0.95,   # metal body / retro-reflective structure
    "metal":       0.98,
    "guardrail":   0.95,
    "pole":        0.90,   # signpost / lamp pole (strong point scatterer)
    "sign":        0.85,
    "building":    0.60,   # concrete / brick facades
    "wall":        0.60,
    "concrete":    0.55,
    "road":        0.35,   # asphalt, grazing-angle weak
    "ground":      0.30,
    "sidewalk":    0.35,
    "vegetation":  0.12,   # foliage, diffuse & weak
    "pedestrian":  0.25,   # low RCS, body water content
    "glass":       0.55,
    "water":       0.20,
}

# CARLA semantic-segmentation tag id -> material key.
# (CARLA 0.9.x CityScapes-style palette; adjust to your build if needed.)
CARLA_TAG_TO_MATERIAL = {
    0:  "unknown",     1:  "building",   2:  "wall",       3:  "unknown",   # fence
    4:  "pedestrian",  5:  "pole",       6:  "road",       7:  "road",      # roadline
    8:  "sidewalk",    9:  "vegetation", 10: "vehicle",    11: "wall",
    12: "sign",        13: "unknown",    14: "ground",     15: "building",  # bridge
    16: "guardrail",   17: "guardrail",  18: "pole",       19: "vehicle",   # dynamic
    20: "unknown",     21: "water",      22: "ground",
}


@dataclass
class ReflectivityModel:
    reference_range: float = 5.0     # R_ref (m): reflectivity normalised here
    specular_weight: float = 0.7     # fraction of energy in the specular lobe
    specular_sharpness: float = 8.0  # higher -> narrower specular lobe
    diffuse_floor: float = 0.15      # minimum orientation gain (backscatter floor)
    area_per_point: float = 1.0      # A_eff, constant unless per-point area given
    range_exponent: float = 4.0      # two-way radar equation exponent
    eps: float = 1e-6

    def gamma_from_material(self, material_id) -> np.ndarray:
        """Map material labels (str keys or int ids) to |Gamma|."""
        material_id = np.asarray(material_id).reshape(-1)
        out = np.empty(material_id.shape[0], dtype=np.float64)
        for i, m in enumerate(material_id):
            if isinstance(m, (int, np.integer)):
                # treat as CARLA tag if within table, else 'unknown'
                key = CARLA_TAG_TO_MATERIAL.get(int(m), "unknown")
            else:
                key = str(m)
            out[i] = MATERIAL_TABLE.get(key, MATERIAL_TABLE["unknown"])
        return out

    def orientation_gain(self, theta_inc_rad: np.ndarray) -> np.ndarray:
        """
        Specular + diffuse orientation lobe.
        theta_inc = 0  -> normal incidence (max return).
        """
        c = np.clip(np.cos(theta_inc_rad), 0.0, 1.0)
        specular = c ** self.specular_sharpness
        diffuse = c  # Lambertian-like
        g = self.specular_weight * specular + (1.0 - self.specular_weight) * diffuse
        return np.maximum(g, self.diffuse_floor)

    def range_attenuation(self, R: np.ndarray) -> np.ndarray:
        R = np.maximum(R, self.eps)
        return (self.reference_range / R) ** self.range_exponent

    def __call__(
        self,
        xyz: np.ndarray,
        normals: np.ndarray = None,
        gamma: np.ndarray = None,
        material_id=None,
        area: np.ndarray = None,
    ) -> np.ndarray:
        """
        Compute linear reflectivity per point.

        Parameters
        ----------
        xyz        : (N,3) point coords in radar frame.
        normals    : (N,3) surface normals (unit). If None, assume the surface
                     faces the radar (theta_inc = 0, i.e. best case).
        gamma      : (N,) precomputed |Gamma| per point. Overrides material_id.
        material_id: (N,) material keys/ids -> |Gamma| via MATERIAL_TABLE.
        area       : (N,) per-point effective area. If None, constant.
        """
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        N = xyz.shape[0]
        R = np.linalg.norm(xyz, axis=1)

        # |Gamma|
        if gamma is not None:
            g_refl = np.asarray(gamma, dtype=np.float64).reshape(N)
        elif material_id is not None:
            g_refl = self.gamma_from_material(material_id)
        else:
            g_refl = np.full(N, MATERIAL_TABLE["unknown"])

        # orientation
        if normals is not None:
            normals = np.asarray(normals, dtype=np.float64).reshape(N, 3)
            los = xyz / np.maximum(R[:, None], self.eps)          # radar->point unit
            # incidence angle between inward normal and incoming ray
            cos_inc = np.abs(np.sum(normals * los, axis=1))
            cos_inc = np.clip(cos_inc, 0.0, 1.0)
            theta_inc = np.arccos(cos_inc)
        else:
            theta_inc = np.zeros(N)
        g_orient = self.orientation_gain(theta_inc)

        # area
        A = self.area_per_point if area is None else np.asarray(area).reshape(N)

        rho = (g_refl ** 2) * g_orient * self.range_attenuation(R) * A
        return np.maximum(rho, 0.0)
