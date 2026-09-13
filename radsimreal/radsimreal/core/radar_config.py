"""
Radar tensor geometry and physical configuration.

RadSimReal never needs the radar's *hardware design* (waveform, antenna
array layout, beamforming, sampling rate). It only needs:

  1. The discretization of the output radar tensor  (range x azimuth x doppler)
  2. The radar's measured Point Spread Function (PSF)  -> see radsimreal.psf
  3. The noise statistics (variance)                  -> see radsimreal.core.noise

This dataclass captures item (1): the coordinate grid that the tensor lives on.
Everything downstream (reflection-point rasterization, PSF convolution, image
conversion) is defined relative to this grid.

Default values follow the RADDet / TI AWR1843 prototype quoted in the paper:
  - azimuth resolution ~3.9 deg
  - range   resolution ~0.28 m
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class RadarConfig:
    # ---- Range axis (metres) ----
    range_min: float = 0.0
    range_max: float = 50.0
    range_bins: int = 256

    # ---- Azimuth axis (degrees) ----
    azimuth_min_deg: float = -90.0
    azimuth_max_deg: float = 90.0
    azimuth_bins: int = 256

    # ---- Doppler axis (m/s) ----
    doppler_min: float = -13.0
    doppler_max: float = 13.0
    doppler_bins: int = 64

    # ---- Optional metadata (not used in math, useful for provenance) ----
    name: str = "generic_fmcw"
    carrier_ghz: float = 77.0

    # cached axes
    _range_axis: np.ndarray = field(default=None, repr=False, compare=False)
    _azimuth_axis: np.ndarray = field(default=None, repr=False, compare=False)
    _doppler_axis: np.ndarray = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self._range_axis = np.linspace(
            self.range_min, self.range_max, self.range_bins, dtype=np.float64
        )
        self._azimuth_axis = np.linspace(
            self.azimuth_min_deg, self.azimuth_max_deg, self.azimuth_bins,
            dtype=np.float64,
        )
        self._doppler_axis = np.linspace(
            self.doppler_min, self.doppler_max, self.doppler_bins, dtype=np.float64
        )

    # ---- axis accessors ----
    @property
    def range_axis(self) -> np.ndarray:
        return self._range_axis

    @property
    def azimuth_axis(self) -> np.ndarray:
        return self._azimuth_axis

    @property
    def doppler_axis(self) -> np.ndarray:
        return self._doppler_axis

    # ---- resolutions ----
    @property
    def range_res(self) -> float:
        return (self.range_max - self.range_min) / max(self.range_bins - 1, 1)

    @property
    def azimuth_res_deg(self) -> float:
        return (self.azimuth_max_deg - self.azimuth_min_deg) / max(
            self.azimuth_bins - 1, 1
        )

    @property
    def doppler_res(self) -> float:
        return (self.doppler_max - self.doppler_min) / max(self.doppler_bins - 1, 1)

    @property
    def tensor_shape(self) -> tuple[int, int, int]:
        """(range, azimuth, doppler)"""
        return (self.range_bins, self.azimuth_bins, self.doppler_bins)

    # ---- coordinate <-> bin index helpers ----
    def range_to_bin(self, r: np.ndarray) -> np.ndarray:
        b = (r - self.range_min) / (self.range_max - self.range_min) * (
            self.range_bins - 1
        )
        return np.rint(b).astype(np.int64)

    def azimuth_to_bin(self, az_deg: np.ndarray) -> np.ndarray:
        b = (az_deg - self.azimuth_min_deg) / (
            self.azimuth_max_deg - self.azimuth_min_deg
        ) * (self.azimuth_bins - 1)
        return np.rint(b).astype(np.int64)

    def doppler_to_bin(self, v: np.ndarray) -> np.ndarray:
        b = (v - self.doppler_min) / (self.doppler_max - self.doppler_min) * (
            self.doppler_bins - 1
        )
        return np.rint(b).astype(np.int64)

    def in_bounds(self, r, az_deg, v=None) -> np.ndarray:
        m = (
            (r >= self.range_min)
            & (r <= self.range_max)
            & (az_deg >= self.azimuth_min_deg)
            & (az_deg <= self.azimuth_max_deg)
        )
        if v is not None:
            m &= (v >= self.doppler_min) & (v <= self.doppler_max)
        return m

    # ---- convenience presets ----
    @classmethod
    def raddet(cls) -> "RadarConfig":
        """TI AWR1843 prototype as used in RADDet (paper values)."""
        return cls(
            name="raddet_ti_awr1843",
            range_min=0.0,
            range_max=50.0,
            range_bins=256,          # ~0.196 m/bin (~0.28 m native res)
            azimuth_min_deg=-85.0,
            azimuth_max_deg=85.0,
            azimuth_bins=256,        # ~0.66 deg/bin (~3.9 deg native res)
            doppler_min=-13.0,
            doppler_max=13.0,
            doppler_bins=64,
            carrier_ghz=77.0,
        )

    @classmethod
    def small(cls) -> "RadarConfig":
        """Tiny grid for fast CPU smoke-tests."""
        return cls(
            name="small_debug",
            range_max=30.0,
            range_bins=96,
            azimuth_min_deg=-60.0,
            azimuth_max_deg=60.0,
            azimuth_bins=96,
            doppler_bins=24,
        )
