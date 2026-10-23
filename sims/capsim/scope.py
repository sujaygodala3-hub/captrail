"""Scope parameters: the numbers S0 measures and S1 to S7 consume.

Every field starts as an ASSUMED default and is flagged in `measured`. Run S0
(`run_s0.py`) on real Celestron captures to overwrite them with measured values.
Anything still flagged False in the output of a sweep is a guess, not a result.
"""
from __future__ import annotations

import json
import math

import numpy as np
from dataclasses import asdict, dataclass, field
from pathlib import Path

FIELDS = ("um_per_px", "sigma_um", "photons", "read_noise", "illum_strength",
          "illum_angle_deg", "contrast", "contrast_oil")


@dataclass
class ScopeParams:
    um_per_px: float = 1.25          # ASSUMED: Berks 2014 research scope sampling, not the Celestron
    sigma_um: float = 2.0            # ASSUMED PSF sigma
    photons: float = 1500.0          # photons/pixel at full scale (255 / conversion gain)
    read_noise: float = 0.01         # read noise, fraction of full scale
    illum_strength: float = 0.10     # peak-to-peak relative lighting variation across the field
    illum_angle_deg: float = 0.0
    contrast: float = 0.20           # fractional drop inside a lumen, no oil
    contrast_oil: float | None = None
    bits: int = 8
    corr_noise: list | None = None   # [[scale_px, std], ...] correlated noise (blank-card frames); None = white only
    texture_amp: float | None = None # skin texture amplitude for S2/S4 (measured on real frames); None = sweep default
    measured: dict = field(default_factory=lambda: {k: False for k in FIELDS})
    notes: dict = field(default_factory=dict)
    uncertainty: dict = field(default_factory=dict)   # absolute 1-SD of each MEASURED value (from S0)

    # ---- bookkeeping -------------------------------------------------
    def mark(self, _sd=None, **values):
        """Set fields from measurements and flag them as measured. `_sd` optionally gives
        {field: 1-SD} so sensitivity.monte_carlo can propagate the measurement uncertainty."""
        for k, v in values.items():
            if k not in FIELDS:
                raise KeyError(k)
            setattr(self, k, v)
            self.measured[k] = True
        for k, v in (_sd or {}).items():
            if k not in FIELDS:
                raise KeyError(k)
            if v is not None and np.isfinite(v):
                self.uncertainty[k] = float(v)
        return self

    def assumed(self):
        return [k for k in FIELDS if not self.measured.get(k, False)
                and not (k == "contrast_oil" and self.contrast_oil is None)]

    def tag(self):
        return "measured" if not self.assumed() else f"assumed-{len(self.assumed())}"

    # ---- simulation glue ---------------------------------------------
    def fine_res(self, target=0.5):
        """Fine simulation grid (um) that divides the pixel size exactly and is <= target."""
        k = max(1, math.ceil(self.um_per_px / target - 1e-9))
        return self.um_per_px / k

    def forward_kwargs(self, oil=False, **over):
        c = self.contrast_oil if (oil and self.contrast_oil is not None) else self.contrast
        kw = dict(sigma_um=self.sigma_um, contrast=c, illum_strength=self.illum_strength,
                  illum_angle_deg=self.illum_angle_deg, pixel_um=self.um_per_px,
                  photons=self.photons, read_noise=self.read_noise, bits=self.bits)
        if self.corr_noise:
            kw["corr_noise"] = [list(map(float, c)) for c in self.corr_noise]
        kw.update(over)
        return kw

    # ---- io ------------------------------------------------------------
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path=None):
        if path is None or not Path(path).exists():
            return cls()
        d = json.loads(Path(path).read_text())
        m = {k: False for k in FIELDS}
        m.update(d.pop("measured", {}))
        d.setdefault("uncertainty", {})
        return cls(measured=m, **d)
