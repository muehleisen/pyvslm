from __future__ import annotations

import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict

from pydantic import BaseModel, Field, ConfigDict

from .constants import Weighting, ResponseSpeed

DEFAULT_SETTINGS_PATH = Path(__file__).parent.parent / "settings.yaml"


class DoseStandard(BaseModel):
    exchange_rate:   float
    criterion_level: float
    threshold_level: float
    shift_hours:     float


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    last_directory: str = str(Path.home())
    ref_pressure:   float = 20e-6

    calibration_factor: float = Field(default=1.0, ge=0)

    weighting: Weighting      = Weighting.A
    speed:     ResponseSpeed  = ResponseSpeed.FAST

    analysis_mode_index:  int   = 0
    leq_interval_index:   int   = 1
    lp_interval_index:    int   = 0
    block_size_ms:        float = Field(default=100.0, gt=0)

    band_filter_order: int = 24

    psd_nfft:   int = 4096
    psd_window: str = "Hanning"

    spec_nfft:   int   = 512
    spec_dt:     float = 1.0
    spec_window: str   = "Hamming"
    spec_cmap:   str   = "plasma"

    current_dose_standard: str = "NIOSH"
    dose_standards: Dict[str, DoseStandard] = Field(
        default_factory=lambda: {
            "NIOSH": DoseStandard(exchange_rate=3.0, criterion_level=85.0,
                                  threshold_level=80.0, shift_hours=8.0),
            "OSHA":  DoseStandard(exchange_rate=5.0, criterion_level=90.0,
                                  threshold_level=80.0, shift_hours=8.0),
        }
    )

    plot_autoscale: bool  = True
    plot_ymin:      float = 0.0
    plot_ymax:      float = 120.0


class SettingsManager:
    def __init__(self, default_path: Path = DEFAULT_SETTINGS_PATH):
        self.default_path = default_path

    def load(self, filepath: Path | None = None) -> AppSettings:
        path = filepath or self.default_path
        if not path.exists():
            return AppSettings()
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            return AppSettings.model_validate(data) if data else AppSettings()
        except Exception as e:
            print(f"Could not load settings from {path}: {e}")
            return AppSettings()

    def save(self, settings: AppSettings, filepath: Path | None = None) -> None:
        path = filepath or self.default_path
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(path, "w") as f:
                f.write(f"# Saved by VSLM on {timestamp}\n")
                yaml.dump(settings.model_dump(mode="json"), f, default_flow_style=False)
        except Exception as e:
            print(f"Could not save settings to {path}: {e}")
