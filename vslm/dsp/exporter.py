"""CSV export for all analysis modes."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import leq as leq_mod
from ..constants import LEQ_INTERVAL_MAP, LeqInterval

if TYPE_CHECKING:
    from ..settings import DoseStandard


def export_lp(filepath: Path, results: list[dict], weighting: str, speed: str) -> None:
    """Export time vs Lp history to CSV."""
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"Time (s)", f"Lp ({weighting}, {speed}) [dB]"])
        for r in results:
            w.writerow([f"{r['time']:.3f}", f"{r['lp']:.2f}"])


def export_leq(
    filepath: Path,
    results: list[dict],
    block_size_ms: float,
    interval_key: LeqInterval,
    weighting: str,
    dose_params: "DoseStandard",
    dose_std_name: str,
    ref_pressure: float = 20e-6,
) -> None:
    """Export integrated Leq time history to CSV."""
    label, interval_s = LEQ_INTERVAL_MAP.get(interval_key, ("1 sec", 1.0))
    stats = leq_mod.calculate(results, block_size_ms, interval_s, dose_params, ref_pressure)

    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# VSLM Export", f"Weighting: {weighting}",
                    f"Interval: {label}", f"Dose standard: {dose_std_name}"])
        w.writerow(["# Overall Leq", f"{stats.overall:.2f} dB"])
        w.writerow(["# Dose", f"{stats.dose['dose']:.1f} %",
                    "TWA", f"{stats.dose['twa']:.1f} dB"])
        w.writerow(["Start Time (s)", "LEQ (dB)"])
        for t, lv in zip(stats.history["time"], stats.history["leq"]):
            w.writerow([f"{t:.2f}", f"{lv:.2f}"])


def export_spectrum(
    filepath: Path,
    results: list[dict],
    weighting: str,
    ref_pressure: float = 20e-6,
) -> None:
    """Export time-averaged octave/third-octave spectrum to CSV."""
    if not results or "bands" not in results[0]:
        return

    freqs = results[0]["band_freqs"]
    p_sum = np.zeros(len(freqs))
    for r in results:
        if "bands" in r:
            p_sum += (10.0 ** (r["bands"] / 10.0)) * (ref_pressure ** 2)

    mean_db = 10.0 * np.log10(p_sum / len(results) / (ref_pressure ** 2) + 1e-30)

    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# VSLM Export", f"Spectrum ({weighting}-weighted)"])
        w.writerow(["Frequency (Hz)", "Average Level (dB)"])
        for freq, level in zip(freqs, mean_db):
            w.writerow([f"{freq:.1f}", f"{level:.2f}"])
