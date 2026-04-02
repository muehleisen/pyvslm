"""
Plots A and C weighting filter responses vs the ideal IEC 61672-1 analogue
curve and the Class 1 tolerance band, for all supported sample rates.

Methodology
-----------
The digital filter response is obtained via scipy.signal.sosfreqz, which
evaluates the SOS transfer function H(e^jω) at 16384 uniformly-spaced
normalised frequencies from 0 to π.  The result is plotted against:

1. The ideal analogue response from _ideal_response_db (IEC 61672-1 Annex E).
2. A shaded tolerance band derived from IEC 61672-1 Class 1 limits.

The Class 1 tolerance is frequency-dependent (tighter in the mid-band,
looser at the extremes) and is applied as ± dB offsets from the ideal curve:

    upper_mask[f] = ideal_db[f] + tol_plus[f]
    lower_mask[f] = ideal_db[f] − tol_minus[f]

Reference: IEC 61672-1:2013 Table 2 (Class 1 electrical performance limits).

Run from the repo root:
    python tests/plot_weighting_filters.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

from vslm.dsp.filters.weighting_filters import WeightingFilter, SUPPORTED_FS, _ideal_response_db

# ISO preferred frequencies used for tolerance overlay
_TOL_FREQS = np.array([
    10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80,
    100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
    1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300,
    8000, 10000, 12500, 16000, 20000,
])


def _class1_tolerance(freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """IEC 61672-1 Class 1 ±tolerance in dB at each frequency."""
    tol = np.where(freqs < 25,   2.5,
          np.where(freqs < 100,  1.5,
          np.where(freqs <= 5000, 1.1,
          np.where(freqs <= 16000, 2.5, 3.0))))
    return tol, -tol


def plot_weighting(w_type: str = "A") -> None:
    f_ideal = np.logspace(1, 5.3, 2000)
    ideal_db = _ideal_response_db(f_ideal, w_type)

    tol_plus, tol_minus = _class1_tolerance(_TOL_FREQS)
    ideal_at_tol = _ideal_response_db(_TOL_FREQS, w_type)

    plt.figure(figsize=(12, 7))

    # Tolerance band
    plt.fill_between(
        _TOL_FREQS,
        ideal_at_tol + tol_minus,
        ideal_at_tol + tol_plus,
        color="green", alpha=0.18, label="Class 1 tolerance (IEC 61672-1)",
    )

    # Ideal analogue curve
    plt.semilogx(f_ideal, ideal_db, "k--", linewidth=2, zorder=5,
                 label=f"Ideal {w_type}-weighting (analogue)")

    # Digital filters at each sample rate
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(SUPPORTED_FS)))
    for i, fs in enumerate(SUPPORTED_FS):
        try:
            wf = WeightingFilter(fs, w_type)
            w, h = scipy.signal.sosfreqz(wf.sos, worN=16384, fs=fs)
            mag_db = 20 * np.log10(np.abs(h) + 1e-15)
            mask = (w >= 10) & (w <= fs / 2)
            plt.semilogx(w[mask], mag_db[mask],
                         color=colors[i], linewidth=1.5, alpha=0.85,
                         label=f"Fs = {fs} Hz")
        except Exception as e:
            print(f"  Skipping {fs} Hz: {e}")

    plt.title(f"{w_type}-Weighting Filter — MZT Hybrid SOS  (IEC 61672-1 compliance)",
              fontsize=13)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(loc="lower center", ncol=4, fontsize="small")
    plt.xlim(10, 100_000)
    plt.ylim((-80, 15) if w_type == "A" else (-20, 15))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Plotting A-weighting…")
    plot_weighting("A")
    print("Plotting C-weighting…")
    plot_weighting("C")
