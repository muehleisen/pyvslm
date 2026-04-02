"""
2×2 grid comparing A and C weighting digital filters against the analogue
reference, with both IEC 61672-1 Class 1 and ANSI S1.4 Type 0 tolerances.

    Top row:    magnitude response (digital overlay on analogue)
    Bottom row: error = digital − analogue

Methodology
-----------
**Analogue reference**
The true analogue A/C weighting responses are evaluated using
scipy.signal.freqs_zpk, which computes H(jω) from continuous-time
zero-pole-gain descriptions of the IEC 61672-1 prototypes.  This avoids
the digital approximation and gives the exact standard curve.

The A-weighting ZPK parameters are:
    zeros:  4 at s=0  (the s⁴ numerator)
    poles:  at −2π·f1 (×2), −2π·f2, −2π·f3, −2π·f4 (×2)
    gain:   Ka = (2π·f4)² · 10^(A_1000 / 20)
where A_1000 = 1.9997 dB is the un-normalised A-weighting gain at 1 kHz,
chosen so that H_A(1000 Hz) = 0 dB after normalisation.

C-weighting uses the same f1 and f4 poles with 2 zeros at s=0 and
Kc = (2π·f4)² · 10^(C_1000 / 20).

**Digital filter response**
scipy.signal.sosfreqz evaluates the digital SOS at the same normalised
frequencies (wn = 2π·f / fs), giving H(e^jω) for comparison with H(jω).

**Error plot**
    error[f] = 20·log10|H_digital(f)| − 20·log10|H_analogue(f)|

This directly shows how many dB the digital filter deviates from the ideal
at each frequency, bounded by the tolerance overlays.

**Tolerance standards**
- IEC 61672-1:2013 Class 1 (red): modern precision SLM standard
- ANSI S1.4-1983 Type 0 (black dotted): legacy laboratory reference

Run from the repo root:
    python tests/plot_weighting_filters_compact.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg

from vslm.dsp.filters.weighting_filters import WeightingFilter, SUPPORTED_FS

# ── Analogue prototypes ────────────────────────────────────────────────────────
_F1, _F2, _F3, _F4 = 20.598997, 107.65265, 737.86223, 12194.217

A1000 = 1.9997
Ka = (2 * np.pi * _F4) ** 2 * 10 ** (A1000 / 20)
Za = [0, 0, 0, 0]
Pa = 2 * np.pi * np.array([_F4, _F4, _F3, _F2, _F1, _F1])

C1000 = 0.0619
Kc = (2 * np.pi * _F4) ** 2 * 10 ** (C1000 / 20)
Zc = [0, 0]
Pc = 2 * np.pi * np.array([_F4, _F1, _F4, _F1])

# ── Tolerance tables ───────────────────────────────────────────────────────────
_TOL_F = np.array([
    10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80,
    100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
    1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300,
    8000, 10000, 12500, 16000, 20000,
])

# ANSI S1.4 Type 0 (±values)
_UP0 = np.array([2, 2, 2, 2, 1.5, 1, 1, 1, 1, 1, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
                 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 1, 1, 1, 2, 2, 2, 2])
_LO0 = np.array([5, 3, 3, 2, 1.5, 1, 1, 1, 1, 1, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
                 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 1, 1.5, 2, 3, 3, 3, 3])

# IEC 61672-1 Class 1 (±values)
_UP1 = np.where(_TOL_F < 25,    2.5,
       np.where(_TOL_F < 100,   1.5,
       np.where(_TOL_F <= 5000, 1.1,
       np.where(_TOL_F <= 16000, 2.5, 3.0))))
_LO1 = _UP1.copy()


def main() -> None:
    f = np.logspace(1, np.log10(30_000), 2048)
    wn = f * 2 * np.pi          # angular frequency for freqs_zpk

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    ax_amag, ax_cmag = axs[0]
    ax_aerr, ax_cerr = axs[1]

    # Analogue reference curves
    _, ha_ref = sg.freqs_zpk(Za, Pa, Ka, wn)
    _, hc_ref = sg.freqs_zpk(Zc, Pc, Kc, wn)
    ha_ref_db = 20 * np.log10(np.abs(ha_ref) + 1e-15)
    hc_ref_db = 20 * np.log10(np.abs(hc_ref) + 1e-15)

    ax_amag.semilogx(f, ha_ref_db, "k--", linewidth=2, label="Analogue ref", alpha=0.65)
    ax_cmag.semilogx(f, hc_ref_db, "k--", linewidth=2, label="Analogue ref", alpha=0.65)

    colors     = plt.cm.jet(np.linspace(0, 0.9, len(SUPPORTED_FS)))
    linestyles = ["-", "--", "-.", ":"] * 4

    for i, fs in enumerate(SUPPORTED_FS):
        lbl = f"{fs} Hz"
        ls  = linestyles[i % len(linestyles)]
        col = colors[i]

        try:
            wf_a = WeightingFilter(fs, "A")
            wf_c = WeightingFilter(fs, "C")

            # Evaluate digital filters at the same normalised frequencies
            wn_d = f / fs * 2 * np.pi
            _, hza = sg.sosfreqz(wf_a.sos, worN=wn_d)
            _, hzc = sg.sosfreqz(wf_c.sos, worN=wn_d)

            hzadb = 20 * np.log10(np.abs(hza) + 1e-15)
            hzcdb = 20 * np.log10(np.abs(hzc) + 1e-15)

            ax_amag.semilogx(f, hzadb, color=col, linestyle=ls, linewidth=1.2, label=lbl)
            ax_cmag.semilogx(f, hzcdb, color=col, linestyle=ls, linewidth=1.2, label=lbl)

            mask = f < fs / 2
            ax_aerr.semilogx(f[mask], (hzadb - ha_ref_db)[mask],
                             color=col, linestyle=ls, linewidth=1.2, label=lbl)
            ax_cerr.semilogx(f[mask], (hzcdb - hc_ref_db)[mask],
                             color=col, linestyle=ls, linewidth=1.2)

        except Exception as e:
            print(f"  Skipping {fs} Hz: {e}")

    # Tolerance overlays on error plots
    for ax in (ax_aerr, ax_cerr):
        ax.fill_between(_TOL_F, -_LO0, _UP0,
                        color="gray", alpha=0.15, label="Type 0 range")
        ax.semilogx(_TOL_F,  _UP1, "r-", linewidth=1.8, alpha=0.45,
                    label="Class 1 (IEC 61672-1)")
        ax.semilogx(_TOL_F, -_LO1, "r-", linewidth=1.8, alpha=0.45)
        ax.semilogx(_TOL_F,  _UP0, "k:", linewidth=1.5, alpha=0.65,
                    label="Type 0 (ANSI S1.4)")
        ax.semilogx(_TOL_F, -_LO0, "k:", linewidth=1.5, alpha=0.65)

    # ── Format axes ────────────────────────────────────────────────────────────
    for ax, title in (
        (ax_amag, "A-Weighting — Magnitude"),
        (ax_cmag, "C-Weighting — Magnitude"),
        (ax_aerr, "A-Weighting — Error (digital − analogue)"),
        (ax_cerr, "C-Weighting — Error (digital − analogue)"),
    ):
        ax.set_title(title)
        ax.set_xlim(10, 30_000)
        ax.grid(True, which="both", alpha=0.3)

    ax_amag.set_ylabel("Magnitude (dB)")
    ax_aerr.set_ylabel("Deviation (dB)")
    ax_aerr.set_xlabel("Frequency (Hz)")
    ax_cerr.set_xlabel("Frequency (Hz)")

    ax_amag.set_ylim(-70, 10)
    ax_cmag.set_ylim(-70, 10)
    ax_aerr.set_ylim(-2, 2)
    ax_cerr.set_ylim(-2, 2)

    ax_amag.legend(fontsize="x-small", ncol=2)
    ax_aerr.legend(loc="upper left", fontsize="x-small")

    fig.suptitle("Weighting Filter Compliance — MZT Hybrid SOS", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
