#!/usr/bin/env python3
"""
Plot ANSI S1.11-2004 octave-band filter responses overlaid on the correct
Class 1 tolerance bands for the 63 Hz, 1 kHz, and 8 kHz bands.

Tolerance bands use the standard's equation (12) log-linear interpolation
(from plot_ansi_tolerance_bands.py), converted from normalized frequency /
attenuation axes to absolute Hz / gain axes used in a frequency response plot.

Run from the repo root:
    python tests/plot_ansi_filter2.py
    python tests/plot_ansi_filter2.py --class 0
    python tests/plot_ansi_filter2.py --class 2
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from vslm.dsp.filters.octave_filters import OctaveFilterBank

# ---------------------------------------------------------------------------
# ANSI S1.11-2004 Table 1 — breakpoint normalized frequencies and limits
# Omega = f / fm (ratio of frequency to band-centre frequency)
# Attenuation sign convention: positive = attenuated below passband level.
# ---------------------------------------------------------------------------
G = 10 ** (3 / 10)   # G = 10^(3/10), the preferred octave base

TABLE1 = [
    # (Omega,            min0,   max0,   min1,   max1,   min2,   max2)
    (G**0,            -0.15,  +0.15,  -0.3,   +0.3,   -0.5,   +0.5),
    (G**(+1/8),       -0.15,  +0.2,   -0.3,   +0.4,   -0.5,   +0.6),
    (G**(+1/4),       -0.15,  +0.4,   -0.3,   +0.6,   -0.5,   +0.8),
    (G**(+3/8),       -0.15,  +1.1,   -0.3,   +1.3,   -0.5,   +1.6),
    # Just inside the band-edge — upper limit has a vertical drop here
    (G**(+1/2)*0.9999,-0.15,  +4.5,   -0.3,   +5.0,   -0.5,   +5.5),
    # At and beyond the band-edge: minimum jumps, maximum -> +inf
    (G**(+1/2),       +2.3,   np.inf, +2.0,   np.inf, +1.6,   np.inf),
    (G**(+1),         +18.0,  np.inf, +17.5,  np.inf, +16.5,  np.inf),
    (G**(+2),         +42.5,  np.inf, +42.0,  np.inf, +41.0,  np.inf),
    (G**(+3),         +62.0,  np.inf, +61.0,  np.inf, +55.0,  np.inf),
    (G**(+4),         +75.0,  np.inf, +70.0,  np.inf, +60.0,  np.inf),
]

CLASS_COLS = {0: (1, 2), 1: (3, 4), 2: (5, 6)}


def _build_breakpoints(cls):
    omegas = np.array([row[0] for row in TABLE1])
    mc, xc = CLASS_COLS[cls]
    mins = np.array([row[mc] for row in TABLE1])
    maxs = np.array([row[xc] for row in TABLE1])
    return omegas, mins, maxs


def _interp_limit(omega_query, omegas_bp, limits_bp):
    """Equation (12): log-linear interpolation between breakpoints."""
    results = np.empty_like(omega_query, dtype=float)
    for i, oq in enumerate(omega_query):
        idx = np.searchsorted(omegas_bp, oq, side='right') - 1
        idx = np.clip(idx, 0, len(omegas_bp) - 2)
        oa, ob = omegas_bp[idx], omegas_bp[idx + 1]
        la, lb = limits_bp[idx], limits_bp[idx + 1]
        if np.isinf(lb) or np.isinf(la):
            results[i] = np.inf
        elif oa == ob:
            results[i] = la
        else:
            frac = np.log10(oq / oa) / np.log10(ob / oa)
            results[i] = la + (lb - la) * frac
    return results


def build_tolerance_gain(fc, cls=1, n_points=2000):
    """
    Build upper and lower GAIN limits (dB) vs absolute frequency (Hz) for one
    band centred at fc.

    Relationship between gain and attenuation:
      gain = –attenuation
      upper_gain = –min_attenuation   (less attenuation allowed -> higher gain)
      lower_gain = –max_attenuation   (more attenuation allowed -> lower gain)

    In the stopband max_attenuation = +inf, so lower_gain = –inf (capped at
    PLOT_FLOOR for display).
    """
    PLOT_FLOOR = -110.0
    omegas_bp, mins_bp, maxs_bp = _build_breakpoints(cls)

    # Positive omega side: G^0 to G^4
    omega_pos = np.logspace(np.log10(1.0), np.log10(G**4), n_points)
    min_pos = _interp_limit(omega_pos, omegas_bp, mins_bp)
    max_pos = _interp_limit(omega_pos, omegas_bp, maxs_bp)

    # Mirror for negative side (symmetric bandpass)
    omega_neg = np.flip(1.0 / omega_pos[1:])
    min_neg   = np.flip(min_pos[1:])
    max_neg   = np.flip(max_pos[1:])

    omega_full = np.concatenate([omega_neg, omega_pos])
    min_full   = np.concatenate([min_neg,   min_pos])
    max_full   = np.concatenate([max_neg,   max_pos])

    # Convert to absolute Hz
    f_hz = omega_full * fc

    # Convert attenuation -> gain, cap infinite values
    upper_gain = -min_full
    lower_gain = np.where(np.isinf(max_full), PLOT_FLOOR, -max_full)

    return f_hz, lower_gain, upper_gain


def compute_filter_response(bank, fc, fs, n=131072):
    """Return (freqs_hz, gain_db) for the band nearest fc, normalised to 0 dB at fc."""
    idx    = np.argmin(np.abs(bank.frequencies - fc))
    fc_act = bank.frequencies[idx]

    impulse = np.zeros(n)
    impulse[0] = 1.0
    bank.initialize_state(np.zeros(1024))
    output = bank.process_chunk(impulse)

    freqs  = np.fft.rfftfreq(n, d=1.0 / fs)
    resp   = np.fft.rfft(output[:, idx])
    mag_db = 20 * np.log10(np.abs(resp) + 1e-15)
    mag_db -= mag_db[np.argmin(np.abs(freqs - fc_act))]

    return freqs, mag_db, fc_act


def plot(filter_class=1, fs=48000, order=12):
    bank = OctaveFilterBank(fs, resolution="octave", order=order)
    print(f"Band centre frequencies: {bank.frequencies}")

    target_bands = [63.0, 1000.0, 8000.0]
    band_colors  = ["tab:red", "tab:green", "tab:blue"]

    fig, ax = plt.subplots(figsize=(14, 8))

    for color, target in zip(band_colors, target_bands):
        # --- tolerance band for this centre frequency ---
        f_tol, lo_gain, hi_gain = build_tolerance_gain(target, cls=filter_class)

        # Restrict to the plotted frequency range
        mask = (f_tol >= 10.0) & (f_tol <= fs / 2)
        f_tol   = f_tol[mask]
        lo_gain = lo_gain[mask]
        hi_gain = hi_gain[mask]

        ax.fill_between(f_tol, lo_gain, hi_gain,
                        color=color, alpha=0.18, linewidth=0)
        ax.plot(f_tol, hi_gain, color=color, linewidth=1.0,
                linestyle="--", alpha=0.7)
        ax.plot(f_tol, lo_gain, color=color, linewidth=1.0,
                linestyle="--", alpha=0.7)

        # --- actual filter response ---
        freqs, mag_db, fc_act = compute_filter_response(bank, target, fs)
        ax.semilogx(freqs, mag_db, color=color, linewidth=2.0,
                    label=f"{target:.0f} Hz")

        ax.text(fc_act, 2.5, f"{target:.0f} Hz", ha="center",
                color=color, fontweight="bold", fontsize=9)

    # Reference lines
    ax.axhline(0,   color="black", linewidth=0.6, linestyle=":", alpha=0.4)
    floor_db = {0: -75, 1: -70, 2: -60}[filter_class]
    ax.axhline(floor_db, color="dimgray", linewidth=1.0, linestyle="-", alpha=0.6,
               label=f"Class {filter_class} stopband floor ({-floor_db} dB)")

    # Legend: combine filter lines + one shaded patch per class
    handles, _ = ax.get_legend_handles_labels()
    tol_patch  = mpatches.Patch(
        facecolor="gray", alpha=0.35, edgecolor="gray",
        linewidth=1.0, linestyle="--",
        label=f"Class {filter_class} tolerance band",
    )
    ax.legend(handles=handles + [tol_patch], loc="lower center",
              ncol=3, fontsize=9, framealpha=0.85)

    ax.set_xscale("log")
    ax.set_title(
        f"Octave Filter Bank — ANSI S1.11-2004 Class {filter_class} Tolerance Bands\n"
        f"(Fs = {fs} Hz, filter order = {order})",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Gain (dB)", fontsize=11)
    ax.set_xlim(10, fs / 2)
    ax.set_ylim(-110, 10)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.grid(True, which="major", linestyle="-",  alpha=0.20)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot octave filter responses vs ANSI S1.11-2004 tolerance bands.",
    )
    parser.add_argument("--class", dest="filter_class", default="1",
                        choices=["0", "1", "2"],
                        help="Filter class (default: 1)")
    parser.add_argument("--fs",    type=int, default=48000,
                        help="Sample rate in Hz (default: 48000)")
    parser.add_argument("--order", type=int, default=8,
                        help="Filter order (default: 8)")
    args = parser.parse_args()

    plot(filter_class=int(args.filter_class), fs=args.fs, order=args.order)
