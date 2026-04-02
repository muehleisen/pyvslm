"""
Plots the frequency response of the octave (and optionally 1/3-octave) filter
bank against the IEC 61260-1 / ANSI S1.11 Class 1 spectral mask.

Three representative bands are shown (63 Hz, 1 kHz, 8 kHz) to demonstrate
compliance across the audio band.

Run from the repo root:
    python tests/plot_ansi_filters.py
    python tests/plot_ansi_filters.py third     # 1/3-octave bank
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from vslm.dsp.filters.octave_filters import OctaveFilterBank


# ── IEC 61260-1 Class 1 spectral mask ────────────────────────────────────────
#
# IEC 61260-1 defines the mask in terms of a normalised frequency Ω:
#
#   Ω = (f/fc − fc/f) / (G^(1/2b) − G^(−1/2b))
#
# where G = 2 (base-2 system) and b = 1 for octave, b = 3 for 1/3-octave.
# The bandwidth factor Δ = G^(1/2b) − G^(−1/2b) differs between the two:
#   octave:      Δ = √2 − 1/√2  ≈ 0.707
#   1/3-octave:  Δ = 2^(1/6) − 2^(−1/6) ≈ 0.232
#
# The dB limits at each Ω breakpoint are the same for both resolutions
# (Class 1): ±0.3 dB passband (|Ω|≤1), −16.1 dB at |Ω|=2, −36.5 at |Ω|=3,
# −54.5 at |Ω|=4, with a deep stopband floor of −70 dB (octave) or −60 dB
# (1/3-octave) beyond |Ω|=5.
#
# _omega_to_r() converts Ω → r = f/fc so the mask can be plotted in Hz.

def _omega_to_r(omega: float, delta: float) -> float:
    """
    Convert normalised frequency Ω to the frequency ratio r = f/fc.

    Methodology
    -----------
    IEC 61260-1 defines Ω implicitly via:

        Ω = (r − 1/r) / Δ     where r = f/fc

    Solving for r:

        r − 1/r = Ω·Δ
        r² − Ω·Δ·r − 1 = 0                 (multiply through by r)
        r = [Ω·Δ + sqrt((Ω·Δ)² + 4)] / 2   (quadratic formula, positive root)

    Only the positive root is physically meaningful (r = f/fc > 0).
    """
    val = omega * delta
    return (val + np.sqrt(val ** 2 + 4.0)) / 2.0


def _class1_mask(fc: float, resolution: str = "octave"
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (f_upper, limit_upper, f_lower, limit_lower) for the IEC 61260-1
    Class 1 spectral mask centred on *fc*, correctly scaled for *resolution*
    ('octave' or 'third').
    """
    b     = 1 if resolution == "octave" else 3
    delta = 2 ** (1 / (2 * b)) - 2 ** (-1 / (2 * b))
    floor = -70.0 if resolution == "octave" else -60.0

    # Ω breakpoints and their upper-limit values (positive side only; mirrored below)
    omegas = np.array([0.0,  1.0,   2.0,    3.0,    4.0,    5.0])
    limits = np.array([0.3,  0.3, -16.1,  -36.5,  -54.5,  floor])

    r_pos = np.array([_omega_to_r(w, delta) for w in omegas])
    r_neg = 1.0 / r_pos[::-1]           # mirror: r_neg[i] = 1/r_pos[n-i]

    r_up = np.concatenate([[0.001], r_neg[:-1], r_pos[1:], [1000.0]])
    l_up = np.concatenate([[floor], limits[::-1][:-1], limits[1:], [floor]])

    # Passband lower limit: 0 dB at band edges (|Ω|=1)
    r_lo = np.array([1.0 / r_pos[1], r_pos[1]])
    l_lo = np.array([-0.3, -0.3])

    return r_up * fc, l_up, r_lo * fc, l_lo


def plot_filter_bank(resolution: str = "octave", fs: int = 48000,
                     order: int = 24) -> None:
    """
    Plot filter bank frequency responses against the IEC 61260-1 Class 1 mask.

    Methodology — impulse-response FFT
    -----------------------------------
    The frequency response of each bandpass filter is obtained by feeding a
    unit impulse (x[0]=1, x[n>0]=0) through the filter and computing the
    DFT of the output:

        H(e^jω) = DFT{ h[n] }     (h = impulse response)

    Using n=131072 points gives a frequency resolution of fs/131072 ≈ 0.37 Hz
    at 48 kHz, which is fine enough to resolve the filter's passband shape
    accurately.  This avoids the need to evaluate the SOS coefficients with
    sosfreqz and is numerically equivalent for stable filters.

    The response is normalised to 0 dB at the centre frequency by subtracting
    the FFT magnitude at the bin closest to fc.
    """
    bank = OctaveFilterBank(fs, resolution=resolution, order=order)
    print(f"{len(bank.frequencies)} bands: {bank.frequencies}")

    # 2^17 = 131072 points → frequency resolution ≈ 0.37 Hz at 48 kHz
    n = 131072
    impulse = np.zeros(n)
    impulse[0] = 1.0   # unit impulse at t=0
    bank.initialize_state(np.zeros(1024))
    output = bank.process_chunk(impulse)   # shape (n, n_bands)
    freqs  = np.fft.rfftfreq(n, d=1 / fs)

    target_bands = [63.0, 1000.0, 8000.0]
    colors = ["tab:red", "tab:green", "tab:blue"]

    plt.figure(figsize=(14, 8))

    for color, target in zip(colors, target_bands):
        idx    = np.argmin(np.abs(bank.frequencies - target))
        fc     = bank.frequencies[idx]
        resp   = np.fft.rfft(output[:, idx])
        mag_db = 20 * np.log10(np.abs(resp) + 1e-15)

        # Normalise to 0 dB at centre frequency
        ref_idx  = np.argmin(np.abs(freqs - fc))
        mag_db  -= mag_db[ref_idx]

        plt.semilogx(freqs, mag_db, color=color, linewidth=2,
                     label=f"{fc:.0f} Hz band")

        # Class 1 mask — correctly scaled for this resolution
        f_up, l_up, f_lo, l_lo = _class1_mask(fc, resolution)
        kw_up = {"color": "k", "linestyle": "--", "linewidth": 1.5, "alpha": 0.75}
        kw_lo = {"color": "k", "linestyle": "-.", "linewidth": 1.5, "alpha": 0.75}
        plt.plot(f_up, l_up, **kw_up,
                 label="Class 1 max limit" if target == target_bands[0] else None)
        plt.plot(f_lo, l_lo, **kw_lo,
                 label="Class 1 min limit" if target == target_bands[0] else None)

        plt.text(fc, 2.5, f"{fc:.0f} Hz", ha="center", color=color,
                 fontweight="bold", fontsize=9)

    res_label = "Octave" if resolution == "octave" else "1/3-Octave"
    plt.title(
        f"{res_label} Filter Bank — IEC 61260-1 / ANSI S1.11 Class 1 Compliance\n"
        f"(Fs = {fs} Hz, filter order = {order})",
        fontsize=13,
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Attenuation (dB)")
    plt.xlim(10, fs / 2)
    plt.ylim(-110, 10)
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend(loc="lower center", ncol=3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    resolution = sys.argv[1] if len(sys.argv) > 1 else "octave"
    plot_filter_bank(resolution=resolution)
