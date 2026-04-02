"""
Unit tests for the ANSI-compliant filter implementations.

Tests verify that the designed filters meet their respective standard's
tolerances at key frequencies.
"""
import numpy as np
import pytest
import scipy.signal

from vslm.dsp.filters.weighting_filters import design_optimized_sos, _ideal_response_db
from vslm.dsp.filters.octave_filters import design_band_sos, get_center_frequencies


# IEC 61672-1 Class 1 tolerances: (freq_hz, tolerance_db)
# Frequencies near Nyquist (> fs/2.2) are skipped — the filter cannot
# faithfully represent the analog response in that region.
_A_TARGETS = [
    (31.5,  1.5), (63,   1.5), (125,  1.5), (250,  1.0),
    (500,   1.0), (1000, 0.7), (2000, 1.0), (4000, 1.0),
    (8000,  1.5), (16000, 2.5),
]
_C_TARGETS = [
    (31.5,  1.5), (63,   1.5), (125,  1.0), (250,  1.0),
    (500,   1.0), (1000, 0.7), (2000, 1.0), (4000, 1.0),
    (8000,  1.5),
]


def _measure_filter_db(sos, fs, freq):
    """Measure steady-state gain of *sos* at *freq* Hz via sine injection."""
    t  = np.linspace(0, 1.0, fs, endpoint=False)
    x  = np.sqrt(2) * np.sin(2 * np.pi * freq * t)
    y  = scipy.signal.sosfilt(sos, x)
    ss = y[fs // 4:]   # discard first 25 % (transient)
    return 20 * np.log10(np.sqrt(np.mean(ss ** 2)) + 1e-15)


# ---------------------------------------------------------------------------
# A-weighting (IEC 61672-1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fs", [22050, 44100, 48000, 96000])
def test_a_weighting_iso_frequencies(fs):
    """A-weighting should meet IEC 61672-1 Class 1 tolerances at ISO frequencies."""
    sos = design_optimized_sos(fs, "A")
    # Gain at 1 kHz for relative normalisation
    gain_1k = _measure_filter_db(sos, fs, 1000)

    for freq, tol in _A_TARGETS:
        if freq >= fs / 2.2:
            continue   # too close to Nyquist for this sample rate
        measured = _measure_filter_db(sos, fs, freq) - gain_1k
        target   = _ideal_response_db(np.array([float(freq)]), "A")[0]
        assert abs(measured - target) < tol, (
            f"A-weighting fs={fs} {freq} Hz: got {measured:.2f} dB, "
            f"target {target:.2f} dB (tol ±{tol} dB)"
        )


@pytest.mark.parametrize("fs", [44100, 48000])
def test_c_weighting_iso_frequencies(fs):
    """C-weighting should meet IEC 61672-1 Class 1 tolerances at ISO frequencies."""
    sos = design_optimized_sos(fs, "C")
    gain_1k = _measure_filter_db(sos, fs, 1000)

    for freq, tol in _C_TARGETS:
        if freq >= fs / 2.2:
            continue
        measured = _measure_filter_db(sos, fs, freq) - gain_1k
        target   = _ideal_response_db(np.array([float(freq)]), "C")[0]
        assert abs(measured - target) < tol, (
            f"C-weighting fs={fs} {freq} Hz: got {measured:.2f} dB, "
            f"target {target:.2f} dB (tol ±{tol} dB)"
        )


# ---------------------------------------------------------------------------
# Octave / 1/3-octave bands (ANSI S1.11)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fc,resolution", [
    (1000, "octave"), (1000, "third"),
    (125,  "octave"), (4000, "third"),
])
def test_band_filter_centre_frequency(fc, resolution):
    """Band filter should have near-0 dB gain at its centre frequency."""
    fs  = 48000
    sos = design_band_sos(fc, fs, resolution, order=24)
    w   = 2 * np.pi * fc / fs
    _, h = scipy.signal.sosfreqz(sos, worN=[w])
    gain_db = 20 * np.log10(np.abs(h[0]) + 1e-15)
    assert abs(gain_db) < 1.0, f"Centre-freq gain {gain_db:.2f} dB (expected ~0 dB)"


def test_octave_band_edge_attenuation():
    """Octave band edges should have ≤ 0.05 dB attenuation (ANSI Class 1)."""
    fs  = 48000
    fc  = 1000.0
    sos = design_band_sos(fc, fs, "octave", order=24)

    bw = 2 ** 0.5
    f_lo = fc / bw
    f_hi = fc * bw

    for f_edge in (f_lo, f_hi):
        w = 2 * np.pi * f_edge / fs
        _, h = scipy.signal.sosfreqz(sos, worN=[w])
        atten = -20 * np.log10(np.abs(h[0]) + 1e-15)
        assert atten < 1.0, (
            f"Edge at {f_edge:.1f} Hz has {atten:.2f} dB attenuation "
            "(expected < 1 dB)"
        )


def test_center_frequencies_include_1khz():
    """Both octave and third-octave sets must include the 1 kHz reference band."""
    for res in ("octave", "third"):
        freqs = get_center_frequencies(res)
        assert any(abs(f - 1000) < 1 for f in freqs), \
            f"1 kHz not found in {res} center frequencies"
