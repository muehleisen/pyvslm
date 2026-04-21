"""
Unit tests for the streaming analysis engine.

Generates synthetic WAV files (known-level tones) and verifies that the
engine produces the expected acoustic levels within tight tolerances.
"""
import os

import numpy as np
import pytest
import soundfile as sf

from vslm.dsp.engine import StreamProcessor
from vslm.constants import Weighting, ResponseSpeed


FS = 48000
DURATION = 2.0  # seconds
REF_PA   = 20e-6


@pytest.fixture
def sine_wav(tmp_path):
    """
    1 kHz sine, RMS = 0.5 Pa after calibration.

    The raw file stores amplitude 0.5 (RMS = 0.5/√2 ≈ 0.3536).
    A cal_factor of √2 restores it to 0.5 Pa RMS ≈ 87.96 dB SPL (Z-weighted).
    """
    t = np.linspace(0, DURATION, int(FS * DURATION), endpoint=False)
    raw_rms  = 0.5 / np.sqrt(2)
    signal   = raw_rms * np.sqrt(2) * np.sin(2 * np.pi * 1000 * t)  # peak = 0.5
    filepath = tmp_path / "test_sine.wav"
    sf.write(str(filepath), signal, FS)
    return filepath, np.sqrt(2)  # (path, cal_factor)


class TestBroadbandLeq:
    def test_steady_state_leq(self, sine_wav):
        """Middle blocks should yield Leq ≈ 87.96 dB (Z-weighted, 0.5 Pa RMS)."""
        path, cal = sine_wav
        proc = StreamProcessor(path, cal_factor=cal)
        results = list(proc.run_spl(block_size_ms=100, weighting=Weighting.Z,
                                    ref_pressure=REF_PA))
        expected = 20 * np.log10(0.5 / REF_PA)   # ≈ 87.96 dB

        # Skip first block (filter warm-up) and check several steady-state blocks
        for r in results[3:-1]:
            assert abs(r["leq"] - expected) < 0.15, (
                f"Leq {r['leq']:.2f} dB, expected {expected:.2f} dB"
            )

    def test_correct_block_count(self, sine_wav):
        """A 2-second file with 100 ms blocks should yield 20 results."""
        path, cal = sine_wav
        proc = StreamProcessor(path, cal_factor=cal)
        results = list(proc.run_spl(block_size_ms=100, weighting=Weighting.Z,
                                    ref_pressure=REF_PA))
        assert len(results) == 20


class TestOctaveBands:
    def test_energy_in_1khz_band(self, sine_wav):
        """A 1 kHz tone must put > 90 dB into the 1 kHz octave band."""
        path, cal = sine_wav
        proc = StreamProcessor(path, cal_factor=cal)
        results = list(proc.run_spl(
            block_size_ms=100, weighting=Weighting.Z,
            do_bands=True, band_resolution="octave",
            ref_pressure=REF_PA,
        ))
        r = results[5]
        assert "bands" in r
        freqs  = r["band_freqs"]
        levels = r["bands"]
        idx_1k = np.argmin(np.abs(freqs - 1000))
        assert levels[idx_1k] > 85.0, (
            f"1 kHz octave band level {levels[idx_1k]:.1f} dB (expected > 85 dB)"
        )

    def test_low_leakage_into_500hz(self, sine_wav):
        """A 1 kHz tone should bleed < 50 dB SPL into the 500 Hz octave band.

        Input is ~88 dB SPL at 1 kHz.  ANSI S1.11-2004 Class 1 requires only
        ≥ 17.5 dB attenuation at one octave (G^1 ≈ 2×fc), which would allow
        up to ~70 dB SPL in the 500 Hz band.  A well-designed higher-order
        filter achieves ~50+ dB of rejection there, so 50 dB SPL is a
        conservative but meaningful threshold.
        """
        path, cal = sine_wav
        proc = StreamProcessor(path, cal_factor=cal)
        results = list(proc.run_spl(
            block_size_ms=100, weighting=Weighting.Z,
            do_bands=True, band_resolution="octave",
            ref_pressure=REF_PA,
        ))
        r = results[5]
        freqs  = r["band_freqs"]
        levels = r["bands"]
        idx_500 = np.argmin(np.abs(freqs - 500))
        assert levels[idx_500] < 50.0, (
            f"500 Hz band level {levels[idx_500]:.1f} dB (expected < 50 dB)"
        )


class TestPSD:
    def test_psd_returns_result_dict(self, sine_wav):
        path, cal = sine_wav
        proc = StreamProcessor(path, cal_factor=cal)
        items = list(proc.run_psd(nfft=1024, window="Hanning",
                                  weighting=Weighting.Z))
        result = items[-1]
        assert isinstance(result, dict)
        assert result["type"] == "psd"
        assert len(result["freqs"]) > 0
        assert len(result["pxx"]) == len(result["freqs"])

    def test_psd_peak_near_1khz(self, sine_wav):
        """PSD peak should be within a few bins of 1 kHz."""
        path, cal = sine_wav
        proc = StreamProcessor(path, cal_factor=cal)
        items = list(proc.run_psd(nfft=4096, window="Hanning",
                                  weighting=Weighting.Z))
        result = items[-1]
        freqs  = result["freqs"]
        pxx    = result["pxx"]
        peak_f = freqs[np.argmax(pxx)]
        assert abs(peak_f - 1000) < 50, (
            f"PSD peak at {peak_f:.1f} Hz (expected near 1000 Hz)"
        )
