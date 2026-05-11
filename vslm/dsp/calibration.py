"""
Utilities for computing and applying acoustic calibration factors.

Acoustic calibration maps raw ADC sample values (dimensionless, ±1.0 full
scale for most WAV files) to physical pressure units (Pascals).  A linear
scale factor *k* is determined so that:

    p(t) = k · x(t)     [Pascals]

where x(t) is the raw sample value from the file and p(t) is the estimated
sound pressure.

Reference
---------
IEC 61672-1:2013  §5.5  "Calibration of sound level meters"
  — recommends using a pistonphone or sound calibrator at a known SPL
    (typically 94 dB SPL at 1 kHz) to determine the instrument sensitivity.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def compute_selection_rms(filepath: Path, start_s: float, end_s: float) -> float:
    """
    Compute the RMS amplitude of an audio selection without loading the whole file.

    Methodology
    -----------
    The RMS (root-mean-square) value of a discrete signal x[n] over N samples is:

        x_rms = sqrt( (1/N) · Σ x[n]² )

    This is computed in one pass over the selected frames using NumPy:

        np.sqrt(np.mean(x**2))

    For multi-channel files the channels are averaged to mono *before* computing
    the RMS, which is consistent with how the analysis engine processes the file.

    The frame indices are derived from the time positions:
        start_frame = int(start_s · sample_rate)
        end_frame   = int(end_s   · sample_rate)

    Using int() (floor) ensures we stay within the valid frame range.

    The 1e-15 guard in the return value prevents a zero RMS (which would cause
    division by zero in factor_from_reference) when the selection is silent.

    Args:
        filepath: Path to the WAV file.
        start_s:  Selection start time in seconds.
        end_s:    Selection end time in seconds.

    Returns:
        RMS amplitude (linear, in raw sample units — not dB).

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        ValueError:        If end_s ≤ start_s (non-positive duration).
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with sf.SoundFile(str(filepath)) as f:
        sr          = f.samplerate
        start_frame = int(start_s * sr)
        end_frame   = int(end_s   * sr)
        n_frames    = end_frame - start_frame
        if n_frames <= 0:
            raise ValueError("Selection duration must be positive.")
        f.seek(start_frame)
        data = f.read(n_frames, always_2d=True)

    # Downmix to mono by averaging channels (same as StreamProcessor)
    mono = np.mean(data, axis=1) if data.shape[1] > 1 else data[:, 0]
    return float(np.sqrt(np.mean(mono ** 2)) + 1e-15)


def factor_from_reference(measured_rms: float, ref_level_db: float,
                           ref_pressure: float = 20e-6) -> float:
    """
    Calculate a linear calibration factor from a reference-tone measurement.

    Methodology
    -----------
    A pistonphone or acoustic calibrator produces a known sound pressure level
    L_ref (dB SPL re 20 µPa).  The equivalent RMS pressure in Pascals is:

        p_target = p_ref · 10^(L_ref / 20)

    The recording of that tone has an RMS of *measured_rms* in raw sample
    units.  The calibration factor k that maps raw units to Pascals is simply:

        k = p_target / measured_rms

    After applying k, the calibrated signal satisfies:

        RMS(k · x) = k · measured_rms = p_target

    and therefore:

        SPL = 20 · log10(p_target / p_ref) = L_ref    ✓

    This is the single-point sensitivity calibration method described in
    IEC 61672-1 §5.5.  It assumes the microphone/ADC chain is linear, which
    is a valid assumption for typical measurement microphones in the normal
    operating range.

    Args:
        measured_rms:  RMS of the uncalibrated recording of the reference tone
                       (in raw sample units, as returned by compute_selection_rms).
        ref_level_db:  Known acoustic level of the calibration tone (dB SPL).
        ref_pressure:  Reference pressure in Pa (default 20 µPa = 0 dB SPL).

    Returns:
        Linear calibration scale factor k  [Pa / sample_unit].
    """
    # Target RMS pressure corresponding to the reference level
    target_pa = ref_pressure * 10.0 ** (ref_level_db / 20.0)
    return target_pa / measured_rms
