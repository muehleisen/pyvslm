"""
ANSI S1.11-2004 octave and 1/3-octave band filter bank.

Filters are Butterworth bandpass designs targeting Class 1 compliance
(≤ 0.05 dB attenuation at nominal band-edge frequencies).  Filter states
are maintained between calls to :meth:`OctaveFilterBank.process_chunk` so
that the bank can process arbitrarily large files in fixed-size blocks.

References
----------
ANSI S1.11-2004  "Specification for Octave-Band and Fractional-Octave-Band
  Analog and Digital Filters"
IEC 61260-1:2014  "Electroacoustics — Octave-band and fractional-octave-band
  filters — Part 1: Specifications"
"""
import numpy as np
import scipy.signal


def get_center_frequencies(resolution: str = "octave", base: int = 10) -> np.ndarray:
    """
    Return ANSI S1.11-2004 preferred centre frequencies.

    Methodology
    -----------
    ANSI S1.11 defines two numbering systems for band centre frequencies.

    **Base-10 (preferred decade) series** — used by default:
        fm = f_ref · 10^(3·x / 10)   for octave bands
        fm = f_ref · 10^(x / 10)     for 1/3-octave bands

    **Base-2 (binary) series**:
        fm = f_ref · 2^x             for octave bands
        fm = f_ref · 2^(x / 3)      for 1/3-octave bands

    In both cases f_ref = 1000 Hz is the reference frequency and x is an
    integer band index.  The index ranges are chosen to span the audible
    spectrum: roughly 16 Hz – 16 kHz for octave, 12.5 Hz – 20 kHz for thirds.

    The base-10 series produces the "preferred" nominal frequencies
    (e.g. 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000 Hz for octaves)
    that appear on standard SLM displays.

    Args:
        resolution: ``'octave'`` or ``'third'``.
        base:       10 (preferred decade series) or 2 (binary series).

    Returns:
        Array of centre frequencies in Hz.
    """
    f_ref = 1000.0
    if resolution == "octave":
        x  = np.arange(-6, 5)     # indices −6 … +4 give 11 bands (~16 Hz – 16 kHz)
        fm = f_ref * ((10.0 ** (3.0 * x / 10.0)) if base == 10 else (2.0 ** x))
    elif resolution == "third":
        x  = np.arange(-19, 14)   # indices −19 … +13 give 33 bands (~12.5 Hz – 20 kHz)
        fm = f_ref * ((10.0 ** (x / 10.0)) if base == 10 else (2.0 ** (x / 3.0)))
    else:
        raise ValueError(f"resolution must be 'octave' or 'third', got {resolution!r}")
    return fm


def design_band_sos(fc: float, fs: float, resolution: str = "third",
                    order: int = 12) -> np.ndarray:
    """
    Design a single Butterworth bandpass filter compliant with ANSI S1.11 Class 1.

    Methodology
    -----------
    **Band-edge frequencies**

    ANSI S1.11 / IEC 61260-1 defines band edges using the preferred base-10
    ratio G = 10^(3/10), where b = 1 for octave and b = 3 for 1/3-octave:

        f_lo = fc · G^(−1/(2b)) = fc / G^(1/(2b))
        f_hi = fc · G^(+1/(2b))

    For octave (b=1):      bw = G^(1/2) = 10^(3/20) ≈ 1.4125
    For 1/3-octave (b=3):  bw = G^(1/6) = 10^(1/20) ≈ 1.1220

    These band edges are used directly as the Butterworth −3 dB cutoff
    frequencies.  The ANSI Class 1 tolerance at the band edge (G^±0.5)
    allows attenuation between −0.3 dB and +5.0 dB, so placing the −3 dB
    point there is well within spec.  For a high-order filter the transition
    band then rolls off rapidly enough to meet all stopband requirements.

    **Filter design**

    scipy.signal.butter designs the Butterworth SOS directly in the digital
    domain using the bilinear transform internally, which is appropriate here
    because the band edges are well below Nyquist for all supported rates.

    Args:
        fc:         Nominal centre frequency in Hz.
        fs:         Sample rate in Hz.
        resolution: ``'octave'`` or ``'third'``.
        order:      Filter order (default 8).

    Returns:
        SOS array of shape (order, 6).
    """
    # ANSI S1.11 preferred base-10 band edges: G = 10^(3/10)
    # octave (b=1):     edges at fc · G^(±1/2) = fc · 10^(±3/20)
    # 1/3-octave (b=3): edges at fc · G^(±1/6) = fc · 10^(±1/20)
    G   = 10.0 ** (3.0 / 10.0)
    bw  = G ** (0.5 if resolution == "octave" else 1.0 / 6.0)
    f_lo = fc / bw
    f_hi = fc * bw

    # Cap upper edge to 99 % of Nyquist to stay within the stable region of
    # the bilinear transform
    nyq  = fs / 2.0
    f_hi = min(f_hi, nyq * 0.99)
    if f_lo >= f_hi:
        raise ValueError(
            f"Band at {fc:.1f} Hz: design edge {f_hi:.1f} Hz too close to Nyquist {nyq:.1f} Hz"
        )

    # scipy butter with output='sos' returns second-order sections directly,
    # which avoids the numerical instability of ba (polynomial) form for
    # high-order filters.
    return scipy.signal.butter(
        N=order, Wn=[f_lo, f_hi], btype="bandpass", fs=fs, output="sos"
    )


class OctaveFilterBank:
    """
    Bank of stateful ANSI S1.11 bandpass filters for streaming analysis.

    Only bands whose upper edge falls below 95 % of Nyquist are included,
    so the bank automatically adapts to the file's sample rate.

    Each band carries its own filter state (zi) between calls to
    :meth:`process_chunk`, enabling block-by-block processing of arbitrarily
    long recordings without resetting the filter memory.
    """

    def __init__(self, fs: float, resolution: str = "octave", order: int = 12):
        self.fs         = fs
        self.resolution = resolution

        # Upper limit: exclude bands whose upper ANSI edge would exceed 95 %
        # of Nyquist.  Beyond that point Butterworth accuracy degrades rapidly.
        G           = 10.0 ** (3.0 / 10.0)
        bw          = G ** (0.5 if resolution == "octave" else 1.0 / 6.0)
        upper_limit = (fs / 2.0) / bw * 0.95

        all_centers   = get_center_frequencies(resolution)
        valid         = all_centers[all_centers < upper_limit]
        self.frequencies = valid

        self._bands: list[dict] = []
        for fc in valid:
            try:
                sos = design_band_sos(fc, fs, resolution, order)
                # sosfilt_zi returns the steady-state initial conditions for a
                # unit-step input — a sensible starting point for the state vector
                self._bands.append({
                    "fc":  fc,
                    "sos": sos,
                    "zi":  scipy.signal.sosfilt_zi(sos),
                })
            except ValueError as e:
                print(f"Warning: skipped band at {fc:.1f} Hz — {e}")

    def reset(self) -> None:
        """Reset all filter states to zero (use between independent recordings)."""
        for b in self._bands:
            b["zi"] = scipy.signal.sosfilt_zi(b["sos"])

    def initialize_state(self, seed: np.ndarray) -> None:
        """
        Warm up all filters using a forward-then-backward pass on *seed*.

        Methodology
        -----------
        Same forward-backward technique as WeightingFilter.initialize_state:
        the forward pass from zero state populates zi_fwd at the end of the
        seed block; the backward pass from zi_fwd estimates what the state
        would have been at the *start* of the block had the filter been running
        on stationary content, suppressing the onset transient.

        Call this on the first audio block before entering the main processing
        loop.
        """
        for b in self._bands:
            _, zi_fwd = scipy.signal.sosfilt(b["sos"], seed, zi=np.zeros_like(b["zi"]))
            _, zi_bwd = scipy.signal.sosfilt(b["sos"], seed[::-1], zi=zi_fwd)
            b["zi"] = zi_bwd

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Filter *chunk* through every band, updating each band's state.

        Args:
            chunk: 1-D audio block of length N.

        Returns:
            Array of shape (N, n_bands) — column i is the output of band i.
            The caller can then compute per-band RMS, Leq, etc. over the block.
        """
        n_bands = len(self._bands)
        output  = np.zeros((len(chunk), n_bands), dtype=chunk.dtype)
        for i, b in enumerate(self._bands):
            out, b["zi"] = scipy.signal.sosfilt(b["sos"], chunk, zi=b["zi"])
            output[:, i] = out
        return output
