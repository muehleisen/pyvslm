"""
Core streaming analysis engine.

All analysis is performed by reading the audio file in fixed-size blocks so
that arbitrarily large files can be processed without loading them into RAM.
Filter states are carried across block boundaries to produce a continuous,
artefact-free output.

Typical usage::

    processor = StreamProcessor("recording.wav", cal_factor=1.23)
    for result in processor.run_spl(block_size_ms=100, weighting="A"):
        print(result["time"], result["lp"])

References
----------
IEC 61672-1:2013  "Electroacoustics — Sound level meters — Part 1: Specifications"
  §3.9  Time-weighted sound level (Lp)
  §3.10 Equivalent continuous sound level (Leq)
  Annex B: Exponential time-weighting detector
P. Welch (1967) "The use of fast Fourier transform for the estimation of power
  spectra", IEEE Trans. Audio Electroacoust. 15(2):70-73.
"""
from __future__ import annotations

from pathlib import Path
from typing import Generator, Any

import numpy as np
import scipy.signal
import soundfile as sf

from .filters.weighting_filters import WeightingFilter, get_weighting_power_response
from .filters.octave_filters import OctaveFilterBank
from ..constants import Weighting, ResponseSpeed, BandResolution

# Mapping from user-facing window names to scipy identifiers.
# Window choice affects the PSD/spectrogram frequency resolution vs
# sidelobe trade-off (Harris, 1978 "On the use of windows for harmonic
# analysis with the DFT").
_WINDOW_MAP: dict[str, str] = {
    "Hanning":         "hann",
    "Hamming":         "hamming",
    "Flattop":         "flattop",      # best amplitude accuracy
    "Blackman":        "blackman",
    "Blackman-Harris": "blackmanharris",
    "Rectangular":     "boxcar",       # no windowing — highest frequency resolution
    "Bartlett":        "bartlett",
}


class TimeWeightingDetector:
    """
    Exponential-averaging detector implementing IEC 61672-1 time weightings.

    Methodology
    -----------
    IEC 61672-1 Annex B defines the time-weighted sound pressure level Lp as
    the output of a first-order exponential averager applied to the
    instantaneous squared (mean-square) pressure signal p²(t):

        s(t) = ∫₋∞ᵗ p²(τ) · exp[−(t − τ) / τ_w] dτ / τ_w

        Lp(t) = 10 · log10[ s(t) / p_ref² ]

    where τ_w is the time constant and p_ref = 20 µPa.

    **Discrete-time implementation**

    The continuous-time exponential averager is discretised as a
    first-order IIR lowpass filter on the squared signal:

        s[n] = s[n−1] + α · (p²[n] − s[n−1])

    with smoothing coefficient:

        α = 1 − exp(−1 / (fs · τ_w))

    This is the exact Z-transform equivalent of the continuous formula
    sampled at rate fs.

    **Impulse mode — asymmetric time constants**

    The Impulse weighting (IEC 61672-1 §3.9 Note 3) uses different
    time constants for rising and falling envelopes:
        τ_rise = 35 ms,  τ_fall = 1500 ms

    The detector therefore maintains two α values and selects based on
    whether the current sample exceeds the running state.

    **Time constants** (IEC 61672-1 Table 1):
        Fast:    τ = 125 ms
        Slow:    τ = 1 s
        Impulse: τ_rise = 35 ms,  τ_fall = 1.5 s

    Maintains state between calls so it can be used on consecutive blocks.
    """

    _TAU: dict[ResponseSpeed, tuple[float, float]] = {
        ResponseSpeed.FAST:    (0.125, 0.125),  # IEC 61672-1 Table 1
        ResponseSpeed.SLOW:    (1.0,   1.0),    # IEC 61672-1 Table 1
        ResponseSpeed.IMPULSE: (0.035, 1.5),    # IEC 61672-1 §3.9 Note 3
    }

    def __init__(self, fs: float, mode: ResponseSpeed = ResponseSpeed.FAST,
                 ref_pressure: float = 20e-6):
        self.ref_pressure = ref_pressure
        tau_rise, tau_fall = self._TAU[mode]
        # α = 1 − exp(−1 / (fs · τ)) — exact discrete equivalent of the
        # continuous RC low-pass with time constant τ
        self._alpha_rise = 1.0 - np.exp(-1.0 / (fs * tau_rise))
        self._alpha_fall = 1.0 - np.exp(-1.0 / (fs * tau_fall))
        self._state: float = 0.0  # running mean-square value s[n]

    def reset(self) -> None:
        """Reset detector state to zero (use between independent recordings)."""
        self._state = 0.0

    def process(self, chunk: np.ndarray) -> float:
        """
        Process one block of frequency-weighted audio samples.

        Applies the sample-by-sample exponential averager and returns the
        *peak* mean-square value encountered in the block, converted to dB SPL.

        The IEC 61672-1 definition of Lp is the *instantaneous* value of the
        averager, but for block-based processing we report the block peak so
        that fast transients are not lost when the block size is large.

        Returns:
            Lp for the block in dB SPL re 20 µPa.
        """
        ar   = self._alpha_rise
        af   = self._alpha_fall
        s    = self._state
        peak = 0.0

        for p2 in chunk ** 2:
            # First-order IIR update: select rise or fall alpha
            s = s + (ar if p2 > s else af) * (p2 - s)
            if s > peak:
                peak = s

        self._state = s  # carry state into next block
        # Convert peak mean-square to dB SPL:  Lp = 10·log10(s / p_ref²)
        # The 1e-30 guard prevents log10(0) when the signal is silent.
        return 10.0 * np.log10(peak / (self.ref_pressure ** 2) + 1e-30)


class StreamProcessor:
    """
    Reads a WAV file in blocks and applies the requested analysis.

    Args:
        filepath:   Path to the audio file.
        cal_factor: Linear amplitude scale factor applied to every sample.
                    Converts raw ADC units to Pascals.
    """

    def __init__(self, filepath: str | Path, cal_factor: float = 1.0):
        self.filepath   = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(self.filepath)
        self.cal_factor = cal_factor
        info            = sf.info(str(self.filepath))
        self.fs         = info.samplerate
        self.duration   = info.duration

    # ── Internal helpers ───────────────────────────────────────────────────

    def _read_mono_blocks(self, blocksize: int) -> Generator[np.ndarray, None, None]:
        """
        Yield calibrated mono blocks of *blocksize* samples.

        Multi-channel audio is downmixed to mono by averaging channels before
        applying the calibration factor.  The fill_value=0.0 argument pads the
        final (potentially short) block with silence so all blocks are the
        same length.
        """
        with sf.SoundFile(str(self.filepath)) as f:
            for chunk in f.blocks(blocksize=blocksize, always_2d=False, fill_value=0.0):
                if chunk.ndim > 1:
                    chunk = np.mean(chunk, axis=1)   # sum-and-scale downmix
                yield chunk * self.cal_factor

    # ── SPL / band analysis ────────────────────────────────────────────────

    def run_spl(
        self,
        block_size_ms: float           = 100.0,
        weighting: str | Weighting     = Weighting.A,
        do_bands: bool                 = False,
        band_resolution: str | BandResolution = BandResolution.OCTAVE,
        band_order: int                = 8,
        speed: str | ResponseSpeed     = ResponseSpeed.FAST,
        ref_pressure: float            = 20e-6,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Block-by-block SPL analysis (Lp and instantaneous Leq).

        Signal chain
        ------------
        raw samples  →  ×cal_factor  →  WeightingFilter (A/C/Z IIR)
                     →  TimeWeightingDetector            → Lp  (dB SPL)
                     →  mean(x²) → 10·log10(·/p_ref²)   → Leq (dB SPL)
                     →  OctaveFilterBank (optional)       → band levels (dB SPL)

        The **instantaneous Leq** per block is:

            Leq_block = 10 · log10[ mean(p²) / p_ref² ]

        where mean(p²) is the mean squared pressure of the frequency-weighted
        signal over the block duration.  This is the short-term equivalent
        level; integrating it over multiple blocks gives the overall Leq.

        The weighting filter and band filter banks are seeded with the first
        audio block before the main loop (see WeightingFilter.initialize_state)
        to suppress the onset transient.

        Yields one result dict per block::

            {
                "time":       float,        # block start time (s)
                "leq":        float,        # instantaneous Leq (dB SPL)
                "lp":         float,        # time-weighted Lp (dB SPL)
                "bands":      np.ndarray,   # per-band Leq (dB SPL) — if do_bands
                "band_freqs": np.ndarray,   # centre frequencies (Hz) — if do_bands
            }
        """
        block_samples = int(self.fs * block_size_ms / 1000.0)
        if block_samples < 1:
            raise ValueError("block_size_ms too small for this sample rate.")

        w_filter  = WeightingFilter(self.fs, str(weighting))
        detector  = TimeWeightingDetector(self.fs, ResponseSpeed(str(speed)), ref_pressure)
        band_bank = OctaveFilterBank(self.fs, str(band_resolution), band_order) if do_bands else None

        # Seed filter states with the first block to eliminate onset transient
        with sf.SoundFile(str(self.filepath)) as f:
            seed_raw = f.read(block_samples, always_2d=False, fill_value=0.0)
        if seed_raw.ndim > 1:
            seed_raw = np.mean(seed_raw, axis=1)
        seed = seed_raw * self.cal_factor

        w_filter.initialize_state(seed)
        if band_bank:
            band_bank.initialize_state(seed)

        current_time = 0.0
        for chunk in self._read_mono_blocks(block_samples):
            # Apply frequency weighting (IIR filter, state preserved)
            weighted = w_filter.process_chunk(chunk)

            # Instantaneous Leq: energy-average over block
            # Leq = 10·log10[ (1/T)·∫p²dt / p_ref² ]
            # In discrete form: mean(p²[n]) / p_ref²
            ms  = np.mean(weighted ** 2)
            leq = 10.0 * np.log10(ms / (ref_pressure ** 2) + 1e-30)

            # Time-weighted Lp from exponential detector
            lp = detector.process(weighted)

            result: dict[str, Any] = {"time": current_time, "leq": leq, "lp": lp}

            if band_bank:
                # Band analysis uses the *unweighted* signal — the band filters
                # provide the spectral selectivity; A/C weighting would be
                # double-applied if done here as well.
                band_out = band_bank.process_chunk(chunk)
                # Per-band Leq: mean squared output of each bandpass filter
                ms_bands = np.mean(band_out ** 2, axis=0)
                result["bands"]      = 10.0 * np.log10(ms_bands / (ref_pressure ** 2) + 1e-30)
                result["band_freqs"] = band_bank.frequencies

            yield result
            current_time += block_size_ms / 1000.0

    # ── PSD ───────────────────────────────────────────────────────────────

    def run_psd(
        self,
        nfft: int                  = 4096,
        window: str                = "Hanning",
        weighting: str | Weighting = Weighting.A,
    ) -> Generator[int | dict[str, Any], None, None]:
        """
        Power spectral density via Welch's method, averaged over the file.

        Methodology
        -----------
        **Welch's method** (P. Welch, 1967) reduces the variance of the raw
        periodogram by averaging multiple overlapping short-time FFTs:

        1. Divide the signal into overlapping segments of length NFFT.
        2. Apply the selected window function to each segment.
        3. Compute the squared magnitude of the DFT of each windowed segment.
        4. Average the periodograms.

        The overlap is fixed at NFFT/2 (50 %), which is the standard choice
        for Hanning/Hamming windows; it ensures that the windowed segments
        together cover all samples with approximately uniform weighting.

        scipy.signal.welch is called with scaling='density', which normalises
        by the window power and sample rate to give units of V²/Hz (here Pa²/Hz
        after calibration).

        **Chunked processing**: the file is read in 10-second blocks.  Each
        block is passed to welch independently, and the resulting periodograms
        are *summed* then divided by the block count to give the time-averaged
        PSD.  This is equivalent to averaging all Welch segments across the
        whole file, assuming the signal is stationary.

        **Frequency-domain weighting**: A or C weighting is applied by
        multiplying the averaged PSD by the squared magnitude of the ideal
        analogue weighting response |H(f)|² (see get_weighting_power_response).
        This is mathematically equivalent to filtering the time-domain signal
        before computing the PSD.

        Yields integers 0–99 (percent complete) then a single result dict::

            {
                "type":      "psd",
                "freqs":     np.ndarray,   # frequency axis (Hz)
                "pxx":       np.ndarray,   # weighted PSD (Pa²/Hz)
                "nfft":      int,
                "window":    str,
                "weighting": str,
            }

        Raises:
            ValueError: If the file is too short for the requested FFT size.
        """
        scipy_win  = _WINDOW_MAP.get(window, "hann")
        chunk_size = int(self.fs * 10.0)   # 10-second processing blocks
        noverlap   = nfft // 2             # 50 % overlap
        total_samp = int(self.duration * self.fs)
        processed  = 0
        pxx_sum    = None
        freqs      = None
        count      = 0

        for chunk in self._read_mono_blocks(chunk_size):
            if len(chunk) < nfft:
                continue   # skip last partial block if shorter than NFFT
            f_c, p_c = scipy.signal.welch(
                chunk, fs=self.fs, window=scipy_win,
                nperseg=nfft, noverlap=noverlap, nfft=nfft, scaling="density",
            )
            # Accumulate for time-average: PSD_avg = (1/K) · Σ PSD_k
            pxx_sum = p_c if pxx_sum is None else pxx_sum + p_c
            freqs   = f_c
            count  += 1
            processed += len(chunk)
            yield int(100 * processed / total_samp)

        if pxx_sum is None or count == 0:
            raise ValueError("File too short for requested FFT size.")

        pxx_avg = pxx_sum / count

        # Apply frequency-domain weighting: multiply by |H_weighting(f)|²
        # Using the ideal analogue response avoids any digital filter
        # roll-off artefacts near Nyquist.
        w_resp = get_weighting_power_response(freqs, str(weighting))
        yield {
            "type":      "psd",
            "freqs":     freqs,
            "pxx":       pxx_avg * w_resp,
            "nfft":      nfft,
            "window":    window,
            "weighting": str(weighting),
        }

    # ── Spectrogram ───────────────────────────────────────────────────────

    def run_spectrogram(
        self,
        nfft: int                  = 512,
        dt: float                  = 1.0,
        window: str                = "Hamming",
        weighting: str | Weighting = Weighting.A,
    ) -> Generator[int | dict[str, Any], None, None]:
        """
        Short-time spectrogram — one Welch-averaged PSD column per *dt* seconds.

        Methodology
        -----------
        Unlike the broadband PSD which averages the entire recording, the
        spectrogram preserves time variation by computing a separate Welch PSD
        for each non-overlapping time slice of duration *dt*.

        For each slice:
          1. Read *dt* seconds of calibrated audio (or NFFT samples, whichever
             is larger, to ensure welch has enough data).
          2. Call scipy.signal.welch with 50 % overlap and the selected window.
          3. Store the resulting PSD vector as one column of the output matrix.

        The result is a 2-D matrix S[time, frequency] of PSD values (Pa²/Hz),
        weighted by |H(f)|² post-hoc (same approach as run_psd).

        **Display**: the matrix is typically rendered as pcolormesh with a
        dB colour scale:  S_dB = 10·log10(S / p_ref²).

        Yields integers 0–99 (percent complete) then a single result dict::

            {
                "type":       "spectrogram",
                "times":      np.ndarray,        # time axis (s)
                "freqs":      np.ndarray,         # frequency axis (Hz)
                "pxx_matrix": np.ndarray,         # (n_times, n_freqs), Pa²/Hz
                "nfft":       int,
                "dt":         float,
                "weighting":  str,
            }

        Raises:
            ValueError: If the file is too short.
        """
        scipy_win    = _WINDOW_MAP.get(window, "hamming")
        noverlap     = nfft // 2
        # Ensure the chunk is at least NFFT samples so welch always has data
        chunk_size   = max(int(self.fs * dt), nfft)
        total_samp   = int(self.duration * self.fs)
        processed    = 0
        current_time = 0.0
        pxx_cols     = []
        time_axis    = []
        freqs        = None

        for chunk in self._read_mono_blocks(chunk_size):
            if len(chunk) < nfft:
                continue
            f_c, p_c = scipy.signal.welch(
                chunk, fs=self.fs, window=scipy_win,
                nperseg=nfft, noverlap=noverlap, nfft=nfft, scaling="density",
            )
            if freqs is None:
                freqs = f_c
            pxx_cols.append(p_c)
            time_axis.append(current_time)
            current_time += len(chunk) / self.fs
            processed    += len(chunk)
            yield int(100 * processed / total_samp)

        if not pxx_cols:
            raise ValueError("File too short for spectrogram analysis.")

        # Stack columns into (n_times, n_freqs) matrix
        S = np.array(pxx_cols)

        # Apply frequency-domain weighting: broadcast |H(f)|² across time axis
        w_resp = get_weighting_power_response(freqs, str(weighting))
        S *= w_resp[np.newaxis, :]

        yield {
            "type":       "spectrogram",
            "times":      np.array(time_axis),
            "freqs":      freqs,
            "pxx_matrix": S,
            "nfft":       nfft,
            "dt":         dt,
            "weighting":  str(weighting),
        }
