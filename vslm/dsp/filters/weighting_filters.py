"""
IEC 61672-1:2013 compliant A and C frequency weighting filters.

Filters are designed dynamically per sample rate using a hybrid bilinear / MZT
transform with minimax optimisation over ISO preferred frequencies.  A passthrough
path is used for Z (flat) weighting.

Reference
---------
IEC 61672-1:2013  "Electroacoustics — Sound level meters — Part 1: Specifications"
  Annex E gives the analogue prototype transfer functions for A and C weighting.
"""
import numpy as np
import scipy.signal
import scipy.optimize

SUPPORTED_FS = [22050, 44100, 48000, 96000, 192000]

# ISO 266:1997 preferred centre frequencies used as optimisation targets
ISO_FREQS = np.array([
    10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80,
    100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
    1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300,
    8000, 10000, 12500, 16000, 20000,
])

# ── IEC 61672-1 Annex E pole frequencies (Hz) ──────────────────────────────
# These four characteristic frequencies define both the A and C weighting
# analogue prototypes.  Values are normative constants from the standard.
#   f1 =  20.598 997 Hz  — low-frequency double pole
#   f2 = 107.652 65  Hz  — A-weighting mid pole (absent from C-weighting)
#   f3 = 737.862 23  Hz  — A-weighting mid pole (absent from C-weighting)
#   f4 = 12 194.217  Hz  — high-frequency double pole
_F1 = 20.598997
_F2 = 107.65265
_F3 = 737.86223
_F4 = 12194.217


def _ideal_response_db(freqs: np.ndarray, w_type: str) -> np.ndarray:
    """
    Theoretical IEC 61672-1 weighting response in dB, normalised to 0 dB at 1 kHz.

    Methodology
    -----------
    IEC 61672-1 Annex E defines the analogue A-weighting transfer function as:

        H_A(f) = (f4² · f²)²
                 ─────────────────────────────────────────────────────────────
                 (f² + f1²) · √[(f² + f2²)(f² + f3²)] · (f² + f4²)

    and C-weighting as:

        H_C(f) = f4² · f²
                 ─────────────────────
                 (f² + f1²) · (f² + f4²)

    Both are normalised so that H(1000 Hz) = 0 dB by dividing by the gain
    evaluated at the 1 kHz reference frequency.

    The f² substitution (x2 = f²) avoids repeated squaring and makes the
    formula structurally identical to the standard's polynomial form.

    Note: DC (f=0) would cause division by zero; the np.where guard replaces
    zero with the smallest positive float before squaring.
    """
    # Guard against DC bin (f=0) which causes log10(0) = -inf
    f2 = np.where(freqs == 0, np.finfo(float).tiny, freqs) ** 2
    r2 = 1000.0 ** 2   # reference frequency squared

    if w_type == "A":
        def _gain(x2):
            # Numerator:   f4² · (f²)²  ← the two zeros at s=0 in the analogue prototype
            # Denominator: double pole at f1, single poles at f2 and f3, double pole at f4
            return (_F4 ** 2) * (x2 ** 2) / (
                (x2 + _F1 ** 2)
                * np.sqrt((x2 + _F2 ** 2) * (x2 + _F3 ** 2))
                * (x2 + _F4 ** 2)
            )
    else:  # C
        def _gain(x2):
            # C-weighting omits the f2 and f3 poles; numerator has one zero pair
            return (_F4 ** 2) * x2 / ((x2 + _F1 ** 2) * (x2 + _F4 ** 2))

    # Normalise to 0 dB at 1 kHz: 20·log10(H(f) / H(1000))
    return 20.0 * np.log10(_gain(f2) / _gain(r2))


def get_weighting_power_response(freqs: np.ndarray, w_type: str | object) -> np.ndarray:
    """
    Linear power gain |H(f)|² for frequency-domain (PSD/spectrogram) weighting.

    Methodology
    -----------
    PSD values are in units of Pa²/Hz.  Applying A or C weighting in the
    frequency domain requires multiplying each bin by the squared magnitude
    of the weighting filter's frequency response:

        PSD_weighted(f) = PSD(f) · |H(f)|²

    Since _ideal_response_db returns the response in dB (20·log10 scale),
    we convert to a linear power factor via:

        |H(f)|² = 10^(dB / 10)

    This is mathematically exact for the analogue prototype; using the ideal
    response rather than the digital filter avoids the slight high-frequency
    error that the digital design introduces near Nyquist.

    Args:
        freqs:  Frequency array in Hz.
        w_type: 'A', 'C', or 'Z'.  Also accepts Weighting enum instances.

    Returns:
        Array of the same length as *freqs* with linear power weights.
    """
    wt = str(w_type).upper()
    if wt in ("Z", "FLAT", "NONE"):
        return np.ones_like(freqs, dtype=float)
    db = _ideal_response_db(freqs, wt)
    return 10.0 ** (db / 10.0)


def _parametric_biquad_sos(f0: float, Q: float, gain_db: float, fs: float) -> np.ndarray:
    """
    Single parametric-EQ biquad as a (1, 6) SOS row.

    Methodology
    -----------
    Uses the Audio EQ Cookbook formulation (R. Bristow-Johnson, 1994):

        H(z) = (1 + α·A  +  (−2·cos ω₀)z⁻¹  +  (1 − α·A)z⁻²)
               ──────────────────────────────────────────────────
               (1 + α/A  +  (−2·cos ω₀)z⁻¹  +  (1 − α/A)z⁻²)

    where:
        ω₀    = 2π·f0 / fs          (digital centre frequency, radians)
        α     = sin(ω₀) / (2·Q)     (bandwidth parameter)
        A     = 10^(gain_dB / 40)   (linear amplitude gain; /40 because it
                                      appears as √A in the biquad coefficients)

    This filter provides a symmetric boost/cut centred at f0 with quality
    factor Q.  It is used as a correction term in the optimisation to trim
    residual error in the high-frequency region of the weighting filter.
    """
    if f0 <= 0 or f0 >= fs / 2:
        # Outside usable range — return unity (pass-through) biquad
        return np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    A  = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2.0 * Q)
    b = np.array([1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A])
    return np.concatenate([b / a[0], a / a[0]]).reshape(1, 6)


def design_optimized_sos(fs: float, w_type: str = "A") -> np.ndarray:
    """
    Design an IEC 61672-1 weighting filter as SOS coefficients.

    Methodology — Hybrid MZT + Minimax Optimisation
    ------------------------------------------------
    A pure bilinear transform of the analogue prototype introduces
    frequency-axis warping that becomes significant near Nyquist, causing
    errors of ~1–2 dB at 8–16 kHz for common sample rates (44.1/48 kHz).
    A pure Matched-Z Transform (MZT) for the high-frequency pole avoids
    warping but shifts DC gain slightly.

    The chosen hybrid approach (following the Gemini implementation):

    1. **Low-frequency sections — Bilinear transform**
       The f1, f2, f3 poles are well below Nyquist for all supported sample
       rates, so bilinear warping is negligible there.  scipy.signal.bilinear_zpk
       applies the standard substitution:

           s  →  2·fs · (z − 1) / (z + 1)

       Poles at −ω1, −ω2, −ω3 (rad/s) map to stable z-plane poles; each pair
       of zeros at s=0 maps to a zero at z=−1 (Nyquist).

    2. **High-frequency section — MZT with free parameter f4**
       The f4 = 12 194 Hz double pole is close to Nyquist at 44.1/48 kHz.
       The Matched-Z Transform places the digital pole at:

           p_d = exp(−ω4 / fs)

       giving a second-order section:

           H_hf(z) = k · (1 − 0·z⁻¹ − 0·z⁻²)
                     ──────────────────────────
                     (1 − 2·p_d·z⁻¹ + p_d²·z⁻²)

       with normalisation k = (1 − p_d)².  f4 is treated as a *free parameter*
       in the optimisation so the solver can adjust it to compensate for the
       bilinear warping in the low-frequency sections.

    3. **Parametric correction biquad**
       A single parametric-EQ biquad (see _parametric_biquad_sos) is added
       to trim any residual high-frequency error.  Its centre frequency,
       gain, and Q are also free optimisation parameters.

    4. **Minimax optimisation — Nelder-Mead**
       The cost function is the *maximum weighted absolute error* (minimax /
       L∞ norm) between the composite digital filter and the ideal analogue
       response across the ISO 266 preferred frequencies.  Errors above 1 kHz
       are penalised 50× more heavily because the standard's Class 1 tolerance
       is tighter in the upper mid-band.  Nelder-Mead is used because the cost
       surface has no gradient information and the problem is low-dimensional
       (4 free parameters).

    5. **Final normalisation**
       The assembled SOS chain is evaluated at exactly 1 kHz and the leading
       biquad's numerator is scaled so that H(1000 Hz) = 1 (0 dB).

    Args:
        fs:     Sample rate in Hz.
        w_type: 'A' or 'C'.

    Returns:
        SOS array of shape (N_sections, 6), normalised to 0 dB at 1 kHz.
    """
    # Convert pole frequencies to rad/s for bilinear_zpk
    w1, w2, w3 = [2.0 * np.pi * f for f in (_F1, _F2, _F3)]

    # ── Step 1: low-frequency fixed sections (bilinear transform) ──────────
    lf_blocks = []
    if w_type == "A":
        # A-weighting analogue prototype:
        #   zeros at s=0 (×4 total), poles at −ω1 (×2), −ω2, −ω3, −ω4 (×2)
        # Split into two biquad sections so each has equal pole/zero order:
        #   Section 1: 2 zeros at 0, poles at −ω1 (double)
        #   Section 2: 2 zeros at 0, poles at −ω2 and −ω3
        z, p, k = scipy.signal.bilinear_zpk([0, 0], [-w1, -w1], 1.0, fs)
        lf_blocks.append(scipy.signal.zpk2sos(z, p, k))
        z, p, k = scipy.signal.bilinear_zpk([0, 0], [-w2, -w3], 1.0, fs)
        lf_blocks.append(scipy.signal.zpk2sos(z, p, k))
    else:  # C
        # C-weighting analogue prototype:
        #   zeros at s=0 (×2), poles at −ω1 (×2) only (f2, f3 absent)
        #   The f4 double pole is handled by the MZT section below
        z, p, k = scipy.signal.bilinear_zpk([0, 0], [-w1, -w1], 1.0, fs)
        lf_blocks.append(scipy.signal.zpk2sos(z, p, k))

    sos_fixed = np.vstack(lf_blocks)

    # ── Step 2: build optimisation target grid ──────────────────────────────
    # Cap at 94 % of Nyquist to stay clear of the transition band where the
    # digital filter inevitably deviates from the analogue prototype.
    nyq_limit = min(20000.0, 0.94 * fs / 2.0)
    mask    = ISO_FREQS <= nyq_limit
    t_freqs = ISO_FREQS[mask].copy()
    t_db    = _ideal_response_db(t_freqs, w_type)

    # For lower sample rates the high-frequency region is sparser in the ISO
    # set, so add 25 uniformly-spaced points above 4 kHz to give the solver
    # adequate resolution in the critical transition band.
    if fs < 50000:
        extra   = np.linspace(4000.0, nyq_limit, 25)
        t_freqs = np.sort(np.concatenate([t_freqs, extra]))
        t_db    = _ideal_response_db(t_freqs, w_type)

    # Pre-compute the fixed section's response at all target frequencies
    w_eval = 2.0 * np.pi * t_freqs / fs
    _, h_fixed = scipy.signal.sosfreqz(sos_fixed, worN=w_eval)

    # ── Step 3: minimax cost function ──────────────────────────────────────
    def _cost(params: np.ndarray) -> float:
        f4_val, cp_f, cp_g, cp_q = params

        # MZT section for the high-frequency double pole at f4_val Hz.
        # The MZT maps a continuous-time pole at s = −ω4 to the z-plane via:
        #   p_d = exp(−ω4 / fs)
        # The second-order section transfer function is:
        #   H(z) = k / (1 − 2·p_d·z⁻¹ + p_d²·z⁻²)
        # with k = (1 − p_d)² chosen to give unity DC gain.
        w4    = 2.0 * np.pi * f4_val
        p_mzt = np.exp(-w4 / fs)
        k_mzt = (1.0 - p_mzt) ** 2
        sos_hf = np.array([[k_mzt, 0.0, 0.0, 1.0, -2.0 * p_mzt, p_mzt ** 2]])

        # Parametric correction biquad (free centre, gain, Q)
        sos_corr = _parametric_biquad_sos(cp_f, cp_q, cp_g, fs)

        _, h_var = scipy.signal.sosfreqz(np.vstack([sos_hf, sos_corr]), worN=w_eval)
        h_total  = h_fixed * h_var

        # Normalise to 0 dB at 1 kHz before computing error
        idx_1k = np.argmin(np.abs(t_freqs - 1000.0))
        mag_db = 20.0 * np.log10(np.abs(h_total) / (np.abs(h_total[idx_1k]) + 1e-15) + 1e-15)
        error  = np.abs(mag_db - t_db)

        # 50× higher penalty above 1 kHz: Class 1 tolerance tightens there
        # and the MZT/bilinear mismatch accumulates in that region.
        weights = np.where(t_freqs > 1000.0, 50.0, 1.0)
        return float(np.max(error * weights))

    # ── Step 4: optimise ────────────────────────────────────────────────────
    # Initial guess: f4 at its standard value, correction biquad inactive
    result = scipy.optimize.minimize(
        _cost, x0=[12194.0, 15000.0, 0.0, 1.0],
        method="Nelder-Mead", tol=1e-5,
        options={"maxiter": 500},
    )
    f4_opt, cf_opt, cg_opt, cq_opt = result.x

    # ── Step 5: assemble final SOS chain ────────────────────────────────────
    w4    = 2.0 * np.pi * f4_opt
    p_mzt = np.exp(-w4 / fs)
    k_mzt = (1.0 - p_mzt) ** 2
    sos_hf   = np.array([[k_mzt, 0.0, 0.0, 1.0, -2.0 * p_mzt, p_mzt ** 2]])
    sos_corr = _parametric_biquad_sos(cf_opt, cq_opt, cg_opt, fs)
    sos_all  = np.vstack([sos_fixed, sos_hf, sos_corr])

    # Normalise numerator of first section so H(1000 Hz) = 1 exactly.
    # This is a scalar gain adjustment; it does not alter the filter shape.
    w_ref = np.array([2.0 * np.pi * 1000.0 / fs])
    _, h_ref = scipy.signal.sosfreqz(sos_all, worN=w_ref)
    sos_all[0, :3] /= np.abs(h_ref[0]) + 1e-15

    return sos_all


# ── Stateful streaming wrapper ─────────────────────────────────────────────────

class WeightingFilter:
    """
    Stateful IEC 61672-1 weighting filter for block-by-block audio processing.

    The filter coefficients are designed once at construction time.  Internal
    state (zi) is updated after each call to :meth:`process_chunk` so that
    filter memory carries across block boundaries — essential for correct
    operation when the analysis block is much shorter than the filter's
    settling time.
    """

    def __init__(self, fs: float, w_type: str = "A"):
        self.fs     = fs
        self.w_type = str(w_type).upper()

        if self.w_type in ("Z", "FLAT", "NONE"):
            # Z-weighting is flat — no filtering needed
            self.passthrough = True
            self.sos = None
            self.zi  = None
            return

        self.passthrough = False
        if fs not in SUPPORTED_FS:
            print(f"Warning: non-standard sample rate {fs} Hz — "
                  "filter accuracy may be reduced near Nyquist.")

        self.sos = design_optimized_sos(fs, self.w_type)
        # sosfilt_zi returns the steady-state initial conditions for a
        # step input, which gives a good starting point for the filter state.
        self.zi  = scipy.signal.sosfilt_zi(self.sos)

    def reset(self) -> None:
        """Reset filter state to zero (use between independent recordings)."""
        if not self.passthrough:
            self.zi = scipy.signal.sosfilt_zi(self.sos)

    def initialize_state(self, seed: np.ndarray) -> None:
        """
        Warm up filter state using a forward-then-backward pass on *seed*.

        Methodology
        -----------
        Starting with a zero initial state causes a transient at the beginning
        of the first analysis block.  The forward-backward warm-up estimates
        the steady-state filter memory for the given input signal:

        1. Forward pass from zero state → yields state zi_fwd at end of seed.
        2. Backward pass starting from zi_fwd → yields state zi_bwd.

        zi_bwd approximates the state that would exist at the *start* of the
        seed block if the filter had been running indefinitely on similar
        content, eliminating the onset transient.
        """
        if self.passthrough:
            return
        _, zi_fwd = scipy.signal.sosfilt(self.sos, seed, zi=np.zeros_like(self.zi))
        _, zi_bwd = scipy.signal.sosfilt(self.sos, seed[::-1], zi=zi_fwd)
        self.zi = zi_bwd

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Filter *chunk*, updating internal state for the next call."""
        if self.passthrough:
            return chunk
        out, self.zi = scipy.signal.sosfilt(self.sos, chunk, zi=self.zi)
        return out
