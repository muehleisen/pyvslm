"""
Leq (equivalent continuous sound level) analysis and noise-dose calculations.

All functions are pure Python/NumPy — no Qt dependency.

References
----------
IEC 61672-1:2013  §3.10  Definition of equivalent continuous sound level (Leq)
NIOSH (1998)  "Criteria for a Recommended Standard: Occupational Noise Exposure"
  DHHS (NIOSH) Publication No. 98-126  — 3 dB exchange rate, 85 dB criterion
OSHA 29 CFR 1910.95  "Occupational Noise Exposure" — 5 dB exchange rate, 90 dB
ISO 1996-1:2016  §3.5  Percentile sound levels (LN)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..settings import DoseStandard


@dataclass
class LeqStats:
    """Results from :func:`calculate`."""
    overall:       float
    maximum:       float
    minimum:       float
    ln:            dict[int, float]       # L10 … L90 percentile levels
    dose:          dict[str, float]       # 'dose' (%), 'twa' (dB)
    history:       dict[str, list[float]] # 'time' (s), 'leq' (dB)
    block_size_ms: float


def calculate(
    block_results: list[dict],
    block_size_ms: float,
    integration_s: float,
    dose_params: "DoseStandard",
    ref_pressure: float = 20e-6,
) -> LeqStats:
    """
    Compute Leq statistics and noise dose from a sequence of block results.

    Methodology
    -----------

    **Overall Leq  (IEC 61672-1 §3.10)**

    The equivalent continuous sound level over the full measurement period T is:

        Leq = 10 · log10[ (1/T) · ∫₀ᵀ p²(t)/p_ref² dt ]

    In discrete block form (each block has duration Δt = block_size_ms/1000 s):

        Leq = 10 · log10[ mean(p²_block) / p_ref² ]

    where mean(p²_block) is the time-average of the instantaneous squared
    pressure values stored in the "leq" field of each block result dict.
    Equivalently:

        p²_k  = p_ref² · 10^(Leq_block_k / 10)     [convert dB back to Pa²]
        Leq   = 10 · log10[ mean_k(p²_k) / p_ref² ]

    **Percentile levels LN  (ISO 1996-1 §3.5)**

    LN is the level exceeded N % of the measurement time — i.e. the (100−N)th
    percentile of the instantaneous level distribution.  For example, L90 is
    the level exceeded 90 % of the time (the background level), and L10 is
    exceeded only 10 % of the time (the near-peak level).

    Implementation: np.percentile(raw_db, 100 − N) gives the level at the
    (100−N)th percentile of the empirical distribution of block Leq values.

    **Noise dose — NIOSH / OSHA formula**

    Both standards express cumulative noise exposure as a *dose* D (%):

        D = 100 · (1/T_criterion) · Σ_k  Δt · C_k

    where T_criterion is the permissible exposure duration at the criterion
    level and C_k is the *fractional contribution* of block k:

        C_k = 10^[(L_k − L_criterion) / q]

    Here q = ER / log10(2), with ER the exchange rate (3 dB for NIOSH,
    5 dB for OSHA).  This q factor comes from the relationship between the
    exchange rate and the equal-energy hypothesis:

        If ER = 3 dB: doubling exposure time doubles energy → q = 3/log10(2) ≈ 9.97
        If ER = 5 dB: q = 5/log10(2) ≈ 16.6

    Only levels above the threshold (80 dB NIOSH, 80 dB OSHA) contribute.

    The Time-Weighted Average (TWA) converts dose back to an equivalent
    8-hour average level:

        TWA = L_criterion + q · log10(D / 100)

    **Time history**

    The block-level Leq values are integrated into coarser intervals by
    energy-averaging groups of consecutive blocks:

        Leq_interval = 10 · log10[ mean(p²_blocks_in_interval) / p_ref² ]

    Args:
        block_results:  List of dicts from StreamProcessor.run_spl, each with
                        at least a ``"leq"`` key (dB SPL).
        block_size_ms:  Duration of each block in milliseconds.
        integration_s:  Time interval for the Leq history (seconds).
        dose_params:    Dose standard parameters (exchange rate, criterion
                        level, threshold level, shift hours).
        ref_pressure:   Reference pressure in Pa (default 20 µPa).

    Returns:
        :class:`LeqStats` with overall Leq, Lmax/Lmin, percentile levels,
        noise dose, and time history.
    """
    _empty = LeqStats(
        overall=-100.0, maximum=-100.0, minimum=-100.0,
        ln={n: -100.0 for n in range(10, 100, 10)},
        dose={"dose": 0.0, "twa": 0.0},
        history={"time": [], "leq": []},
        block_size_ms=block_size_ms,
    )
    if not block_results:
        return _empty

    # Convert per-block dB levels back to mean-square pressures (Pa²)
    # p² = p_ref² · 10^(Leq_dB / 10)
    raw_db = np.array([b.get("leq", -100.0) for b in block_results])
    p_sq   = (10.0 ** (raw_db / 10.0)) * (ref_pressure ** 2)

    # Overall Leq: energy average over all blocks
    overall = 10.0 * np.log10(np.mean(p_sq) / (ref_pressure ** 2) + 1e-30)

    # Percentile levels: LN = (100−N)th percentile of the level distribution
    ln = {n: float(np.percentile(raw_db, 100 - n)) for n in range(10, 100, 10)}

    # ── Noise dose ──────────────────────────────────────────────────────────
    er    = dose_params.exchange_rate    # 3 dB (NIOSH) or 5 dB (OSHA)
    cl    = dose_params.criterion_level  # 85 dB (NIOSH) or 90 dB (OSHA)
    tl    = dose_params.threshold_level  # 80 dB both standards
    hours = dose_params.shift_hours      # typically 8 hours
    dt_s  = block_size_ms / 1000.0
    T_s   = hours * 3600.0              # criterion shift duration in seconds

    # q converts exchange rate to the base-10 logarithm scaling factor:
    #   q = ER / log10(2)
    # so that a level increase of ER dB doubles the fractional dose
    q = er / np.log10(2.0)

    above_threshold = raw_db[raw_db > tl]
    if above_threshold.size > 0:
        # Accumulated exposure in seconds at criterion level:
        #   Acc = Σ Δt · 10^[(L_k − L_crit) / q]
        accumulated = float(np.sum(dt_s * 10.0 ** ((above_threshold - cl) / q)))
        dose_pct    = 100.0 * accumulated / T_s
        # TWA: inverse of the dose formula, solved for level
        #   TWA = L_crit + q · log10(D / 100)
        twa = cl + q * np.log10(dose_pct / 100.0) if dose_pct > 0 else 0.0
    else:
        dose_pct = 0.0
        twa      = 0.0

    # ── Time history at the requested integration interval ──────────────────
    # Group consecutive blocks into intervals; energy-average within each
    blocks_per_interval = max(1, int(round(integration_s / dt_s)))
    n_intervals         = len(raw_db) // blocks_per_interval

    if n_intervals > 0:
        # Reshape into (n_intervals, blocks_per_interval) then average energy
        trimmed  = p_sq[: n_intervals * blocks_per_interval].reshape(n_intervals, blocks_per_interval)
        agg_leq  = 10.0 * np.log10(np.mean(trimmed, axis=1) / (ref_pressure ** 2) + 1e-30)
        agg_time = np.arange(n_intervals) * integration_s
    else:
        agg_leq  = np.array([])
        agg_time = np.array([])

    return LeqStats(
        overall=float(overall),
        maximum=float(np.max(raw_db)),
        minimum=float(np.min(raw_db)),
        ln=ln,
        dose={"dose": dose_pct, "twa": float(twa)},
        history={"time": agg_time.tolist(), "leq": agg_leq.tolist()},
        block_size_ms=block_size_ms,
    )
