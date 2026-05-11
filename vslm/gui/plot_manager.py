"""
Renders analysis results onto a matplotlib Figure.

All methods are static so the plotter is stateless and can be called from
any context that holds a Figure reference.
"""
from __future__ import annotations

import traceback

import numpy as np
from matplotlib.figure import Figure

from ..dsp import leq as leq_mod
from ..constants import LEQ_INTERVAL_MAP, AnalysisMode, MODE_ID_MAP


class ResultPlotter:

    @staticmethod
    def plot(
        figure: Figure,
        results: list,
        mode_id: int,
        weighting: str,
        speed: str,
        leq_interval_key,
        block_size_ms: float,
        dose_params,
        dose_std_name: str,
        ref_pressure: float,
        autoscale: bool = True,
        ymin: float = 0.0,
        ymax: float = 120.0,
        spec_cmap: str = "plasma",
    ) -> None:
        figure.clear()
        if not results:
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return

        mode = MODE_ID_MAP[mode_id]
        try:
            match mode:
                case AnalysisMode.LP:
                    ResultPlotter._plot_lp(figure, results, weighting, speed, autoscale, ymin, ymax)
                case AnalysisMode.LEQ:
                    ResultPlotter._plot_leq(
                        figure, results, weighting, leq_interval_key,
                        block_size_ms, dose_params, dose_std_name, ref_pressure,
                        autoscale, ymin, ymax,
                    )
                case AnalysisMode.OCTAVE | AnalysisMode.THIRD_OCTAVE:
                    ResultPlotter._plot_spectrum(
                        figure, results, weighting,
                        is_third=(mode == AnalysisMode.THIRD_OCTAVE),
                        ref_pressure=ref_pressure, autoscale=autoscale, ymin=ymin, ymax=ymax,
                    )
                case AnalysisMode.PSD:
                    ResultPlotter._plot_psd(figure, results[0], ref_pressure, autoscale, ymin, ymax)
                case AnalysisMode.SPECTROGRAM:
                    ResultPlotter._plot_spectrogram(
                        figure, results[0], ref_pressure, autoscale, ymin, ymax, spec_cmap
                    )
        except Exception:
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Plot error:\n{traceback.format_exc()}",
                    ha="center", va="center", color="red", transform=ax.transAxes,
                    fontsize=8, wrap=True)

        figure.tight_layout()

    # ------------------------------------------------------------------

    @staticmethod
    def _apply_ylim(ax, autoscale: bool, ymin: float, ymax: float) -> None:
        if autoscale:
            ax.autoscale(axis="y")
        else:
            ax.set_ylim(ymin, ymax)

    # ------------------------------------------------------------------

    @staticmethod
    def _plot_lp(figure, results, weighting, speed, autoscale, ymin, ymax) -> None:
        ax = figure.add_subplot(111)
        t = [r["time"] for r in results]
        l = [r["lp"]   for r in results]
        ax.plot(t, l, linewidth=1.2)
        ax.set_title(f"Sound Pressure Level — {weighting}-weighted, {speed}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Level (dB SPL)")
        ax.grid(True)
        ResultPlotter._apply_ylim(ax, autoscale, ymin, ymax)

    @staticmethod
    def _plot_leq(figure, results, weighting, interval_key,
                  block_size_ms, dose_params, dose_std_name, ref_pressure,
                  autoscale, ymin, ymax) -> None:
        label, interval_s = LEQ_INTERVAL_MAP.get(interval_key, ("1 sec", 1.0))
        stats = leq_mod.calculate(results, block_size_ms, interval_s, dose_params, ref_pressure)

        ax1 = figure.add_subplot(2, 1, 1)
        if stats.history["time"]:
            t = list(stats.history["time"]) + [stats.history["time"][-1] + interval_s]
            l = list(stats.history["leq"])  + [stats.history["leq"][-1]]
            ax1.step(t, l, where="post", color="#1f77b4", linewidth=1.5)
        ax1.set_title(f"Leq History — {label} intervals, {weighting}-weighted")
        ax1.set_ylabel("Leq (dB SPL)")
        ax1.grid(True)
        ResultPlotter._apply_ylim(ax1, autoscale, ymin, ymax)

        ax2 = figure.add_subplot(2, 1, 2)
        ax2.axis("off")

        # Summary statistics table
        col = [0.05, 0.38, 0.68]
        row_h = 0.12
        ax2.text(0.5, 0.96, f"Overall Leq:  {stats.overall:.1f} dB",
                 ha="center", fontsize=13, fontweight="bold", color="#1e40af",
                 transform=ax2.transAxes)

        stat_rows = [
            (col[0], [f"Lmax: {stats.maximum:.1f} dB",
                      f"L10:  {stats.ln[10]:.1f} dB",
                      f"L20:  {stats.ln[20]:.1f} dB",
                      f"L30:  {stats.ln[30]:.1f} dB"]),
            (col[1], [f"Lmin: {stats.minimum:.1f} dB",
                      f"L40:  {stats.ln[40]:.1f} dB",
                      f"L50:  {stats.ln[50]:.1f} dB",
                      f"L60:  {stats.ln[60]:.1f} dB"]),
            (col[2], [f"Dose ({dose_std_name})",
                      f"Dose: {stats.dose['dose']:.1f} %",
                      f"TWA:  {stats.dose['twa']:.1f} dB",
                      f"L70:  {stats.ln[70]:.1f} dB"]),
        ]
        for x, lines in stat_rows:
            for i, line in enumerate(lines):
                bold = (i == 0 and x == col[2])
                ax2.text(x, 0.80 - i * row_h, line,
                         fontweight="bold" if bold else "normal",
                         transform=ax2.transAxes)

    @staticmethod
    def _plot_spectrum(figure, results, weighting, is_third, ref_pressure,
                       autoscale, ymin, ymax) -> None:
        ax = figure.add_subplot(111)
        freqs = results[-1].get("band_freqs", [])
        if not len(freqs):
            return

        p_sum = np.zeros(len(freqs))
        n = 0
        for r in results:
            if "bands" in r:
                p_sum += (10.0 ** (r["bands"] / 10.0)) * (ref_pressure ** 2)
                n += 1
        mean_db = 10.0 * np.log10(p_sum / max(n, 1) / (ref_pressure ** 2) + 1e-30)

        x = np.arange(len(freqs))
        ax.bar(x, mean_db, color="#2ca02c", alpha=0.85)
        ax.set_xticks(x)
        labels = [f"{f/1000:.0f}k" if f >= 1000 else f"{f:.0f}" for f in freqs]
        if is_third:
            labels = [lb if i % 3 == 0 else "" for i, lb in enumerate(labels)]
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_title(f"Average Spectrum — {weighting}-weighted")
        ax.set_ylabel("Level (dB SPL)")
        ax.grid(axis="y")
        ResultPlotter._apply_ylim(ax, autoscale, ymin, ymax)

    @staticmethod
    def _plot_psd(figure, data, ref_pressure, autoscale, ymin, ymax) -> None:
        ax     = figure.add_subplot(111)
        freqs  = data["freqs"]
        pxx    = data["pxx"]
        nfft   = data["nfft"]
        win    = data["window"]
        wt     = data.get("weighting", "Z")

        lpxx = 10.0 * np.log10(pxx / (ref_pressure ** 2) + 1e-30)
        # Clamp dynamic range for display (60 dB below peak)
        lpxx = np.maximum(lpxx, np.max(lpxx) - 60.0)

        ax.semilogx(freqs, lpxx)
        ax.set_title(f"Power Spectral Density — {nfft}-pt FFT, {win} window, {wt}-weighted")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (dB/Hz re (20 µPa)²/Hz)")
        ax.grid(True, which="both", linestyle="-", alpha=0.4)
        ax.set_xlim(left=freqs[1])
        ResultPlotter._apply_ylim(ax, autoscale, ymin, ymax)

    @staticmethod
    def _plot_spectrogram(figure, data, ref_pressure, autoscale, ymin, ymax, cmap) -> None:
        ax    = figure.add_subplot(111)
        times = data["times"]
        freqs = data["freqs"]
        S     = 10.0 * np.log10(data["pxx_matrix"] / (ref_pressure ** 2) + 1e-30)
        nfft  = data["nfft"]
        dt    = data["dt"]
        wt    = data.get("weighting", "Z")

        vmin = None if autoscale else ymin
        vmax = None if autoscale else ymax
        pcm = ax.pcolormesh(times, freqs, S.T, shading="auto",
                            cmap=cmap, vmin=vmin, vmax=vmax)
        figure.colorbar(pcm, ax=ax, label="Level (dB SPL)")
        ax.set_title(f"Spectrogram — {nfft}-pt FFT, {dt} s slices, {wt}-weighted")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(0, freqs[-1])
