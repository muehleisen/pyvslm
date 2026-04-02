"""
Background analysis worker thread.

Wraps :class:`~vslm.dsp.engine.StreamProcessor` in a QThread so the GUI
remains responsive during long analyses.  Progress is reported via signals
rather than by blocking the main thread.
"""
from __future__ import annotations

from contextlib import closing
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from ..constants import AnalysisMode
from ..dsp.engine import StreamProcessor


class AnalysisWorker(QThread):
    sig_progress    = Signal(int)
    sig_total_blocks = Signal(int)
    sig_finished    = Signal(list)
    sig_error       = Signal(str)

    def __init__(
        self,
        filepath: Path,
        cal_factor: float,
        mode: AnalysisMode,
        weighting: str,
        speed: str,
        block_size_ms: float,
        do_bands: bool,
        band_res: str,
        band_order: int,
        ref_pressure: float,
        psd_nfft: int = 4096,
        psd_window: str = "Hanning",
        spec_nfft: int = 512,
        spec_dt: float = 1.0,
        spec_window: str = "Hamming",
    ) -> None:
        super().__init__()
        self.filepath      = filepath
        self.cal_factor    = cal_factor
        self.mode          = mode
        self.weighting     = weighting
        self.speed         = speed
        self.block_size_ms = block_size_ms
        self.do_bands      = do_bands
        self.band_res      = band_res
        self.band_order    = band_order
        self.ref_pressure  = ref_pressure
        self.psd_nfft      = psd_nfft
        self.psd_window    = psd_window
        self.spec_nfft     = spec_nfft
        self.spec_dt       = spec_dt
        self.spec_window   = spec_window
        self._running      = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        try:
            processor = StreamProcessor(self.filepath, self.cal_factor)

            match self.mode:
                case AnalysisMode.PSD:
                    self._run_single_result(
                        processor.run_psd(self.psd_nfft, self.psd_window, self.weighting)
                    )
                case AnalysisMode.SPECTROGRAM:
                    self._run_single_result(
                        processor.run_spectrogram(self.spec_nfft, self.spec_dt,
                                                  self.spec_window, self.weighting)
                    )
                case _:
                    self._run_spl(processor)

        except Exception as e:
            import traceback
            self.sig_error.emit(f"{e}\n\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_spl(self, processor: StreamProcessor) -> None:
        """Stream block-by-block SPL analysis, emitting results as a list."""
        total = int(processor.duration * 1000.0 / self.block_size_ms)
        self.sig_total_blocks.emit(total)

        results = []
        gen = processor.run_spl(
            block_size_ms  = self.block_size_ms,
            weighting      = self.weighting,
            do_bands       = self.do_bands,
            band_resolution = self.band_res,
            band_order     = self.band_order,
            speed          = self.speed,
            ref_pressure   = self.ref_pressure,
        )
        with closing(gen):
            for i, block in enumerate(gen):
                if not self._running:
                    return
                results.append(block)
                if i % 10 == 0:
                    self.sig_progress.emit(i + 1)

        self.sig_progress.emit(total)
        self.sig_finished.emit(results)

    def _run_single_result(self, gen) -> None:
        """Drive a progress-then-result generator (PSD / Spectrogram)."""
        self.sig_total_blocks.emit(100)
        final = None
        with closing(gen):
            for item in gen:
                if not self._running:
                    return
                if isinstance(item, int):
                    self.sig_progress.emit(item)
                elif isinstance(item, dict):
                    final = item
        if final is not None:
            self.sig_progress.emit(100)
            self.sig_finished.emit([final])
