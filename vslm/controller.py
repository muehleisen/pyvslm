"""
Application controller (MVC middle layer).

Owns the settings, the active file path, and the analysis worker thread.
Communicates with the GUI exclusively via Qt signals — no GUI imports here.
"""
from __future__ import annotations

import time
import threading
import traceback
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import QObject, Signal, Slot

from .settings import AppSettings, SettingsManager
from .constants import MODE_ID_MAP, AnalysisMode, LEQ_INTERVAL_MAP
from .gui.worker import AnalysisWorker
from .dsp import exporter


class VSLMController(QObject):
    # Emitted after a file is successfully loaded
    sig_file_loaded = Signal(object, object)   # (Path, soundfile.info)
    # Analysis lifecycle
    sig_analysis_started  = Signal()
    sig_analysis_progress = Signal(int)
    sig_total_blocks      = Signal(int)
    sig_analysis_finished = Signal(list)
    sig_analysis_error    = Signal(str)
    # Export / settings
    sig_export_done    = Signal()
    sig_status_message = Signal(str)
    # Playback state: "playing" | "paused" | "stopped"
    sig_playback_state_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._settings_mgr = SettingsManager()
        self.settings: AppSettings = self._settings_mgr.load()

        self.filepath:   Path | None  = None
        self.start_time: float        = 0.0
        self.end_time:   float | None = None
        self.cal_factor: float        = self.settings.calibration_factor
        self.last_results: list       = []
        self._worker: AnalysisWorker | None = None

        # Playback state
        self._play_data:         np.ndarray | None = None
        self._play_fs:           int               = 0
        self._play_frame_offset: int               = 0
        self._play_wall_time:    float             = 0.0
        self._is_playing:        bool              = False
        self._is_paused:         bool              = False

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------

    def load_file(self, filepath: str) -> None:
        path = Path(filepath)
        try:
            info = sf.info(str(path))
        except Exception as e:
            self.sig_analysis_error.emit(f"Cannot open file: {e}")
            return
        self.filepath  = path
        self.start_time = 0.0
        self.end_time   = info.duration
        self.last_results = []
        self.settings.last_directory = str(path.parent)
        self.sig_file_loaded.emit(path, info)
        self.sig_status_message.emit("File loaded.")

    def set_analysis_range(self, start: float, end: float) -> None:
        self.start_time = start
        self.end_time   = end
        self.sig_status_message.emit(f"Range: {start:.2f}s – {end:.2f}s")

    def update_calibration(self, factor: float) -> None:
        self.cal_factor = factor
        self.settings.calibration_factor = factor
        self.sig_status_message.emit(f"Calibration factor: {factor:.4f}")

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def play_audio(self) -> None:
        """Play the selected range, or resume from pause position."""
        if not self.filepath:
            return
        if self._is_paused and self._play_data is not None:
            self._resume_playback()
            return
        try:
            info = sf.info(str(self.filepath))
            fs   = info.samplerate
            start_frame = int(self.start_time * fs)
            stop_frame  = int(self.end_time * fs) if self.end_time else None
            if stop_frame is not None and stop_frame <= start_frame:
                self.sig_status_message.emit("Invalid selection range.")
                return
            data, _ = sf.read(str(self.filepath), start=start_frame,
                               stop=stop_frame, always_2d=True)
        except Exception as e:
            self.sig_analysis_error.emit(f"Playback error: {e}")
            return

        self._play_data         = data
        self._play_fs           = fs
        self._play_frame_offset = 0
        self._is_paused         = False
        sd.stop()
        sd.play(data, fs, blocking=False)
        self._play_wall_time = time.monotonic()
        self._is_playing     = True
        self.sig_playback_state_changed.emit("playing")
        end_t = self.end_time or info.duration
        self.sig_status_message.emit(
            f"Playing {self.start_time:.2f}s – {end_t:.2f}s…"
        )
        threading.Thread(target=self._monitor_playback, daemon=True).start()

    def pause_audio(self) -> None:
        """Freeze playback; resume position is preserved for Play."""
        if not self._is_playing:
            return
        elapsed = int((time.monotonic() - self._play_wall_time) * self._play_fs)
        data_len = len(self._play_data) if self._play_data is not None else 0
        self._play_frame_offset = min(self._play_frame_offset + elapsed, data_len)
        sd.stop()
        self._is_playing = False
        self._is_paused  = True
        self.sig_playback_state_changed.emit("paused")
        self.sig_status_message.emit("Paused.")

    def stop_audio(self) -> None:
        """Stop playback and reset to start of selection."""
        sd.stop()
        self._play_data         = None
        self._play_frame_offset = 0
        self._is_playing        = False
        self._is_paused         = False
        self.sig_playback_state_changed.emit("stopped")
        self.sig_status_message.emit("Playback stopped.")

    def _resume_playback(self) -> None:
        remaining = self._play_data[self._play_frame_offset:]
        if len(remaining) == 0:
            self.stop_audio()
            return
        sd.play(remaining, self._play_fs, blocking=False)
        self._play_wall_time = time.monotonic()
        self._is_playing     = True
        self._is_paused      = False
        self.sig_playback_state_changed.emit("playing")
        self.sig_status_message.emit("Resumed…")
        threading.Thread(target=self._monitor_playback, daemon=True).start()

    def _monitor_playback(self) -> None:
        """Daemon thread: detect natural end of playback and emit stopped."""
        sd.wait()
        if self._is_playing:
            self._is_playing        = False
            self._play_frame_offset = 0
            self._play_data         = None
            self.sig_playback_state_changed.emit("stopped")

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def run_analysis(self, mode_id: int) -> None:
        if not self.filepath:
            return
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait()

        s = self.settings
        mode = MODE_ID_MAP[mode_id]

        # For Lp mode the block size comes from the Lp interval selector
        block_ms = s.block_size_ms
        if mode == AnalysisMode.LP:
            keys = list(LEQ_INTERVAL_MAP.keys())
            if 0 <= s.lp_interval_index < len(keys):
                _, block_ms = LEQ_INTERVAL_MAP[keys[s.lp_interval_index]]
                block_ms *= 1000.0  # convert seconds → ms

        self._worker = AnalysisWorker(
            filepath      = self.filepath,
            cal_factor    = self.cal_factor,
            mode          = mode,
            weighting     = s.weighting,
            speed         = s.speed,
            block_size_ms = block_ms,
            do_bands      = mode in (AnalysisMode.OCTAVE, AnalysisMode.THIRD_OCTAVE),
            band_res      = "third" if mode == AnalysisMode.THIRD_OCTAVE else "octave",
            band_order    = s.band_filter_order,
            ref_pressure  = s.ref_pressure,
            psd_nfft      = s.psd_nfft,
            psd_window    = s.psd_window,
            spec_nfft     = s.spec_nfft,
            spec_dt       = s.spec_dt,
            spec_window   = s.spec_window,
        )
        self._worker.sig_total_blocks.connect(self.sig_total_blocks.emit)
        self._worker.sig_progress.connect(self.sig_analysis_progress.emit)
        self._worker.sig_finished.connect(self._on_worker_finished)
        self._worker.sig_error.connect(self.sig_analysis_error.emit)
        self._worker.start()
        self.sig_analysis_started.emit()
        self.sig_status_message.emit("Analysing…")

    def stop_analysis(self) -> None:
        if self._worker:
            self._worker.stop()
            self._worker.wait()
            self.sig_status_message.emit("Analysis stopped.")

    @Slot(list)
    def _on_worker_finished(self, results: list) -> None:
        # SPL-mode results: filter to the user-selected time range
        if results and isinstance(results[0], dict) and results[0].get("type") not in ("psd", "spectrogram"):
            if self.end_time is not None:
                results = [r for r in results if self.start_time <= r["time"] <= self.end_time]
        self.last_results = results
        self.sig_analysis_finished.emit(results)
        self.sig_status_message.emit("Analysis complete.")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_results(self, path: Path, mode_id: int, leq_interval_key) -> None:
        if not self.last_results:
            return
        s    = self.settings
        mode = MODE_ID_MAP[mode_id]
        try:
            match mode:
                case AnalysisMode.LP:
                    exporter.export_lp(path, self.last_results, str(s.weighting), str(s.speed))
                case AnalysisMode.LEQ:
                    dose_params = s.dose_standards[s.current_dose_standard]
                    exporter.export_leq(
                        path, self.last_results, s.block_size_ms,
                        leq_interval_key, str(s.weighting),
                        dose_params, s.current_dose_standard, s.ref_pressure,
                    )
                case AnalysisMode.OCTAVE | AnalysisMode.THIRD_OCTAVE:
                    exporter.export_spectrum(path, self.last_results, str(s.weighting), s.ref_pressure)
                case _:
                    self.sig_status_message.emit("Export not supported for this mode.")
                    return
            self.sig_export_done.emit()
            self.sig_status_message.emit(f"Exported → {path.name}")
        except Exception as e:
            self.sig_analysis_error.emit(f"Export failed:\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def save_settings(self, path: Path | None = None) -> None:
        self._settings_mgr.save(self.settings, path)
        self.sig_status_message.emit("Settings saved.")

    def load_settings(self, path: Path) -> bool:
        new = self._settings_mgr.load(path)
        if new:
            self.settings = new
            self.cal_factor = new.calibration_factor
            return True
        return False

    def shutdown(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait()
        self._is_playing = False
        self._is_paused  = False
        try:
            sd.stop()
        except Exception:
            pass
        self._settings_mgr.save(self.settings)
