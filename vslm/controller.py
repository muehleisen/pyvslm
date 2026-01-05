import traceback
import sounddevice as sd  # <--- Added
import soundfile as sf    # <--- Added
from pathlib import Path
from PySide6.QtCore import QObject, Signal, Slot
from .settings_manager import SettingsManager, AppSettings
from .gui.analysis_worker import AnalysisWorker
from .result_exporter import ResultsExporter
from .constants import Weighting, ResponseSpeed, LEQ_INTERVAL_MAP

class VSLMController(QObject):
    sig_file_loaded = Signal(object, object)
    sig_analysis_started = Signal(str)
    sig_analysis_progress = Signal(int)
    sig_analysis_finished = Signal(list)
    sig_analysis_error = Signal(str)
    sig_status_message = Signal(str)
    sig_export_finished = Signal()
    sig_total_blocks = Signal(int)

    def __init__(self):
        super().__init__()
        self.settings_mgr = SettingsManager()
        self.settings = self.settings_mgr.load()
        
        self.filepath: Path | None = None
        self.start_time: float = 0.0
        self.end_time: float | None = None
        
        self.last_results: list = []
        self.worker: AnalysisWorker | None = None
        self.cal_factor = self.settings.calibration_factor
        
        self.total_blocks_estimate = 100

    def load_file(self, filepath: str):
        path = Path(filepath)
        try:
            from soundfile import info
            inf = info(str(path))
            
            self.filepath = path
            self.settings.last_directory = str(path.parent)
            self.start_time = 0.0
            self.end_time = inf.duration
            self.last_results = []
            
            self.sig_file_loaded.emit(path, inf)
            self.sig_status_message.emit("File loaded successfully.")
        except Exception as e:
            self.sig_analysis_error.emit(f"Failed to load file: {e}")

    def update_calibration(self, new_factor: float):
        self.cal_factor = new_factor
        self.settings.calibration_factor = new_factor
        self.sig_status_message.emit(f"Calibration updated: {new_factor:.4f}")

    def set_analysis_range(self, start: float, end: float):
        self.start_time = start
        self.end_time = end
        self.sig_status_message.emit(f"Analysis range set: {start:.2f}s - {end:.2f}s")

    def play_audio(self):
        """Plays the selected range of audio asynchronously."""
        if not self.filepath: return
        try:
            # Stop any existing playback first
            sd.stop()
            
            # Read file info to get sample rate
            info = sf.info(str(self.filepath))
            fs = info.samplerate
            
            # Calculate frames to read based on selection
            start_frame = int(self.start_time * fs)
            stop_frame = int(self.end_time * fs) if self.end_time else None
            
            # Check for valid range
            if stop_frame and (stop_frame <= start_frame):
                self.sig_status_message.emit("Invalid selection range.")
                return

            # Read only the selected segment
            data, _ = sf.read(str(self.filepath), start=start_frame, stop=stop_frame, always_2d=True)
            
            # Play asynchronously (returns immediately)
            # We assume default output device
            sd.play(data, fs)
            self.sig_status_message.emit(f"Playing selection ({self.start_time:.2f}s - {self.end_time or info.duration:.2f}s)...")
            
        except Exception as e:
            self.sig_analysis_error.emit(f"Playback error: {e}")

    def stop_audio(self):
        """Stops audio playback."""
        try:
            sd.stop()
            self.sig_status_message.emit("Playback stopped.")
        except Exception as e:
            print(f"Error stopping audio: {e}")

    def run_analysis(self, mode_id: int):
        if not self.filepath: return

        is_lp = (mode_id == 0)
        is_psd = (mode_id == 4)
        is_spec = (mode_id == 5)
        
        do_bands = (mode_id in [2, 3])
        band_res = 'third' if mode_id == 3 else 'octave'
        
        w = self.settings.weighting
        weighting_val = w.value if hasattr(w, 'value') else w
        
        s = self.settings.speed
        speed_val = s.value if hasattr(s, 'value') else s
        
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        
        # --- Determine Block Size based on Mode ---
        calc_block_ms = self.settings.block_size_ms 
        
        if is_lp:
            try:
                keys = list(LEQ_INTERVAL_MAP.keys())
                idx = self.settings.lp_interval_index
                if 0 <= idx < len(keys):
                    key = keys[idx]
                    interval_sec = LEQ_INTERVAL_MAP[key][1]
                    calc_block_ms = interval_sec * 1000.0
            except Exception as e:
                print(f"Error setting Lp interval: {e}, using default.")
                calc_block_ms = 100.0

        # Grab settings
        psd_nfft = getattr(self.settings, 'psd_nfft', 4096)
        psd_window = getattr(self.settings, 'psd_window', 'Hanning')
        
        spec_nfft = getattr(self.settings, 'spec_nfft', 512)
        spec_dt = getattr(self.settings, 'spec_dt', 1.0)
        spec_window = getattr(self.settings, 'spec_window', 'Hamming')

        self.worker = AnalysisWorker(
            filepath=self.filepath,
            cal_factor=self.cal_factor,
            block_size_ms=calc_block_ms,
            weighting=weighting_val,
            do_bands=do_bands,
            band_res=band_res,
            speed=speed_val,
            band_order=self.settings.band_filter_order,
            ref_pressure=self.settings.ref_pressure,
            mode_is_psd=is_psd,
            psd_nfft=psd_nfft,
            psd_window=psd_window,
            mode_is_spec=is_spec,
            spec_nfft=spec_nfft,
            spec_dt=spec_dt,
            spec_window=spec_window
        )

        self.worker.sig_total_blocks.connect(self.sig_total_blocks.emit)
        self.worker.sig_progress.connect(self.sig_analysis_progress.emit)
        self.worker.sig_error.connect(self.sig_analysis_error.emit)
        self.worker.sig_finished.connect(self._on_worker_finished)
        
        self.worker.start()
        self.sig_analysis_started.emit(str(speed_val))

    def stop_analysis(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.sig_status_message.emit("Analysis stopped by user.")

    def _on_worker_finished(self, results):
        if results and isinstance(results[0], dict) and results[0].get('type') in ['psd', 'spectrogram']:
            filtered = results
        else:
            if self.end_time:
                filtered = [r for r in results if self.start_time <= r['time'] <= self.end_time]
            else:
                filtered = results
            
        self.last_results = filtered
        self.sig_analysis_finished.emit(filtered)
        self.sig_status_message.emit("Analysis Complete.")

    def export_results(self, path: Path, mode_id: int, leq_interval_key):
        if not self.last_results: return

        try:
            w = self.settings.weighting
            weighting = w.value if hasattr(w, 'value') else w
            
            s = self.settings.speed
            speed = s.value if hasattr(s, 'value') else s
            
            dose_std_name = self.settings.current_dose_standard
            dose_params = self.settings.dose_standards.get(dose_std_name)
            
            if mode_id == 1: # LEQ
                 ResultsExporter.export_leq(
                    path, 
                    self.last_results, 
                    self.settings.block_size_ms, 
                    leq_interval_key, 
                    weighting,
                    dose_params,
                    dose_std_name,
                    self.settings.ref_pressure
                )
            elif mode_id == 0: # Lp
                ResultsExporter.export_lp(path, self.last_results, weighting, speed)
            elif mode_id in [2, 3]: # Spectrum
                ResultsExporter.export_spectrum(path, self.last_results, weighting, self.settings.ref_pressure)
                
            self.sig_export_finished.emit()
            self.sig_status_message.emit(f"Exported to {path.name}")
        except Exception as e:
            self.sig_analysis_error.emit(f"Export failed: {str(e)}\n{traceback.format_exc()}")

    def save_settings(self, path: Path):
        self.settings_mgr.save(self.settings, path)
        self.sig_status_message.emit("Settings saved.")

    def load_settings(self, path: Path) -> bool:
        new_settings = self.settings_mgr.load(path)
        if new_settings:
            self.settings = new_settings
            self.cal_factor = self.settings.calibration_factor
            return True
        return False

    def shutdown(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        # Also stop audio on shutdown
        try:
            sd.stop()
        except:
            pass
        self.settings_mgr.save(self.settings)