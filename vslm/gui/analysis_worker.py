from PySide6.QtCore import QThread, Signal
from contextlib import closing
import traceback

# VSLM Imports
from ..analysis_engine import StreamProcessor

class AnalysisWorker(QThread):
    """
    Background worker thread to run the audio analysis without freezing the GUI.
    """
    # Signals to communicate with the main thread
    sig_progress = Signal(int)       # Emits current block number
    sig_total_blocks = Signal(int)   # Emits total number of blocks
    sig_finished = Signal(list)      # Emits the full list of results on completion
    sig_error = Signal(str)          # Emits error message if something fails

    def __init__(self, filepath, cal_factor, block_size_ms, weighting, 
                 do_bands, band_res, speed, band_order, ref_pressure,
                 # PSD Params
                 mode_is_psd=False, psd_nfft=4096, psd_window='Hanning',
                 # Spectrogram Params
                 mode_is_spec=False, spec_nfft=512, spec_dt=1.0):
        super().__init__()
        self.filepath = filepath
        self.cal_factor = cal_factor
        self.block_size_ms = block_size_ms
        self.weighting = weighting
        self.do_bands = do_bands
        self.band_res = band_res
        self.speed = speed
        self.band_order = band_order
        self.ref_pressure = ref_pressure
        
        self.mode_is_psd = mode_is_psd
        self.psd_nfft = psd_nfft
        self.psd_window = psd_window
        
        self.mode_is_spec = mode_is_spec
        self.spec_nfft = spec_nfft
        self.spec_dt = spec_dt
        
        self._is_running = True

    def run(self):
        try:
            processor = StreamProcessor(self.filepath, self.cal_factor)
            
            if self.mode_is_spec:
                # --- SPECTROGRAM PATH ---
                self.sig_total_blocks.emit(100)
                gen = processor.calculate_spectrogram(
                    nfft=self.spec_nfft,
                    dt=self.spec_dt,
                    weighting=self.weighting
                )
                self._run_generator(gen)

            elif self.mode_is_psd:
                # --- PSD PATH ---
                self.sig_total_blocks.emit(100) 
                gen = processor.calculate_psd(
                    nfft=self.psd_nfft,
                    window_type=self.psd_window,
                    weighting=self.weighting
                )
                self._run_generator(gen)

            else:
                # --- STANDARD PATH (Time History) ---
                total_blocks = int((processor.duration * 1000) / self.block_size_ms)
                self.sig_total_blocks.emit(total_blocks)
                
                results = []
                gen = processor.run_analysis(
                    block_size_ms=self.block_size_ms,
                    weighting=self.weighting,
                    do_band_analysis=self.do_bands,
                    band_resolution=self.band_res,
                    time_weighting=self.speed,
                    band_order=self.band_order,
                    ref_pressure=self.ref_pressure
                )
                
                with closing(gen):
                    for i, block in enumerate(gen):
                        if not self._is_running: break
                        results.append(block)
                        if i % 10 == 0: self.sig_progress.emit(i + 1)
                
                if self._is_running:
                    self.sig_progress.emit(total_blocks)
                    self.sig_finished.emit(results)
                
        except Exception as e:
            error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            self.sig_error.emit(error_msg)

    def _run_generator(self, gen):
        """Helper to iterate single-result generators (PSD/Spectrogram)"""
        final_result = None
        with closing(gen):
            for item in gen:
                if not self._is_running: break
                
                if isinstance(item, int):
                    self.sig_progress.emit(item)
                elif isinstance(item, dict):
                    final_result = item
        
        if self._is_running and final_result:
            self.sig_progress.emit(100)
            self.sig_finished.emit([final_result])

    def stop(self):
        """Signals the thread to stop processing safely."""
        self._is_running = False