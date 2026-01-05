import numpy as np
import soundfile as sf
import scipy.signal
from pathlib import Path
from typing import Generator, Any

from .filters.weighting_filters import WeightingFilter, get_weighting_power_response
from .filters.octave_filters import OctaveFilterBank
from .constants import Weighting, ResponseSpeed, BandResolution

class TimeWeightingDetector:
    # Use | for Union types (Python 3.10+)
    def __init__(self, fs: float, mode: ResponseSpeed = ResponseSpeed.FAST, ref_pressure: float = 20e-6):
        self.fs = fs
        self.mode = mode
        self.ref_pressure = ref_pressure
        self.state = 0.0 
        
        # Implement Structural Pattern Matching (Python 3.10+)
        match mode:
            case ResponseSpeed.FAST:
                tau_rise = tau_fall = 0.125
            case ResponseSpeed.IMPULSE:
                tau_rise = 0.035
                tau_fall = 1.5
            case _: 
                # Default to Slow or fallback
                tau_rise = tau_fall = 1.0
            
        self.alpha_rise = 1.0 - np.exp(-1.0 / (fs * tau_rise))
        self.alpha_fall = 1.0 - np.exp(-1.0 / (fs * tau_fall))

    def process(self, chunk: np.ndarray) -> float:
        p2 = chunk**2
        current_val = self.state
        max_val = 0.0
        
        a_rise = self.alpha_rise
        a_fall = self.alpha_fall
        
        for s in p2:
            if s > current_val:
                current_val = (1 - a_rise) * current_val + a_rise * s
            else:
                current_val = (1 - a_fall) * current_val + a_fall * s
            if current_val > max_val:
                max_val = current_val
        
        self.state = current_val
        return 10 * np.log10(max_val / (self.ref_pressure**2) + 1e-30)

class StreamProcessor:
    # Use | for Union types (Python 3.10+)
    def __init__(self, filepath: str | Path, cal_factor: float = 1.0):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
            
        self.cal_factor = cal_factor
        try:
            info = sf.info(str(self.filepath))
            self.fs = info.samplerate
            self.duration = info.duration
        except Exception as e:
            raise ValueError(f"Could not read file info: {e}")

    def calculate_psd(self, 
                      nfft: int = 4096, 
                      window_type: str = 'Hanning', 
                      weighting: Weighting = Weighting.A
                      ) -> Generator[dict[str, Any], None, None]:
        """
        Calculates PSD using Welch's method (averaging periodograms of chunks).
        Applies frequency weighting in the frequency domain.
        Yields progress integers, then finally yields the result dict.
        """
        win_map = {'Hanning': 'hann', 'Hamming': 'hamming', 'Flattop': 'flattop'}
        scipy_window = win_map.get(window_type, 'hann')
        
        # Read in large chunks (e.g., 10 seconds) to update progress
        chunk_size_sec = 10.0
        chunk_samples = int(self.fs * chunk_size_sec)
        
        # Accumulators
        pxx_sum = None
        count = 0
        freqs = None
        
        total_samples = int(self.duration * self.fs)
        processed_samples = 0
        
        nperseg = nfft
        noverlap = nfft // 2
        
        with sf.SoundFile(str(self.filepath)) as f:
            for chunk in f.blocks(blocksize=chunk_samples, always_2d=False, fill_value=0.0):
                if len(chunk) < nperseg:
                    continue # Skip tail if too short for one FFT
                
                # Apply Calibration
                chunk = chunk * self.cal_factor
                if chunk.ndim > 1: chunk = np.mean(chunk, axis=1) # Mono mixdown
                
                # Compute Welch for this chunk
                f_c, pxx_c = scipy.signal.welch(chunk, fs=self.fs, window=scipy_window,
                                                nperseg=nperseg, noverlap=noverlap,
                                                nfft=nfft, scaling='density')
                
                if pxx_sum is None:
                    pxx_sum = pxx_c
                    freqs = f_c
                else:
                    pxx_sum += pxx_c
                
                count += 1
                processed_samples += len(chunk)
                
                # Yield progress (0-100)
                progress = int(100 * processed_samples / total_samples)
                yield progress

        if pxx_sum is None or count == 0:
             raise ValueError("Data too short for specified FFT size.")

        # Average Pxx across all chunks
        pxx_avg = pxx_sum / count
        
        # Apply Frequency Weighting using the filter module
        # vslm.m logic: Pxx = Pxx * |H(f)|^2
        w_response = get_weighting_power_response(freqs, weighting)
        pxx_weighted = pxx_avg * w_response
        
        # Yield Final Result
        yield {
            'type': 'psd',
            'freqs': freqs,
            'pxx': pxx_weighted,
            'nfft': nfft,
            'window': window_type,
            'weighting': weighting
        }
        
    def run_analysis(self, 
                     block_size_ms: float = 100.0, 
                     weighting: Weighting = Weighting.A, 
                     do_band_analysis: bool = False, 
                     band_resolution: BandResolution = BandResolution.OCTAVE,
                     band_order: int = 24,
                     time_weighting: ResponseSpeed = ResponseSpeed.FAST,
                     ref_pressure: float = 20e-6
                     ) -> Generator[dict[str, Any], None, None]:
        
        weighting_filter = WeightingFilter(self.fs, weighting)
        lp_detector = TimeWeightingDetector(self.fs, time_weighting, ref_pressure)
        
        band_bank = None
        if do_band_analysis:
            band_bank = OctaveFilterBank(self.fs, resolution=band_resolution, order=band_order)

        block_samples = int(self.fs * (block_size_ms / 1000.0))
        if block_samples == 0: raise ValueError("Block size too small.")
        
        with sf.SoundFile(str(self.filepath)) as f:
            seed_data = f.read(block_samples, always_2d=False, fill_value=0.0)
            if seed_data.ndim > 1: seed_data = np.mean(seed_data, axis=1)
            seed_data = seed_data * self.cal_factor
            
            weighting_filter.initialize_state(seed_data)
            if band_bank: band_bank.initialize_state(seed_data)
            lp_detector.process(seed_data) 
            
            f.seek(0)

            current_time = 0.0
            block_gen = f.blocks(blocksize=block_samples, always_2d=False, fill_value=0.0)
            
            for chunk in block_gen:
                if chunk.ndim > 1: chunk = np.mean(chunk, axis=1)
                calibrated_chunk = chunk * self.cal_factor
                weighted_chunk = weighting_filter.process_chunk(calibrated_chunk)
                ms_broadband = np.mean(weighted_chunk**2)
                leq_block = 10 * np.log10(ms_broadband / (ref_pressure**2) + 1e-30)
                lp_block = lp_detector.process(weighted_chunk)
                
                result = {'time': current_time, 'leq': leq_block, 'lp': lp_block}
                
                if band_bank:
                    filtered_bands = band_bank.process_chunk(calibrated_chunk)
                    ms_bands = np.mean(filtered_bands**2, axis=0)
                    result['bands'] = 10 * np.log10(ms_bands / (ref_pressure**2) + 1e-30)
                    result['band_freqs'] = band_bank.frequencies

                yield result
                current_time += (block_size_ms / 1000.0)