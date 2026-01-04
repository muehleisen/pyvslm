import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg

# --- Path Setup ---
# Ensure we can import from the 'vslm' package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from vslm.filters.octave_filters import OctaveFilterBank
from vslm.constants import BandResolution

def get_ansi_s111_class1_limits(f_center):
    """
    Returns frequency (Hz) and limit (dB) arrays for ANSI S1.11 / IEC 61260 Class 1.
    Defined relative to center frequency fc.
    """
    # Breakpoints for Class 1 Octave Band (Base 10) from IEC 61260-1:2014
    # G = 10^(3/10) (Base 10 system) or 2 (Base 2)
    # Ratios f/fc
    
    # These approximate the mask for plotting:
    # (Ratio, Max dB, Min dB)
    # The standard defines these piecewise.
    
    # Passband region
    ratios = np.array([0.1, 0.18, 0.4, 0.6, 0.8, 1/np.sqrt(2), 1.0, np.sqrt(2), 1.3, 1.8, 3.0, 6.0, 10.0])
    
    # We will generate a dense mask for smoother plotting
    f_ratio = np.logspace(np.log10(0.1), np.log10(10), 500)
    
    # IEC 61260 Class 1 Limits (Simplified logic for visualization)
    upper_limit = np.zeros_like(f_ratio)
    lower_limit = np.zeros_like(f_ratio)
    
    for i, r in enumerate(f_ratio):
        # Symmetrical usually, working with log ratio Q = (f/fc)
        # 1/1 Octave Bandwidth Limits
        
        # --- Upper Limit ---
        if 1/np.sqrt(2) <= r <= np.sqrt(2):
            upper_limit[i] = 0.3
        else:
            # Transition/Stopband upper limit
            # This is a loose approximation of the spectral mask
            upper_limit[i] = 0.3 # Technically extends, but usually we just care about passband ripple here
            
            # The standard has specific equations for the skirts, 
            # but usually just checking the passband box is sufficient for basic verification.
            if r < 0.5 or r > 2.0:
                 upper_limit[i] = -0.1 # Should drop off, but max limit is usually unbounded downwards in stopband
        
        # --- Lower Limit ---
        # Passband: +/- 0.3 dB within the central region
        if (1/np.sqrt(2))*1.02 <= r <= (np.sqrt(2))/1.02: # Slightly narrower than band edges
            lower_limit[i] = -0.3
        elif r < 0.1 or r > 10.0:
            lower_limit[i] = -np.inf
        else:
            # Transition region (Slope)
            # A rough guide for 1-octave Class 1:
            # Attenuation >= 70dB at f/fc < 0.12 or f/fc > 8
            lower_limit[i] = -np.inf # Complex to plot full mask procedurally without lookup table
            
    return f_ratio * f_center, upper_limit, lower_limit

def get_tolerance_box(fc):
    """
    Returns the box corners for the strict ANSI S1.11 Class 1 Passband Tolerance.
    Passband: fc / sqrt(2) to fc * sqrt(2)
    Ripple: +/- 0.3 dB
    """
    f_lower = fc / np.sqrt(2)
    f_upper = fc * np.sqrt(2)
    
    return [f_lower, f_upper, f_upper, f_lower, f_lower], [0.3, 0.3, -0.3, -0.3, 0.3]

def plot_octave_response():
    print("Initializing Octave Filter Bank...")
    
    fs = 48000
    # Create the bank
    # Note: We assume order=6 or similar which is typical for Class 1 compliance
    bank = OctaveFilterBank(fs, resolution=BandResolution.OCTAVE, order=6)
    
    print(f"Generated {len(bank.frequencies)} bands: {bank.frequencies}")

    # --- Generate Frequency Response via Impulse ---
    # We use an impulse response to characterize the filter regardless of implementation (SOS/BA)
    n_samples = 32768 # Sufficient frequency resolution
    impulse = np.zeros((n_samples, 1)) # 2D array [samples, channels]
    impulse[0, 0] = 1.0 * np.sqrt(2) # Unit impulse, scaled? No, just 1.0. 
    # NOTE: analysis_engine usually expects [samples] or [samples, channels]
    # The filter bank expects [samples]
    
    impulse_flat = impulse.flatten()
    
    # Initialize state (flush filters)
    bank.initialize_state(np.zeros(1024))
    
    # Process
    # This returns [samples, n_bands]
    output_bands = bank.process_chunk(impulse_flat)
    
    # Compute FFT
    freqs = np.fft.rfftfreq(n_samples, d=1/fs)
    
    plt.figure(figsize=(12, 8))
    
    # --- Plot specific bands ---
    # Let's verify 1kHz, 63Hz, and 8kHz
    target_freqs = [63.0, 1000.0, 8000.0]
    
    colors = ['r', 'g', 'b']
    
    for i, target in enumerate(target_freqs):
        # Find nearest band index
        idx = np.argmin(np.abs(bank.frequencies - target))
        actual_fc = bank.frequencies[idx]
        
        # Get response
        resp = np.fft.rfft(output_bands[:, idx])
        mag_db = 20 * np.log10(np.abs(resp) + 1e-15)
        
        # Normalize to peak (Insertion loss should be ~0dB)
        peak_val = np.max(mag_db)
        mag_db -= peak_val
        
        col = colors[i % len(colors)]
        plt.semilogx(freqs, mag_db, color=col, linewidth=1.5, label=f'Band {actual_fc:.0f} Hz')
        
        # Overlay Tolerance Box (Passband)
        box_f, box_a = get_tolerance_box(actual_fc)
        if i == 0: # Only label one
            plt.plot(box_f, box_a, 'k--', linewidth=2, label='ANSI S1.11 Class 1 Limits')
        else:
            plt.plot(box_f, box_a, 'k--', linewidth=2)
            
        # Optional: Add text marker
        plt.text(actual_fc, 1.5, f"{actual_fc:.0f}Hz", ha='center', color=col, fontweight='bold')

    # Formatting
    plt.title(f"Octave Filter Bank Response (Fs={fs}Hz) vs ANSI S1.11 Class 1 Limits")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, which='both', alpha=0.3)
    plt.xlim(20, 24000)
    plt.ylim(-60, 5) # Focus on passband and skirts
    
    # Zoom in on Y-axis?
    # plt.ylim(-2, 1) # Uncomment to see passband ripple clearly
    
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_octave_response()