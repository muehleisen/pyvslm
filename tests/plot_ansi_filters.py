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

def get_class1_mask(fc):
    """
    Returns frequency arrays and limit values for the ANSI S1.11 / IEC 61260 Class 1 
    Octave Band spectral mask.
    """
    # Ratios (f / fc) for Class 1 Octave Band (Base 2 definition G=2)
    # derived from IEC 61260-1:2014 Table 2
    
    # --- Upper Limit (Maximum Permitted Transmission) ---
    r_upper = np.array([
        0.001,  # Extension
        0.125,  # 3 Octaves down (f/fc = 1/8)
        0.25,   # 2 Octaves down (f/fc = 1/4)
        0.5,    # 1 Octave down (f/fc = 1/2)
        0.7071, # Lower Band Edge (1/sqrt(2))
        1.4142, # Upper Band Edge (sqrt(2))
        2.0,    # 1 Octave up
        4.0,    # 2 Octaves up
        8.0,    # 3 Octaves up
        1000.0  # Extension
    ])
    
    # Limits in dB
    # -70dB is required at +/- 3 octaves for Class 1 (approximate practical limit)
    # -61dB at +/- 2 octaves
    # -16.1dB at +/- 1 octave
    l_upper = np.array([
        -70.0, # < 0.125
        -70.0, # 0.125
        -61.0, # 0.25
        -16.1, # 0.5
        0.3,   # Passband (+0.3)
        0.3,   # Passband (+0.3)
        -16.1, # 2.0
        -61.0, # 4.0
        -70.0, # 8.0
        -70.0  # > 8.0
    ])
    
    # --- Lower Limit (Minimum Permitted Transmission) ---
    r_lower = np.array([
        0.7071, # Lower Edge
        1.4142  # Upper Edge
    ])
    
    l_lower = np.array([
        -0.3,
        -0.3
    ])
    
    # Convert Ratios to Frequency
    f_upper = r_upper * fc
    f_lower = r_lower * fc
    
    return f_upper, l_upper, f_lower, l_lower

def plot_octave_response():
    print("Initializing Octave Filter Bank...")
    
    fs = 48000
    
    # --- UPDATE: Use Order 24 to meet Class 1 Requirements ---
    # Order 6 is too shallow for the strict IEC 61260 transition bands.
    filter_order = 24 
    
    # Create the bank
    bank = OctaveFilterBank(fs, resolution=BandResolution.OCTAVE, order=filter_order)
    
    print(f"Generated {len(bank.frequencies)} bands: {bank.frequencies}")

    # --- Generate Frequency Response via Impulse ---
    # Use 2^17 samples for high frequency resolution in FFT
    n_samples = 131072 
    impulse = np.zeros(n_samples)
    impulse[0] = 1.0 
    
    # Initialize state (flush filters)
    bank.initialize_state(np.zeros(1024))
    
    # Process
    output_bands = bank.process_chunk(impulse)
    
    # Compute FFT
    freqs = np.fft.rfftfreq(n_samples, d=1/fs)
    
    plt.figure(figsize=(14, 9))
    
    # --- Plot specific bands ---
    target_freqs = [63.0, 1000.0, 8000.0]
    colors = ['r', 'g', 'b']
    
    for i, target in enumerate(target_freqs):
        idx = np.argmin(np.abs(bank.frequencies - target))
        actual_fc = bank.frequencies[idx]
        
        # Get response
        resp = np.fft.rfft(output_bands[:, idx])
        mag_db = 20 * np.log10(np.abs(resp) + 1e-15)
        
        # Normalize to peak for clear shape comparison
        peak_idx = np.argmin(np.abs(freqs - actual_fc))
        ref_level = mag_db[peak_idx]
        norm_mag_db = mag_db - ref_level
        
        col = colors[i % len(colors)]
        
        # Plot Filter Response
        plt.semilogx(freqs, norm_mag_db, color=col, linewidth=2, label=f'Band {actual_fc:.0f} Hz')
        
        # --- Overlay Class 1 Mask ---
        mask_f_up, mask_l_up, mask_f_lo, mask_l_lo = get_class1_mask(actual_fc)
        
        lbl_up = 'Class 1 Max Limit' if i == 0 else None
        lbl_lo = 'Class 1 Min Limit' if i == 0 else None
            
        plt.plot(mask_f_up, mask_l_up, 'k--', linewidth=1.5, label=lbl_up, alpha=0.8)
        plt.plot(mask_f_lo, mask_l_lo, 'k-.', linewidth=1.5, label=lbl_lo, alpha=0.8)
        
        plt.text(actual_fc, 2.5, f"{actual_fc:.0f}Hz", ha='center', color=col, fontweight='bold')

    # Formatting
    plt.title(f"Octave Filter Bank Response vs ANSI S1.11 / IEC 61260 Class 1 Limits\n(Fs={fs} Hz, Order={filter_order})", fontsize=14)
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Attenuation (dB)", fontsize=12)
    plt.grid(True, which='both', alpha=0.3, linestyle='--')
    
    plt.xlim(10, 24000)
    plt.ylim(-100, 10) # Extended Y-range to see deep stopband
    
    plt.legend(loc='lower center', ncol=3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_octave_response()