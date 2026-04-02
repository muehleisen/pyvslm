import sys
import os
import argparse
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

def get_ansi_mask(fc, filter_class=1):
    """
    Returns absolute frequency arrays and transmission gain limit values 
    for the ANSI S1.11-2004 Octave Band spectral mask.
    """
    # Base-ten system G value for octave ratio
    G = 10**0.3

    # Define normalized frequencies (f/fm) as exponents of G
    epsilon = 1e-6
    exponents = np.array([
        0, 1/8, 1/4, 3/8, 1/2 - epsilon, 1/2, 1, 2, 3, 4, 6
    ])

    # Max attenuation of +infinity in the stopband is represented by a large number (100 dB)
    inf_db = 100.0

    # Limits (Minimum Attenuation, Maximum Attenuation) in dB 
    # Extracted from ANSI S1.11-2004 Table 1
    limits = {
        0: {
            'min': np.array([-0.15, -0.15, -0.15, -0.15, -0.15, 2.3, 18.0, 42.5, 62.0, 75.0, 75.0]),
            'max': np.array([ 0.15,  0.2,   0.4,   1.1,   4.5,   4.5, inf_db, inf_db, inf_db, inf_db, inf_db])
        },
        1: {
            'min': np.array([-0.3, -0.3, -0.3, -0.3, -0.3, 2.0, 17.5, 42.0, 61.0, 70.0, 70.0]),
            'max': np.array([ 0.3,  0.4,   0.6,   1.3,   5.0,   5.0, inf_db, inf_db, inf_db, inf_db, inf_db])
        },
        2: {
            'min': np.array([-0.5, -0.5, -0.5, -0.5, -0.5, 1.6, 16.5, 41.0, 55.0, 60.0, 60.0]),
            'max': np.array([ 0.5,  0.6,   0.8,   1.6,   5.5,   5.5, inf_db, inf_db, inf_db, inf_db, inf_db])
        }
    }

    selected_min = limits[filter_class]['min']
    selected_max = limits[filter_class]['max']

    # Mirror the exponents and limits for the negative exponents (f/fm < 1)
    neg_exponents = -exponents[::-1]
    
    # Combine negative and positive sides (removing the duplicate G^0 point)
    all_exponents = np.concatenate((neg_exponents[:-1], exponents))
    all_omega = G ** all_exponents
    all_min_att = np.concatenate((selected_min[::-1][:-1], selected_min))
    all_max_att = np.concatenate((selected_max[::-1][:-1], selected_max))

    # Convert Normalized Frequency to Absolute Frequency
    f_absolute = all_omega * fc
    
    # Invert Attenuation to Transmission Gain
    gain_upper_limit = -all_min_att
    gain_lower_limit = -all_max_att

    return f_absolute, gain_upper_limit, gain_lower_limit


def plot_octave_response(filter_class=1):
    print("Initializing Octave Filter Bank...")
    
    fs = 48000
    filter_order = 8 # Ensure you use the updated design_compliant_sos
    
    bank = OctaveFilterBank(fs, resolution=BandResolution.OCTAVE, order=filter_order)
    print(f"Generated {len(bank.frequencies)} bands: {bank.frequencies}")

    # Use 2^17 samples for high frequency resolution in FFT
    n_samples = 131072 
    impulse = np.zeros(n_samples)
    impulse[0] = 1.0 
    
    bank.initialize_state(np.zeros(1024))
    output_bands = bank.process_chunk(impulse)
    freqs = np.fft.rfftfreq(n_samples, d=1/fs)
    
    plt.figure(figsize=(14, 9))
    
    target_freqs = [63.0, 1000.0, 8000.0]
    colors = ['r', 'g', 'b']
    
    # Standard nominal frequencies from ANSI S1.11-2004 Table A1
    ansi_nominal_fcs = np.array([16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
    
    for i, target in enumerate(target_freqs):
        idx = np.argmin(np.abs(bank.frequencies - target))
        actual_fc = bank.frequencies[idx]
        
        # Snap to closest nominal frequency for the label
        nom_idx = np.argmin(np.abs(ansi_nominal_fcs - actual_fc))
        nominal_fc = ansi_nominal_fcs[nom_idx]
        
        # Format 31.5 correctly, and format the rest as integers
        nom_label = "31.5" if nominal_fc == 31.5 else f"{int(nominal_fc)}"
        
        # Get response
        resp = np.fft.rfft(output_bands[:, idx])
        mag_db = 20 * np.log10(np.abs(resp) + 1e-15)
        
        # Normalize to peak
        peak_idx = np.argmin(np.abs(freqs - actual_fc))
        ref_level = mag_db[peak_idx]
        norm_mag_db = mag_db - ref_level
        
        col = colors[i % len(colors)]
        
        # Plot Filter Response using the nominal label
        plt.semilogx(freqs, norm_mag_db, color=col, linewidth=2, label=f'Band {nom_label} Hz')
        
        # Overlay ANSI Mask (using the exact actual_fc for accurate bounds)
        mask_f, mask_gain_up, mask_gain_lo = get_ansi_mask(actual_fc, filter_class)
        
        lbl_up = f'Class {filter_class} Max Transmission' if i == 0 else None
        lbl_lo = f'Class {filter_class} Min Transmission' if i == 0 else None
        lbl_fill = f'Class {filter_class} Tolerance Region' if i == 0 else None
        
        # Shade the region between the bounds
        plt.fill_between(mask_f, mask_gain_lo, mask_gain_up, color='gray', alpha=0.15, label=lbl_fill)
            
        plt.plot(mask_f, mask_gain_up, 'k--', linewidth=1.5, label=lbl_up, alpha=0.8)
        plt.plot(mask_f, mask_gain_lo, 'k-.', linewidth=1.5, label=lbl_lo, alpha=0.8)
        
        # Update text annotation to use the nominal label
        plt.text(actual_fc, 5.0, f"{nom_label}Hz", ha='center', color=col, fontweight='bold')

    # Formatting
    plt.title(f"Octave Filter Bank Response vs ANSI S1.11-2004 Class {filter_class} Limits\n(Fs={fs} Hz, Order={filter_order})", fontsize=14)
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Normalized Transmission (dB)", fontsize=12)
    plt.grid(True, which='both', alpha=0.3, linestyle='--')
    
    plt.xlim(10, 24000)
    plt.ylim(-100, 10)
    
    plt.legend(loc='lower center', ncol=3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Octave-Band Filter Responses against ANSI S1.11-2004 Limits.")
    parser.add_argument('--class', dest='filter_class', type=int, choices=[0, 1, 2], default=1,
                        help="Filter accuracy class to plot (0, 1, or 2). Default is 1.")
    args = parser.parse_args()

    plot_octave_response(filter_class=args.filter_class)