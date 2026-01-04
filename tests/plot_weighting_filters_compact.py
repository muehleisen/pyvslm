# plot_weighting_filters_grid.py
#
# Updates plot_weighting_filters.py to use the 4-subplot layout.
# - Plots ALL supported sampling rates.
# - Compares against Analog Reference.
# - Shows IEC 61672 Class 1 Tolerances (Modern Standard).
# - Shows ANSI S1.4 Type 0 Tolerances (Legacy Laboratory Standard).
# - Adds shading for Type 0 Tolerance range.

import sys
import os
import scipy.signal as sg
import numpy as np
import matplotlib.pyplot as plt

# --- Path Setup to import vslm ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(current_dir, 'vslm')):
    sys.path.insert(0, current_dir)
else:
    sys.path.insert(0, os.path.dirname(current_dir))

from vslm.filters.weighting_filters import WeightingFilter, SUPPORTED_FS

# --- 1. Define Analog Prototypes (ANSI S1.42 Reference) ---

# A-weighting poles/zeros
fa1 = 20.598997
fa2 = 107.65265
fa3 = 737.86223
fa4 = 12194.217

A1000 = 1.9997
Ka = (2*np.pi*fa4)**2 * (10**(A1000/20))
Za = [0, 0, 0, 0]
Pa = 2*np.pi*np.array([fa4, fa4, fa3, fa2, fa1, fa1])

# C-weighting poles/zeros
C1000 = 0.0619
Kc = (2*np.pi*fa4)**2 * (10**(C1000/20))
Zc = [0, 0]
Pc = 2*np.pi*np.array([fa4, fa1, fa4, fa1])

# --- 2. Define Tolerances ---

def get_tolerances():
    """
    Returns frequency arrays and tolerance masks.
    """
    freqs = np.array([10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 
                      100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 
                      1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 
                      8000, 10000, 12500, 16000, 20000])

    # --- ANSI S1.4 Type 0 (Laboratory Reference) ---
    up0 = np.array([2, 2, 2, 2, 1.5, 1, 1, 1, 1, 1, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
                    0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 1, 1, 1, 2, 2, 2, 2])
    low0 = np.array([5, 3, 3, 2, 1.5, 1, 1, 1, 1, 1, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
                    0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 1, 1.5, 2, 3, 3, 3, 3])

    # --- IEC 61672-1 Class 1 (Modern Precision) ---
    up1 = []
    low1 = []
    for f in freqs:
        if f < 25:
             p, m = 2.5, np.inf
        elif f < 100:
            p, m = 1.5, 1.5
        elif f <= 5000:
            p, m = 1.1, 1.1 
        elif f <= 16000:
            p, m = 2.5, 2.5
        else:
            p, m = 3.0, 5.0
        up1.append(p)
        low1.append(m)
        
    return freqs, up0, low0, np.array(up1), np.array(low1)

ftol, up0, low0, up1, low1 = get_tolerances()

# --- 3. Setup Plotting ---
FS_list = SUPPORTED_FS
linestyles = ['-', '--', '-.', ':'] * 2 
colors = plt.cm.jet(np.linspace(0, 0.9, len(FS_list)))

# Create Figure with 2x2 Layout
fig, axs = plt.subplots(2, 2, figsize=(12, 6))
ax_amag = axs[0, 0]  # Top Left: A Mag
ax_cmag = axs[0, 1]  # Top Right: C Mag
ax_aerr = axs[1, 0]  # Bottom Left: A Error
ax_cerr = axs[1, 1]  # Bottom Right: C Error

f = np.logspace(1, np.log10(30000), 2048)

# --- 4. Main Loop Over Sampling Rates ---
for i, fs in enumerate(FS_list):
    wn = f * 2 * np.pi / fs
    ls = linestyles[i % len(linestyles)]
    color = colors[i]
    lbl = f'{fs} Hz'

    try:
        # Design Filters
        wf_a = WeightingFilter(fs, 'A')
        wf_c = WeightingFilter(fs, 'C')
        
        # Calculate Analog Response (Reference)
        _, ha = sg.freqs_zpk(Za, Pa, Ka, wn * fs)
        _, hc = sg.freqs_zpk(Zc, Pc, Kc, wn * fs)
        
        hadb = 20 * np.log10(np.abs(ha))
        hcdb = 20 * np.log10(np.abs(hc))

        # Calculate Digital Response
        _, hza = sg.sosfreqz(wf_a.sos, worN=wn)
        _, hzc = sg.sosfreqz(wf_c.sos, worN=wn)
        
        hzadb = 20 * np.log10(np.abs(hza) + 1e-15)
        hzcdb = 20 * np.log10(np.abs(hzc) + 1e-15)

        # Plot Magnitude (Digital)
        ax_amag.semilogx(f, hzadb, label=lbl, color=color, linestyle=ls, linewidth=1.2)
        ax_cmag.semilogx(f, hzcdb, label=lbl, color=color, linestyle=ls, linewidth=1.2)

        # Plot Error (Digital - Analog)
        diffa = hzadb - hadb
        diffc = hzcdb - hcdb
        
        # Limit error plot range to Nyquist
        mask = f < fs/2
        ax_aerr.semilogx(f[mask], diffa[mask], label=lbl, color=color, linestyle=ls, linewidth=1.2)
        ax_cerr.semilogx(f[mask], diffc[mask], label=lbl, color=color, linestyle=ls, linewidth=1.2)

    except Exception as e:
        print(f"Skipping {fs}: {e}")

# --- 5. Add Analog Reference & Tolerances ---

# Plot Analog Reference on Magnitude plots
_, ha_ref = sg.freqs_zpk(Za, Pa, Ka, f * 2 * np.pi)
_, hc_ref = sg.freqs_zpk(Zc, Pc, Kc, f * 2 * np.pi)
ha_ref_db = 20 * np.log10(np.abs(ha_ref))
hc_ref_db = 20 * np.log10(np.abs(hc_ref))

ax_amag.semilogx(f, ha_ref_db, 'k--', linewidth=2, label='Analog Ref', alpha=0.6)
ax_cmag.semilogx(f, hc_ref_db, 'k--', linewidth=2, label='Analog Ref', alpha=0.6)

# Plot Tolerance Masks on Error plots
for ax in [ax_aerr, ax_cerr]:
    # 1. Shading for Type 0 Tolerance (Light Gray)
    ax.fill_between(ftol, -low0, up0, color='gray', alpha=0.15, label='Type 0 Range')

    # 2. Modern Class 1 Limits (Red)
    ax.semilogx(ftol, up1, 'r-', linewidth=2, alpha=0.3, label='Class 1 (IEC 61672)')
    ax.semilogx(ftol, -low1, 'r-', linewidth=2, alpha=0.3)
    
    # 3. Legacy Type 0 Lines (Black Dotted)
    ax.semilogx(ftol, up0, 'k:', linewidth=1.5, alpha=0.6, label='Type 0 (ANSI S1.4)')
    ax.semilogx(ftol, -low0, 'k:', linewidth=1.5, alpha=0.6)

# --- 6. Formatting ---

# A-Weighting Magnitude
ax_amag.set_title('A-Weighting Magnitude Response')
ax_amag.set_ylabel('Magnitude (dB)')
ax_amag.set_ylim(-70, 10)
ax_amag.set_xlim(10, 30000)
ax_amag.grid(True, which='both', alpha=0.3)
ax_amag.legend(fontsize='x-small', ncol=2)

# C-Weighting Magnitude
ax_cmag.set_title('C-Weighting Magnitude Response')
ax_cmag.set_ylim(-70, 10)
ax_cmag.set_xlim(10, 30000)
ax_cmag.grid(True, which='both', alpha=0.3)

# A-Weighting Error
ax_aerr.set_title('A-Weighting Error (Digital - Analog)')
ax_aerr.set_ylabel('Deviation (dB)')
ax_aerr.set_xlabel('Frequency (Hz)')
ax_aerr.set_ylim(-2, 2)
ax_aerr.set_xlim(10, 30000)
ax_aerr.grid(True, which='both', alpha=0.3)
ax_aerr.legend(loc='upper left', fontsize='x-small')

# C-Weighting Error
ax_cerr.set_title('C-Weighting Error (Digital - Analog)')
ax_cerr.set_xlabel('Frequency (Hz)')
ax_cerr.set_ylim(-2, 2)
ax_cerr.set_xlim(10, 30000)
ax_cerr.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.show()