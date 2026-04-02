import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Plot ANSI S1.11-2004 Octave-Band Filter Tolerance Limits.")
    parser.add_argument('--class', dest='filter_class', type=int, choices=[0, 1, 2], default=1,
                        help="Filter accuracy class to plot (0, 1, or 2). Default is 1.")
    args = parser.parse_args()

    # Base-ten system G value for octave ratio
    G = 10**0.3

    # Define normalized frequencies (f/fm) as exponents of G
    # We use a small epsilon for the discontinuity at the G^(+/- 1/2) bandedge
    epsilon = 1e-6
    exponents = np.array([
        0, 1/8, 1/4, 3/8, 1/2 - epsilon, 1/2, 1, 2, 3, 4, 6
    ])

    # Max attenuation of +infinity in the stopband is represented by a large number
    inf_db = 100

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

    # Select the requested class
    selected_min = limits[args.filter_class]['min']
    selected_max = limits[args.filter_class]['max']

    # Mirror the exponents and limits for the negative exponents (f/fm < 1)
    neg_exponents = -exponents[::-1]
    neg_min_att = selected_min[::-1]
    neg_max_att = selected_max[::-1]

    # Combine negative and positive sides (removing the duplicate G^0 point)
    all_exponents = np.concatenate((neg_exponents[:-1], exponents))
    all_omega = G ** all_exponents
    all_min_att = np.concatenate((neg_min_att[:-1], selected_min))
    all_max_att = np.concatenate((neg_max_att[:-1], selected_max))

    # Plotting the bounds
    plt.figure(figsize=(10, 6))

    # Fill the valid tolerance region
    plt.fill_between(all_omega, all_min_att, all_max_att, color='skyblue', alpha=0.3, 
                     label=f'Class {args.filter_class} Valid Tolerance Region')

    # Plot the bounding lines
    plt.plot(all_omega, all_min_att, 'b--', linewidth=1.5, label='Minimum Attenuation Limit')
    plt.plot(all_omega, all_max_att, 'r-', linewidth=1.5, label='Maximum Attenuation Limit')

    # Format the plot to match ANSI S1.11-2004 visual standards
    plt.xscale('log')
    plt.xlim(G**-5, G**5)
    plt.ylim(80, -5) # Inverted y-axis to match Figure 1 standard

    plt.title(f'ANSI S1.11-2004 Octave-Band Filter Tolerance Limits (Class {args.filter_class})')
    plt.xlabel('Normalized Frequency ($f/f_m$)')
    plt.ylabel('Relative Attenuation $\Delta A$ (dB)')

    # Add gridlines for easier visual checking
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='lower center')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()