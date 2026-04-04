#!/usr/bin/env python3
"""
Plot ANSI S1.11-2004 octave-band filter tolerance bands for Class 0, 1, or 2.

Table 1 from the standard defines the minimum and maximum relative attenuation
limits at breakpoint normalized frequencies f/fm = Omega for octave-band filters.
Between breakpoints, limits are interpolated linearly on a log-frequency scale
per equation (12) of the standard.

Usage:
    python plot_ansi_s111_filter.py              # Class 0 (default)
    python plot_ansi_s111_filter.py --class 1
    python plot_ansi_s111_filter.py --class 2
    python plot_ansi_s111_filter.py --class all  # Overlay all three classes
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Table 1: Breakpoint normalized frequencies and attenuation limits
# ---------------------------------------------------------------------------
# G = 10^(3/10) (base-ten octave ratio, preferred per the standard)
G = 10 ** (3 / 10)

# Each row: (Omega, min_class0, max_class0, min_class1, max_class1, min_class2, max_class2)
# +inf is represented as np.inf.  The bandedge discontinuity is handled specially.
#
# Per Table 1 footnote (a): at frequencies OUTSIDE the bandedge (i.e. |Omega| > G^(1/2))
# the maximum limit jumps to +inf.  The passband maximum caps are listed for the
# region just inside the bandedge.

TABLE1 = [
    # Omega           min0    max0    min1    max1    min2    max2
    (G**0,            -0.15,  +0.15,  -0.3,   +0.3,   -0.5,   +0.5),
    (G**(+1/8),       -0.15,  +0.2,   -0.3,   +0.4,   -0.5,   +0.6),
    (G**(+1/4),       -0.15,  +0.4,   -0.3,   +0.6,   -0.5,   +0.8),
    (G**(+3/8),       -0.15,  +1.1,   -0.3,   +1.3,   -0.5,   +1.6),
    # Limit approaching bandedge from inside (lim e->0 of G^(1/2 - e))
    (G**(+1/2)*0.9999,-0.15,  +4.5,   -0.3,   +5.0,   -0.5,   +5.5),
    # At and beyond the bandedge: minimum jumps up, maximum -> +inf
    (G**(+1/2),       +2.3,   np.inf, +2.0,   np.inf, +1.6,   np.inf),
    (G**(+1),         +18.0,  np.inf, +17.5,  np.inf, +16.5,  np.inf),
    (G**(+2),         +42.5,  np.inf, +42.0,  np.inf, +41.0,  np.inf),
    (G**(+3),         +62.0,  np.inf, +61.0,  np.inf, +55.0,  np.inf),
    (G**(+4),         +75.0,  np.inf, +70.0,  np.inf, +60.0,  np.inf),
]

# Class index helpers
CLASS_COLS = {
    0: (1, 2),   # (min_col, max_col) indices into TABLE1 rows
    1: (3, 4),
    2: (5, 6),
}

CLASS_COLORS = {
    0: "#1f77b4",   # blue
    1: "#ff7f0e",   # orange
    2: "#2ca02c",   # green
}

CLASS_LABELS = {
    0: "Class 0",
    1: "Class 1",
    2: "Class 2",
}


def build_breakpoints():
    """Return arrays of (omega, min0, max0, min1, max1, min2, max2) for positive Omega."""
    omegas = np.array([row[0] for row in TABLE1])
    data = {}
    for cls, (mc, xc) in CLASS_COLS.items():
        mins = np.array([row[mc] for row in TABLE1])
        maxs = np.array([row[xc] for row in TABLE1])
        data[cls] = (mins, maxs)
    return omegas, data


def interpolate_limit(omega_query, omegas_bp, limits_bp):
    """
    Equation (12): linear interpolation of attenuation limits on a log-frequency axis.
    For omega_query beyond the last breakpoint, return the last limit value.
    Handles +inf limits by returning +inf directly.
    """
    results = np.empty_like(omega_query, dtype=float)
    for i, oq in enumerate(omega_query):
        # Find surrounding breakpoints
        idx = np.searchsorted(omegas_bp, oq, side='right') - 1
        idx = np.clip(idx, 0, len(omegas_bp) - 2)

        oa, ob = omegas_bp[idx], omegas_bp[idx + 1]
        la, lb = limits_bp[idx], limits_bp[idx + 1]

        if np.isinf(lb) or np.isinf(la):
            results[i] = np.inf
        elif oa == ob:
            results[i] = la
        else:
            # Eq. (12): linear in log(omega)
            frac = np.log10(oq / oa) / np.log10(ob / oa)
            results[i] = la + (lb - la) * frac
    return results


def build_full_curve(cls, n_points=2000):
    """
    Build symmetric (positive and negative omega) arrays for the tolerance band
    of a given class.  Returns (omega_plot, min_dB, max_dB) where omega_plot
    goes from G^{-4} to G^{+4} (i.e. several octaves either side of midband).
    """
    omegas_bp, data = build_breakpoints()
    mins_bp, maxs_bp = data[cls]

    # Positive side: from 1.0 to G^4
    omega_pos = np.logspace(np.log10(1.0), np.log10(G**4), n_points)
    min_pos = interpolate_limit(omega_pos, omegas_bp, mins_bp)
    max_pos = interpolate_limit(omega_pos, omegas_bp, maxs_bp)

    # Negative side (mirror): from G^{-4} to 1.0
    # By symmetry of bandpass filters, omega_neg = 1/omega_pos
    omega_neg = np.flip(1.0 / omega_pos[1:])   # exclude duplicate at 1.0
    min_neg = np.flip(min_pos[1:])
    max_neg = np.flip(max_pos[1:])

    omega_full = np.concatenate([omega_neg, omega_pos])
    min_full   = np.concatenate([min_neg,   min_pos])
    max_full   = np.concatenate([max_neg,   max_pos])

    return omega_full, min_full, max_full


def cap_for_plot(arr, cap=85.0):
    """Replace +inf with a finite cap for plotting purposes."""
    out = arr.copy()
    out[np.isinf(out)] = cap
    return out


def plot_classes(classes_to_plot, title_suffix=""):
    fig, ax = plt.subplots(figsize=(11, 7))

    CAP = 85.0   # dB cap for +inf regions

    for cls in classes_to_plot:
        color = CLASS_COLORS[cls]
        label = CLASS_LABELS[cls]
        omega, lo, hi = build_full_curve(cls)

        lo_plot = cap_for_plot(lo, CAP)
        hi_plot = cap_for_plot(hi, CAP)

        # Shade the valid (tolerance) region between min and max attenuation
        ax.fill_between(
            omega, lo_plot, hi_plot,
            alpha=0.20, color=color, linewidth=0,
            label=f"{label} tolerance band",
        )
        # Draw boundary lines
        ax.plot(omega, lo_plot, color=color, linewidth=1.5, linestyle="--",
                label=f"{label} min limit")
        ax.plot(omega, hi_plot, color=color, linewidth=1.5, linestyle="-",
                label=f"{label} max limit")

    # Mark the bandedge frequencies
    for x in [G**(+0.5), G**(-0.5)]:
        ax.axvline(x, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)

    ax.axvline(1.0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)

    # Axis formatting
    ax.set_xscale("log")
    ax.invert_yaxis()          # convention: more attenuation downward
    ax.set_xlabel("Normalized frequency  f / f$_m$  (log scale)", fontsize=12)
    ax.set_ylabel("Relative attenuation  ΔA  (dB)", fontsize=12)
    ax.set_xlim(G**-4, G**4)
    ax.set_ylim(CAP + 2, -3)   # inverted: passband near top, stopband at bottom

    # Ticks at octave-ratio breakpoints
    tick_exps = [-4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4]
    tick_vals = [G**e for e in tick_exps]
    tick_lbls = [
        f"G$^{{{{-4}}}}$", f"G$^{{{{-3}}}}$", f"G$^{{{{-2}}}}$",
        f"G$^{{{{-1}}}}$", f"G$^{{{{-1/2}}}}$", "G$^0$",
        f"G$^{{{{+1/2}}}}$", f"G$^{{{{+1}}}}$", f"G$^{{{{+2}}}}$",
        f"G$^{{{{+3}}}}$", f"G$^{{{{+4}}}}$",
    ]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_lbls, fontsize=9)

    # Y-ticks at standard attenuation values
    y_ticks = [0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{v}" for v in y_ticks])

    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.grid(True, which="major", linestyle="-", alpha=0.25)

    # Annotate the ±inf cap line
    ax.axhline(CAP, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.text(G**3.7, CAP - 1.5, f"+∞ (shown as {CAP} dB)",
            fontsize=7, color="gray", va="top", ha="right")

    # Annotate bandedge
    ax.text(G**0.5, -2.0, "f$_2$ = G$^{+1/2}$f$_m$",
            fontsize=7.5, color="gray", ha="center")
    ax.text(G**-0.5, -2.0, "f$_1$ = G$^{-1/2}$f$_m$",
            fontsize=7.5, color="gray", ha="center")

    # Legend – deduplicate entries produced by fill_between + 2 lines per class
    handles, lbls = ax.get_legend_handles_labels()
    # Group into one patch per class for cleaner legend
    legend_handles = []
    for cls in classes_to_plot:
        color = CLASS_COLORS[cls]
        patch = mpatches.Patch(facecolor=color, alpha=0.4,
                               edgecolor=color, linewidth=1.5,
                               label=CLASS_LABELS[cls])
        legend_handles.append(patch)

    ax.legend(handles=legend_handles, loc="lower center",
              fontsize=10, framealpha=0.85)

    title = "ANSI S1.11-2004  –  Octave-Band Filter Tolerance Bands"
    if title_suffix:
        title += f"  ({title_suffix})"
    ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot ANSI S1.11-2004 octave-band filter tolerance bands.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--class", dest="filter_class", default="0",
        choices=["0", "1", "2", "all"],
        help="Filter class to plot: 0, 1, 2, or 'all' (default: 0)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save figure to this file instead of displaying it (e.g. filter.png)",
    )
    args = parser.parse_args()

    if args.filter_class == "all":
        classes = [0, 1, 2]
        suffix = "All Classes"
    else:
        cls = int(args.filter_class)
        classes = [cls]
        suffix = CLASS_LABELS[cls]

    fig = plot_classes(classes, title_suffix=suffix)

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()