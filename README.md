# pyvslm — Virtual Sound Level Meter

A Python implementation of a virtual sound level meter (VSLM) for post-processing digital audio recordings as if they were measured by an ANSI-compliant sound level meter.  Supports A, C, and Z frequency weightings; Fast, Slow, and Impulse time weightings; octave and 1/3-octave band analysis; PSD; and spectrogram modes.

---

## Requirements

- [Miniforge](https://github.com/conda-forge/miniforge) or Anaconda with the `conda-forge` channel
- Windows, macOS, or Linux

---

## Environment Setup

```bash
conda create -n vslm_env -c conda-forge \
    python=3.12 numpy scipy matplotlib pyside6 \
    pysoundfile python-sounddevice numba \
    pydantic pyyaml pyinstaller pytest

conda activate vslm_env
```

---

## Running the Application

```bash
conda activate vslm_env
python run_pyvslm.py
```

The GUI will open.  Load a WAV file with the **Load WAV** button, optionally select a sub-section with **Select Section**, set your analysis mode and parameters, then click **ANALYSE**.

---

## Running the Tests

### Unit tests

```bash
conda activate vslm_env

# Run all unit tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_engine.py -v
python -m pytest tests/test_filters.py -v

# Run a specific test
python -m pytest tests/test_engine.py::TestBroadbandLeq::test_steady_state_leq
```

### Filter visualisation plots

These scripts display matplotlib figures — useful for visually verifying filter compliance.

```bash
# A and C weighting responses vs IEC 61672-1 ideal curve + Class 1 tolerance band
python tests/plot_weighting_filters.py

# 2×2 grid: magnitude and error (digital − analogue) for all sample rates,
# with both IEC 61672-1 Class 1 and ANSI S1.4 Type 0 tolerance overlays
python tests/plot_weighting_filters_compact.py

# Octave filter bank response vs IEC 61260-1 / ANSI S1.11 Class 1 spectral mask
python tests/plot_ansi_filters.py

# 1/3-octave version of the above
python tests/plot_ansi_filters.py third
```

---

## Analysis Modes

| Mode | Description |
|------|-------------|
| Level vs Time (Lp) | Time-weighted instantaneous SPL history |
| Leq Analysis | Equivalent continuous level with noise dose |
| Octave Bands | 1-octave IIR filter bank (ANSI S1.11) |
| 1/3 Octave Bands | 1/3-octave IIR filter bank (ANSI S1.11) |
| PSD | Power spectral density via Welch's method |
| Spectrogram | Short-time PSD vs time |

---

## Project Structure

```
run_pyvslm.py          Application entry point
vslm/
├── constants.py       Enumerations, interval map, mode map
├── settings.py        Pydantic settings model, load/save
├── controller.py      MVC controller
├── dsp/
│   ├── engine.py      StreamProcessor — core analysis engine
│   ├── calibration.py Acoustic calibration utilities
│   ├── leq.py         Leq statistics and noise dose
│   ├── exporter.py    CSV export
│   └── filters/
│       ├── weighting_filters.py   IEC 61672-1 A/C/Z filters
│       └── octave_filters.py      Octave / 1/3-octave filter banks
└── gui/
    ├── main_window.py
    ├── worker.py
    ├── plot_widget.py
    ├── plot_manager.py
    └── dialogs/
        ├── calibration.py
        ├── waveform.py
        └── about.py
tests/
├── test_engine.py                  StreamProcessor unit tests
├── test_filters.py                 Filter compliance tests (IEC 61672-1 Class 1)
├── plot_weighting_filters.py       A/C weighting response plots
├── plot_weighting_filters_compact.py  2×2 magnitude + error grid
└── plot_ansi_filters.py            Octave / 1/3-octave bank compliance plots
```

---

## Settings

Application settings are saved automatically to `~/.vslm_settings.yaml` on exit and restored on next launch.  You can also save and load named settings files via **File → Settings**.
