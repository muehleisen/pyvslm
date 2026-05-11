# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pyvslm** is a Python-based Virtual Sound Level Meter (VSLM) — a port of the MATLAB `vslm.m` application.
The active implementation lives in the **root directory**.

---

## Environment Setup

Uses miniconda to manage the Python environment.

```bash
conda create -n pyvslm -c conda-forge python=3.12 numpy scipy matplotlib pyside6 pysoundfile python-sounddevice numba pydantic pyyaml pyinstaller pytest
conda activate pyvslm
```

Always install packages with `conda install -c conda-forge`; only use pip if a package is unavailable there.

## Running the Application

```bash
python run_pyvslm.py
```

## Running Tests

```bash
python -m pytest tests/
python -m pytest tests/test_engine.py                                       # single file
python -m pytest tests/test_engine.py::TestBroadbandLeq::test_steady_state_leq  # single test
python -m pytest tests/ -v
```

## Filter Visualisation Plots

```bash
python tests/plot_weighting_filters.py          # A/C weighting vs IEC 61672-1
python tests/plot_weighting_filters_compact.py  # 2×2 magnitude + error grid
python tests/plot_ansi_filters.py               # Octave bank vs IEC 61260-1
python tests/plot_ansi_filters.py third         # 1/3-octave version
```

---

## Architecture

The application follows an MVC pattern with Qt signals/slots for decoupling:

```
MainWindow (View)                         vslm/gui/main_window.py
    └── VSLMController (Controller)       vslm/controller.py
            ├── SettingsManager           vslm/settings.py
            ├── AnalysisWorker (QThread)  vslm/gui/worker.py
            └── exporter functions        vslm/dsp/exporter.py
```

**`StreamProcessor`** (`vslm/dsp/engine.py`) is the core DSP engine. It reads WAV files via `soundfile`,
applies calibration, and yields results block-by-block as a generator. Supports six analysis modes:

| mode_id | Type |
|---------|------|
| 0 | Lp (time-weighted SPL) |
| 1 | Leq (equivalent level) |
| 2 | Octave band |
| 3 | 1/3-octave band |
| 4 | PSD |
| 5 | Spectrogram |

**Signal chain for SPL modes**: raw audio → calibration factor → `WeightingFilter` (A/C/Z IIR) →
`TimeWeightingDetector` (Fast/Slow/Impulse exponential averaging) → dB SPL re 20 µPa.

**Band analysis**: `OctaveFilterBank` (`vslm/dsp/filters/octave_filters.py`) applies IIR bandpass filters
per centre frequency; state is seeded with the first block to avoid transient artefacts.

**PSD/Spectrogram**: Use `scipy.signal.welch` with configurable NFFT and window. PSD averages across chunks;
spectrogram builds a time×frequency matrix. Both are weighted post-FFT via `get_weighting_power_response`.

**Settings**: `AppSettings` is a Pydantic `BaseModel` serialised to `~/.vslm_settings.yaml` on shutdown
and loaded on startup. `SettingsManager` handles load/save with graceful fallback to defaults.

**Plot layer**: `ResultPlotter` (`vslm/gui/plot_manager.py`) renders all matplotlib figures into
`MatplotlibWidget` (`vslm/gui/plot_widget.py`), a `QWidget` embedding a `FigureCanvas` with a custom
toolbar that includes a Y-axis scaling button.

---

## Package Layout

```
vslm/
├── constants.py          StrEnum definitions, LEQ_INTERVAL_MAP, MODE_ID_MAP
├── settings.py           AppSettings (Pydantic), SettingsManager
├── controller.py         VSLMController — MVC glue, owns worker lifecycle
├── dsp/
│   ├── engine.py         StreamProcessor — block-by-block WAV analysis
│   ├── calibration.py    RMS measurement, factor_from_reference
│   ├── leq.py            LeqStats calculation, noise dose
│   ├── exporter.py       CSV export for all modes
│   └── filters/
│       ├── weighting_filters.py   IEC 61672-1 A/C/Z (MZT hybrid + minimax optimisation)
│       └── octave_filters.py      ANSI S1.11 octave / 1/3-octave IIR banks
└── gui/
    ├── main_window.py    MainWindow — all controls and layout
    ├── worker.py         AnalysisWorker (QThread)
    ├── plot_widget.py    MatplotlibWidget + custom toolbar
    ├── plot_manager.py   ResultPlotter — stateless rendering
    └── dialogs/
        ├── calibration.py  CalibrationDialog (manual + from-selection)
        ├── waveform.py     WaveformDialog (SpanSelector section picker)
        └── about.py        AboutDialog
tests/
├── test_engine.py                  StreamProcessor unit tests (synthetic WAV)
├── test_filters.py                 Weighting + octave filter compliance tests
├── plot_weighting_filters.py       A/C weighting response plots
├── plot_weighting_filters_compact.py  2×2 magnitude + error grid
└── plot_ansi_filters.py            Octave / 1/3-octave bank compliance plots
```

---

## GUI Layout Pattern

The left panel (330 px fixed) has a top section of global controls (file, weighting) and then an
**Analysis Mode group** that pairs mode radio buttons on the left with a `QStackedWidget` of
mode-specific options on the right.  Each stack page uses a vertical `QVBoxLayout` of
`QLabel + QComboBox` pairs followed by a stretch.  Mode-to-page mapping:

| mode_id | Stack page | Controls |
|---------|-----------|----------|
| 0 (Lp) | 0 | Plot interval combo |
| 1 (Leq) | 1 | Leq interval combo + dose standard combo |
| 2, 3 (bands) | 2 | *(empty)* |
| 4 (PSD) | 3 | FFT size combo + window combo |
| 5 (Spectrogram) | 4 | FFT size + slice duration + window + colour palette combos |

The colourmap combo has a live `currentTextChanged` connection that re-draws the spectrogram
immediately without re-running analysis (`_on_cmap_changed`).

---

## Key Conventions

- Python 3.12. Uses `match`/`case`, `X | Y` union types, `StrEnum`.
- All enums in `vslm/constants.py` are `StrEnum` — values serialise cleanly to YAML.
- `LEQ_INTERVAL_MAP` in `constants.py` is the single source of truth for interval labels and durations.
- Weighting filter tests use **IEC 61672-1 Class 1 tolerances** per frequency (not a flat ±0.5 dB).
  Frequencies above `fs / 2.2` are skipped — bilinear transform cannot replicate the analog response there.
- Tests generate temporary WAV files and clean up after themselves; no installed package required.

---

## Known Gaps / Future Work

- No PyInstaller build script yet
- PSD and spectrogram CSV export not implemented
- No integration test with a real calibrated recording
- Waveform dialog does not restore a previously-saved selection on re-open
- No dark-mode / HiDPI stylesheet applied
- `plot_manager.py` Leq plot stats layout could be improved for long dose standard names
