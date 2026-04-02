# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pyvslm** is a Python-based Virtual Sound Level Meter (VSLM) — a port of the MATLAB `vslm.m` application.
The active implementation lives in the **root directory**. `gemini_conversion/` is the prior implementation kept for reference.

## Environment Setup

```bash
conda create -n vslm_env -c conda-forge python=3.12 numpy scipy matplotlib pyside6 pysoundfile python-sounddevice numba pydantic pyyaml pyinstaller pytest
conda activate vslm_env
```

## Running the Application

```bash
python run_pyvslm.py
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run a single file
python -m pytest tests/test_engine.py

# Run a specific test
python -m pytest tests/test_engine.py::TestBroadbandLeq::test_steady_state_leq
```

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
applies calibration, and yields results as a generator. Supports six analysis modes:

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
├── test_engine.py        StreamProcessor unit tests (synthetic WAV)
└── test_filters.py       Weighting + octave filter compliance tests
```

## Key Conventions

- Python 3.12. Uses `match`/`case`, `X | Y` union types, `StrEnum`.
- All enums in `vslm/constants.py` are `StrEnum` — values serialise cleanly to YAML.
- `LEQ_INTERVAL_MAP` in `constants.py` is the single source of truth for interval labels and durations.
- Weighting filter tests use **IEC 61672-1 Class 1 tolerances** per frequency (not a flat ±0.5 dB).
  Frequencies above `fs / 2.2` are skipped — bilinear transform cannot replicate the analog response there.
- Tests generate temporary WAV files and clean up after themselves; no installed package required.
