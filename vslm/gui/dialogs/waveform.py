"""
Waveform viewer dialog for selecting a time region within a WAV file.

Uses matplotlib with a SpanSelector widget — no pyqtgraph dependency.
The waveform is displayed as a decimated min/max envelope so that even
multi-hour files render quickly.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("QtAgg")

import numpy as np
import soundfile as sf
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QVBoxLayout,
)

_TARGET_POINTS = 10_000   # envelope resolution


def _build_envelope(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Return (time_axis, env_min, env_max, duration) for a decimated waveform.

    Reads the file in streaming blocks so RAM usage stays low.
    """
    info  = sf.info(filepath)
    total = info.frames
    step  = max(1, total // _TARGET_POINTS)
    mins, maxs = [], []

    with sf.SoundFile(filepath) as f:
        while f.tell() < total:
            data = f.read(step * 10, always_2d=True)
            mono = np.mean(data, axis=1) if data.shape[1] > 1 else data[:, 0]
            trim = len(mono) % step
            if trim:
                mono = mono[:-trim]
            if len(mono) == 0:
                continue
            shaped = mono.reshape(-1, step)
            mins.append(shaped.min(axis=1))
            maxs.append(shaped.max(axis=1))

    env_min = np.concatenate(mins)
    env_max = np.concatenate(maxs)
    t_axis  = np.linspace(0.0, info.duration, len(env_min))
    return t_axis, env_min, env_max, info.duration


class WaveformDialog(QDialog):
    """
    Modal dialog showing a waveform envelope with a draggable selection region.

    After ``exec()`` returns Accepted, call :meth:`get_selection` for the
    chosen start/end times.
    """

    def __init__(self, filepath: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select File Section")
        self.resize(920, 420)
        self._filepath = filepath
        self._start    = 0.0
        self._end      = 0.0
        self._duration = 0.0
        self._build_ui()
        self._load()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._figure = Figure(figsize=(9, 3), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        layout.addWidget(self._canvas)

        self._lbl = QLabel()
        self._lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._lbl)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _load(self) -> None:
        t, env_min, env_max, dur = _build_envelope(self._filepath)
        self._duration = dur
        self._start    = 0.0
        self._end      = dur

        ax = self._figure.add_subplot(111)
        ax.fill_between(t, env_min, env_max, color="#1f77b4", alpha=0.55)
        ax.plot(t, env_max, color="#1f77b4", linewidth=0.8)
        ax.plot(t, env_min, color="#1f77b4", linewidth=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

        self._span = SpanSelector(
            ax, self._on_select, "horizontal",
            useblit=True,
            props={"alpha": 0.3, "facecolor": "orange"},
            interactive=True,
        )
        self._span.extents = (0.0, dur)
        self._update_label()
        self._canvas.draw()

    def _on_select(self, xmin: float, xmax: float) -> None:
        self._start = max(0.0, xmin)
        self._end   = min(self._duration, xmax)
        self._update_label()

    def _update_label(self) -> None:
        dur = self._end - self._start
        self._lbl.setText(
            f"Selected:  {self._start:.2f} s  –  {self._end:.2f} s  "
            f"(duration: {dur:.2f} s)"
        )

    def get_selection(self) -> tuple[float, float]:
        return self._start, self._end
