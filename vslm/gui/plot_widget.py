"""
Matplotlib canvas widget with a custom toolbar that includes a scaling button.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QToolButton, QVBoxLayout, QWidget,
)


class PlotScalingDialog(QDialog):
    """Simple dialog to configure Y-axis min/max or autoscale."""

    def __init__(self, autoscale: bool, ymin: float, ymax: float, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Plot Scaling")
        self.resize(300, 150)

        layout = QVBoxLayout(self)

        self.chk_auto = QCheckBox("Autoscale Y-axis")
        self.chk_auto.setChecked(autoscale)
        self.chk_auto.toggled.connect(self._on_auto_toggled)
        layout.addWidget(self.chk_auto)

        form = QFormLayout()
        self.spin_min = QDoubleSpinBox()
        self.spin_min.setRange(-200, 200)
        self.spin_min.setValue(ymin)
        form.addRow("Min (dB):", self.spin_min)

        self.spin_max = QDoubleSpinBox()
        self.spin_max.setRange(-200, 200)
        self.spin_max.setValue(ymax)
        form.addRow("Max (dB):", self.spin_max)
        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._on_auto_toggled(autoscale)

    def _on_auto_toggled(self, checked: bool) -> None:
        self.spin_min.setEnabled(not checked)
        self.spin_max.setEnabled(not checked)

    def values(self) -> tuple[bool, float, float]:
        return self.chk_auto.isChecked(), self.spin_min.value(), self.spin_max.value()


class _CustomToolbar(NavigationToolbar):
    """Navigation toolbar with an extra scaling button (⇕)."""
    sig_open_scaling = Signal()

    def __init__(self, canvas, parent=None) -> None:
        super().__init__(canvas, parent)
        btn = QToolButton(self)
        btn.setToolTip("Configure plot scaling")
        btn.setText("⇕")
        f = btn.font()
        f.setWeight(QFont.Weight.Black)
        f.setPointSize(24)
        btn.setFont(f)
        btn.clicked.connect(self.sig_open_scaling.emit)

        # Insert before the coordinate label if present
        target = next(
            (a for a in self.actions() if self.widgetForAction(a) is self.locLabel),
            None,
        )
        if target:
            self.insertSeparator(target)
            self.insertWidget(target, btn)
        else:
            self.addSeparator()
            self.addWidget(btn)


class MatplotlibWidget(QWidget):
    """
    QWidget embedding a matplotlib figure with toolbar.

    Emits :attr:`sig_scaling_changed` ``(autoscale, ymin, ymax)`` when the
    user confirms new scaling settings.
    """
    sig_scaling_changed = Signal(bool, float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        vbox = QVBoxLayout(self)

        self.figure  = Figure()
        self.canvas  = FigureCanvas(self.figure)
        self.toolbar = _CustomToolbar(self.canvas, self)
        self.toolbar.sig_open_scaling.connect(self._open_scaling_dialog)

        vbox.addWidget(self.toolbar)
        vbox.addWidget(self.canvas)

        self._autoscale = True
        self._ymin      = 0.0
        self._ymax      = 120.0

    def configure_scaling(self, autoscale: bool, ymin: float, ymax: float) -> None:
        """Called by MainWindow to synchronise internal state with saved settings."""
        self._autoscale = autoscale
        self._ymin      = ymin
        self._ymax      = ymax

    def _open_scaling_dialog(self) -> None:
        dlg = PlotScalingDialog(self._autoscale, self._ymin, self._ymax, self)
        if dlg.exec():
            auto, ymin, ymax = dlg.values()
            self._autoscale = auto
            self._ymin      = ymin
            self._ymax      = ymax
            self.sig_scaling_changed.emit(auto, ymin, ymax)

    def refresh(self) -> None:
        self.canvas.draw()
