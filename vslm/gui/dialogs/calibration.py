"""Calibration manager dialog."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout,
)

from ...dsp.calibration import compute_selection_rms, factor_from_reference


class CalibrationDialog(QDialog):
    """
    Two-method calibration:

    1. **Manual** — type in a linear scale factor directly.
    2. **From selection** — measure the RMS of a loaded calibration tone and
       compute the factor from a known reference level (e.g. 94 dB from a
       pistonphone).
    """

    def __init__(self, current_factor: float, filepath: Path | None = None,
                 start: float = 0.0, end: float = 0.0, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibration Manager")
        self.resize(420, 330)
        self._factor   = current_factor
        self._filepath = filepath
        self._start    = start
        self._end      = end
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._lbl_current = QLabel()
        self._lbl_current.setAlignment(Qt.AlignCenter)
        self._lbl_current.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #1e40af; margin-bottom: 8px;"
        )
        layout.addWidget(self._lbl_current)
        self._refresh_label()

        # --- Method 1: Manual ---
        grp1 = QGroupBox("Method 1 — Manual entry")
        row1 = QHBoxLayout()
        self._spin_manual = QDoubleSpinBox()
        self._spin_manual.setRange(1e-4, 1e4)
        self._spin_manual.setDecimals(4)
        self._spin_manual.setValue(self._factor)
        self._spin_manual.setSuffix("  (linear)")
        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(self._on_apply_manual)
        row1.addWidget(QLabel("Factor:"))
        row1.addWidget(self._spin_manual)
        row1.addWidget(btn_apply)
        grp1.setLayout(row1)
        layout.addWidget(grp1)

        # --- Method 2: From selection ---
        grp2 = QGroupBox("Method 2 — From selected audio")
        vbox2 = QVBoxLayout()
        has_selection = (self._filepath is not None and
                         (self._end - self._start) > 0.1)
        if has_selection:
            dur = self._end - self._start
            info = f"Selection: {self._start:.2f} s – {self._end:.2f} s  ({dur:.2f} s)"
        else:
            info = "No file loaded or selection too short."
        vbox2.addWidget(QLabel(info))

        form = QFormLayout()
        self._spin_ref = QDoubleSpinBox()
        self._spin_ref.setRange(0.0, 194.0)
        self._spin_ref.setValue(94.0)
        self._spin_ref.setSuffix(" dB SPL")
        form.addRow("Reference level:", self._spin_ref)
        vbox2.addLayout(form)

        self._btn_calc = QPushButton("Calculate & Apply")
        self._btn_calc.setStyleSheet("background-color: #dcfce7; font-weight: bold;")
        self._btn_calc.setEnabled(has_selection)
        self._btn_calc.clicked.connect(self._on_calculate)
        vbox2.addWidget(self._btn_calc)
        grp2.setLayout(vbox2)
        layout.addWidget(grp2)

        layout.addStretch()
        btn_done = QPushButton("Done")
        btn_done.clicked.connect(self.accept)
        layout.addWidget(btn_done)

    def _refresh_label(self) -> None:
        self._lbl_current.setText(f"Current factor: {self._factor:.4f}")

    def _on_apply_manual(self) -> None:
        self._factor = self._spin_manual.value()
        self._refresh_label()

    def _on_calculate(self) -> None:
        try:
            rms = compute_selection_rms(self._filepath, self._start, self._end)
            new_factor = factor_from_reference(rms, self._spin_ref.value())
            self._factor = new_factor
            self._spin_manual.setValue(new_factor)
            self._refresh_label()
            QMessageBox.information(
                self, "Calibration",
                f"Measured RMS: {rms:.6f}\n"
                f"Reference: {self._spin_ref.value():.1f} dB\n"
                f"New factor: {new_factor:.4f}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Calibration Failed", str(e))

    def get_factor(self) -> float:
        return self._factor
