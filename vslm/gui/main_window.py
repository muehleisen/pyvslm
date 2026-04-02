"""Main application window."""
from __future__ import annotations

import sys
from pathlib import Path

import soundfile as sf
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QComboBox, QFileDialog, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QRadioButton, QSizePolicy,
    QStackedWidget, QVBoxLayout, QWidget,
)
from PySide6.QtCore import QUrl

from .plot_widget import MatplotlibWidget
from .plot_manager import ResultPlotter
from .dialogs.calibration import CalibrationDialog
from .dialogs.waveform import WaveformDialog
from .dialogs.about import AboutDialog
from ..constants import LEQ_INTERVAL_MAP
from ..controller import VSLMController


# Display labels for the six analysis modes (index must match MODE_ID_MAP)
_MODE_LABELS = [
    "Level vs Time (Lp)",
    "Leq Analysis",
    "Octave Bands",
    "1/3 Octave Bands",
    "Power Spectral Density",
    "Spectrogram",
]

_WINDOW_OPTIONS  = ["Hanning", "Hamming", "Flattop", "Blackman",
                    "Blackman-Harris", "Rectangular", "Bartlett"]
_PALETTE_OPTIONS = ["plasma", "viridis", "cividis", "inferno", "jet", "coolwarm"]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VSLM 2.0 — Virtual Sound Level Meter")
        self.resize(1200, 820)

        self.controller = VSLMController()

        self._build_menu()
        self._build_ui()
        self._connect_signals()
        self._apply_settings()

        self.statusBar().showMessage("Ready.  Load a WAV file to begin.")

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        mb = self.menuBar()

        m_file = mb.addMenu("File")

        m_settings = m_file.addMenu("Settings")
        act_load_settings = QAction("Load Settings…", self)
        act_load_settings.triggered.connect(self._on_load_settings)
        act_save_settings = QAction("Save Settings…", self)
        act_save_settings.triggered.connect(self._on_save_settings)
        m_settings.addAction(act_load_settings)
        m_settings.addAction(act_save_settings)

        m_file.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        m_file.addAction(act_quit)

        self._m_export = mb.addMenu("Export")
        self._m_export.setEnabled(False)
        act_csv = QAction("Save Results (CSV)…", self)
        act_csv.triggered.connect(self._on_export_csv)
        act_fig = QAction("Save Plot Figure…", self)
        act_fig.triggered.connect(self._on_save_figure)
        self._m_export.addAction(act_csv)
        self._m_export.addAction(act_fig)

        m_help = mb.addMenu("Help")
        act_about = QAction("About VSLM", self)
        act_about.triggered.connect(lambda: AboutDialog(self).exec())
        m_help.addAction(act_about)

    # ------------------------------------------------------------------
    # Central UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_row = QHBoxLayout(central)

        # ---- Left control panel ----
        left = QWidget()
        left.setFixedWidth(330)
        col = QVBoxLayout(left)
        self._left = left

        # File group
        grp_file = QGroupBox("File && Selection")
        row_btns = QHBoxLayout()
        self._btn_load = QPushButton("Load\nWAV")
        self._btn_load.setMinimumHeight(48)
        self._btn_load.clicked.connect(self._on_load)
        self._btn_select = QPushButton("Select\nSection")
        self._btn_select.setMinimumHeight(48)
        self._btn_select.setEnabled(False)
        self._btn_select.clicked.connect(self._on_select_section)
        self._btn_cal = QPushButton("Calibrate")
        self._btn_cal.setMinimumHeight(48)
        self._btn_cal.clicked.connect(self._on_calibrate)
        row_btns.addWidget(self._btn_load)
        row_btns.addWidget(self._btn_select)
        row_btns.addWidget(self._btn_cal)

        self._lbl_info = QLabel("No file loaded.")
        self._lbl_info.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self._lbl_info.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._lbl_info.setMinimumHeight(70)

        vf = QVBoxLayout()
        vf.addLayout(row_btns)
        vf.addWidget(self._lbl_info)
        grp_file.setLayout(vf)
        col.addWidget(grp_file)

        # Weighting group
        grp_w = QGroupBox("Weighting")
        row_w = QHBoxLayout()
        self._bg_weight = QButtonGroup()
        for i, label in enumerate(["A", "C", "Z"]):
            rb = QRadioButton(label)
            self._bg_weight.addButton(rb, i)
            row_w.addWidget(rb)
        grp_w.setLayout(row_w)
        col.addWidget(grp_w)

        # Speed group
        grp_s = QGroupBox("Speed (Lp mode)")
        row_s = QHBoxLayout()
        self._bg_speed = QButtonGroup()
        for i, label in enumerate(["Slow", "Fast", "Impulse"]):
            rb = QRadioButton(label)
            self._bg_speed.addButton(rb, i)
            row_s.addWidget(rb)
        grp_s.setLayout(row_s)
        col.addWidget(grp_s)

        # Analysis mode + per-mode settings
        grp_mode = QGroupBox("Analysis Mode")
        mode_row = QHBoxLayout()

        modes_col = QVBoxLayout()
        self._bg_mode = QButtonGroup()
        for i, label in enumerate(_MODE_LABELS):
            rb = QRadioButton(label)
            self._bg_mode.addButton(rb, i)
            modes_col.addWidget(rb)
        modes_col.addStretch()
        mode_row.addLayout(modes_col, stretch=1)

        self._stack = QStackedWidget()

        # Page 0 — Lp settings
        pg_lp = QWidget()
        vl = QVBoxLayout(pg_lp)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.addWidget(QLabel("Plot interval:"))
        self._combo_lp_interval = QComboBox()
        for key, (label, _) in LEQ_INTERVAL_MAP.items():
            self._combo_lp_interval.addItem(label, key)
        vl.addWidget(self._combo_lp_interval)
        vl.addStretch()
        self._stack.addWidget(pg_lp)

        # Page 1 — Leq settings
        pg_leq = QWidget()
        vl = QVBoxLayout(pg_leq)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.addWidget(QLabel("Leq interval:"))
        self._combo_leq_interval = QComboBox()
        for key, (label, _) in LEQ_INTERVAL_MAP.items():
            self._combo_leq_interval.addItem(label, key)
        vl.addWidget(self._combo_leq_interval)
        vl.addWidget(QLabel("Dose standard:"))
        self._combo_dose = QComboBox()
        self._combo_dose.addItems(["NIOSH", "OSHA"])
        vl.addWidget(self._combo_dose)
        vl.addStretch()
        self._stack.addWidget(pg_leq)

        # Page 2 — Bands (no extra settings)
        self._stack.addWidget(QWidget())

        # Page 3 — PSD settings
        pg_psd = QWidget()
        vl = QVBoxLayout(pg_psd)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.addWidget(QLabel("FFT size:"))
        self._combo_psd_nfft = QComboBox()
        self._combo_psd_nfft.addItems(["256", "512", "1024", "2048", "4096", "8192", "16384"])
        self._combo_psd_nfft.setCurrentText("4096")
        vl.addWidget(self._combo_psd_nfft)
        vl.addWidget(QLabel("Window:"))
        self._combo_psd_window = QComboBox()
        self._combo_psd_window.addItems(_WINDOW_OPTIONS)
        vl.addWidget(self._combo_psd_window)
        vl.addStretch()
        self._stack.addWidget(pg_psd)

        # Page 4 — Spectrogram settings
        pg_spec = QWidget()
        vl = QVBoxLayout(pg_spec)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.addWidget(QLabel("FFT size:"))
        self._combo_spec_nfft = QComboBox()
        self._combo_spec_nfft.addItems(["128", "256", "512", "1024", "2048", "4096"])
        self._combo_spec_nfft.setCurrentText("512")
        vl.addWidget(self._combo_spec_nfft)
        vl.addWidget(QLabel("Slice duration (s):"))
        self._combo_spec_dt = QComboBox()
        self._combo_spec_dt.addItem("0.1 s",  0.1)
        self._combo_spec_dt.addItem("1.0 s",  1.0)
        self._combo_spec_dt.addItem("10.0 s", 10.0)
        self._combo_spec_dt.setCurrentIndex(1)
        vl.addWidget(self._combo_spec_dt)
        vl.addWidget(QLabel("Window:"))
        self._combo_spec_window = QComboBox()
        self._combo_spec_window.addItems(_WINDOW_OPTIONS)
        self._combo_spec_window.setCurrentText("Hamming")
        vl.addWidget(self._combo_spec_window)
        vl.addWidget(QLabel("Colour palette:"))
        self._combo_spec_cmap = QComboBox()
        self._combo_spec_cmap.addItems(_PALETTE_OPTIONS)
        self._combo_spec_cmap.setCurrentText("plasma")
        self._combo_spec_cmap.currentTextChanged.connect(self._on_cmap_changed)
        vl.addWidget(self._combo_spec_cmap)
        vl.addStretch()
        self._stack.addWidget(pg_spec)

        mode_row.addWidget(self._stack, stretch=1)
        grp_mode.setLayout(mode_row)
        col.addWidget(grp_mode)

        col.addSpacing(12)

        self._progress = QProgressBar()
        self._progress.setTextVisible(False)
        col.addWidget(self._progress)

        self._btn_analyze = QPushButton("ANALYSE")
        self._btn_analyze.setStyleSheet(
            "font-weight: bold; font-size: 14px; height: 42px; background-color: #dbeafe;"
        )
        self._btn_analyze.setEnabled(False)
        self._btn_analyze.clicked.connect(self._on_analyze)
        col.addWidget(self._btn_analyze)
        col.addStretch()

        main_row.addWidget(left)

        # ---- Plot panel ----
        self._plot = MatplotlibWidget()
        self._plot.sig_scaling_changed.connect(self._on_scaling_changed)
        main_row.addWidget(self._plot, stretch=1)

        # Connect mode radio buttons after stack is built
        self._bg_mode.idToggled.connect(self._on_mode_changed)

    # ------------------------------------------------------------------
    # Signals ↔ controller
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        c = self.controller
        c.sig_file_loaded.connect(self._on_file_loaded)
        c.sig_analysis_started.connect(self._on_analysis_started)
        c.sig_analysis_progress.connect(self._progress.setValue)
        c.sig_total_blocks.connect(self._progress.setMaximum)
        c.sig_analysis_finished.connect(self._on_analysis_finished)
        c.sig_analysis_error.connect(self._on_error)
        c.sig_status_message.connect(self.statusBar().showMessage)
        c.sig_export_done.connect(lambda: QMessageBox.information(
            self, "Export", "Results exported successfully."))

    # ------------------------------------------------------------------
    # Settings sync
    # ------------------------------------------------------------------

    def _apply_settings(self) -> None:
        s = self.controller.settings

        # Weighting
        w_text = str(s.weighting)
        for btn in self._bg_weight.buttons():
            if btn.text() == w_text:
                btn.setChecked(True)
                break
        else:
            self._bg_weight.button(0).setChecked(True)

        # Speed
        sp_text = str(s.speed)
        for btn in self._bg_speed.buttons():
            if btn.text() == sp_text:
                btn.setChecked(True)
                break
        else:
            self._bg_speed.button(1).setChecked(True)

        self._bg_mode.button(s.analysis_mode_index).setChecked(True)

        self._combo_lp_interval.setCurrentIndex(s.lp_interval_index)
        self._combo_leq_interval.setCurrentIndex(s.leq_interval_index)
        self._combo_dose.setCurrentText(s.current_dose_standard)

        self._combo_psd_nfft.setCurrentText(str(s.psd_nfft))
        self._combo_psd_window.setCurrentText(s.psd_window)

        self._combo_spec_nfft.setCurrentText(str(s.spec_nfft))
        idx = self._combo_spec_dt.findData(s.spec_dt)
        if idx >= 0:
            self._combo_spec_dt.setCurrentIndex(idx)
        self._combo_spec_window.setCurrentText(s.spec_window)
        self._combo_spec_cmap.setCurrentText(s.spec_cmap)

        self._plot.configure_scaling(s.plot_autoscale, s.plot_ymin, s.plot_ymax)

    def _scrape_settings(self) -> None:
        s = self.controller.settings
        w_btn = self._bg_weight.checkedButton()
        if w_btn:
            s.weighting = w_btn.text()
        sp_btn = self._bg_speed.checkedButton()
        if sp_btn:
            s.speed = sp_btn.text()

        s.analysis_mode_index  = self._bg_mode.checkedId()
        s.lp_interval_index    = self._combo_lp_interval.currentIndex()
        s.leq_interval_index   = self._combo_leq_interval.currentIndex()
        s.current_dose_standard = self._combo_dose.currentText()

        s.psd_nfft   = int(self._combo_psd_nfft.currentText())
        s.psd_window = self._combo_psd_window.currentText()

        s.spec_nfft   = int(self._combo_spec_nfft.currentText())
        s.spec_dt     = self._combo_spec_dt.currentData()
        s.spec_window = self._combo_spec_window.currentText()
        s.spec_cmap   = self._combo_spec_cmap.currentText()

    # ------------------------------------------------------------------
    # UI event handlers
    # ------------------------------------------------------------------

    def _on_mode_changed(self, mode_id: int, checked: bool) -> None:
        if not checked:
            return
        # Map mode_id to stack page
        page = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 4}.get(mode_id, 2)
        self._stack.setCurrentIndex(page)

    def _on_cmap_changed(self, cmap: str) -> None:
        self.controller.settings.spec_cmap = cmap
        if self.controller.last_results:
            self._redraw()

    def _on_load(self) -> None:
        start_dir = self.controller.settings.last_directory
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open WAV file", start_dir, "WAV Files (*.wav)")
        if fname:
            self.controller.load_file(fname)

    def _on_select_section(self) -> None:
        if not self.controller.filepath:
            return
        dlg = WaveformDialog(str(self.controller.filepath), self)
        if self.controller.end_time:
            dlg._start = self.controller.start_time
            dlg._end   = self.controller.end_time
        if dlg.exec():
            s, e = dlg.get_selection()
            self.controller.set_analysis_range(s, e)
            self._update_file_label()

    def _on_calibrate(self) -> None:
        if not self.controller.filepath:
            return
        dlg = CalibrationDialog(
            self.controller.cal_factor,
            self.controller.filepath,
            self.controller.start_time,
            self.controller.end_time or 0.0,
            self,
        )
        if dlg.exec():
            self.controller.update_calibration(dlg.get_factor())
            self._update_file_label()

    def _on_analyze(self) -> None:
        if self._btn_analyze.text() == "STOP":
            self.controller.stop_analysis()
            return
        self._scrape_settings()
        self.controller.run_analysis(self._bg_mode.checkedId())

    def _on_export_csv(self) -> None:
        if not self.controller.last_results:
            return
        fname, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "results.csv", "CSV Files (*.csv)")
        if fname:
            self.controller.export_results(
                Path(fname),
                self._bg_mode.checkedId(),
                self._combo_leq_interval.currentData(),
            )

    def _on_save_figure(self) -> None:
        if self._plot.toolbar:
            self._plot.toolbar.save_figure()

    def _on_save_settings(self) -> None:
        self._scrape_settings()
        fname, _ = QFileDialog.getSaveFileName(
            self, "Save Settings",
            self.controller.settings.last_directory,
            "YAML Files (*.yaml)",
        )
        if fname:
            self.controller.save_settings(Path(fname))

    def _on_load_settings(self) -> None:
        fname, _ = QFileDialog.getOpenFileName(
            self, "Load Settings",
            self.controller.settings.last_directory,
            "YAML Files (*.yaml);;All Files (*)",
        )
        if fname and self.controller.load_settings(Path(fname)):
            self._apply_settings()

    def _on_scaling_changed(self, auto: bool, ymin: float, ymax: float) -> None:
        s = self.controller.settings
        s.plot_autoscale = auto
        s.plot_ymin      = ymin
        s.plot_ymax      = ymax
        if self.controller.last_results:
            self._redraw()

    # ------------------------------------------------------------------
    # Controller signal slots
    # ------------------------------------------------------------------

    @Slot(object, object)
    def _on_file_loaded(self, path, info) -> None:
        self._btn_select.setEnabled(True)
        self._btn_analyze.setEnabled(True)
        self._m_export.setEnabled(False)
        self._update_file_label(info)

    @Slot()
    def _on_analysis_started(self) -> None:
        self._set_inputs_enabled(False)
        self._btn_analyze.setText("STOP")
        self._btn_analyze.setStyleSheet(
            "font-weight: bold; font-size: 14px; height: 42px; background-color: #fca5a5;"
        )
        self._progress.setValue(0)

    @Slot(list)
    def _on_analysis_finished(self, results: list) -> None:
        self._set_inputs_enabled(True)
        self._btn_analyze.setText("ANALYSE")
        self._btn_analyze.setStyleSheet(
            "font-weight: bold; font-size: 14px; height: 42px; background-color: #dbeafe;"
        )
        self._btn_analyze.setEnabled(True)
        self._progress.setValue(0)
        self._m_export.setEnabled(bool(results))
        self._redraw()

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        if self._btn_analyze.text() == "STOP":
            self._set_inputs_enabled(True)
            self._btn_analyze.setText("ANALYSE")
            self._btn_analyze.setStyleSheet(
                "font-weight: bold; font-size: 14px; height: 42px; background-color: #dbeafe;"
            )
        QMessageBox.critical(self, "Analysis Error", msg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_inputs_enabled(self, enabled: bool) -> None:
        self._btn_load.setEnabled(enabled)
        self._btn_select.setEnabled(enabled)
        self._btn_cal.setEnabled(enabled)
        for grp in self._left.findChildren(QGroupBox):
            if grp.title() != "File && Selection":
                grp.setEnabled(enabled)
        can_export = enabled and bool(self.controller.last_results)
        self._m_export.setEnabled(can_export)

    def _update_file_label(self, info=None) -> None:
        c = self.controller
        if not c.filepath:
            self._lbl_info.setText("No file loaded.")
            return
        if info is None:
            try:
                info = sf.info(str(c.filepath))
            except Exception:
                pass
        fs_str  = f"{info.samplerate} Hz" if info else ""
        dur_str = f"{info.duration:.2f} s" if info else ""
        self._lbl_info.setText(
            f"File:  {c.filepath.name}\n"
            f"Fs:    {fs_str}\n"
            f"Dur:   {dur_str}\n"
            f"Cal:   {c.cal_factor:.4f}"
        )

    def _redraw(self) -> None:
        results = self.controller.last_results
        if not results:
            return
        s        = self.controller.settings
        mode_id  = self._bg_mode.checkedId()
        w_btn    = self._bg_weight.checkedButton()
        sp_btn   = self._bg_speed.checkedButton()
        weighting = w_btn.text() if w_btn else "A"
        speed     = sp_btn.text() if sp_btn else "Fast"
        dose_params = s.dose_standards.get(s.current_dose_standard)

        ResultPlotter.plot(
            self._plot.figure,
            results,
            mode_id,
            weighting,
            speed,
            self._combo_leq_interval.currentData(),
            s.block_size_ms,
            dose_params,
            s.current_dose_standard,
            s.ref_pressure,
            s.plot_autoscale,
            s.plot_ymin,
            s.plot_ymax,
            spec_cmap=s.spec_cmap,
        )
        self._plot.refresh()

    def closeEvent(self, event) -> None:
        self._scrape_settings()
        self.controller.shutdown()
        event.accept()
