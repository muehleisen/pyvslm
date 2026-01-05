import sys
from pathlib import Path
import soundfile as sf
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QGroupBox, 
                               QFileDialog, QMessageBox, QFrame, QButtonGroup, 
                               QRadioButton, QProgressBar, QComboBox, QGridLayout,
                               QStackedWidget, QSizePolicy)
from PySide6.QtGui import QAction, QDesktopServices, QIcon
from PySide6.QtCore import QUrl, Slot

from .waveform_dialog import WaveformDialog
from .calibration_dialog import CalibrationDialog
from .about_dialog import AboutDialog 
from .plot_widget import MatplotlibWidget
from .plot_manager import ResultPlotter
from ..constants import LEQ_INTERVAL_MAP, AnalysisMode
from ..controller import VSLMController


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VSLM 2.0 (Python)")
        self.resize(1150, 800)
        
        self.controller = VSLMController()
        
        self._init_menu_bar()
        self._init_ui()
        
        self._connect_controller_signals()
        self._apply_settings_to_ui()
        
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready. Load a file to begin.")

    def _connect_controller_signals(self):
        self.controller.sig_file_loaded.connect(self.on_file_loaded_update_ui)
        self.controller.sig_analysis_started.connect(self.on_analysis_started_ui)
        self.controller.sig_analysis_progress.connect(self.progress.setValue)
        self.controller.sig_total_blocks.connect(self.progress.setMaximum)
        self.controller.sig_analysis_finished.connect(self.on_analysis_finished_ui)
        self.controller.sig_analysis_error.connect(self.on_error_message)
        self.controller.sig_status_message.connect(self.update_status_bar)
        self.controller.sig_export_finished.connect(self.on_export_success)

    def _init_menu_bar(self):
        menu_bar = self.menuBar()
        menu_file = menu_bar.addMenu("File")
        
        menu_settings = menu_file.addMenu("Settings")
        act_load_sets = QAction("Load Settings...", self)
        act_load_sets.triggered.connect(self.on_action_load_settings)
        menu_settings.addAction(act_load_sets)
        
        act_save_sets = QAction("Save Settings...", self)
        act_save_sets.triggered.connect(self.on_action_save_settings)
        menu_settings.addAction(act_save_sets)
        
        menu_file.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close) 
        menu_file.addAction(act_quit)
        
        self.menu_export = menu_bar.addMenu("Export")
        self.menu_export.setEnabled(False) 
        
        act_export_csv = QAction("Save Results (CSV)...", self)
        act_export_csv.triggered.connect(self.on_export_csv)
        self.menu_export.addAction(act_export_csv)
        
        act_save_fig = QAction("Save Plot Figure...", self)
        act_save_fig.triggered.connect(self.on_action_save_figure)
        self.menu_export.addAction(act_save_fig)
        
        menu_help = menu_bar.addMenu("Help")
        act_docs = QAction("Documentation", self)
        act_docs.triggered.connect(lambda: self.on_open_url("https://example.com/docs"))
        menu_help.addAction(act_docs)
        
        act_about = QAction("About VSLM", self)
        act_about.triggered.connect(self.on_about)
        menu_help.addAction(act_about)

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        left_panel = QWidget()
        left_panel.setFixedWidth(320)
        left_layout = QVBoxLayout(left_panel)
        self.left_panel = left_panel 
        
        # --- 1. File & Selection Group ---
        grp_file = QGroupBox("File & Selection")
        layout_file = QVBoxLayout()
        layout_btns = QHBoxLayout()
        layout_btns.setSpacing(5) 
        layout_btns.setContentsMargins(0, 0, 0, 0)

        self.btn_load = QPushButton("Load\nWav")
        self.btn_load.setMinimumHeight(45)
        self.btn_load.clicked.connect(self.on_btn_load_click)
        layout_btns.addWidget(self.btn_load)

        self.btn_select = QPushButton("Select\nSection")
        self.btn_select.setMinimumHeight(45)
        self.btn_select.clicked.connect(self.on_select_section)
        self.btn_select.setEnabled(False)
        layout_btns.addWidget(self.btn_select)

        self.btn_cal = QPushButton("Calibrate")
        self.btn_cal.setMinimumHeight(45)
        self.btn_cal.clicked.connect(self.on_calibrate)
        self.btn_cal.setStyleSheet("background-color: #f3f4f6;")
        layout_btns.addWidget(self.btn_cal)

        layout_file.addLayout(layout_btns)
        
        # --- Playback Buttons (New Row) ---
        layout_play = QHBoxLayout()
        layout_play.setSpacing(5)
        
        self.btn_play = QPushButton("Play\nSelection")
        self.btn_play.setMinimumHeight(40)
        self.btn_play.clicked.connect(self.on_play_click)
        self.btn_play.setEnabled(False)
        layout_play.addWidget(self.btn_play)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.clicked.connect(self.on_stop_click)
        self.btn_stop.setEnabled(False) # Enabled when playing potentially, but for now enables with file
        layout_play.addWidget(self.btn_stop)
        
        layout_file.addLayout(layout_play)

        self.lbl_info = QLabel()
        self.lbl_info.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        layout_file.addWidget(QLabel("File Info"))
        layout_file.addWidget(self.lbl_info)
        
        grp_file.setLayout(layout_file)
        left_layout.addWidget(grp_file)
        
        # --- 2. Weighting Group ---
        grp_weight = QGroupBox("Weighting")
        layout_weight = QHBoxLayout()
        self.bg_weight = QButtonGroup()
        for i, text in enumerate(['A', 'C', 'Z']):
            rb = QRadioButton(text)
            self.bg_weight.addButton(rb, i)
            layout_weight.addWidget(rb)
        grp_weight.setLayout(layout_weight)
        left_layout.addWidget(grp_weight)
        
        # --- 3. Speed Group ---
        grp_speed = QGroupBox("Speed (Lp Mode)")
        layout_speed = QHBoxLayout()
        self.bg_speed = QButtonGroup()
        for i, text in enumerate(['Slow', 'Fast', 'Impulse']):
            rb = QRadioButton(text)
            self.bg_speed.addButton(rb, i)
            layout_speed.addWidget(rb)
        grp_speed.setLayout(layout_speed)
        left_layout.addWidget(grp_speed)
        
        # --- 4. Analysis Mode Group ---
        grp_mode = QGroupBox("Analysis Mode")
        layout_mode = QHBoxLayout()
        layout_mode.setSpacing(10)
        
        col_radios = QVBoxLayout()
        self.bg_mode = QButtonGroup()
        
        modes = ["Level vs Time (Lp)", "LEQ Analysis", "Octave Bands", "1/3 Octave Bands", 
                 "Power Spectral Density", "Spectrogram"]
        for i, m in enumerate(modes):
            rb = QRadioButton(m)
            self.bg_mode.addButton(rb, i)
            col_radios.addWidget(rb)
        
        col_radios.addStretch()
        layout_mode.addLayout(col_radios, stretch=1)
        
        self.stack_settings = QStackedWidget()
        
        window_options = ["Hanning", "Hamming", "Flattop", "Blackman", "Blackman-Harris", "Rectangular", "Bartlett"]
        palette_options = ["plasma", "viridis", "cividis", "inferno", "jet", "coolwarm"]

        # -- Page 0: Lp Settings --
        page_lp = QWidget()
        lay_lp = QVBoxLayout(page_lp)
        lay_lp.setContentsMargins(0,0,0,0)
        lay_lp.addWidget(QLabel("Plot Interval:"))
        self.combo_lp_sampling = QComboBox()
        for key, (label, _) in LEQ_INTERVAL_MAP.items():
            self.combo_lp_sampling.addItem(label, key)
        lay_lp.addWidget(self.combo_lp_sampling)
        lay_lp.addStretch()
        self.stack_settings.addWidget(page_lp)
        
        # -- Page 1: LEQ Settings --
        page_leq = QWidget()
        lay_leq = QVBoxLayout(page_leq)
        lay_leq.setContentsMargins(0,0,0,0)
        
        lay_leq.addWidget(QLabel("LEQ Time:"))
        self.combo_leq_int = QComboBox()
        for key, (label, _) in LEQ_INTERVAL_MAP.items():
            self.combo_leq_int.addItem(label, key)
        lay_leq.addWidget(self.combo_leq_int)
        
        lay_leq.addWidget(QLabel("Dose Standard:"))
        self.combo_dose_std = QComboBox()
        self.combo_dose_std.addItems(["NIOSH", "OSHA"])
        lay_leq.addWidget(self.combo_dose_std)
        lay_leq.addStretch()
        self.stack_settings.addWidget(page_leq)
        
        # -- Page 2: PSD Settings --
        page_psd = QWidget()
        lay_psd = QVBoxLayout(page_psd)
        lay_psd.setContentsMargins(0,0,0,0)
        lay_psd.addWidget(QLabel("FFT Size:"))
        self.combo_psd_nfft = QComboBox()
        self.combo_psd_nfft.addItems(["256", "512", "1024", "2048", "4096", "8192", "16384"])
        self.combo_psd_nfft.setCurrentText("4096")
        lay_psd.addWidget(self.combo_psd_nfft)
        
        lay_psd.addWidget(QLabel("Window:"))
        self.combo_psd_window = QComboBox()
        self.combo_psd_window.addItems(window_options)
        lay_psd.addWidget(self.combo_psd_window)
        lay_psd.addStretch()
        self.stack_settings.addWidget(page_psd)
        
        # -- Page 3: Spectrogram Settings --
        page_spec = QWidget()
        lay_spec = QVBoxLayout(page_spec)
        lay_spec.setContentsMargins(0,0,0,0)
        lay_spec.addWidget(QLabel("FFT Size:"))
        self.combo_spec_nfft = QComboBox()
        self.combo_spec_nfft.addItems(["128", "256", "512", "1024", "2048", "4096"])
        self.combo_spec_nfft.setCurrentText("512")
        lay_spec.addWidget(self.combo_spec_nfft)
        
        lay_spec.addWidget(QLabel("Slice (s):"))
        self.combo_spec_dt = QComboBox()
        self.combo_spec_dt.addItem("0.1 s", 0.1)
        self.combo_spec_dt.addItem("1.0 s", 1.0)
        self.combo_spec_dt.addItem("10.0 s", 10.0)
        self.combo_spec_dt.setCurrentIndex(1)
        lay_spec.addWidget(self.combo_spec_dt)
        
        lay_spec.addWidget(QLabel("Window:"))
        self.combo_spec_window = QComboBox()
        self.combo_spec_window.addItems(window_options)
        self.combo_spec_window.setCurrentText("Hamming")
        lay_spec.addWidget(self.combo_spec_window)
        
        # Palette Selector
        lay_spec.addWidget(QLabel("Palette:"))
        self.combo_spec_cmap = QComboBox()
        self.combo_spec_cmap.addItems(palette_options)
        self.combo_spec_cmap.setCurrentText("plasma")
        self.combo_spec_cmap.currentTextChanged.connect(self.on_cmap_changed)
        lay_spec.addWidget(self.combo_spec_cmap)
        
        lay_spec.addStretch()
        self.stack_settings.addWidget(page_spec)
        
        # -- Page 4: Empty (Octave/Third) --
        self.stack_settings.addWidget(QWidget())
        
        layout_mode.addWidget(self.stack_settings, stretch=1)
        
        self.bg_mode.idToggled.connect(self.on_mode_changed)
        grp_mode.setLayout(layout_mode)
        left_layout.addWidget(grp_mode)
        
        left_layout.addSpacing(20)
        
        # --- Analyze Button & Progress ---
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        left_layout.addWidget(self.progress)
        
        self.btn_analyze = QPushButton("ANALYZE")
        self.btn_analyze.setStyleSheet("font-weight: bold; font-size: 14px; height: 40px; background-color: #dbeafe;")
        self.btn_analyze.clicked.connect(self.on_analyze_click)
        self.btn_analyze.setEnabled(False)
        left_layout.addWidget(self.btn_analyze)
        
        left_layout.addStretch()
        main_layout.addWidget(left_panel)
        
        self.plot_panel = MatplotlibWidget()
        self.plot_panel.sig_scaling_changed.connect(self.on_scaling_changed)
        main_layout.addWidget(self.plot_panel, stretch=1)

    def _apply_settings_to_ui(self):
        settings = self.controller.settings
        
        w_val = settings.weighting.value if hasattr(settings.weighting, 'value') else settings.weighting
        s_val = settings.speed.value if hasattr(settings.speed, 'value') else settings.speed

        for btn in self.bg_weight.buttons():
            if btn.text() == w_val:
                btn.setChecked(True)
                break
        else:
            self.bg_weight.button(0).setChecked(True)

        for btn in self.bg_speed.buttons():
            if btn.text() == s_val:
                btn.setChecked(True)
                break
        else:
            self.bg_speed.button(1).setChecked(True) 

        self.bg_mode.button(settings.analysis_mode_index).setChecked(True)
        self.combo_leq_int.setCurrentIndex(settings.leq_interval_index)
        
        if hasattr(settings, 'lp_interval_index'):
            self.combo_lp_sampling.setCurrentIndex(settings.lp_interval_index)
        
        if hasattr(settings, 'current_dose_standard'):
            self.combo_dose_std.setCurrentText(settings.current_dose_standard)
        
        if hasattr(settings, 'psd_nfft'): self.combo_psd_nfft.setCurrentText(str(settings.psd_nfft))
        if hasattr(settings, 'psd_window'): self.combo_psd_window.setCurrentText(settings.psd_window)
        
        if hasattr(settings, 'spec_nfft'): self.combo_spec_nfft.setCurrentText(str(settings.spec_nfft))
        if hasattr(settings, 'spec_dt'):
            idx = self.combo_spec_dt.findData(settings.spec_dt)
            if idx >= 0: self.combo_spec_dt.setCurrentIndex(idx)
        if hasattr(settings, 'spec_window'):
            self.combo_spec_window.setCurrentText(settings.spec_window)
        if hasattr(settings, 'spec_cmap'):
            self.combo_spec_cmap.setCurrentText(settings.spec_cmap)
        
        self.plot_panel.set_plot_settings(
            settings.plot_autoscale,
            settings.plot_ymin,
            settings.plot_ymax
        )
        if self.controller.filepath:
            from soundfile import info
            self.on_file_loaded_update_ui(self.controller.filepath, info(str(self.controller.filepath)))
        else:
            self.on_file_loaded_update_ui(None, None)

    def _scrape_ui_to_settings(self):
        w_btn = self.bg_weight.checkedButton()
        if w_btn: self.controller.settings.weighting = w_btn.text()
        
        s_btn = self.bg_speed.checkedButton()
        if s_btn: self.controller.settings.speed = s_btn.text()
        
        self.controller.settings.analysis_mode_index = self.bg_mode.checkedId()
        self.controller.settings.leq_interval_index = self.combo_leq_int.currentIndex()
        
        # Lp
        self.controller.settings.lp_interval_index = self.combo_lp_sampling.currentIndex()
        
        # LEQ
        self.controller.settings.current_dose_standard = self.combo_dose_std.currentText()
        
        # PSD
        self.controller.settings.psd_nfft = int(self.combo_psd_nfft.currentText())
        self.controller.settings.psd_window = self.combo_psd_window.currentText()
        
        # Spectrogram
        self.controller.settings.spec_nfft = int(self.combo_spec_nfft.currentText())
        self.controller.settings.spec_dt = self.combo_spec_dt.currentData()
        self.controller.settings.spec_window = self.combo_spec_window.currentText()
        self.controller.settings.spec_cmap = self.combo_spec_cmap.currentText()

    def on_mode_changed(self, mode_id, checked):
        if not checked: return
        page_map = {
            0: 0, # Lp -> Page 0
            1: 1, # LEQ -> Page 1
            2: 4, # Octave -> Page 4 (Empty)
            3: 4, # 3rd Oct -> Page 4 (Empty)
            4: 2, # PSD -> Page 2
            5: 3  # Spectrogram -> Page 3
        }
        idx = page_map.get(mode_id, 4)
        self.stack_settings.setCurrentIndex(idx)

    def on_cmap_changed(self, text):
        self.controller.settings.spec_cmap = text
        if self.controller.last_results:
            self._redraw_plot()

    def on_play_click(self):
        self.controller.play_audio()

    def on_stop_click(self):
        self.controller.stop_audio()

    def on_btn_load_click(self):
        start_dir = self.controller.settings.last_directory
        fname, _ = QFileDialog.getOpenFileName(self, "Open WAV", start_dir, "WAV Files (*.wav)")
        if fname:
            self.controller.load_file(fname)

    def on_select_section(self):
        if not self.controller.filepath: return
        
        dlg = WaveformDialog(str(self.controller.filepath), self)
        if self.controller.end_time:
             dlg.viewer.region.setRegion([self.controller.start_time, self.controller.end_time])
             
        if dlg.exec():
            s, e = dlg.get_selection()
            self.controller.set_analysis_range(s, e)
            self._update_file_label_text() 

    def on_calibrate(self):
        if not self.controller.filepath: return
        
        dlg = CalibrationDialog(
            self.controller.cal_factor, 
            self.controller.filepath, 
            self.controller.start_time, 
            self.controller.end_time or 0.0, 
            self
        )
        if dlg.exec():
            new_factor = dlg.get_factor()
            self.controller.update_calibration(new_factor)
            self._update_file_label_text()

    def on_analyze_click(self):
        if self.btn_analyze.text() == "STOP":
            self.controller.stop_analysis()
            return

        self._scrape_ui_to_settings()
        mode_id = self.bg_mode.checkedId()
        self.controller.run_analysis(mode_id)

    def on_export_csv(self):
        if not self.controller.last_results: return
        
        path_str, _ = QFileDialog.getSaveFileName(self, "Export CSV", "results.csv", "CSV Files (*.csv)")
        if not path_str: return
        
        mode_id = self.bg_mode.checkedId()
        leq_key = self.combo_leq_int.currentData()
        
        self.controller.export_results(Path(path_str), mode_id, leq_key)

    def on_action_save_figure(self):
        if self.plot_panel.toolbar:
            self.plot_panel.toolbar.save_figure()
        else:
            QMessageBox.information(self, "Info", "Use the floppy disk icon on the plot to save.")

    def on_scaling_changed(self, auto, ymin, ymax):
        self.controller.settings.plot_autoscale = auto
        self.controller.settings.plot_ymin = ymin
        self.controller.settings.plot_ymax = ymax
        if self.controller.last_results:
            self._redraw_plot()

    def on_action_save_settings(self):
        self._scrape_ui_to_settings()
        fname, _ = QFileDialog.getSaveFileName(self, "Save Settings", self.controller.settings.last_directory, "YAML Files (*.yaml)")
        if fname:
            self.controller.save_settings(Path(fname))

    def on_action_load_settings(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Settings", self.controller.settings.last_directory, "YAML Files (*.yaml);;All Files (*)")
        if fname:
            if self.controller.load_settings(Path(fname)):
                self._apply_settings_to_ui()
                self.update_status_bar(f"Settings loaded from {Path(fname).name}")

    def on_open_url(self, url):
        QDesktopServices.openUrl(QUrl(url))

    def on_about(self):
        AboutDialog(self).exec()

    @Slot(object, object)
    def on_file_loaded_update_ui(self, path, info):
        self.btn_select.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.btn_play.setEnabled(True)  # <--- Enable Play
        self.btn_stop.setEnabled(True)  # <--- Enable Stop
        self.menu_export.setEnabled(False)
        self._update_file_label_text(info)

    def _update_file_label_text(self, info=None):
        fname = "None"
        if self.controller.filepath:
             fname = self.controller.filepath.name

        cal = f"{self.controller.cal_factor:.4f}"
        
        fs_str = ""
        dur_str = ""
        
        if self.controller.filepath:
            # Attempt to re-read info if not provided (e.g. after cal dialog)
            if info is None:
                try:
                    info = sf.info(str(self.controller.filepath))
                except Exception:
                    pass
            
            if info:
                fs_str = f"{info.samplerate} Hz"
                dur_str = f"{info.duration:.2f} s"
        
        # Always 4 lines
        txt = (f"File: {fname}\n"
               f"Cal Factor: {cal}\n"
               f"Fs: {fs_str}\n"
               f"Dur: {dur_str}")
             
        self.lbl_info.setText(txt)

    @Slot(str)
    def on_analysis_started_ui(self, speed_str):
        self.toggle_inputs(False)
        self.btn_analyze.setText("STOP")
        self.btn_analyze.setStyleSheet("background-color: #fca5a5;")
        self.progress.setValue(0)
        # Note: We do NOT setMaximum here anymore because we wait for sig_total_blocks
        self.update_status_bar(f"Analyzing ({speed_str})...")

    @Slot(list)
    def on_analysis_finished_ui(self, results):
        self.toggle_inputs(True)
        self.btn_analyze.setText("ANALYZE")
        self.btn_analyze.setStyleSheet("background-color: #dbeafe;")
        self.btn_analyze.setEnabled(True)
        
        self.progress.setValue(0)
        
        self.menu_export.setEnabled(True)
        self._redraw_plot()

    @Slot(str)
    def on_error_message(self, msg):
        if self.btn_analyze.text() == "STOP":
            self.toggle_inputs(True)
            self.btn_analyze.setText("ANALYZE")
            self.btn_analyze.setStyleSheet("background-color: #dbeafe;")
            
        QMessageBox.critical(self, "Error", msg)

    @Slot(str)
    def update_status_bar(self, msg):
        self.status_bar.showMessage(msg)
        
    @Slot()
    def on_export_success(self):
        QMessageBox.information(self, "Export", "Data exported successfully.")

    def toggle_inputs(self, enabled: bool):
        self.btn_load.setEnabled(enabled)
        self.btn_select.setEnabled(enabled)
        self.btn_cal.setEnabled(enabled) 
        self.combo_leq_int.setEnabled(enabled)
        
        # Play controls should remain enabled if file loaded, even if analysis runs?
        # Typically yes, you might want to listen while analyzing.
        # But toggle_inputs is called with False during analysis.
        # If we want async play, we should probably force them enabled or skip disabling them.
        # For safety, let's let them be disabled during analysis to avoid file access conflicts if any.
        self.btn_play.setEnabled(enabled)
        self.btn_stop.setEnabled(enabled)
        
        can_export = (len(self.controller.last_results) > 0)
        self.menu_export.setEnabled(enabled and can_export)
        
        for child in self.left_panel.findChildren(QGroupBox):
             if child.title() != "File & Selection": 
                 child.setEnabled(enabled)

    def _redraw_plot(self):
        results = self.controller.last_results
        if not results: return
        
        mode_id = self.bg_mode.checkedId()
        weighting = self.bg_weight.checkedButton().text()
        speed = self.bg_speed.checkedButton().text()
        
        leq_key = self.combo_leq_int.currentData()
        dose_std = self.controller.settings.current_dose_standard
        dose_params = self.controller.settings.dose_standards.get(dose_std)
        
        ResultPlotter.plot(
            self.plot_panel.figure,
            results,
            mode_id,
            weighting,
            speed,
            leq_key,
            self.controller.settings.block_size_ms,
            dose_params,
            dose_std,
            self.controller.settings.ref_pressure,
            self.controller.settings.plot_autoscale,
            self.controller.settings.plot_ymin,
            self.controller.settings.plot_ymax,
            spec_cmap=self.controller.settings.spec_cmap
        )
        self.plot_panel.draw()

    def closeEvent(self, event):
        self._scrape_ui_to_settings()
        self.controller.shutdown()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())