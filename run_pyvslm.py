"""VSLM application entry point."""
import sys
from PySide6.QtWidgets import QApplication
from vslm.gui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
