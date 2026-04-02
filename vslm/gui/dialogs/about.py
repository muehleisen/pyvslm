from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QFrame, QLabel, QPushButton, QVBoxLayout,
)


class AboutDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About VSLM")
        self.resize(320, 260)

        layout = QVBoxLayout(self)

        title = QLabel("VSLM — Virtual Sound Level Meter")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #1e3a8a;")
        layout.addWidget(title)

        version = QLabel("Version 2.0")
        version.setAlignment(Qt.AlignCenter)
        version.setStyleSheet("color: #555;")
        layout.addWidget(version)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        info = QLabel(
            "<b>Purpose:</b><br>"
            "Process digital recordings as if measured by an ANSI-compliant "
            "1/3-octave sound level meter.<br><br>"
            "<b>Filters:</b><br>"
            "IEC 61672-1 A/C weighting &nbsp;·&nbsp; ANSI S1.11 octave / 1/3-oct bands<br><br>"
            "<b>GUI:</b> PySide6 (Qt 6) &nbsp;·&nbsp; <b>Plots:</b> Matplotlib<br>"
            "<b>License:</b> MIT"
        )
        info.setAlignment(Qt.AlignLeft)
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addStretch()
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)
