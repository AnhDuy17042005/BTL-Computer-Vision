from __future__ import annotations
import sys
from pathlib import Path
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

ROOT = Path(__file__).resolve().parents[2]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My GUI")
        self.setFixedSize(QSize(1200, 900))

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()