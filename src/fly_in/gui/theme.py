"""Shared Tokyo Night palette and application widget styling."""

CANVAS = "#16161e"
TEXT = "#f8f8f2"
MUTED = "#9aa5ce"
ACCENT = "#7dcfff"
STYLESHEET = """
QMainWindow, QWidget { background: #1a1b26; color: #f8f8f2;
    font-size: 16px; }
QDockWidget, QGroupBox { background: #1f2335; }
QLabel { background: transparent; }
QPushButton, QComboBox, QLineEdit { background: #24283b;
    border: 1px solid #3b4261; border-radius: 5px; padding: 6px; }
QPushButton:hover { border-color: #7dcfff; }
QPushButton:disabled { color: #9aa5ce; }
QTreeWidget, QListWidget { background: #1f2335; }
QTreeWidget::item:selected, QListWidget::item:selected {
    background: #3b4261; color: #f8f8f2; }
QToolTip { background: #24283b; color: #f8f8f2;
    border: 1px solid #7dcfff; padding: 6px; }
QProgressBar { border: 1px solid #3b4261; text-align: center; }
QProgressBar::chunk { background: #3b4261; }
QWidget#header { background: #16161e; border-bottom: 1px solid #3b4261; }
QWidget#sidebar { background: #1f2335; }
QWidget#workspace { background: #1a1b26; }
QLabel#brand { font-size: 24px; font-weight: bold; color: #7dcfff; }
QLabel#mapTitle { font-size: 22px; font-weight: bold; }
QLabel#sectionTitle { font-size: 18px; font-weight: bold; }
QLabel#rootLabel, QLabel#quietText, QLabel#statLabel {
    color: #9aa5ce; font-size: 14px; }
QLabel#simulationState { font-size: 16px; color: #7dcfff; }
QLabel#detailText, QLabel#legend { font-size: 14px; }
QFrame#separator { background: #3b4261; max-height: 1px; }
QPushButton#primaryButton { padding: 12px; font-size: 14px;
    font-weight: bold; background: #7dcfff; color: #16161e;
    border: 1px solid #7dcfff; }
QPushButton#primaryButton:hover { background: #a6e0ff; }
QPushButton#primaryButton[playing="true"] {
    background: #e0af68; color: #16161e; border-color: #e0af68; }
QPushButton#primaryButton[playing="true"]:hover { background: #f0c788; }
QPushButton#primaryButton:disabled { background: #24283b;
    color: #9aa5ce; border-color: #3b4261; }

"""
