from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


class MyPicBox(QWidget):
    def __init__(self, parent:QWidget = None):
        QWidget.__init__(self, parent=parent)
        lay:QGridLayout = parent.layout()
        lay.children().clear()
        lay.addWidget(self)
        self.p = QPixmap()

    def setPixmap(self, p):
        self.p = p
        self.update()

    def paintEvent(self, event):
        if not self.p.isNull():
            painter = QPainter()
            painter.begin(self)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            pix = self.p.scaled(
                QSize(int(self.width()), int(self.height())),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation)
            painter.drawPixmap(0, 0, self.width(), self.height(), pix,
                               0, 0, self.width(), self.height())
            painter.end()
