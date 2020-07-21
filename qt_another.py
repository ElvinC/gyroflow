# tweaked test code from https://stackoverflow.com/questions/44404349/pyqt-showing-video-stream-from-opencv/44404713

import cv2
import sys
from PySide2.QtWidgets import  QWidget, QLabel, QApplication
from PySide2.QtCore import QThread, Qt, Signal, Slot
from PySide2.QtGui import QImage, QPixmap

class Thread(QThread):
    changePixmap = Signal(QImage)

    def run(self):
        cap = cv2.VideoCapture("hero5.mp4")
        while True:
            ret, frame = cap.read()
            if ret:
                # https://stackoverflow.com/a/55468544/6622587
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(1280, 720, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    @Slot(QImage)
    def setImage(self, image):
        pixmap = QPixmap.fromImage(image).scaledToWidth(100)
        self.label.setPixmap(pixmap)

    def initUI(self):
        self.setWindowTitle("tes")
        self.resize(1800, 1200)
        # create a label
        self.label = QLabel(self)
        self.label.move(0, 0)
        self.label.resize(640, 480)
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()
        self.show()


appli = QApplication([])

widget = App()
widget.resize(800, 600)
widget.show()
sys.exit(appli.exec_())