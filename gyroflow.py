"""Main file containing UI code"""

import sys
import random
import cv2
from PySide2 import QtCore, QtWidgets, QtGui
from PySide2.QtMultimediaWidgets import QVideoWidget
from _version import __version__
import calibrate_video
import time

class Launcher(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Gyroflow {} Launcher".format(__version__))

        self.setFixedWidth(450)

        self.hello = ["Hai", "Hello there", "What's up", "Hello world"]
        self.drop = QtWidgets.QComboBox()
        self.drop.addItems(self.hello)
        self.drop.currentIndexChanged.connect(self.what)

        self.calibrator_button = QtWidgets.QPushButton("Camera Calibrator")
        self.calibrator_button.setMinimumSize(300,50)
        self.calibrator_button.setToolTip("Use this to generate camera calibration files")

        self.stabilizer_button = QtWidgets.QPushButton("Video Stabilizer")
        self.stabilizer_button.setMinimumSize(300,50)

        self.stretch_button = QtWidgets.QPushButton("Nonlinear Stretch")
        self.stretch_button.setMinimumSize(300,50)

        self.text = QtWidgets.QLabel("<h1>Gyroflow {}</h1>".format(__version__))
        self.text.setAlignment(QtCore.Qt.AlignCenter)

        self.footer = QtWidgets.QLabel('''Developed by Elvin. <a href='https://github.com/ElvinC/gyroflow'>Contribute or support on Github</a>''')
        self.footer.setOpenExternalLinks(True)
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.calibrator_button)
        self.layout.addWidget(self.stabilizer_button)
        self.layout.addWidget(self.stretch_button)
        
        self.layout.addWidget(self.drop)
        self.setLayout(self.layout)

        self.layout.addWidget(self.footer)

        self.calibrator_button.clicked.connect(self.open_calib_utility)




        # Placeholder for utility windows.
        self.calibrator_utility = None
        self.stabilizer_utility = None

    def open_calib_utility(self):

        # Only open if not already open
        if self.calibrator_utility:
            if self.calibrator_utility.isVisible():
                return
        
        self.calibrator_utility = CalibratorWidget()
        self.calibrator_utility.show()



    def what(self):
        self.text.setText(self.hello[self.drop.currentIndex()])
        print(self.calibrator_utility.isVisible())


class VideoThread(QtCore.QThread):
    changePixmap = QtCore.Signal(QtGui.QImage)
    playing = False

    def run(self):
        self.cap = cv2.VideoCapture("hero5.mp4")

        while True:
            if self.playing:
                ret, frame = self.cap.read()
                if ret:
                    time.sleep(1/24)
                    # https://stackoverflow.com/a/55468544/6622587
                    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgbImage.shape
                    bytesPerLine = ch * w
                    convertToQtFormat = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
                    self.changePixmap.emit(convertToQtFormat.copy())


# based on https://robonobodojo.wordpress.com/2018/07/01/automatic-image-sizing-with-pyside/
# and https://stackoverflow.com/questions/44404349/pyqt-showing-video-stream-from-opencv/44404713
class VideoPlayer(QtWidgets.QLabel):
    def __init__(self, img):
        super(VideoPlayer, self).__init__()
        self.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        self.pixmap = QtGui.QPixmap("cat_placeholder.jpg")
        #self.setPixmap(self.pixmap)
        
        self.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)


    def paintEvent(self, event = None):
        size = self.size()
        painter = QtGui.QPainter(self)
        point = QtCore.QPoint(0,0)
        scaledPix = self.pixmap.scaled(size, QtCore.Qt.KeepAspectRatio, transformMode = QtCore.Qt.SmoothTransformation)
        point.setX((size.width() - scaledPix.width())/2)
        point.setY((size.height() - scaledPix.height())/2)
        # print point.x(), ' ', point.y()
        painter.drawPixmap(point, scaledPix)



class VideoPlayerWidget(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.player = VideoPlayer("cat_placeholder.jpg")

        vb_layout = QtWidgets.QVBoxLayout()
        vb_layout.addWidget(self.player)
        self.setStyleSheet("background-color:rgb(100, 100, 100);")
        self.setLayout(vb_layout)

        self.thread = VideoThread(self)
        self.thread.changePixmap.connect(self.setImage)
        self.thread.start()
        
    @QtCore.Slot(QtGui.QImage)
    def setImage(self, image):
        pixmap = QtGui.QPixmap.fromImage(image)
        self.player.pixmap = pixmap
        self.player.setPixmap(pixmap)

class CalibratorUtility(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize UI
        self.setWindowTitle("Gyroflow Calibrator {}".format(__version__))

        self.main_widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QVBoxLayout()
        self.main_widget.setLayout(self.layout)

        self.calibrator_button = QtWidgets.QPushButton("Camera Calibrator")
        self.calibrator_button.setMinimumSize(300,50)
        self.calibrator_button.setToolTip("Use this to generate camera calibration files")
        self.calibrator_button.clicked.connect(self.toggle_play)


        self.text = QtWidgets.QLabel("<h1>Calibrator</h1>")
        self.text.setAlignment(QtCore.Qt.AlignCenter)

        self.picture = QtGui.QPixmap("cat_placeholder.jpg")

        #self.cat_pic = QtWidgets.QLabel(self)
        #self.cat_pic.setPixmap(self.picture.setDevicePixelRatio(1))

        #self.layout.addWidget(self.cat_pic)  
        
        self.video_viewer = VideoPlayerWidget()
        self.layout.addWidget(self.video_viewer)

        self.layout.addWidget(self.text)
        self.layout.addWidget(self.calibrator_button)


        # control bar init

        self.control_bar = QtWidgets.QWidget()
        self.control_layout = QtWidgets.QHBoxLayout()
        self.control_bar.setLayout(self.control_layout)

        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)

        self.add_frame_button = QtWidgets.QPushButton("Analyze current frame")


        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        
        self.control_layout.addWidget(self.play_button)
        self.control_layout.addWidget(self.time_slider)

        self.layout.addWidget(self.control_bar)


        self.setCentralWidget(self.main_widget)

        self.open_file = QtWidgets.QAction(QtGui.QIcon('exit24.png'), 'Open file', self)
        self.open_file.setShortcut("Ctrl+O")
        self.open_file.triggered.connect(self.open_file_func)

        self.statusBar()

        menubar = self.menuBar()
        filemenu = menubar.addMenu('&File')
        filemenu.addAction(self.open_file)

        self.show()

        self.main_widget.show()

    def toggle_play(self):
        #self.video_viewer.thread.cap.set(cv2.CAP_PROP_POS_MSEC, 2)
        self.video_viewer.thread.playing = not self.video_viewer.thread.playing
        self.play_button.setText("Pause" if self.video_viewer.thread.playing else "Play")

    def open_file_func(self):
        print("HEY")
        path = QtWidgets.QFileDialog.getOpenFileName(self, "Open video file", filter="Video (*.mp4 *.avi, *.mov)")
        print(path[0])

    def closeEvent(self, event):
        print("Closing")
        event.accept()


    def save_preset_file(self):
        pass

    

def main():
    app = QtWidgets.QApplication([])

    widget = CalibratorUtility()
    widget.resize(500, 500)
    import time


    widget.show()

    

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()