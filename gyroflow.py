"""Main file containing UI code"""

import sys
import random
import cv2
from PySide2 import QtCore, QtWidgets, QtGui
from _version import __version__
import calibrate_video
import time

import nonlinear_stretch

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
        
        #self.layout.addWidget(self.drop)
        self.setLayout(self.layout)

        self.layout.addWidget(self.footer)

        self.calibrator_button.clicked.connect(self.open_calib_util)
        self.stabilizer_button.clicked.connect(self.open_stab_util)
        self.stretch_button.clicked.connect(self.open_stretch_util)




        # Placeholder for utility windows.
        self.calibrator_utility = None
        self.stabilizer_utility = None
        self.stretch_utility = None

    def open_calib_util(self):

        # Only open if not already open
        if self.calibrator_utility:
            if self.calibrator_utility.isVisible():
                return
        
        self.calibrator_utility = CalibratorWidget()
        self.calibrator_utility.show()

    def open_stab_util(self):
        pass

    def open_stretch_util(self):
        # Only open if not already open
        if self.stretch_utility:
            if self.calibrator_utility.isVisible():
                return
        
        self.stretch_utility = StretchUtility()
        self.stretch_utility.show()

    def what(self):
        self.text.setText(self.hello[self.drop.currentIndex()])
        print(self.stretch_utility.isVisible())


class VideoThread(QtCore.QThread):
    changePixmap = QtCore.Signal(QtGui.QImage)
    playing = False
    update_once = False
    next_frame = False



    def run(self):

        self.map1s = []
        self.map2s = []

        # Draw vertical lines at given coords
        self.vert_line_coords = []

        self.cap = cv2.VideoCapture("chessboard.mp4")

        self.frame = None

        while True:
            if self.playing or self.next_frame:
                self.next_frame = False
                ret, self.frame = self.cap.read()
                if ret:
                    time.sleep(1/24)
                    self.update_frame()

            elif self.update_once:
                self.update_once = False
                self.update_frame()


    def update_frame(self):

        if self.frame is None :
            self.next_frame
            return

        # https://stackoverflow.com/a/55468544/6622587
        rgbImage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

        for i in range(len(self.map1s)):
            # apply the maps using linear interpolation for now
            rgbImage = cv2.remap(rgbImage, self.map1s[i].astype('float32'), self.map2s[i].astype('float32'), cv2.INTER_LINEAR)
            
        for line_pos in self.vert_line_coords:
            cv2.line(rgbImage,(int(line_pos), 0),(int(line_pos),rgbImage.shape[0]),(255,255,0),2)

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

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.player)
        #self.setStyleSheet("background-color:rgb(100, 100, 100);")
        self.setLayout(self.layout)

        self.control_bar = QtWidgets.QWidget()
        self.control_layout = QtWidgets.QHBoxLayout()
        self.control_bar.setLayout(self.control_layout)

        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)

        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)

        self.control_layout.addWidget(self.play_button)
        self.control_layout.addWidget(self.time_slider)

        self.layout.addWidget(self.control_bar)

        self.frame_width = 1920 # placeholder
        self.frame_height = 1080
        

        self.thread = VideoThread(self)
        self.thread.changePixmap.connect(self.setImage)
        self.thread.start()
        
    @QtCore.Slot(QtGui.QImage)
    def setImage(self, image):
        pixmap = QtGui.QPixmap.fromImage(image)
        self.player.pixmap = pixmap
        self.player.setPixmap(pixmap)

    def stop(self):
        self.thread.playing = False
        self.play_button.setText("Play")

    def play(self):
        self.thread.playing = True
        self.play_button.setText("Pause")

    def toggle_play(self):
        #self.video_viewer.thread.cap.set(cv2.CAP_PROP_POS_MSEC, 2)
        self.thread.playing = not self.thread.playing
        self.play_button.setText("Pause" if self.thread.playing else "Play")

    def set_video_path(self, path):
        self.stop()
        self.thread.cap.release()
        self.thread.cap = cv2.VideoCapture(path)
        self.frame_width = self.thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_height = self.thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def reset_maps(self):
        self.thread.map1s = []
        self.thread.map2s = []

    def add_maps(self, map1, map2):
        self.thread.map1s.append(map1)
        self.thread.map2s.append(map2)

    def reset_lines(self):
        self.thread.vert_line_coords = []

    def add_vert_lines(self, xcoord):
        self.thread.vert_line_coords.append(xcoord)

    def update_frame(self):
        self.thread.update_once = True


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
        #self.calibrator_button.clicked.connect(self.toggle_play)


        #self.text = QtWidgets.QLabel("<h1>Calibrator</h1>")
        #self.text.setAlignment(QtCore.Qt.AlignCenter)

        
        self.video_viewer = VideoPlayerWidget()
        self.layout.addWidget(self.video_viewer)


        # control bar init

        self.add_frame_button = QtWidgets.QPushButton("Analyze current frame")
        
        #self.layout.addWidget(self.text)
        self.layout.addWidget(self.calibrator_button)

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

    def open_file_func(self):
        print("HEY")
        path = QtWidgets.QFileDialog.getOpenFileName(self, "Open video file", filter="Video (*.mp4 *.avi, *.mov)")
        self.video_viewer.set_video_path(path[0])
        self.video_viewer.update_frame()
        print(path[0])

    def closeEvent(self, event):
        print("Closing")
        event.accept()


    def save_preset_file(self):
        pass



class StretchUtility(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize UI
        self.setWindowTitle("Gyroflow Stretcher {}".format(__version__))

        self.main_widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QVBoxLayout()
        self.main_widget.setLayout(self.layout)

        # video player with controls
        self.video_viewer = VideoPlayerWidget()
        self.layout.addWidget(self.video_viewer)

        self.setCentralWidget(self.main_widget)

        # control buttons for stretching [Safe area slider] [expo slider] [X]View safe area [recompute maps] [render to file]
        self.stretch_controls = QtWidgets.QWidget()
        self.stretch_controls_layout = QtWidgets.QHBoxLayout()
        self.stretch_controls.setLayout(self.stretch_controls_layout)

        # slider for adjusting unwarped area
        self.safe_area_text = QtWidgets.QLabel("Safe area (0%):")
        self.safe_area_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.safe_area_slider.setMinimum(0)
        self.safe_area_slider.setValue(0)
        self.safe_area_slider.setMaximum(100)
        self.safe_area_slider.setSingleStep(1)
        self.safe_area_slider.setTickInterval(1)
        self.safe_area_slider.valueChanged.connect(self.safe_area_changed)

        self.stretch_controls_layout.addWidget(self.safe_area_text)
        self.stretch_controls_layout.addWidget(self.safe_area_slider)

        # slider for adjusting non linear expo
        self.expo_text = QtWidgets.QLabel("Stretch expo (2.0):")
        self.expo_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.expo_slider.setMinimum(10)
        self.expo_slider.setValue(20)
        self.expo_slider.setMaximum(40)
        self.expo_slider.setSingleStep(1)
        self.expo_slider.setTickInterval(1)
        self.expo_slider.valueChanged.connect(self.stretch_expo_changed)


        self.stretch_controls_layout.addWidget(self.expo_text)
        self.stretch_controls_layout.addWidget(self.expo_slider)        

        # checkmark for showing untouched area
        self.show_safe_check = QtWidgets.QCheckBox("Show safe area: ")
        self.show_safe_check.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.show_safe_check.stateChanged.connect(self.safe_check_change)

        self.stretch_controls_layout.addWidget(self.show_safe_check)

        # button for recomputing image stretching maps
        self.recompute_stretch_button = QtWidgets.QPushButton("Apply settings")
        self.recompute_stretch_button.clicked.connect(self.recompute_stretch)
        
        self.stretch_controls_layout.addWidget(self.recompute_stretch_button)

        # button for recomputing image stretching maps
        self.export_button = QtWidgets.QPushButton("Export video")
        self.export_button.clicked.connect(self.export_video)
        
        self.stretch_controls_layout.addWidget(self.export_button)

        # add control bar to main layout
        self.layout.addWidget(self.stretch_controls)

        # file menu setup
        self.open_file = QtWidgets.QAction(QtGui.QIcon('exit24.png'), 'Open file', self)
        self.open_file.setShortcut("Ctrl+O")
        self.open_file.triggered.connect(self.open_file_func)

        self.infile_path = ""

        self.statusBar()

        menubar = self.menuBar()
        filemenu = menubar.addMenu('&File')
        filemenu.addAction(self.open_file)

        self.show()

        self.main_widget.show()

        # non linear setup
        self.nonlin = nonlinear_stretch.NonlinearStretch((1280,720), (1920,1080))

    def open_file_func(self):
        path = QtWidgets.QFileDialog.getOpenFileName(self, "Open video file", filter="Video (*.mp4 *.avi *.mov)")
        self.infile_path = path[0]
        self.video_viewer.set_video_path(path[0])


        # recompute non linear stretch maps
        self.recompute_stretch()

        self.video_viewer.play()

        

    def closeEvent(self, event):
        print("Closing")
        event.accept()

    def safe_area_changed(self):
        safe_area_val = self.safe_area_slider.value()
        self.nonlin.set_safe_area(safe_area_val/100)
        self.safe_area_text.setText("Safe area ({}%): ".format(safe_area_val))
        self.safe_check_change()

    def stretch_expo_changed(self):
        stretch_expo_val = self.expo_slider.value() / 10
        self.nonlin.set_expo(stretch_expo_val)
        self.expo_text.setText("Stretch expo ({}):".format(stretch_expo_val))

    def recompute_stretch(self):
        self.nonlin.set_in_size((self.video_viewer.frame_width, self.video_viewer.frame_height))
        self.nonlin.recompute_maps()
        self.video_viewer.reset_maps()
        self.video_viewer.add_maps(self.nonlin.map1, self.nonlin.map2)
        self.safe_check_change()


    def safe_check_change(self):
        self.video_viewer.reset_lines()

        # if checked show safe area
        if self.show_safe_check.isChecked():
            midpoint = self.nonlin.out_size[0] / 2
            safe_dist = self.nonlin.safe_area * self.nonlin.out_size[0] / 2
            line1 = int(midpoint + safe_dist)
            line2 = int(midpoint - safe_dist)

            self.video_viewer.add_vert_lines(line1)
            self.video_viewer.add_vert_lines(line2)
        
        self.video_viewer.update_frame()

    def export_video(self):
        # no input file opened
        if self.infile_path == "":
            self.show_error("No video file loaded")
            return

        self.video_viewer.stop()

        # get file

        filename = QtWidgets.QFileDialog.getSaveFileName(self, "Export video", filter="Video (*.mp4 *.avi, *.mov)")
        print(filename[0])

        if len(filename) == 0:
            self.show_error("No output chosen")
            return
        
        self.nonlin.stretch_save_video(self.infile_path, filename[0])

    def show_error(msg):
        err_window = QtWidgets.QMessageBox(self)
        err_window.setIcon(QtWidgets.QMessageBox.Critical)
        err_window.setText(msg)
        err_window.setWindowTitle("Something's gone awry")
        err_window.show()

def main():
    app = QtWidgets.QApplication([])

    widget = Launcher()
    widget.resize(500, 500)
    import time


    widget.show()

    

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
    # Pack to exe using:
    # pyinstaller gyroflow.py --add-binary <path-to-python>\Python38\Lib\site-packages\cv2\opencv_videoio_ffmpeg430_64.dll
    # in my case:
    # pyinstaller -F gyroflow.py --add-binary C:\Users\elvin\AppData\Local\Programs\Python\Python38\Lib\site-packages\cv2\opencv_videoio_ffmpeg430_64.dll;.
    # -F == one file, -w == no command window