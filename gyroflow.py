"""Main file containing UI code"""

import sys
import random
import cv2
import os
import numpy as np
from PySide2 import QtCore, QtWidgets, QtGui
from matplotlib import colors
from _version import __version__
from vidgear.gears.helper import get_valid_ffmpeg_path
import calibrate_video
import time
import nonlinear_stretch
import urllib.request
import json
import re
import subprocess
import bundled_images
import insta360_utility as insta360_util
import stabilizer
import smoothing_algos
from datetime import datetime
import gyrolog
from UI_elements import sync_ui

# area for environment variables
try:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
    os.environ.pop("QT_QPA_FONTDIR")
except:
    pass


# https://en.wikipedia.org/wiki/List_of_digital_camera_brands
cam_company_list = ["GoPro", "Runcam", "Insta360", "Caddx", "Foxeer", "DJI", "RED", "Canon", "Arri",
                    "Blackmagic", "Casio", "Nikon", "Panasonic", "Sony", "Jvc", "Olympus", "Fujifilm",
                    "Phone"]

class Launcher(QtWidgets.QWidget):
    """Main launcher with options to open different utilities
    """
    def __init__(self):

        super().__init__()

        self.setWindowTitle("Gyroflow {} Launcher".format(__version__))
        self.setWindowIcon(QtGui.QIcon(':/media/icon.png'))

        self.setFixedWidth(450)
        # image
        pixmap = QtGui.QPixmap(':/media/logo_rev0_w400.png')
        self.top_logo = QtWidgets.QLabel()
        self.top_logo.setPixmap(pixmap.scaled(400,450,QtCore.Qt.KeepAspectRatio))
        self.top_logo.setAlignment(QtCore.Qt.AlignCenter)

        self.text = QtWidgets.QLabel("<h2>Version {}</h2>".format(__version__))
        self.text.setAlignment(QtCore.Qt.AlignCenter)

        self.calibrator_button = QtWidgets.QPushButton("Camera Calibrator")
        self.calibrator_button.setMinimumSize(300,50)
        self.calibrator_button.setStyleSheet("font-size: 14px;")
        self.calibrator_button.setToolTip("Use this to generate camera calibration files")

        self.stabilizer_button = QtWidgets.QPushButton("Video Stabilizer (Fancy version)")
        self.stabilizer_button.setMinimumSize(300,50)
        self.stabilizer_button.setEnabled(True)
        self.stabilizer_button.setStyleSheet("font-size: 14px;")

        self.stabilizer_barebone_button = QtWidgets.QPushButton("Video Stabilizer (old)")
        self.stabilizer_barebone_button.setMinimumSize(300,30)
        self.stabilizer_barebone_button.setStyleSheet("font-size: 13px;")

        self.stretch_button = QtWidgets.QPushButton("Non-linear Stretch")
        self.stretch_button.setMinimumSize(300,30)
        self.stretch_button.setStyleSheet("font-size: 13px;")


        self.version_button = QtWidgets.QPushButton("Check for updates")
        self.version_button.setMinimumSize(300,30)
        self.version_button.setStyleSheet("font-size: 13px;")

        self.footer = QtWidgets.QLabel('''Developed by Elvin & Contributors | <a href='http://gyroflow.xyz/'>gyroflow.xyz</a> | <a href='https://github.com/ElvinC/gyroflow'>Git repo</a> | <a href='http://gyroflow.xyz/donate'>Donate</p>''')
        self.footer.setOpenExternalLinks(True)
        self.footer.setAlignment(QtCore.Qt.AlignCenter)


        self.layout = QtWidgets.QVBoxLayout()

        self.layout.addWidget(self.top_logo)
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.stabilizer_button)
        
        self.layout.addWidget(self.calibrator_button)
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        line.setMinimumHeight(20)
        self.layout.addWidget(line)
        self.layout.addWidget(self.stabilizer_barebone_button)
        self.layout.addWidget(self.stretch_button)
        self.layout.addWidget(self.version_button)
        self.layout.addWidget(self.footer)

        self.setLayout(self.layout)

        self.calibrator_button.clicked.connect(self.open_calib_util)
        self.stabilizer_button.clicked.connect(self.open_stab_util)
        self.stabilizer_barebone_button.clicked.connect(self.open_stab_util_barebone)
        self.stretch_button.clicked.connect(self.open_stretch_util)
        self.version_button.clicked.connect(self.check_version)

        # Placeholder for utility windows.
        self.calibrator_utility = None
        self.stabilizer_utility = None
        self.stabilizer_utility_barebone = None
        self.stretch_utility = None

        self.check_version(True)

    def open_calib_util(self):
        """Open the camera calibration utility in a new window
        """

        # Only open if not already open
        if self.calibrator_utility:
            if self.calibrator_utility.isVisible():
                return
        self.calibrator_utility = CalibratorUtility()
        self.calibrator_utility.resize(500, 500)
        self.calibrator_utility.show()

    def open_stab_util(self):
        """Open video stabilization utility in new window
        """
        # Only open if not already open
        if self.stabilizer_utility:
            if self.stabilizer_utility.isVisible():
                return
        self.stab_utility = StabUtility()
        self.stab_utility.resize(500, 500)
        self.stab_utility.show()

    def open_stab_util_barebone(self):
        if self.stabilizer_utility_barebone:
            if self.stabilizer_utility_barebone.isVisible():
                return

        self.stabilizer_utility_barebone = StabUtilityBarebone()
        self.stabilizer_utility_barebone.resize(500, 800)
        self.stabilizer_utility_barebone.show()

    def open_stretch_util(self):
        """Open non-linear stretch utility in new window
        """
        # Only open if not already open
        if self.stretch_utility:
            if self.stretch_utility.isVisible():
                return
        self.stretch_utility = StretchUtility()
        self.stretch_utility.resize(500, 500)
        self.stretch_utility.show()

    def check_version(self, background=False):
        try:
            with urllib.request.urlopen("https://api.github.com/repos/elvinc/gyroflow/releases") as url:

                releases = json.loads(url.read())
                newest_version = releases[0]["tag_name"]
                new = re.match("(\d+).(\d+).(\d+).*", newest_version).groups()
                current = re.match("(\d+).(\d+).(\d+).*", __version__).groups()

                extra = ""

                diff = [int(A) - int(B) for A,B in zip(new,current)]
                val = diff[0] if diff[0] != 0 else diff[1] if diff[1] != 0 else diff[2] if diff[2] != 0 else 0  
                
                showpopup = not background

                if newest_version.strip() == __version__.strip():
                    extra = "Not much to see here:"
                elif val > 0:
                    extra = "Oh look, there's a shiny new update. <a href='https://elvinchen.com/gyroflow/download/'>Here's a link just for you.</a> "
                    showpopup = True
                elif val < 0:
                    extra = "Looks like somebody is time traveling..."
                else:
                    extra = "Spot the difference:"

                if showpopup:
                    msg_window = QtWidgets.QMessageBox(self)
                    msg_window.setIcon(QtWidgets.QMessageBox.Information)
                    msg_window.setText("{}<br>Your version: <b>{}</b>, newest release: <b>{}</b>".format(extra, __version__, newest_version))
                    msg_window.setWindowTitle("Version check")
                    msg_window.show()
        except: 
            print("Unable to check version")


class VideoThread(QtCore.QThread):
    changePixmap = QtCore.Signal(QtGui.QImage)

    def __init__(self, parent, frame_pos_update = None):
        """Video Thread

        Args:
            parent ([type]):
            frame_pos_update (function, optional): Function to call for frame num update. Defaults to None.
        """
        super().__init__(parent)

        self.playing = False
        self.update_once = False
        self.next_frame = False
        self.frame_pos_update = frame_pos_update
        self.map1s = []
        self.map2s = []
        self.frame_delay = 1/30

        self.map_function = None
        self.map_function_enable = True

        self.map_preview_res = (1280, 720)

        # Draw vertical lines at given coords
        self.vert_line_coords = []

        self.cap = None
        self.frame = None

        # used for scaling
        self.max_width = 1280

        self.stretch_enable = False
        self.horizontal_stretch = 1

    def run(self):
        """
        Run the videoplayer using the thread
        """

        self.cap = cv2.VideoCapture()

        while True:
            if self.playing or self.next_frame:
                self.next_frame = False
                self.this_frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                ret, self.frame = self.cap.read()
                if ret:
                    time.sleep(self.frame_delay)
                    self.update_frame()


            elif self.update_once:
                self.update_once = False
                self.update_frame()

            else:
                time.sleep(1/20)


    def update_frame(self):
        """Opdate the current video frame shown
        """

        if self.frame is None:
            return

        # https://stackoverflow.com/a/55468544/6622587
        rgbImage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        

        for i in range(len(self.map1s)):
            # apply the maps using linear interpolation for now
            rgbImage = cv2.remap(rgbImage, self.map1s[i], self.map2s[i], cv2.INTER_LINEAR)
        for line_pos in self.vert_line_coords:
            cv2.line(rgbImage,(int(line_pos), 0),(int(line_pos),rgbImage.shape[0]),(255,255,0),2)

        # Resize to match pixel aspect

        if self.map_function and self.map_function_enable:
            tmap1, tmap2 = self.map_function(self.this_frame_num, out_size = self.map_preview_res)
            rgbImage = cv2.remap(rgbImage, tmap1, tmap2, cv2.INTER_LINEAR)


        if rgbImage.shape[1] > self.max_width:
            new_height = self.max_width/rgbImage.shape[1] * rgbImage.shape[0]

            if self.stretch_enable:
                new_height = new_height /  self.horizontal_stretch

            rgbImage = cv2.resize(rgbImage, (self.max_width, round(new_height)))

        elif self.stretch_enable:
            new_height = round(rgbImage.shape[0] / self.horizontal_stretch)
            rgbImage = cv2.resize(rgbImage, (rgbImage.shape[1], new_height))

        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        self.changePixmap.emit(convertToQtFormat.copy())

        
        if self.this_frame_num % 5 == 0 and self.frame_pos_update:
            self.frame_pos_update(self.this_frame_num)



# based on https://robonobodojo.wordpress.com/2018/07/01/automatic-image-sizing-with-pyside/
# and https://stackoverflow.com/questions/44404349/pyqt-showing-video-stream-from-opencv/44404713
class VideoPlayer(QtWidgets.QLabel):
    def __init__(self, img = "placeholder.jpg"):
        super(VideoPlayer, self).__init__()
        self.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        self.pixmap = QtGui.QPixmap(img)
        #self.setPixmap(self.pixmap)
        self.setMinimumSize(1,1)

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)


    def paintEvent(self, event = None):
        size = self.size()
        painter = QtGui.QPainter(self)
        point = QtCore.QPoint(0,0)
        if not self.pixmap.isNull():
            scaledPix = self.pixmap.scaled(size, QtCore.Qt.KeepAspectRatio, transformMode = QtCore.Qt.SmoothTransformation)
            point.setX((size.width() - scaledPix.width())/2)
            point.setY((size.height() - scaledPix.height())/2)
            # print point.x(), ' ', point.y()
            painter.drawPixmap(point, scaledPix)


class VideoPlayerWidget(QtWidgets.QWidget):
    def __init__(self):
        """Widget containing videoplayer and controls
        """
        QtWidgets.QWidget.__init__(self)
        self.player = VideoPlayer("placeholder.jpg")

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.player)
        #self.setStyleSheet("background-color:rgb(100, 100, 100);")
        self.setLayout(self.layout)

        self.control_bar = QtWidgets.QWidget()
        self.control_layout = QtWidgets.QHBoxLayout()
        self.control_bar.setLayout(self.control_layout)

        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)


        # seek bar with value from 0-200
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.time_slider.setMinimum(0)
        self.time_slider.setValue(0)

        self.seek_ticks = 200
        self.time_slider.setMaximum(self.seek_ticks)
        self.time_slider.setSingleStep(1)
        self.time_slider.setTickInterval(1)

        #self.time_slider.valueChanged.connect(self.seek)
        self.time_slider.sliderPressed.connect(self.start_seek)
        self.time_slider.sliderReleased.connect(self.stop_seek)

        self.is_seeking = False
        self.was_playing_before = False
        self.last_seek_time = time.time()


        self.time_stamp_display = QtWidgets.QLabel("0 s (--:-- / --:--)")
        self.time_stamp_display.setStyleSheet("font-size:12px;")


        self.control_layout.addWidget(self.play_button)
        self.control_layout.addWidget(self.time_slider)
        self.control_layout.addWidget(self.time_stamp_display)

        self.layout.addWidget(self.control_bar)

        self.frame_width = 1920 # placeholder
        self.frame_height = 1080
        self.num_frames = 0
        self.fps = 30
        self.video_length = 0


        # initialize thread for video player with frame update function
        self.thread = VideoThread(self, self.set_seek_frame)
        self.thread.changePixmap.connect(self.setImage)
        self.thread.start()

    @QtCore.Slot(QtGui.QImage)
    def setImage(self, image):
        pixmap = QtGui.QPixmap.fromImage(image)
        self.player.pixmap = pixmap
        self.player.setPixmap(pixmap)

    def set_cv_frame(self, frame):
        self.thread.frame = frame
        self.thread.update_once = True

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
        self.num_frames = self.thread.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.thread.cap.get(cv2.CAP_PROP_FPS)

        self.video_length = int(self.num_frames / self.fps)

        self.thread.frame_delay = max(1/self.fps, 0.005)

    def reset_maps(self):
        self.thread.map1s = []
        self.thread.map2s = []

    def add_maps(self, map1, map2):
        self.thread.map1s.append(map1)
        self.thread.map2s.append(map2)

    def set_map_function(self, map_function):
        self.thread.map_function = map_function

    def reset_map_function(self):
        self.thread.map_function = None

    def enable_map_function(self, enabled = True):
        self.thread.map_function_enable = enabled

    def set_horizontal_stretch(self, stretch=1):
        self.thread.horizontal_stretch = stretch

    def enable_stretch(self, enabled = True):
        self.thread.stretch_enable = enabled

    def reset_lines(self):
        self.thread.vert_line_coords = []

    def add_vert_lines(self, xcoord):
        self.thread.vert_line_coords.append(xcoord)

    def update_frame(self):
        self.thread.update_once = True

    def next_frame(self):
        self.thread.next_frame = True

    def start_seek(self):
        self.is_seeking = True
        self.was_playing_before = self.thread.playing
        self.stop()

    def stop_seek(self):
        self.is_seeking = False

        self.seek()

    def seek(self):
        """Handler for seek bar update
        """

        # only update when not dragging:
        #if self.is_seeking:
        #    return

        # prevent video overread using 0.2 sec cooldown
        timenow = time.time()
        if (timenow - self.last_seek_time) < 0.5:
            return

        self.last_seek_time = timenow

        was_playing = self.thread.playing
        if was_playing:
            self.stop()
        print(self.time_slider.value())
        selected_frame = int(self.num_frames * self.time_slider.value() / self.seek_ticks)
        print(selected_frame)
        self.thread.cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)

        # restart if it was playing
        if (was_playing or self.was_playing_before) and not self.is_seeking:
            self.play()
        else:
            self.next_frame()

    def set_seek_frame(self, frame_pos):
        """Set the seek bar position to match frame

        Args:
            frame (int): Frame number
        """

        # only update when slider not in use
        if self.is_seeking:
            return

        slider_val = int(frame_pos * self.seek_ticks / (max(self.num_frames, 1)))
        timestamp = frame_pos / self.fps 
        # update slider without triggering valueChange
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(slider_val)

        self.time_stamp_display.setText(f"{timestamp:.2f} s ({self.time_string(timestamp)} / {self.time_string(self.video_length)})")
        
        self.time_slider.blockSignals(False)

    def time_string(self, t):
        return f"{int(t / 60):02d}:{int(t) % 60:02d}"

    def destroy_thread(self):
        self.thread.terminate()

    def get_current_timestamp(self):
        if self.thread.cap:
            return self.thread.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        return 0

class CalibratorUtility(QtWidgets.QMainWindow):
    def __init__(self):
        """Qt window containing camera calibration utility
        """
        super().__init__()

        calib_input = QtWidgets.QInputDialog.getText(self, "Calibration setting","Calibration chessboard size. w, h",
                                                     QtWidgets.QLineEdit.Normal, "14,8")[0].split(",")

        try:
            w, h = [min(max(int(x), 1),30) for x in calib_input]
            self.chessboard_size = (w,h)

        except:
            print("setting to default 14,8 pattern")
            self.chessboard_size = (14,8)


        # Initialize UI
        self.setWindowTitle("Gyroflow Calibrator {}".format(__version__))
        self.setWindowIcon(QtGui.QIcon(':/media/icon.png'))

        self.main_widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout()
        self.main_widget.setLayout(self.layout)

        # left half of screen with player/buttons
        self.left_side_widget = QtWidgets.QWidget()
        self.left_layout = QtWidgets.QVBoxLayout()
        self.left_side_widget.setLayout(self.left_layout)

        self.layout.addWidget(self.left_side_widget)

        # right half of screen with export options
        self.right_side_widget = QtWidgets.QWidget()
        self.right_layout = QtWidgets.QVBoxLayout()
        self.right_side_widget.setLayout(self.right_layout)
        self.right_side_widget.setFixedWidth(250)
        self.right_layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.addWidget(self.right_side_widget)



        # video player with controls
        self.video_viewer = VideoPlayerWidget()
        self.video_viewer.enable_stretch()
        self.left_layout.addWidget(self.video_viewer)

        self.setCentralWidget(self.main_widget)

        # control buttons for stretching [Safe area slider] [expo slider] [X]View safe area [recompute maps] [render to file]
        self.calib_controls = QtWidgets.QWidget()
        self.calib_controls_layout = QtWidgets.QHBoxLayout()
        self.calib_controls.setLayout(self.calib_controls_layout)

        self.button_height = 40

        self.calib_msg = ""


        
        #self.show_chessboard_btn = QtWidgets.QPushButton("Calibration target")
        #self.show_chessboard_btn. setMinimumHeight(self.button_height)
        #self.show_chessboard_btn.clicked.connect(self.chessboard_func)
        #self.show_chessboard_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogListView))
        #self.calib_controls_layout.addWidget(self.show_chessboard_btn)


        self.open_file_btn = QtWidgets.QPushButton("Open file")
        self.open_file_btn.setMinimumHeight(self.button_height)
        self.open_file_btn.clicked.connect(self.open_file_func)
        self.open_file_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon))
        self.calib_controls_layout.addWidget(self.open_file_btn)

         # button for recomputing image stretching maps
        self.add_frame_button = QtWidgets.QPushButton("Add current frame")
        self.add_frame_button.setMinimumHeight(self.button_height)
        self.add_frame_button.clicked.connect(self.add_current_frame)

        self.calib_controls_layout.addWidget(self.add_frame_button)

        self.del_frame_button = QtWidgets.QPushButton("Remove last frame")
        self.del_frame_button.setMinimumHeight(self.button_height)
        self.del_frame_button.clicked.connect(self.remove_frame)

        self.calib_controls_layout.addWidget(self.del_frame_button)
        
        # button for recomputing image stretching maps
        self.process_frames_btn = QtWidgets.QPushButton("Process loaded frames")
        self.process_frames_btn.setMinimumHeight(self.button_height)
        self.process_frames_btn.setEnabled(False)
        self.process_frames_btn.clicked.connect(self.calibrate_frames)
        self.calib_controls_layout.addWidget(self.process_frames_btn)

        # button for auto lens calibration. TODO: Move to menu
        #self.start_lens_calibration_btn = QtWidgets.QPushButton("Start lens calibration")
        #self.start_lens_calibration_btn.setMinimumHeight(self.button_height)
        #self.start_lens_calibration_btn.setEnabled(False)
        #self.start_lens_calibration_btn.clicked.connect(self.start_lens_calibration)

        #self.calib_controls_layout.addWidget(self.start_lens_calibration_btn)


        # info text box
        self.info_text = QtWidgets.QLabel("No frames loaded")
        self.calib_controls_layout.addWidget(self.info_text)



        # horizontal destretching
        self.destretch_text = QtWidgets.QLabel("Horizontal desqueeze:")
        self.calib_controls_layout.addWidget(self.destretch_text)

        self.destretch_control = QtWidgets.QDoubleSpinBox(self)
        self.destretch_control.setMinimum(0.01)
        self.destretch_control.setMaximum(4)
        self.destretch_control.setValue(1)
        self.destretch_control.setDecimals(5)
        self.destretch_control.setSingleStep(0.05)
        self.destretch_control.setToolTip("For processing stretched footage with non-square pixel aspect ratio. 0.75 corresponds to converting 16:9 to 4:3")
        self.destretch_control.valueChanged.connect(self.update_destretch)

        self.calib_controls_layout.addWidget(self.destretch_control)

        self.fov_scale = 1.4

        # slider for adjusting FOV
        self.fov_text = QtWidgets.QLabel("FOV scale ({}):".format(self.fov_scale))
        self.fov_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.fov_slider.setMinimum(8)
        self.fov_slider.setValue(14)
        self.fov_slider.setMaximum(30)
        self.fov_slider.setMaximumWidth(300)
        self.fov_slider.setSingleStep(1)
        self.fov_slider.setTickInterval(1)
        self.fov_slider.valueChanged.connect(self.fov_changed)
        self.fov_slider.sliderReleased.connect(self.update_preview)

        self.calib_controls_layout.addWidget(self.fov_text)
        self.calib_controls_layout.addWidget(self.fov_slider)

        # checkbox to preview lens distortion correction
        self.preview_toggle_btn = QtWidgets.QCheckBox("Toggle lens correction: ")
        self.preview_toggle_btn.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.preview_toggle_btn.stateChanged.connect(self.update_preview)
        self.preview_toggle_btn.setEnabled(False)

        self.calib_controls_layout.addWidget(self.preview_toggle_btn)

        # add control bar to main layout
        self.left_layout.addWidget(self.calib_controls)



        # Right layout: Export settings

        text = QtWidgets.QLabel("<h2>Preset parameters</h2>")
        text.setAlignment(QtCore.Qt.AlignCenter)
        self.right_layout.addWidget(text)

        self.right_layout.addWidget(QtWidgets.QLabel("Camera brand (*):"))
        completer = QtWidgets.QCompleter(cam_company_list)
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        # create line edit and add auto complete

        self.cam_company_input = QtWidgets.QLineEdit()
        self.cam_company_input.setCompleter(completer)
        self.cam_company_input.setPlaceholderText(random.choice(cam_company_list + ["Potatocam"]))
        self.right_layout.addWidget(self.cam_company_input)

        self.right_layout.addWidget(QtWidgets.QLabel("Camera make (*)"))
        self.cam_model_input = QtWidgets.QLineEdit()
        self.cam_model_input.setPlaceholderText(random.choice(["Hero5", "D5100", "Hero8", "Komodo", "Pocket Cinema", "2000D", "Alexa", "Potato"])) # Don't ask...
        self.right_layout.addWidget(self.cam_model_input)

        self.right_layout.addWidget(QtWidgets.QLabel("Lens name (leave blank if not relevant)"))
        self.cam_lens_input = QtWidgets.QLineEdit()
        self.cam_lens_input.setPlaceholderText(random.choice(["Nikkor 35mm f/1.8G", "EF-S 18-55mm", "Sigma 16mm F/1.4", "E 50mm F/1.8 OSS", "Tamron 17-28mm f/2.8 Di", "Rokinon 14mm T3.1", "PotatoGlass deluxe"]))
        self.right_layout.addWidget(self.cam_lens_input)

        self.right_layout.addWidget(QtWidgets.QLabel("Recording setting (*)"))
        self.cam_setting_input = QtWidgets.QLineEdit()
        self.cam_setting_input.setPlaceholderText("2160p 4by3 wide")
        self.right_layout.addWidget(self.cam_setting_input)

        self.right_layout.addWidget(QtWidgets.QLabel("Other relevant note"))
        self.cam_note_input = QtWidgets.QLineEdit()
        self.cam_note_input.setPlaceholderText(random.choice(["ND filter installed", "Bad light conditions", "Test calibration, don't use", "Can't believe potatoes can record video"]))
        self.right_layout.addWidget(self.cam_note_input)

        self.right_layout.addWidget(QtWidgets.QLabel("Name of calibrator (*)"))
        self.calibrated_by_input = QtWidgets.QLineEdit()
        self.calibrated_by_input.setText("Anonymous")
        self.right_layout.addWidget(self.calibrated_by_input)


        # button for exporting preset
        self.right_layout.addWidget(QtWidgets.QLabel("Check before saving:\nLow RMS error (<5 works, <1 is best)\nImage looks right after 'toggle lens correction'"))
        self.export_button = QtWidgets.QPushButton("Export preset file")
        self.export_button.setMinimumHeight(self.button_height)
        self.export_button.clicked.connect(self.save_preset_file)
        self.export_button.setEnabled(False)

        self.right_layout.addWidget(self.export_button, alignment=QtCore.Qt.AlignBottom)
        self.right_layout.addWidget(QtWidgets.QLabel("Want to help? Send new presets here:<br><a href='http://gyroflow.xyz/lens'>http://gyroflow.xyz/lens</a>"))

        # file menu setup
        menubar = self.menuBar()
        filemenu = menubar.addMenu('&File')

        # https://joekuan.wordpress.com/2015/09/23/list-of-qt-icons/

        icon = self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon)
        self.open_file = QtWidgets.QAction(icon, 'Open file', self)
        self.open_file.setShortcut("Ctrl+O")
        self.open_file.triggered.connect(self.open_file_func)
        filemenu.addAction(self.open_file)

        icon = self.style().standardIcon(QtWidgets.QStyle.SP_FileLinkIcon)
        self.open_preset = QtWidgets.QAction(icon, 'Open calibration preset', self)
        self.open_preset.triggered.connect(self.open_preset_func)
        filemenu.addAction(self.open_preset)

        icon = self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogListView)
        self.show_chessboard = QtWidgets.QAction(icon, 'Calibration target', self)
        self.show_chessboard.triggered.connect(self.chessboard_func)
        filemenu.addAction(self.show_chessboard)

        icon = self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay)
        self.start_lens_calibration_btn = QtWidgets.QAction(icon, 'Start auto calibration', self)
        self.start_lens_calibration_btn.triggered.connect(self.start_lens_calibration)
        self.start_lens_calibration_btn.setEnabled(False)
        filemenu.addAction(self.start_lens_calibration_btn)

        



        self.chess_window = None

        self.statusBar()

        self.infile_path = ""


        self.show()

        self.main_widget.show()

        # initialize instance of calibrator class
        self.calibrator = calibrate_video.FisheyeCalibrator(chessboard_size=self.chessboard_size)


    def update_destretch(self):
        print(f"Update destretch to {self.destretch_control.value()}")
        self.calibrator.set_horizontal_stretch(self.destretch_control.value())
        self.video_viewer.set_horizontal_stretch(self.destretch_control.value())
        self.video_viewer.update_frame()

    def open_file_func(self):
        """Open file using Qt filedialog
        """
        path = QtWidgets.QFileDialog.getOpenFileName(self, "Open video file", filter="Video (*.mp4 *.avi *.mov *.MP4 *.AVI *.MOV)")
        self.infile_path = path[0]
        self.video_viewer.set_video_path(path[0])

        self.video_viewer.next_frame()
        self.start_lens_calibration_btn.setEnabled(True)
        # reset calibrator and info
        self.calibrator = calibrate_video.FisheyeCalibrator(chessboard_size=self.chessboard_size)
        self.update_calib_info()


    def open_preset_func(self):
        """Load in calibration preset
        """
        path = QtWidgets.QFileDialog.getOpenFileName(self, "Open preset file", filter="JSON preset (*.json)")

        if (len(path[0]) == 0):
            print("No file selected")
            return

        self.calibrator.load_calibration_json(path[0])

        self.update_calib_info()

    def chessboard_func(self):
        """Function to show the calibration chessboard in a new window
        """
        print("Showing chessboard")

        board_width = self.chessboard_size[0]
        board_height = self.chessboard_size[1]

        self.chess_window = QtWidgets.QWidget()
        self.chess_window.setWindowTitle(f"Calibration target ({board_width}x{board_height})")
        self.chess_window.setStyleSheet("background-color:white;")

        self.chess_layout = QtWidgets.QVBoxLayout()
        self.chess_window.setLayout(self.chess_layout)

        # VideoPlayer class doubles as a auto resizing image viewer

        # generate chessboard pattern so no external images are needed


        chess_pic = np.zeros((board_height + 3,board_width + 3), np.uint8)

        # Set white squares
        chess_pic[::2,::2] = 255
        chess_pic[1::2,1::2] = 255

        # Borders to white
        chess_pic[0,:] = 255
        chess_pic[-1,:]= 255
        chess_pic[:,0]= 255
        chess_pic[:,-1]= 255

        # double size and reduce borders slightly
        chess_pic = cv2.resize(chess_pic,((board_width+3)*2, (board_height+3)*2), interpolation=cv2.INTER_NEAREST)
        chess_pic = chess_pic[1:-1,:]

        # convert to Qt image
        h, w = chess_pic.shape
        convertToQtFormat = QtGui.QImage(chess_pic.data, w, h, w, QtGui.QImage.Format_Grayscale8)

        # VideoPlayer doubles as a autoresiznig image viewer
        chess_viewer = VideoPlayer(convertToQtFormat.copy())

        self.chess_layout.addWidget(chess_viewer)

        self.chess_window.resize(500, 500)
        self.chess_window.showMaximized()


    def closeEvent(self, event):
        print("Closing now")
        self.video_viewer.destroy_thread()
        event.accept()

    def fov_changed(self):
        self.fov_scale = self.fov_slider.value()/10
        self.fov_text.setText("FOV scale ({}):".format(self.fov_scale))
        self.video_viewer.update_frame()

    def save_preset_file(self):
        """save camera preset file
        """
        print("Exporting preset")

        # Window to set export data

        cam_brand = self.cam_company_input.text()
        cam_model = self.cam_model_input.text()
        cam_lens = self.cam_lens_input.text()
        cam_setting = self.cam_setting_input.text()
        cam_note = self.cam_note_input.text()
        calibrated_by = self.calibrated_by_input.text()

        if not (cam_brand and cam_model and cam_setting):
            self.show_error("Missing information about the camera system")
            return

        if not calibrated_by:
            self.show_error("You went through all the trouble to make a calibration profile but don't want credit? I'll just fill in 'Anonymous' for you, but feel free to write a (nick)name or a handle instead.")
            self.calibrated_by_input.setText("Anonymous")
            return



        calib_name = f"{cam_brand}_{cam_model}_{cam_lens}_{cam_setting}".replace("__", "_").replace("__", "_")
        calib_name = " ".join(calib_name.replace("_", " ").split())

        # make sure name works
        default_file_name = calib_name.replace("@", "At") # 18-55mm@18mm -> 18-55mmAt18mm, eh works I guess
        default_file_name = default_file_name.replace("/", "_").replace(".", "_").replace(" ","_") # f/1.8 -> f_1_8
        default_file_name = "".join([c for c in default_file_name if c.isalpha() or c.isdigit() or c in "_-"]).rstrip()
        default_file_name = "_".join(default_file_name.replace("_", " ").split())

        filename = QtWidgets.QFileDialog.getSaveFileName(self, "Export calibration preset", default_file_name,
                                                        filter="JSON preset (*.json)")
        print(filename[0])

        if len(filename[0]) == 0:
            self.show_warning("No output file given")
            return

        self.calibrator.save_calibration_json(filename[0], calib_name=calib_name, camera_brand=cam_brand, camera_model=cam_model,
                                              lens_model=cam_lens, camera_setting=cam_setting, note=cam_note, calibrated_by=calibrated_by)

    def show_error(self, msg):
        QtWidgets.QMessageBox.critical(self, "Something's gone awfully wrong", msg)

        return
        #self.err_window = QtWidgets.QMessageBox(self)
        #self.err_window.setIcon(QtWidgets.QMessageBox.Warning)
        #self.err_window.setText(msg)
        #self.err_window.setWindowTitle("Something's gone awry")
        #self.err_window.exec_()
        #self.err_window.close()

    def show_warning(self, msg):
        QtWidgets.QMessageBox.critical(self, "Something's gone awry", msg)


    def add_current_frame(self):
        print("Adding frame")

        ret, self.calib_msg, corners = self.calibrator.add_calib_image(self.video_viewer.thread.frame)

        if ret:
            self.video_viewer.set_cv_frame(cv2.drawChessboardCorners(self.video_viewer.thread.frame, self.calibrator.chessboard_size,corners,True) )

        self.update_calib_info()

        if self.calibrator.num_images > 0:
            self.process_frames_btn.setEnabled(True)

    def remove_frame(self):
        """Remove last calibration frame
        """
        self.calibrator.remove_calib_image()

        self.update_calib_info()

    def start_lens_calibration(self):
        self.calibrator.new_calibration()
        n_calibration_frames = 50
        cap = cv2.VideoCapture(self.infile_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Starting lens calibration with {num_frames} frames")
        t = datetime.now()
        good_frames = []
        for n in np.linspace(0, num_frames - 1, n_calibration_frames):
            n = int(n)
            cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            ret, frame = cap.read()
            self.calibrator.num_processed_images += 1
            if ret:
                print("\n\nFrame:", n)
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                ret, message, corners = self.calibrator.add_calib_image(frame)
                print(message)
                self.update_calib_info()
                if ret:
                    cv2.drawChessboardCorners(rgbImage, self.calibrator.chessboard_size, corners, True)
                    scaled = cv2.resize(rgbImage, (960, 720))
                    cv2.imshow('Chessboard detection', scaled)
                    cv2.waitKey(1)
                    rms = self.calibrator.compute_calibration()
                    print("RMS:", rms)
                    if rms > 3:
                        self.calibrator.remove_calib_image()
                        if rms != 100:
                            print("rms too high, removing image")
                    else:
                        good_frames.append(n)
                        self.calibrator.num_images_used += 1
                else:
                    self.calibrator.remove_calib_image()
                    print("corners not detected, removing image")
                self.update_calib_info()
        cv2.destroyWindow('Chessboard detection')
        print(f"Good frames: {len(good_frames)} of {n_calibration_frames} ({len(good_frames) / n_calibration_frames * 100} %)")
        cap.release()

        print("Time for lens calibration: ", datetime.now() - t)



    def remove_frame(self):
        """Remove last calibration frame
        """
        self.calibrator.remove_calib_image()

        self.update_calib_info()


    def update_calib_info(self):
        """ Update the status text in the utility
        """

        txt = "Good frames: {}\nProcessed frames: {}\nRMS error: {:.5f}\n{}".format(
            self.calibrator.num_images_used,
            self.calibrator.num_processed_images,
            float(self.calibrator.RMS_error),
            self.calib_msg)
        self.info_text.setText(txt)

        # enable/disable buttons
        if self.calibrator.num_images > 0:
            self.process_frames_btn.setEnabled(True)
            self.destretch_control.setEnabled(False)
        else:
            self.process_frames_btn.setEnabled(False)
            self.destretch_control.setEnabled(True)

        if self.calibrator.num_images_used > 0:
            self.preview_toggle_btn.setEnabled(True)
            self.export_button.setEnabled(True)
        else:
            self.preview_toggle_btn.setChecked(False)
            self.preview_toggle_btn.setEnabled(False)
            self.export_button.setEnabled(False)

    def calibrate_frames(self):
        self.calibrator.compute_calibration()
        self.update_calib_info()
        self.preview_toggle_btn.setEnabled(True)
        self.update_preview()



    def update_preview(self):
        self.video_viewer.reset_maps()
        if self.preview_toggle_btn.isChecked():
            img_dim = (int(self.video_viewer.frame_width), int(self.video_viewer.frame_height))
            map1, map2 = self.calibrator.get_maps(fov_scale=self.fov_scale, new_img_dim=img_dim)
            self.video_viewer.add_maps(map1, map2)
            self.video_viewer.enable_stretch(False)

            #map1, map2 = self.calibrator.get_rotation_map()
            #self.video_viewer.add_maps(map1, map2)

        else:
            self.video_viewer.enable_stretch(True)

        self.video_viewer.update_frame()





class StretchUtility(QtWidgets.QMainWindow):
    def __init__(self):
        """Qt window containing utility for nonlinear stretch
        """
        super().__init__()

        # Initialize UI
        self.setWindowTitle("Gyroflow Stretcher {}".format(__version__))
        self.setWindowIcon(QtGui.QIcon(':/media/icon.png'))

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

        # output size choice
        self.out_size_text = QtWidgets.QLabel("Output size: ")
        self.stretch_controls_layout.addWidget(self.out_size_text)

        self.out_width_control = QtWidgets.QSpinBox(self)
        self.out_width_control.setMinimum(16)
        self.out_width_control.setMaximum(7680) # 8K max is probably fine
        self.out_width_control.setValue(1920)
        self.out_width_control.valueChanged.connect(self.update_out_size)

        self.stretch_controls_layout.addWidget(self.out_width_control)

        # output size choice
        self.out_height_control = QtWidgets.QSpinBox(self)
        self.out_height_control.setMinimum(9)
        self.out_height_control.setMaximum(4320)
        self.out_height_control.setValue(1080)
        self.out_height_control.valueChanged.connect(self.update_out_size)

        self.stretch_controls_layout.addWidget(self.out_height_control)

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
        menubar = self.menuBar()
        filemenu = menubar.addMenu('&File')

        # https://joekuan.wordpress.com/2015/09/23/list-of-qt-icons/
        icon = self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon)

        self.open_file = QtWidgets.QAction(icon, 'Open file', self)
        self.open_file.setShortcut("Ctrl+O")
        self.open_file.triggered.connect(self.open_file_func)
        filemenu.addAction(self.open_file)


        self.statusBar()

        self.infile_path = ""


        self.show()

        self.main_widget.show()

        # non linear setup
        self.nonlin = nonlinear_stretch.NonlinearStretch((1280,720), (1920,1080))

    def open_file_func(self):
        """Open file using Qt filedialog
        """
        path = QtWidgets.QFileDialog.getOpenFileName(self, "Open video file", filter="Video (*.mp4 *.avi *.mov *.MP4 *.AVI *.MOV)")
        self.infile_path = path[0]
        self.video_viewer.set_video_path(path[0])



        # recompute non linear stretch maps
        self.recompute_stretch()

        self.video_viewer.next_frame()


    def closeEvent(self, event):
        print("Closing now")
        self.video_viewer.destroy_thread()
        event.accept()

    def safe_area_changed(self):
        """Nonlinear stretch safe area has changed
        """
        safe_area_val = self.safe_area_slider.value()
        self.nonlin.set_safe_area(safe_area_val/100)
        self.safe_area_text.setText("Safe area ({}%): ".format(safe_area_val))
        self.safe_check_change()

    def stretch_expo_changed(self):
        """Nonlinear expo has changed
        """
        stretch_expo_val = self.expo_slider.value() / 10
        self.nonlin.set_expo(stretch_expo_val)
        self.expo_text.setText("Stretch expo ({}):".format(stretch_expo_val))

    def update_out_size(self):
        """Update image size used for nonlinear stretch computation
        """
        #print(self.out_width_control.value())
        self.nonlin.set_out_size((self.out_width_control.value(), self.out_height_control.value()))

    def recompute_stretch(self):
        """Update nonlinear stretch maps
        """
        self.nonlin.set_in_size((self.video_viewer.frame_width, self.video_viewer.frame_height))
        self.nonlin.recompute_maps()
        self.video_viewer.reset_maps()
        self.video_viewer.add_maps(self.nonlin.map1, self.nonlin.map2)
        self.safe_check_change()


    def safe_check_change(self):
        """Handler for change of safe area checkbox
        """
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
        """Gives save location using filedialog
           and saves video to given location
        """
        # no input file opened
        if self.infile_path == "":
            self.show_error("No video file loaded")
            return

        self.video_viewer.stop()

        # get file

        filename = QtWidgets.QFileDialog.getSaveFileName(self, "Export video", filter="mp4 (*.mp4);; Quicktime (*.mov)")
        print(filename[0])

        if len(filename[0]) == 0:
            self.show_error("No output file given")
            return
        self.nonlin.stretch_save_video(self.infile_path, filename[0])

    def show_error(self, msg):
        QtWidgets.QMessageBox.critical(self, "Something's gone awry", msg)

        return
        err_window = QtWidgets.QMessageBox(self)
        err_window.setIcon(QtWidgets.QMessageBox.Critical)
        err_window.setText(msg)
        err_window.setWindowTitle("Something's gone awry")
        err_window.show()


class StabUtilityBarebone(QtWidgets.QMainWindow):
    def __init__(self, with_UI = True):
        """Qt window containing barebone utility for stabilization. (No video viewer)
        """
        super().__init__()
        if with_UI:
            # Initialize UI
            self.setWindowTitle("Gyroflow Stabilizer Barebone {}".format(__version__))
            self.setWindowIcon(QtGui.QIcon(':/media/icon.png'))

            self.main_setting_widget = QtWidgets.QTabWidget()
            self.layout = QtWidgets.QHBoxLayout()
            self.main_setting_widget.setLayout(self.layout)
            self.main_setting_widget.setStyleSheet("font-size: 12px")
            self.setCentralWidget(self.main_setting_widget)
        
            self.init_UI()
            self.show()
            self.main_setting_widget.show()

    def init_UI(self):

        # input tab
        self.input_controls = QtWidgets.QWidget()
        self.input_controls_layout = QtWidgets.QVBoxLayout()
        self.input_controls.setLayout(self.input_controls_layout)
        self.input_controls_layout.setAlignment(QtCore.Qt.AlignTop)
        self.input_controls.setMinimumWidth(500)

        # sync tab
        self.sync_controls = QtWidgets.QWidget()
        self.sync_controls_layout = QtWidgets.QVBoxLayout()
        self.sync_controls.setLayout(self.sync_controls_layout)
        self.sync_controls_layout.setAlignment(QtCore.Qt.AlignTop)
        self.sync_controls.setMinimumWidth(500)

        # stab tab
        self.stab_controls = QtWidgets.QWidget()
        self.stab_controls_layout = QtWidgets.QVBoxLayout()
        self.stab_controls.setLayout(self.stab_controls_layout)
        self.stab_controls_layout.setAlignment(QtCore.Qt.AlignTop)
        self.stab_controls.setMinimumWidth(500)

        self.export_controls = QtWidgets.QWidget()
        self.export_controls_layout = QtWidgets.QVBoxLayout()
        self.export_controls.setLayout(self.export_controls_layout)
        self.export_controls_layout.setAlignment(QtCore.Qt.AlignTop)
        self.export_controls.setMinimumWidth(500)

        #self.export_keyframes_controls = QtWidgets.QWidget()
        #self.export_keyframes_controls_layout = QtWidgets.QVBoxLayout()
        #self.export_keyframes_controls.setLayout(self.export_keyframes_controls_layout)
        #self.export_keyframes_controls_layout.setAlignment(QtCore.Qt.AlignTop)
        #self.export_keyframes_controls.setMinimumWidth(500)


        text = QtWidgets.QLabel("<h2>Input parameters:</h2>".format(__version__))
        text.setAlignment(QtCore.Qt.AlignCenter)
        self.input_controls_layout.addWidget(text)

        self.open_vid_button = QtWidgets.QPushButton("Open video file")
        self.open_vid_button.setMinimumHeight(40)
        self.open_vid_button.clicked.connect(self.open_video_func)
        self.input_controls_layout.addWidget(self.open_vid_button)

        # lens preset
        #self.input_controls_layout.addWidget(QtWidgets.QLabel("Search preset"))

        preset_full_paths = calibrate_video.get_all_preset_paths()
        self.preset_trunc_paths = [pathname.lstrip("camera_presets/") for pathname in preset_full_paths]
        #print(preset_trunc_paths)
        completer = QtWidgets.QCompleter(self.preset_trunc_paths)
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        completer.setFilterMode(QtCore.Qt.MatchContains)  

        self.preset_search_input = QtWidgets.QLineEdit()
        self.preset_search_input.setPlaceholderText("Search preset")
        self.preset_search_input.setCompleter(completer)
        self.preset_search_input.textChanged.connect(self.preset_search_handler)
        self.input_controls_layout.addWidget(self.preset_search_input)
        

        self.open_preset_button = QtWidgets.QPushButton("Browse lens preset")
        self.open_preset_button.setMinimumHeight(40)
        self.open_preset_button.clicked.connect(self.open_preset_func)
        self.input_controls_layout.addWidget(self.open_preset_button)


        self.open_gyro_button = QtWidgets.QPushButton("Open Gyro log")
        self.open_gyro_button.setMinimumHeight(40)
        self.open_gyro_button.clicked.connect(self.open_gyro_func)
        self.input_controls_layout.addWidget(self.open_gyro_button)

        self.video_as_log_button = QtWidgets.QPushButton("Set input video as log (GoPro, Insta360)")
        self.video_as_log_button.setMinimumHeight(30)
        self.video_as_log_button.clicked.connect(self.video_as_log_func)
        self.input_controls_layout.addWidget(self.video_as_log_button)


        explaintext = QtWidgets.QLabel("<b>Note:</b> BBL and CSV files in video folder with identical names are detected automatically.<br>If BBL doesn't work, use exported CSV file from blackbox explorer")
        explaintext.setWordWrap(True)
        explaintext.setMinimumHeight(60)
        self.input_controls_layout.addWidget(explaintext)

        self.input_controls_layout.addWidget(QtWidgets.QLabel('Rotate video:'))
        self.input_video_rotate_select = QtWidgets.QComboBox()
        self.input_video_rotate_select.addItem("None", -1)
        self.input_video_rotate_select.addItem("90° Clockwise", cv2.ROTATE_90_CLOCKWISE) # 0
        self.input_video_rotate_select.addItem("90° Counterclockwise", cv2.ROTATE_90_COUNTERCLOCKWISE) # 2
        self.input_video_rotate_select.addItem("180°", cv2.ROTATE_180) # 1
        self.input_video_rotate_select.currentIndexChanged.connect(self.reset_stab)
        self.input_controls_layout.addWidget(self.input_video_rotate_select)
        
        data =  [(name, name) for name in gyrolog.get_log_reader_names()]

        #data = [("rawblackbox", "Raw Betaflight Blackbox"), ("csvblackbox", "Betaflight Blackbox CSV"), ("csvgyroflow", "Gyroflow CSV log (in dev)"), ("csvruncam", "RC CSV log (in dev)"), ("csvgocam", "RC GOCAM CSV log (in dev)"), ("gpmf", "GoPro metadata"), ("insta360", "Insta360 metadata")]

        self.gyro_log_format_text = QtWidgets.QLabel("Gyro log type:")
        self.gyro_log_format_select = QtWidgets.QComboBox()

        #self.gyro_log_model = QtGui.QStandardItemModel()
        for i, text in data:
            #itm = QtGui.QStandardItem(text)
            #itm.setData(i)
            #self.gyro_log_model.appendRow(itm)
            self.gyro_log_format_select.addItem(text, i)

        self.gyro_log_format_select.setMinimumHeight(20)

        self.gyro_log_format_text.setVisible(False)
        self.gyro_log_format_select.setVisible(False)
        self.gyro_log_format_select.currentIndexChanged.connect(self.update_gyro_input_settings_dropdown)

        self.input_controls_layout.addWidget(self.gyro_log_format_text)
        self.input_controls_layout.addWidget(self.gyro_log_format_select)


        self.gyro_variant_text = QtWidgets.QLabel('Gyro source variant')
        self.input_controls_layout.addWidget(self.gyro_variant_text)

        self.gyro_variant_control = QtWidgets.QComboBox()
        self.gyro_variant_control.setToolTip('Different setups/models can have different gyro orientations. Make sure to select the right one!')
        self.gyro_variant_control.currentIndexChanged.connect(self.reset_stab)
        self.gyro_variant_control.clear()
        
        self.log_reader = gyrolog.GyroflowGyroLog()
        #self.gyro_variant_control.addItem("hero5")
        #self.gyro_variant_control.addItem("hero6")
        #self.gyro_variant_control.addItem("hero7")
        #self.gyro_variant_control.addItem("hero8")
        #self.gyro_variant_control.addItem("hero9")

        #self.gyro_variant_control.addItems(gyrolog.GyroflowGyroLog().get_variants())

        self.input_controls_layout.addWidget(self.gyro_variant_control)


        self.fpv_tilt_text = QtWidgets.QLabel("Camera to gyro angle:")
        
        self.fpv_tilt_control = QtWidgets.QDoubleSpinBox(self)
        self.fpv_tilt_control.setToolTip('FPV drones typically have the HD camera tilted from the flight controller. Positive=upwards')
        self.fpv_tilt_control.setMinimum(-90)
        self.fpv_tilt_control.setMaximum(180)
        self.fpv_tilt_control.setValue(0)
        self.fpv_tilt_control.valueChanged.connect(self.reset_stab)


        # Only show when blackbox file is loaded
        #self.fpv_tilt_text.setVisible(False) # Always show, for flexibility
        #self.fpv_tilt_control.setVisible(False)

        self.input_controls_layout.addWidget(self.fpv_tilt_text)
        self.input_controls_layout.addWidget(self.fpv_tilt_control)

        #self.gyro_log_use_raw_data_text = QtWidgets.QLabel("Use raw gyro data (debug_mode = GYRO_SCALED):")
        #self.gyro_log_use_raw_data_control = QtWidgets.QCheckBox(self)
        #self.gyro_log_use_raw_data_control.setChecked(False)

        #self.gyro_log_use_raw_data_text.setVisible(False)
        #self.gyro_log_use_raw_data_control.setVisible(False)

        #self.input_controls_layout.addWidget(self.gyro_log_use_raw_data_text)
        #self.input_controls_layout.addWidget(self.gyro_log_use_raw_data_control)


        self.input_controls_layout.addWidget(QtWidgets.QLabel('Input low-pass filter cutoff (Hz). Set to -1 to disable'))
        self.input_lpf_control = QtWidgets.QSpinBox(self)
        self.input_lpf_control.setMinimum(-1)
        self.input_lpf_control.setMaximum(1000)
        self.input_lpf_control.setValue(-1)
        self.input_lpf_control.setToolTip('Enable the filter if excessive noise is seen in the gyro data. Frequencies below the cutoff are kept, so lower value means more filtering. 50 Hz is a good starting point.')
        self.input_lpf_control.valueChanged.connect(self.reset_stab)
        self.input_controls_layout.addWidget(self.input_lpf_control)


        self.input_plot_button = QtWidgets.QPushButton("Plot and analyze gyro data")
        self.input_plot_button.setMinimumHeight(30)
        self.input_plot_button.setToolTip('Click here if you want to analyze the frequencies in the gyro data after applying the filter')
        self.input_plot_button.clicked.connect(self.input_gyro_plot)
        self.input_controls_layout.addWidget(self.input_plot_button)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.input_controls_layout.addWidget(line)

        #text = QtWidgets.QLabel("<h2>Video information:</h2>")
        #text.setAlignment(QtCore.Qt.AlignCenter)
        #self.input_controls_layout.addWidget(text)

        self.video_info_dict = {
            "fps": 0,
            "width": 0,
            "height": 0,
            "aspect": 0,
            "time": 0,
        }
        self.video_info_template = "<h2>Video information:</h2>" \
                                   "Framerate: {fps:.2f} fps <br>" \
                                   "Resolution:  {width}x{height} ({aspect:.3f}:1)<br>" \
                                   "Length: {time} s<br>" \
                                   "Bitrate: {bitrate} kbps"

        self.video_info_text = QtWidgets.QLabel()
        self.input_controls_layout.addWidget(self.video_info_text)

        #self.display_video_info()

        #text = QtWidgets.QLabel("<h2>Preset information:</h2>")
        #text.setAlignment(QtCore.Qt.AlignCenter)
        #self.input_controls_layout.addWidget(text)

        self.preset_info_template = "<h2>Preset information:</h2>" \
                                    "Preset name: {name}<br>" \
                                    "Calibrator version: {calibrator_version} <br>" \
                                    "Calibrated by {calibrated_by} on date {date}<br>" \
                                    "Camera: {camera_brand} {camera_model}<br>" \
                                    "Lens: {lens_model}<br>" \
                                    "Extra calibration note: {note}<br>" \
                                    "Resolution:  {width}x{height} ({aspect:.3f}:1)<br>" \
                                    "Number of calibration frames: {num_images}"

        self.preset_info_text = QtWidgets.QLabel()
        self.input_controls_layout.addWidget(self.preset_info_text)

        self.aspect_warning_text = QtWidgets.QLabel()
        self.aspect_warning_text.setStyleSheet("color: #ee3333")
        self.aspect_warning_text.setWordWrap(True)
        self.input_controls_layout.addWidget(self.aspect_warning_text)

        # SYNC AND STABILIZATION SETTINGS

        text = QtWidgets.QLabel("<h2>Sync and stabilization:</h2>")
        text.setAlignment(QtCore.Qt.AlignCenter)
        self.sync_controls_layout.addWidget(text)
        #self.input_controls_layout.addStretch()


        self.sync_controls_layout.addWidget(QtWidgets.QLabel("Initial rough gyro offset in seconds:"))

        self.offset_control = QtWidgets.QDoubleSpinBox(self)
        self.offset_control.setToolTip('Positive if gyro logging starts after video. Just has to be within a few seconds')
        self.offset_control.setMinimum(-1000)
        self.offset_control.setMaximum(1000)
        self.offset_control.setValue(0)

        self.sync_controls_layout.addWidget(self.offset_control)


        self.sync_controls_layout.addWidget(QtWidgets.QLabel("Sync search size (seconds)"))
        self.sync_search_size = QtWidgets.QDoubleSpinBox(self)
        self.sync_search_size.setMinimum(0)
        self.sync_search_size.setMaximum(60)
        self.sync_search_size.setValue(10)
        self.sync_controls_layout.addWidget(self.sync_search_size)

        self.sync_controls_layout.addWidget(QtWidgets.QLabel("Auto sync error margin (seconds):"))
        self.max_fitting_control = QtWidgets.QDoubleSpinBox(self)
        self.max_fitting_control.setMinimum(0)
        self.max_fitting_control.setMaximum(1)
        self.max_fitting_control.setValue(0.02)
        self.max_fitting_control.setToolTip('Auto sync works by attempting to fit a line through all syncs. This value determines how close the points can be to the line')

        self.sync_controls_layout.addWidget(self.max_fitting_control)

        self.sync_controls_layout.addWidget(QtWidgets.QLabel("Max analysis slices:"))
        self.max_sync_control = QtWidgets.QSpinBox(self)
        self.max_sync_control.setToolTip('More points take longer to analyze, but can give better results')
        self.max_sync_control.setMinimum(4)
        self.max_sync_control.setMaximum(20)
        self.max_sync_control.setValue(9)

        self.sync_controls_layout.addWidget(self.max_sync_control)

        self.sync_controls_layout.addWidget(QtWidgets.QLabel("Analyze every Nth frame (increase for high fps footage)"))
        self.num_frames_skipped_control = QtWidgets.QSpinBox(self)
        self.num_frames_skipped_control.setMinimum(1)
        self.num_frames_skipped_control.setMaximum(50)
        self.num_frames_skipped_control.setValue(1)
        self.num_frames_skipped_control.setToolTip("High fps means less movement between frames. This option allows for faster and more reliable sync of high speed footage. A value of 5 works fine for 300 fps")

        self.sync_controls_layout.addWidget(self.num_frames_skipped_control)

        # button for (re)computing sync
        self.auto_sync_button = QtWidgets.QPushButton("Attempt auto sync")
        self.auto_sync_button.setMinimumHeight(30)
        self.auto_sync_button.clicked.connect(self.auto_sync_start)
        self.sync_controls_layout.addWidget(self.auto_sync_button)


        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.sync_controls_layout.addWidget(line)


        #self.fpv_tilt_text = QtWidgets.QLabel("")
        #self.fpv_tilt_control = QtWidgets.QDoubleSpinBox(self)
        #self.fpv_tilt_control.setMinimum(-90)
        #self.fpv_tilt_control.setMaximum(90)
        #self.fpv_tilt_control.setValue(0)


        #self.sync_controls_layout.addWidget(QtWidgets.QLabel("Auto sync timestamp 2"))
        #self.sync2_control = QtWidgets.QDoubleSpinBox(self)
        #self.sync2_control.setMinimum(0)
        #self.sync2_control.setMaximum(10000)
        #self.sync2_control.setValue(30)
        #self.sync_controls_layout.addWidget(self.sync2_control)

        # How many frames to analyze using optical flow each slice
        self.sync_controls_layout.addWidget(QtWidgets.QLabel("Number of frames to analyze per sync:"))
        self.OF_frames_control = QtWidgets.QSpinBox(self)
        self.OF_frames_control.setToolTip("This number of frames is analyzed for each manual sync. More frames take longer, but can give a better sync. 60 is usually more than fine")
        self.OF_frames_control.setMinimum(10)
        self.OF_frames_control.setMaximum(600)
        self.OF_frames_control.setValue(60)

        self.sync_controls_layout.addWidget(self.OF_frames_control)
 

        # slider for adjusting smoothness. 0 = no stabilization. 100 = locked. Scaling is a bit weird still and depends on gyro sample rate.
        #self.smooth_max_period = 30 # seconds
        #self.smooth_text_template = "Smoothness (time constant: {:.3f} s, {}%):"
        #self.smooth_text = QtWidgets.QLabel(self.smooth_text_template.format((20/100)**3 * self.smooth_max_period  ,20))
        #self.smooth_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        #self.smooth_slider.setMinimum(0)
        #self.smooth_slider.setValue(20)
        #self.smooth_slider.setMaximum(100)
        #self.smooth_slider.setSingleStep(1)
        #self.smooth_slider.setTickInterval(1)
        #self.smooth_slider.valueChanged.connect(self.smooth_changed)

        #self.sync_controls_layout.addWidget(self.smooth_text)
        #self.sync_controls_layout.addWidget(self.smooth_slider)

        #explaintext = QtWidgets.QLabel("<b>Note:</b> 0% corresponds to no smoothing and 100% corresponds to a locked camera. " \
        #"intermediate values are non-linear and depend on gyro sample rate in current implementation.")
        #explaintext.setWordWrap(True)
        #explaintext.setMinimumHeight(60)
        #self.sync_controls_layout.addWidget(explaintext)


        # slider for adjusting non linear crop
        #self.fov_text = QtWidgets.QLabel("FOV scale (1.5):")
        #self.fov_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        #self.fov_slider.setMinimum(10)
        #self.fov_slider.setValue(15)
        #self.fov_slider.setMaximum(40)
        #self.fov_slider.setSingleStep(1)
        #self.fov_slider.setTickInterval(1)
        #self.fov_slider.valueChanged.connect(self.fov_scale_changed)
        #self.fov_text.setVisible(False)
        #self.fov_slider.setVisible(False)


        #self.sync_controls_layout.addWidget(self.fov_text)
        #self.sync_controls_layout.addWidget(self.fov_slider)


        self.sync_controls_layout.addWidget(QtWidgets.QLabel("Sync timestamp (video time in seconds)"))
        self.sync1_control = QtWidgets.QDoubleSpinBox(self)
        self.sync1_control.setMinimum(0)
        self.sync1_control.setMaximum(10000)
        self.sync1_control.setValue(5)
        self.sync_controls_layout.addWidget(self.sync1_control)

        # button for (re)computing sync
        self.add_sync_button_barebone = QtWidgets.QPushButton("Add sync at specified timestamp")
        self.add_sync_button_barebone.setMinimumHeight(30)
        self.add_sync_button_barebone.clicked.connect(self.add_sync_barebone_handler)
        self.sync_controls_layout.addWidget(self.add_sync_button_barebone)



        self.multiSyncUI = sync_ui.MultiSyncUI(None)
        self.sync_controls_layout.addWidget(self.multiSyncUI)

        #self.multiSyncUI.update_from_stab_data()

        self.sync_debug_select = QtWidgets.QCheckBox("Display sync plots")
        self.sync_debug_select.setChecked(False)
        self.sync_controls_layout.addWidget(self.sync_debug_select)

        # button for (re)computing sync
        self.recompute_stab_button = QtWidgets.QPushButton("Apply settings and compute sync")
        self.recompute_stab_button.setMinimumHeight(30)
        self.recompute_stab_button.clicked.connect(self.recompute_stab)
        self.sync_controls_layout.addWidget(self.recompute_stab_button)

        #explaintext = QtWidgets.QLabel("<b>Note:</b> Check console for info after clicking. A number of plots will appear during the" \
        #                                " process showing the difference between gyro and optical flow. Just close these after you're done looking at them.")
        #explaintext.setWordWrap(True)
        #explaintext.setMinimumHeight(60)
        #self.sync_controls_layout.addWidget(explaintext)

        #self.sync_controls_layout.addWidget(QtWidgets.QLabel("Delay for sync 1"))
        #self.d1_control = QtWidgets.QDoubleSpinBox(self)
        #self.d1_control.setDecimals(5)
        #self.d1_control.setMinimum(-1000)
        #self.d1_control.setMaximum(1000)
        #self.d1_control.setValue(0)
        #self.d1_control.setSingleStep(0.01)
        #self.sync_controls_layout.addWidget(self.d1_control)

        #self.sync_controls_layout.addWidget(QtWidgets.QLabel("Delay for sync 2"))
        #self.d2_control = QtWidgets.QDoubleSpinBox(self)
        #self.d2_control.setDecimals(5)
        #self.d2_control.setMinimum(-1000)
        #self.d2_control.setMaximum(1000)
        #self.d2_control.setValue(0)
        #self.d2_control.setSingleStep(0.01)
        #self.sync_controls_layout.addWidget(self.d2_control)


        #self.sync_correction_button = QtWidgets.QPushButton("Sync correction/update smoothness")
        #self.sync_correction_button.setMinimumHeight(30)
        #self.sync_correction_button.setEnabled(False)
        #self.sync_correction_button.clicked.connect(self.correct_sync)
        #self.sync_controls_layout.addWidget(self.sync_correction_button)


        # Select method for doing low-pass filtering
        self.stab_controls_layout.addWidget(QtWidgets.QLabel("Smoothing method"))
        self.stabilization_algo_select = QtWidgets.QComboBox()

        self.stab_algo_names = smoothing_algos.get_stab_algo_names()
        self.stab_algo_instances = smoothing_algos.get_all_stab_algo_instances()
        self.stab_algo_instance_current = self.stab_algo_instances[0]
        for name in self.stab_algo_names:
            self.stabilization_algo_select.addItem(name)
            
        self.stabilization_algo_select.currentIndexChanged.connect(self.stab_algo_change)

        self.stab_controls_layout.addWidget(self.stabilization_algo_select)

        self.stab_algo_settings_widget = QtWidgets.QWidget()
        self.stab_algo_settings_widget_layout = QtWidgets.QVBoxLayout()
        self.stab_algo_settings_widget.setLayout(self.stab_algo_settings_widget_layout)
        self.stab_algo_settings_widget_layout.setAlignment(QtCore.Qt.AlignTop)
        self.stab_controls_layout.addWidget(self.stab_algo_settings_widget)

        for inst in self.stab_algo_instances:
            self.stab_algo_settings_widget_layout.addWidget(inst.get_ui_widget())
            inst.get_ui_widget().setVisible(False)

        self.stab_algo_change()

        self.update_smoothness_button = QtWidgets.QPushButton("Apply smoothness settings")
        self.update_smoothness_button.setMinimumHeight(30)
        self.update_smoothness_button.setEnabled(False)
        self.update_smoothness_button.clicked.connect(self.update_smoothness)
        self.stab_controls_layout.addWidget(self.update_smoothness_button)

        # OUTPUT OPTIONS

        text = QtWidgets.QLabel("<h2>Output parameters:</h2>".format(__version__))
        text.setAlignment(QtCore.Qt.AlignCenter)
        self.export_controls_layout.addWidget(text)

        # output size choice presets
        names = ["4K", "2.7K", "1440p", "1080p", "720p", "4K 2.39:1"]
        self.resolutions = [(3840, 2160), (2704, 1520), (2560, 1440), (1920, 1080), (1280, 720), (4096, 1716)]
        self.export_controls_layout.addWidget(QtWidgets.QLabel("Preset video resolutions:"))
        self.preset_resolution_combo = QtWidgets.QComboBox()
        self.preset_resolution_combo.addItem(f"Original")
        for name, res in zip(names, self.resolutions):
            self.preset_resolution_combo.addItem(f"{name} ({res[0]}x{res[1]}px)")
        self.preset_resolution_combo.currentIndexChanged.connect(self.preset_resolution_selected)

        self.export_controls_layout.addWidget(self.preset_resolution_combo)

        self.out_width_control = QtWidgets.QSpinBox(self)
        self.out_width_control.setMinimum(16)
        self.out_width_control.setMaximum(7680) # 8K max is probably fine
        self.out_width_control.setValue(1920)
        self.out_width_control.setToolTip("For action cameras, try exporting to 16:9 if recorded in 4:3")
        self.out_width_control.valueChanged.connect(self.update_out_size)


        # output size choice
        self.out_size_text = QtWidgets.QLabel("Output dimensions: ")
        self.export_controls_layout.addWidget(self.out_size_text)
        self.out_height_control = QtWidgets.QSpinBox(self)
        self.out_height_control.setMinimum(9)
        self.out_height_control.setMaximum(4320)
        self.out_height_control.setValue(1080)
        self.out_height_control.valueChanged.connect(self.update_out_size)

        self.export_controls_layout.addWidget(self.out_width_control)
        self.export_controls_layout.addWidget(self.out_height_control)

        #self.export_controls_layout.addWidget(QtWidgets.QLabel("Output upscale"))

        #self.out_scale_control = QtWidgets.QSpinBox(self)
        #self.out_scale_control.setMinimum(1)
        #self.out_scale_control.setMaximum(4)
        #self.out_scale_control.setValue(1)
        #self.export_controls_layout.addWidget(self.out_scale_control)


        #explaintext = QtWidgets.QLabel("<b>Note:</b> The current code uses two image remappings for lens correction " \
        #"and perspective transform, so output must be cropped separately to avoid black borders. These steps can be combined later. For now fov_scale = 1.5 with appropriate crop depending on resolution works.")
        #explaintext.setWordWrap(True)
        #explaintext.setMinimumHeight(60)
        #self.export_controls_layout.addWidget(explaintext)

        self.export_controls_layout.addWidget(QtWidgets.QLabel("Video export start and stop (seconds)"))
        self.export_starttime = QtWidgets.QDoubleSpinBox(self)
        self.export_starttime.setMinimum(0)
        self.export_starttime.setMaximum(10000)
        self.export_starttime.setValue(0)
        self.export_controls_layout.addWidget(self.export_starttime)


        self.export_stoptime = QtWidgets.QDoubleSpinBox(self)
        self.export_stoptime.setMinimum(0)
        self.export_stoptime.setMaximum(10000)
        self.export_stoptime.setValue(30)
        self.export_controls_layout.addWidget(self.export_stoptime)

        self.enableAdaptiveZoom = QtWidgets.QCheckBox("Adaptive zoom  (if disabled, use zoom to set desired FOV)")
        self.enableAdaptiveZoom.setChecked(True)
        self.enableAdaptiveZoom.clicked.connect(self.enableAdaptiveZoomClicked)
        self.export_controls_layout.addWidget(self.enableAdaptiveZoom)

        self.fov_smoothing_text = QtWidgets.QLabel("Smoothing Window FOV (sec): 4.0")
        self.export_controls_layout.addWidget(self.fov_smoothing_text)
        self.fov_smoothing = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.fov_smoothing.setToolTip("Determines how fast dynamic zoom acts. Maximum value means a constant zoom")
        self.fov_smoothing.setMinimum(0)
        self.fov_smoothing.setValue(40)
        self.fov_smoothing.setMaximum(150)
        self.fov_smoothing.setSingleStep(1)
        self.fov_smoothing.setTickInterval(1)
        self.fov_smoothing.valueChanged.connect(self.fov_smoothing_changed)
        self.export_controls_layout.addWidget(self.fov_smoothing)

        self.zoom_text = QtWidgets.QLabel("Zoom Factor (with adaptive zoom) or FOV scale (same as preview): 1.0")
        self.export_controls_layout.addWidget(self.zoom_text)
        self.zoom = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.zoom.setMinimum(5)
        self.zoom.setValue(10)
        self.zoom.setMaximum(70)
        self.zoom.setSingleStep(1)
        self.zoom.setTickInterval(1)
        self.zoom.valueChanged.connect(self.zoom_changed)
        self.export_controls_layout.addWidget(self.zoom)

        # Check for available encoders and grey out those who are not available
        self.available_encoders = self.get_available_encoders()
        # TODO: Use subprocess or lib to import these dynamically directly from FFmpeg. They dont really change much but would be more robust in terms
        # of different FFmpeg versions etc
        supported_encoders = {
            "libx264": ["baseline", "main", "high", "high10", "high422", "hight444"],
            "h264_nvenc": ["baseline", "main", "high", "high444p"],
            "h264_amf": ["main", "high", "constrained_baseline", "constrained_high"],
            "h264_vaapi": ["baseline", "main", "high"],
            "h264_videotoolbox": ["baseline", "main", "high", "extended"],
            "prores_ks": ["auto", "proxy", "lt", "standard", "hq", "4444", "4444xq"] #,
            #"v210": ["raw"] # todo
        }

        self.encoder_model = QtGui.QStandardItemModel()
        self.video_encoder_text = QtWidgets.QLabel('Video encoder')
        self.video_encoder_select = QtWidgets.QComboBox()
        self.video_encoder_select.setModel(self.encoder_model)

        self.encoder_profile_text = QtWidgets.QLabel('Encoder profile')
        self.encoder_profile_select = QtWidgets.QComboBox()
        self.encoder_profile_select.setModel(self.encoder_model)

        for encoder, profiles in supported_encoders.items():
            encoder_item = QtGui.QStandardItem(encoder)
            # Disable encoders not listed by ffmpeg -encoders
            if encoder not in self.available_encoders:
                encoder_item.setEnabled(False)
            self.encoder_model.appendRow(encoder_item)
            for profile in profiles:
                profile_item = QtGui.QStandardItem(profile)
                encoder_item.appendRow(profile_item)

        # Prevent a unsupported/disabled item to be default selection
        for i in range(0, self.video_encoder_select.count()):
            if self.encoder_model.item(i).isEnabled():
                self.video_encoder_select.setCurrentIndex(i)
                break

        self.video_encoder_select.currentIndexChanged.connect(self.update_profile_select)
        self.video_encoder_select.currentIndexChanged.connect(self.update_bitrate_visibility)
        self.video_encoder_select.currentIndexChanged.connect(self.update_container_selection)
        self.update_container_selection()
        self.update_profile_select()
        self.export_controls_layout.addWidget(self.video_encoder_text)
        self.export_controls_layout.addWidget(self.video_encoder_select)
        self.export_controls_layout.addWidget(self.encoder_profile_text)
        self.export_controls_layout.addWidget(self.encoder_profile_select)


        self.split_screen_select = QtWidgets.QCheckBox("Export split screen (temporarily disabled)")
        self.split_screen_select.setChecked(False)
        self.split_screen_select.setVisible(False)
        self.export_controls_layout.addWidget(self.split_screen_select)

        self.display_preview = QtWidgets.QCheckBox("Display preview during rendering")
        self.display_preview.setToolTip("Disabling this means a faster render")
        self.display_preview.setChecked(True)
        self.export_controls_layout.addWidget(self.display_preview)

        self.export_audio = QtWidgets.QCheckBox("Export audio")
        self.export_audio.setChecked(True)
        self.export_controls_layout.addWidget(self.export_audio)

        self.export_debug_text = QtWidgets.QCheckBox("Render with extra info (for debugging)")
        self.export_debug_text.setChecked(False)
        self.export_controls_layout.addWidget(self.export_debug_text)

        # TODO: Should consider hiding this widget if prores is selected
        self.export_bitrate_text = QtWidgets.QLabel("Export bitrate [Mbit/s]")
        self.export_controls_layout.addWidget(self.export_bitrate_text)
        self.export_bitrate = QtWidgets.QDoubleSpinBox(self)
        self.export_bitrate.setDecimals(0)
        self.export_bitrate.setMinimum(1)
        self.export_bitrate.setMaximum(200)
        self.export_bitrate.setValue(20)
        self.export_bitrate.setVisible(True)
        self.export_controls_layout.addWidget(self.export_bitrate)
        self.update_bitrate_visibility()

        #self.export_controls_layout.addWidget(QtWidgets.QLabel("FFmpeg color space selection (Try 'yuv420p' if output doesn't play):"))
        self.pixfmt_select = QtWidgets.QLineEdit()
        #self.export_controls_layout.addWidget(self.pixfmt_select) # Shouldn't be required

        
        bg_description = QtWidgets.QLabel("Background color. #HexCode, CSS color name, REPLICATE (Extend edge), HISTORY (Keep previous frame):")
        bg_description.setWordWrap(True)
        self.export_controls_layout.addWidget(bg_description)
        colornames = list(colors.cnames.keys())
        completer = QtWidgets.QCompleter(colornames + ["REPLICATE","HISTORY"])
        self.bg_color_select = QtWidgets.QLineEdit()
        self.bg_color_select.setCompleter(completer)
        self.bg_color_select.setPlaceholderText(random.choice(colornames))
        self.bg_color_select.setText("REPLICATE")
        self.export_controls_layout.addWidget(self.bg_color_select)

        example_ffmpeg_pipeline = '{"-vcodec": "prores_ks","-profile:v": "hq"}'
        self.export_controls_layout.addWidget(QtWidgets.QLabel("FFmpeg custom pipeline, overwrites all settings above."))
        self.custom_ffmpeg_pipeline = QtWidgets.QLineEdit()
        self.custom_ffmpeg_pipeline.setPlaceholderText(example_ffmpeg_pipeline)
        self.export_controls_layout.addWidget(self.custom_ffmpeg_pipeline)


        self.hyperlapse_text = QtWidgets.QLabel("Hyperlapse speed multipler")
        self.export_controls_layout.addWidget(self.hyperlapse_text)
        self.hyperlapse_multiplier = QtWidgets.QSpinBox(self)
        self.hyperlapse_multiplier.setToolTip("What it says on the tin. How much to speed up the video by")
        self.hyperlapse_multiplier.setMinimum(1)
        self.hyperlapse_multiplier.setMaximum(128)
        self.hyperlapse_multiplier.setValue(1)
        self.hyperlapse_multiplier.setVisible(True)
        self.export_controls_layout.addWidget(self.hyperlapse_multiplier)

        self.hyperlapse_blend_text = QtWidgets.QLabel("Hyperlapse number of blended frames")
        self.export_controls_layout.addWidget(self.hyperlapse_blend_text)
        self.hyperlapse_blend = QtWidgets.QSpinBox(self)
        self.hyperlapse_blend.setToolTip("Fakes motion blur. Set to half of the speed multiplier for a simulated 180 deg shutter")
        self.hyperlapse_blend.setMinimum(1)
        self.hyperlapse_blend.setMaximum(128)
        self.hyperlapse_blend.setValue(1)
        self.hyperlapse_blend.setVisible(True)
        self.export_controls_layout.addWidget(self.hyperlapse_blend)

        # button for exporting video
        self.export_button = QtWidgets.QPushButton("Export (hopefully) stabilized video")
        self.export_button.setMinimumHeight(30)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_video)
        self.export_controls_layout.addWidget(self.export_button)

        self.export_gyroflow_button = QtWidgets.QPushButton("Export Gyroflow Data")
        self.export_gyroflow_button.setToolTip("This file contains the synced motion data, and can be loaded by Gyroflow again")
        self.export_gyroflow_button.setMinimumHeight(30)
        self.export_gyroflow_button.setEnabled(True)
        self.export_gyroflow_button.clicked.connect(self.export_gyroflow)
        self.export_controls_layout.addWidget(self.export_gyroflow_button)

        self.export_keyframes_button = QtWidgets.QPushButton("Export (hopefully) stabilized keyframes for the whole clip")
        self.export_keyframes_button.setMinimumHeight(30)
        self.export_keyframes_button.setEnabled(False)
        self.export_keyframes_button.clicked.connect(self.export_keyframes)
        
        self.export_controls_layout.addWidget(self.export_keyframes_button)

        # warning for HW encoding
        render_description = QtWidgets.QLabel(
        "<b>Note:</b> videotoolbox, vaapi and nvenc are HW accelerated encoders and require FFmpeg with hardware acceleration support!")
        render_description.setWordWrap(True)
        self.export_controls_layout.addWidget(render_description)


        # add control bar to main layout
        #self.layout.addWidget(self.input_controls)
        #self.layout.addWidget(self.sync_controls)

        self.input_controls_scroll = QtWidgets.QScrollArea()
        self.input_controls_scroll_layout = QtWidgets.QVBoxLayout(self.input_controls_scroll.widget())
        self.input_controls_scroll.setLayout(self.input_controls_scroll_layout)
        self.input_controls_scroll.setWidget(self.input_controls)
        self.input_controls_scroll.setWidgetResizable(True)

        self.sync_controls_scroll = QtWidgets.QScrollArea()
        self.sync_controls_scroll_layout = QtWidgets.QVBoxLayout(self.sync_controls_scroll.widget())
        self.sync_controls_scroll.setLayout(self.sync_controls_scroll_layout)
        self.sync_controls_scroll.setWidget(self.sync_controls)
        self.sync_controls_scroll.setWidgetResizable(True)

        self.stab_controls_scroll = QtWidgets.QScrollArea()
        self.stab_controls_scroll_layout = QtWidgets.QVBoxLayout(self.stab_controls_scroll.widget())
        self.stab_controls_scroll.setLayout(self.stab_controls_scroll_layout)
        self.stab_controls_scroll.setWidget(self.stab_controls)
        self.stab_controls_scroll.setWidgetResizable(True)

        self.export_controls_scroll = QtWidgets.QScrollArea()
        self.export_controls_scroll_layout = QtWidgets.QVBoxLayout(self.export_controls_scroll.widget())
        self.export_controls_scroll.setLayout(self.export_controls_scroll_layout)
        self.export_controls_scroll.setWidget(self.export_controls)
        self.export_controls_scroll.setWidgetResizable(True)

        self.main_setting_widget.addTab(self.input_controls_scroll, "Input")
        self.main_setting_widget.addTab(self.sync_controls_scroll, "Sync")
        self.main_setting_widget.addTab(self.stab_controls_scroll, "Stabilization")
        self.main_setting_widget.addTab(self.export_controls_scroll, "Export")

        self.infile_path = ""
        self.preset_path = ""
        self.gyro_log_path = ""
        self.gyro_log_type = ""
        self.stab = None
        self.analyzed = False

        self.use_gyroflow_data_file = False

        

        self.has_player = False

        self.preview_fov_scale = 1.4

        self.update_gyro_input_settings()

    def stab_algo_change(self):
        idx = self.stabilization_algo_select.currentIndex()
        self.stab_algo_instance_current.get_ui_widget().setVisible(False)
        self.stab_algo_instance_current = self.stab_algo_instances[idx]
        self.stab_algo_instance_current.get_ui_widget().setVisible(True)
        

    def open_video_func(self):
        """Open file using Qt filedialog
        """
        #path = QtWidgets.QFileDialog.getOpenFileName(self, "Open video file")
        if self.has_player:
            self.video_viewer.stop()


        dialog = QtWidgets.QFileDialog()
        dialog.setMimeTypeFilters(["video/mp4", "video/x-msvideo", "video/quicktime", "application/octet-stream"])
        dialog.exec_()
        path = dialog.selectedFiles()
        if (len(path) == 0 or len(path[0]) == 0):
            print("No file selected")
            return False
        self.infile_path = path[0]
        self.open_vid_button.setText("Video file: {}".format(self.infile_path.split("/")[-1]))
        self.open_vid_button.setStyleSheet("font-weight:bold;")

        # Extract information about the clip

        cap = cv2.VideoCapture(self.infile_path)
        self.video_info_dict["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_info_dict["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_info_dict["fps"] = cap.get(cv2.CAP_PROP_FPS)
        self.video_info_dict["time"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.video_info_dict["fps"])

        self.video_info_dict["aspect"] = 0 if self.video_info_dict["height"] == 0 else self.video_info_dict["width"]/self.video_info_dict["height"]
        try:
            self.video_info_dict["bitrate"] = int(cap.get(cv2.CAP_PROP_BITRATE))
        except:
            # estimate based on filesize
            size = os.path.getsize(self.infile_path)
            self.video_info_dict["bitrate"] = (size / self.video_info_dict.get("time", 1)) / 1000 # to kbps
        cap.release()

        self.display_video_info()

        # Reset the stabilizer
        self.reset_stab()

        #no_suffix = os.path.splitext(self.infile_path)[0]
        # check if gyroflow data file exists
        gyroflow_data_path = stabilizer.find_gyroflow_data_file(self.infile_path)
        if gyroflow_data_path:
            
                reply = QtWidgets.QMessageBox.question(self, 'Message', 
                "Load the associated .gyroflow data file?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
                if reply == QtWidgets.QMessageBox.Yes:
                    self.use_gyroflow_data_file = True
                    self.gyro_log_path = gyroflow_data_path
                    self.preset_path = gyroflow_data_path
                    try:
                        self.stab = stabilizer.Stabilizer(self.infile_path,gyroflow_file=self.gyro_log_path)
                        self.update_gyro_input_settings()
                        self.display_preset_info()
                        self.recompute_stab()
                        if self.has_player:
                            self.update_player_maps()
                        return True

                    except RuntimeError as e:
                        print(e)



        self.use_gyroflow_data_file = False

        if self.gyro_log_path.endswith(".gyroflow"):
            self.gyro_log_path = ""
        if self.preset_path.endswith(".gyroflow"):
            self.preset_path = ""
            self.display_preset_info()

        log_guess, log_type, variant = gyrolog.guess_log_type_from_video(self.infile_path)

        if log_guess:
            self.gyro_log_path = log_guess
            self.gyro_log_type = log_type
            print("Automatically detected gyro log file: {}".format(self.gyro_log_path.split("/")[-1]))

            self.log_reader = gyrolog.get_log_reader_by_name(self.gyro_log_type)
            self.log_reader.set_variant(variant)

        # check gyro logs by priority
        #log_suffixes = [".bbl.csv", ".bfl.csv", ".csv", ".bbl"]
        #for suffix in log_suffixes:
        #    if os.path.isfile(no_suffix + suffix):
        #        self.gyro_log_path = no_suffix + suffix
        #        print("Automatically detected gyro log file: {}".format(self.gyro_log_path.split("/")[-1]))
        #        break

        self.update_gyro_input_settings()
        return True

    def video_as_log_func(self):
        self.gyro_log_path = self.infile_path
        # check if Insta360
        if insta360_util.isInsta360Video(self.infile_path):
            self.gyro_log_format_select.setCurrentIndex(self.gyro_log_format_select.findData("Insta360 IMU metadata"))
            #self.camera_type_control.setCurrentText('smo4k')
            self.input_lpf_control.setValue(40)

        else: # Probably gopro
            self.gyro_log_format_select.setCurrentIndex(self.gyro_log_format_select.findData("GoPro GPMF metadata"))
            #self.camera_type_control.setCurrentText('hero8')
            self.input_lpf_control.setValue(-1)

        # Reset the stabilizer
        self.reset_stab()

        self.update_gyro_input_settings()

    def preset_search_handler(self):
        selected_text = self.preset_search_input.text()
        if selected_text in self.preset_trunc_paths:
            preset_path = "camera_presets/" + selected_text
            self.preset_path = preset_path
            self.reset_stab()
            self.display_preset_info()


    def display_video_info(self):


        info = self.video_info_template.format(fps = self.video_info_dict["fps"],
                                               width = self.video_info_dict["width"],
                                               height = self.video_info_dict["height"],
                                               time = self.video_info_dict["time"],
                                               aspect = self.video_info_dict["aspect"],
                                               bitrate = self.video_info_dict["bitrate"])

        self.video_info_text.setText(info)

        # set default sync and export options
        self.out_width_control.setValue(self.video_info_dict["width"])
        self.out_height_control.setValue(self.video_info_dict["height"])
        self.export_starttime.setValue(0)
        self.export_stoptime.setValue(int(self.video_info_dict["time"])) # round down
        #self.sync1_control.setValue(5)
        #self.sync2_control.setValue(int(self.video_info_dict["time"] - 5)) # 5 seconds before end
        self.export_bitrate.setValue(max(int(self.video_info_dict["bitrate"]) / 1000, 20))
        

        self.check_aspect()

    def open_preset_func(self):
        path = QtWidgets.QFileDialog.getOpenFileName(self, "Open preset file", filter="JSON preset (*.json *.JSON)")

        if (len(path[0]) == 0):
            print("No file selected")
            return
        #print(path)
        self.preset_path = path[0]

        # Reset the stabilizer
        self.reset_stab()


        self.display_preset_info()



    def display_preset_info(self):
        if not self.preset_path:
            self.open_preset_button.setText("Browse lens preset")
            self.open_preset_button.setStyleSheet("font-weight:normal;")
            return
        self.preset_path = self.preset_path.replace("\\", "/")
        self.open_preset_button.setText("Preset file: {}".format(self.preset_path.split("/")[-1]))
        self.open_preset_button.setStyleSheet("font-weight:bold;")

        if type(self.stab) == type(None):
            self.use_gyroflow_data_file = False

        if self.use_gyroflow_data_file:
            self.preset_info_dict = self.stab.undistort.extra_cam_info
        else:    
            self.preset_info_dict = calibrate_video.FisheyeCalibrator().load_calibration_json(self.preset_path)
        #print(self.preset_info_dict)

        info = self.preset_info_template.format(**self.preset_info_dict)
        self.preset_info_text.setText(info)

        self.check_aspect()

    def check_aspect(self):
        if self.preset_path and self.infile_path:
            # Check if the aspect ratios match
            self.preset_info_dict.get("aspect")
            v_aspect = self.video_info_dict.get("aspect")
            p_aspect = self.preset_info_dict.get("aspect")
            if abs(v_aspect - p_aspect) > 0.01:
                self.aspect_warning_text.setText(f"<h3>Seems like the aspect ratios don't quite match. Video: {v_aspect:.3f}, preset: {p_aspect:.3f}</h3>")
            else:
                self.aspect_warning_text.setText("")


    def update_gyro_input_settings(self):
        # display/hide relevant gyro log settings

        self.gyro_log_format_text.setVisible(True)
        self.gyro_log_format_select.setVisible(True)

        selected_log_type = self.gyro_log_format_select.currentData()
        # Treat everything as "external"
        #external =  selected_log_type in ["rawblackbox", "csvblackbox", "csvgyroflow", "csvruncam", "csvgocam"] # display more settings if external source is used

        videofile_selected = bool(self.infile_path)
        gyrofile_selected = bool(self.gyro_log_path)
        #internal = not external

        #self.fpv_tilt_text.setVisible(True)
        #self.fpv_tilt_control.setVisible(True)

        #self.camera_type_control.setVisible(True)
        #self.camera_type_text.setVisible(True)

        #self.gyro_log_use_raw_data_text.setVisible(selected_log_type == "csvblackbox")
        #self.gyro_log_use_raw_data_control.setVisible(selected_log_type == "csvblackbox")

        if gyrofile_selected:
            self.open_gyro_button.setText("Gyro data: {} (click to remove)".format(self.gyro_log_path.split("/")[-1]))
            self.open_gyro_button.setStyleSheet("font-weight:bold;")
        else:
            self.open_gyro_button.setText("Open Gyro log")
            self.open_gyro_button.setStyleSheet("font-weight: normal;")

        if self.log_reader and not self.use_gyroflow_data_file:
            self.gyro_log_format_select.setCurrentIndex(self.gyro_log_format_select.findData(self.log_reader.name))
            self.gyro_variant_control.clear()
            #print(self.log_reader.get_variants())
            self.gyro_variant_control.addItems(self.log_reader.get_variants())
            self.gyro_variant_control.setCurrentIndex(self.gyro_variant_control.findText(self.log_reader.variant))
            self.input_lpf_control.setValue(self.log_reader.default_filter)
            self.sync_search_size.setValue(self.log_reader.default_search_size)

    def update_gyro_input_settings_dropdown(self):
        selected_log_format = self.gyro_log_format_select.currentData()

        self.reset_stab()

        if self.log_reader:
            if self.log_reader.name != selected_log_format:
                self.log_reader = gyrolog.get_log_reader_by_name(selected_log_format)
                self.gyro_variant_control.clear()
                self.gyro_variant_control.addItems(self.log_reader.get_variants())
                self.gyro_variant_control.setCurrentIndex(self.gyro_variant_control.findText(self.log_reader.variant))


    def open_gyro_func(self):
        # Remove file if already added

        if self.gyro_log_path:
            self.gyro_log_path = ""
            self.update_gyro_input_settings()
            return


        path = QtWidgets.QFileDialog.getOpenFileName(self, "Open motion data file",
                                                     filter="Blackbox log (*.bbl *.bfl *.bbl.csv *.BBL .BFL *.BBL.CSV);; CSV file (*.csv *.CSV);; MP4 file (*.mp4 *.MP4);; All files (*)")

        if (len(path[0]) == 0):
            print("No file selected")
            return

        self.gyro_log_path = path[0]

        

        if self.gyro_log_path.endswith(".gyroflow"):
            self.use_gyroflow_data_file = True
        else:
            # Reset the stabilizer
            self.reset_stab()

            self.use_gyroflow_data_file = False

            log_guess, log_type, variant = gyrolog.guess_log_type_from_log(self.gyro_log_path)

            if log_guess:
                self.gyro_log_type = log_type
                print("Automatically detected gyro log file: {}".format(self.gyro_log_path.split("/")[-1]))

                self.log_reader = gyrolog.get_log_reader_by_name(self.gyro_log_type)
                self.log_reader.set_variant(variant)


        self.update_gyro_input_settings()


    def input_gyro_plot(self):
        if self.use_gyroflow_data_file:
            print("N/A for .gyroflow data file")
            return

        if not self.gyro_log_path:
            print("No input log file")
            return

        selected_log_type = self.gyro_log_format_select.currentData()
        logvariant = self.gyro_variant_control.currentText()

        temp_log_reader = gyrolog.get_log_reader_by_name(selected_log_type)
        temp_log_reader.set_variant(logvariant)
        temp_log_reader.set_pre_filter(self.input_lpf_control.value())
        success = temp_log_reader.extract_log(self.gyro_log_path)

        if not success:
            print("Log extraction failed")
            return

        temp_log_reader.plot_gyro(blocking=stabilizer.BLOCKING_PLOTS)

    def reset_stab(self):
        #print("Reset stabilization class")
        if type(self.stab) != type(None):
            self.multiSyncUI.UI_reset()
            self.stab.release()
            self.stab = None


    def closeEvent(self, event):
        print("Closing now")
        #self.video_viewer.destroy_thread()
        self.stab.release()
        event.accept()

    def smooth_changed(self):
        """Smoothness has changed
        """
        return
        #raw_val = self.smooth_slider.value()
        #smooth_val = (raw_val/100)**3 * self.smooth_max_period
        #self.smooth_text.setText(self.smooth_text_template.format(smooth_val, raw_val))

    def get_smoothness_timeconstant(self):
        """ Nonlinear smoothness slider
        """
        return
        #return (self.smooth_slider.value()/100)**3 * self.smooth_max_period

    def fov_scale_changed(self):
        """Undistort FOV scale changed
        """
        fov_val = self.fov_slider.value() / 10
        self.fov_text.setText("FOV scale ({}):".format(fov_val))

    def fov_smoothing_changed(self):
        
        if self.fov_smoothing.value() == self.fov_smoothing.maximum():
            val = "fixed"
        else:
            val = self.fov_smoothing.value() / 10
        self.fov_smoothing_text.setText("Smoothing Window FOV (sec): {}".format(val))

    def enableAdaptiveZoomClicked(self):
        if self.enableAdaptiveZoom.isChecked():
            self.fov_smoothing.setDisabled(False)
        else:
            self.fov_smoothing.setDisabled(True)

    def zoom_changed(self):
        val = self.zoom.value() / 10
        self.zoom_text.setText("Zoom Factor (with adaptive zoom) or FOV scale (same as preview): {}".format(val))

    def preset_resolution_selected(self):
        index = self.preset_resolution_combo.currentIndex()
        if index == 0:
            self.out_width_control.setValue(self.video_info_dict["width"])
            self.out_height_control.setValue(self.video_info_dict["height"])
        else:
            self.out_width_control.setValue(self.resolutions[index - 1][0])
            self.out_height_control.setValue(self.resolutions[index - 1][1])

    def update_out_size(self):
        """Update export image size
        """
        #print(self.out_width_control.value())
        pass

    def check_inputs_valid(self):
        if self.infile_path == "" or self.preset_path == "":
            self.show_error("Hey, looks like you forgot to open a video file and/or camera calibration preset. I guess this button could've been grayed out, but whatever.")
            self.export_button.setEnabled(False)
            self.export_keyframes_button.setEnabled(False)
            #self.sync_correction_button.setEnabled(False)
            self.update_smoothness_button.setEnabled(False)
            return False

        if self.gyro_log_path == "":
            self.show_error("No gyro log given. If you want to use the internal gyro data, there's a convenient button for using the input video as the gyro log.")
            return False
        
        return True
    
    def add_sync_barebone_handler(self):
        self.add_sync_at_time(self.sync1_control.value())

    def add_sync_at_time(self, time=0):
        if not self.check_inputs_valid():
            return

        fov_val = 1.4
        #smoothness_time_constant = self.get_smoothness_timeconstant()
        OF_slice_length = self.OF_frames_control.value()

        gyro_lpf = self.input_lpf_control.value()

        selected_log_type = self.gyro_log_format_select.currentData()
        uptilt = self.fpv_tilt_control.value()
        print("Going skiing?" if uptilt < 0 else "That's a lotta angle" if uptilt > 70 else "{} degree gyro/camera angle".format(uptilt))

        rotate_code = self.input_video_rotate_select.currentData()
        logvariant=self.gyro_variant_control.currentText()

        if self.use_gyroflow_data_file:
            if type(self.stab) == type(None):
                self.stab = stabilizer.Stabilizer(self.infile_path,gyroflow_file=self.gyro_log_path)
        elif type(self.stab) == type(None):
            self.stab = stabilizer.MultiStabilizer(self.infile_path, self.preset_path, self.gyro_log_path, fov_scale=fov_val, cam_angle_degrees=uptilt,
                                            gyro_lpf_cutoff = gyro_lpf, logtype=selected_log_type, logvariant=logvariant, video_rotation=rotate_code)

            self.stab.multi_sync_init()

            self.multiSyncUI.update_stab_instance(self.stab)


        self.stab.set_initial_offset(self.offset_control.value())
        self.stab.set_rough_search(self.sync_search_size.value())
        self.stab.set_num_frames_skipped(self.num_frames_skipped_control.value())

        if time * self.stab.fps + OF_slice_length + 10 > self.stab.num_frames:
            print("Sync not allowed here")
            return False

        self.stab.multi_sync_add_slice(time * self.stab.fps, OF_slice_length, debug_plots = False)

        self.multiSyncUI.update_from_stab_data()

    def auto_sync_start(self):
        if not self.check_inputs_valid():
            return


        fov_val = 1.4
        #smoothness_time_constant = self.get_smoothness_timeconstant()
        OF_slice_length = self.OF_frames_control.value()

        #cap = cv2.VideoCapture(self.infile_path)
        #width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #fps = cap.get(cv2.CAP_PROP_FPS)
        #num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        #cap.release()

        gyro_lpf = self.input_lpf_control.value()

        selected_log_type = self.gyro_log_format_select.currentData()
        uptilt = self.fpv_tilt_control.value()
        print("Going skiing?" if uptilt < 0 else "That's a lotta angle" if uptilt > 70 else "{} degree gyro/camera angle".format(uptilt))

        rotate_code = self.input_video_rotate_select.currentData()
        logvariant=self.gyro_variant_control.currentText()

        if self.use_gyroflow_data_file:
            if type(self.stab) == type(None):
                self.stab = stabilizer.Stabilizer(self.infile_path,gyroflow_file=self.gyro_log_path)
        else:
            self.stab = stabilizer.MultiStabilizer(self.infile_path, self.preset_path, self.gyro_log_path, fov_scale=fov_val, cam_angle_degrees=uptilt,
                                               gyro_lpf_cutoff = gyro_lpf, logtype=selected_log_type, logvariant=logvariant, video_rotation=rotate_code)

            self.multiSyncUI.update_stab_instance(self.stab)

        self.stab.set_initial_offset(self.offset_control.value())
        self.stab.set_rough_search(self.sync_search_size.value())
        self.stab.set_num_frames_skipped(self.num_frames_skipped_control.value())

        self.stab.set_smoothing_algo(self.stab_algo_instance_current)

        max_fitting_error = self.max_fitting_control.value()
        max_syncs = self.max_sync_control.value()
        success = self.stab.full_auto_sync(max_fitting_error,max_syncs)

        if not success:
            return

        print("Finished computing")

        self.multiSyncUI.update_from_stab_data()

        #self.recompute_stab_button.setText("Recompute sync")
        self.export_button.setEnabled(True)
        self.export_keyframes_button.setEnabled(True)

        # Show estimated delays in UI
        #self.sync_correction_button.setEnabled(True)
        self.update_smoothness_button.setEnabled(True)
        #self.d1_control.setValue(self.stab.d1)
        #self.d2_control.setValue(self.stab.d2)

        self.stab.set_map_func_scale(self.preview_fov_scale)

        if self.has_player:
            self.update_player_maps()
        self.analyzed = True

    def recompute_stab(self):
        """Update sync and stabilization
        """

        if not self.check_inputs_valid():
            return

        fov_val = 1.4
        #smoothness_time_constant = self.get_smoothness_timeconstant()
        OF_slice_length = self.OF_frames_control.value()


        cap = cv2.VideoCapture(self.infile_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()

        #sync1_frame = int(self.sync1_control.value() * fps)
        #sync2_frame = int(self.sync2_control.value() * fps)

        gyro_lpf = self.input_lpf_control.value()

        #if max(sync1_frame, sync2_frame) + OF_slice_length + 5 > num_frames:
        #    self.show_error("You're trying to analyze frames after the end of video. Video length: {} s, latest allowable sync time: {}".format(num_frames/fps, (num_frames - OF_slice_length-1)/fps))
        #    return


        selected_log_type = self.gyro_log_format_select.currentData()
        uptilt = self.fpv_tilt_control.value()
        print("Going skiing?" if uptilt < 0 else "That's a lotta angle" if uptilt > 70 else "{} degree gyro/camera angle".format(uptilt))

        #self.stab = stabilizer.BBLStabilizer(self.infile_path, self.preset_path, self.gyro_log_path, fov_scale=fov_val, cam_angle_degrees=uptilt,
        #                                        use_csv=use_csv, gyro_lpf_cutoff = gyro_lpf, logtype=logtype, use_raw_gyro_data=use_raw_gyro_data)

        rotate_code = self.input_video_rotate_select.currentData()
        logvariant=self.gyro_variant_control.currentText()


        if self.use_gyroflow_data_file:
            if type(self.stab) == type(None):
                self.stab = stabilizer.Stabilizer(self.infile_path,gyroflow_file=self.gyro_log_path)
        elif type(self.stab) == type(None):
            self.stab = stabilizer.MultiStabilizer(self.infile_path, self.preset_path, self.gyro_log_path, fov_scale=fov_val, cam_angle_degrees=uptilt,
                                               gyro_lpf_cutoff = gyro_lpf, logtype=selected_log_type, logvariant=logvariant, video_rotation=rotate_code)

        #self.stab.set_initial_offset(self.offset_control.value())
        #self.stab.set_rough_search(self.sync_search_size.value())
        #self.stab.set_num_frames_skipped(self.num_frames_skipped_control.value())

        #print("Starting sync. sync1: {} (frame {}), sync2: {} (frame {}), OF slices of {} frames".format(
        #        self.sync1_control.value(), sync1_frame, self.sync2_control.value(), sync2_frame, OF_slice_length))


        self.stab.set_smoothing_algo(self.stab_algo_instance_current)
        #self.stab.auto_sync_stab(sync1_frame, sync2_frame,
        #                         OF_slice_length, debug_plots=self.sync_debug_select.isChecked())

        self.stab.multi_sync_compute(debug_plots=self.sync_debug_select.isChecked())

        print("Finished computing")

        self.recompute_stab_button.setText("Recompute sync")
        self.export_button.setEnabled(True)

        self.export_keyframes_button.setEnabled(True)

        # Show estimated delays in UI
        #self.sync_correction_button.setEnabled(True)
        self.update_smoothness_button.setEnabled(True)
        #self.d1_control.setValue(self.stab.d1)
        #self.d2_control.setValue(self.stab.d2)

        self.stab.set_map_func_scale(self.preview_fov_scale)

        self.analyzed = True


    #def correct_sync(self):
    #    d1 = self.d1_control.value()
    #    d2 = self.d2_control.value()
    #    #smoothness = self.smooth_slider.value() / 100
    #    self.stab.set_smoothing_algo(self.stab_algo_instance_current)
    #    self.stab.manual_sync_correction(d1, d2)
    #    print("Finished computing")

    def update_smoothness(self):
        print("Updating smoothness setting...")
        self.stab.set_smoothing_algo(self.stab_algo_instance_current)
        self.stab.update_smoothing()
        print("Smoothness updated")


    def export_keyframes(self):
        export_file_filter = "Comma-separated values (*.csv)"
        
        filename = QtWidgets.QFileDialog.getSaveFileName(self, "Export keyframes", filter=export_file_filter)
        print("Output file: {}".format(filename[0]))

        if len(filename[0]) == 0:
            self.show_error("No output file given")
            return

        self.stab.export_stabilization(filename[0])

    def export_video(self):
        """Gives save location using filedialog
           and saves video to given location
        """



        out_size = (self.out_width_control.value(), self.out_height_control.value())

        #if out_size[0] > self.stab.width:
        #    self.show_error("The given output cropped width ({}) is greater than the video width ({})".format(out_size[0], self.stab.width))
        #    return
        #if out_size[1] > self.stab.height:
        #    self.show_error("The given output cropped height ({}) is greater than the video height ({})".format(out_size[1], self.stab.height))
        #    return

        start_time = self.export_starttime.value()
        stop_time = self.export_stoptime.value()

        if (stop_time < start_time):
            self.show_error("Start time is later than stop time.")
            return

        video_length = self.stab.num_frames / self.stab.fps

        if stop_time > video_length:
            self.show_error("Stop time ({}) is after end of video ({})".format(stop_time, video_length))
            return

        # get file
        export_file_filter = ""
        if self.enable_mp4_export:
            export_file_filter+="mp4 (*.mp4)"
        if self.enable_mov_export:
            if export_file_filter == "":
                export_file_filter+="Quicktime (*.mov)"
            else:
                export_file_filter+=";; Quicktime (*.mov)"
        if export_file_filter == "":
            export_file_filter+="All files (*)" # Should not happen (lost state)
        else:
            # Add "All files" option in case only one file format is supported to force the option list
            # to be present, if not it dissapears. Should probably look more into setMimeTypeFilters at a later stage
            export_file_filter+=";; All files (*)"

        filename = QtWidgets.QFileDialog.getSaveFileName(self, "Export video", filter=export_file_filter)
        print("Output file: {}".format(filename[0]))
        time.sleep(0.5) # Time to close file dialog
        # Handled in stabilizer
        #if filename[0] == self.infile_path:
        #    self.show_error("You can't overwride the input file")
        #    return

        if len(filename[0]) == 0:
            self.show_error("No output file given")
            return

        split_screen = self.split_screen_select.isChecked()
        #hardware_acceleration = self.hw_acceleration_select.isChecked()
        vcodec = self.video_encoder_select.currentText()
        vprofile = self.encoder_profile_select.currentText()
        bitrate = self.export_bitrate.value()  # Bitrate in Mbit/s
        preview = self.display_preview.isChecked()
        audio = self.export_audio.isChecked()
        #output_scale = int(self.out_scale_control.value())
        debug_text = self.export_debug_text.isChecked()
        pix_fmt = self.pixfmt_select.text()
        custom_ffmpeg = self.custom_ffmpeg_pipeline.text()
        smoothingFocus=self.fov_smoothing.value()/10


        zoomVal = self.zoom.value() /10
        if not self.enableAdaptiveZoom.isChecked():
            smoothingFocus = -2 # -2 means adaptive zoom is totally disabled
            zoomVal = zoomVal * 1280 / self.out_width_control.value() # scale to match preview window. A bit hacky...
        elif self.fov_smoothing.value() == self.fov_smoothing.maximum():
            smoothingFocus = -1 # -1 means minimum sufficient cropping
        
        bg_color = self.bg_color_select.text()

        
        self.stab.set_hyperlapse(hyperlapse_multiplier = self.hyperlapse_multiplier.value(), hyperlapse_num_blended_frames = self.hyperlapse_blend.value())

        viewer_thread = None
        if self.has_player:
            self.video_viewer.stop()
            viewer_thread = self.video_viewer.thread


        self.stab.renderfile(start_time, stop_time, filename[0], out_size = out_size,
                             split_screen = split_screen, bitrate_mbits = bitrate,
                             display_preview=preview, vcodec=vcodec, vprofile=vprofile,
                             pix_fmt = pix_fmt, debug_text=debug_text, custom_ffmpeg=custom_ffmpeg,
                             smoothingFocus=smoothingFocus, fov_scale=zoomVal, bg_color=bg_color, audio=audio, viewer_thread=viewer_thread)

    def export_gyroflow(self):
        self.stab.export_gyroflow_file()

    def show_error(self, msg):
        err_window = QtWidgets.QMessageBox(self)
        err_window.setIcon(QtWidgets.QMessageBox.Critical)
        err_window.setText(msg)
        err_window.setWindowTitle("Something's gone awry")
        err_window.show()

    def show_warning(self, msg):
        QtWidgets.QMessageBox.critical(self, "Something's gone awry", msg)

    def get_available_encoders(self):
        if(get_valid_ffmpeg_path()):  # Helper function from VidGear
            ffmpeg_encoders_sp = subprocess.run([get_valid_ffmpeg_path(),'-encoders'], check=True, stdout=subprocess.PIPE, universal_newlines=True)
            return ffmpeg_encoders_sp.stdout
        else:
            self.show_warning("Could not find FFmpeg installation")
            return ""

    def update_profile_select(self):
        index = self.video_encoder_select.currentIndex()
        encoder_index = self.encoder_model.index(index, 0, self.video_encoder_select.rootModelIndex())
        self.encoder_profile_select.setRootModelIndex(encoder_index)
        h264_encoders = ["libx264", "h264_videotoolbox", "h264_vaapi", "h264_nvenc"]
        if self.video_encoder_select.currentText() in h264_encoders and self.encoder_profile_select.count() > 2:
            self.encoder_profile_select.setCurrentIndex(2)  # Make "high" default profile for standard h264 encoders
        else:
            self.encoder_profile_select.setCurrentIndex(0)

    def update_bitrate_visibility(self):
        encoders_without_bitrate_control = ["prores_ks"]
        if self.video_encoder_select.currentText() in encoders_without_bitrate_control:
            enable_bitrate = False
        else:
            enable_bitrate = True
        self.export_bitrate_text.setVisible(enable_bitrate)
        self.export_bitrate.setVisible(enable_bitrate)

    def update_container_selection(self):
        encoders_with_only_mp4_support = [""]
        encoders_with_only_mov_support = ["prores_ks"]
        if self.video_encoder_select.currentText() in encoders_with_only_mp4_support:
            self.enable_mp4_export = True
            self.enable_mov_export = False
        elif self.video_encoder_select.currentText() in encoders_with_only_mov_support:
            self.enable_mp4_export = False
            self.enable_mov_export = True
        else:
            self.enable_mp4_export = True
            self.enable_mov_export = True

class StabUtility(StabUtilityBarebone):
    def __init__(self):
        """Qt window containing camera calibration utility
        """

        super().__init__(False)

        self.preview_fov_scale = 3.0

        # Initialize UI
        self.setWindowTitle("Gyroflow Stabilizer {}".format(__version__))
        self.setWindowIcon(QtGui.QIcon(':/media/icon.png'))
        
        self.main_widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout()
        self.main_widget.setLayout(self.layout)

        # left half of screen with player/buttons
        self.left_side_widget = QtWidgets.QWidget()
        self.left_layout = QtWidgets.QVBoxLayout()
        self.left_side_widget.setLayout(self.left_layout)

        self.layout.addWidget(self.left_side_widget)
        

        # right half of screen with export options
        self.right_side_widget = QtWidgets.QWidget()
        self.right_layout = QtWidgets.QVBoxLayout()
        self.right_side_widget.setLayout(self.right_layout)
        self.right_side_widget.setFixedWidth(550)
        self.right_layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.addWidget(self.right_side_widget)



        # video player with controls
        self.video_viewer = VideoPlayerWidget()
        self.left_layout.addWidget(self.video_viewer)

        self.setCentralWidget(self.main_widget)

        # control buttons for stretching [Safe area slider] [expo slider] [X]View safe area [recompute maps] [render to file]
        self.calib_controls = QtWidgets.QWidget()
        self.calib_controls_layout = QtWidgets.QHBoxLayout()
        self.calib_controls.setLayout(self.calib_controls_layout)

        self.button_height = 40

        self.calib_msg = ""

        # info text box
        self.info_text = QtWidgets.QLabel("[info]")
        self.calib_controls_layout.addWidget(self.info_text)


        self.add_sync_button = QtWidgets.QPushButton("Add sync here")
        self.add_sync_button.setMinimumHeight(self.button_height)
        self.add_sync_button.clicked.connect(self.add_sync_here)
        self.calib_controls_layout.addWidget(self.add_sync_button)

        # button for recomputing image stretching maps
        #self.add_sync1_button = QtWidgets.QPushButton("Sync 1 here")
        #self.add_sync1_button.setMinimumHeight(self.button_height)
        #self.add_sync1_button.clicked.connect(self.synchere1)
        #self.calib_controls_layout.addWidget(self.add_sync1_button)

        #self.add_sync2_button = QtWidgets.QPushButton("Sync 2 here")
        #self.add_sync2_button.setMinimumHeight(self.button_height)
        #self.add_sync2_button.clicked.connect(self.synchere2)
        #self.calib_controls_layout.addWidget(self.add_sync2_button)

        self.trim_start_button = QtWidgets.QPushButton("Trim start")
        self.trim_start_button.setMinimumHeight(self.button_height)
        self.trim_start_button.clicked.connect(self.trimstart)
        self.calib_controls_layout.addWidget(self.trim_start_button)

        self.trim_end_button = QtWidgets.QPushButton("Trim end")
        self.trim_end_button.setMinimumHeight(self.button_height)
        self.trim_end_button.clicked.connect(self.trimend)
        self.calib_controls_layout.addWidget(self.trim_end_button)

        # slider for adjusting FOV
        self.preview_fov_text = QtWidgets.QLabel("FOV scale ({}):".format(self.preview_fov_scale))
        self.preview_fov_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.preview_fov_slider.setMinimum(8)
        self.preview_fov_slider.setValue(round(self.preview_fov_scale * 10))
        self.preview_fov_slider.setMaximum(70)
        self.preview_fov_slider.setMaximumWidth(300)
        self.preview_fov_slider.setSingleStep(1)
        self.preview_fov_slider.setTickInterval(1)
        self.preview_fov_slider.valueChanged.connect(self.fov_changed)
        self.preview_fov_slider.sliderReleased.connect(self.update_preview)

        self.calib_controls_layout.addWidget(self.preview_fov_text)
        self.calib_controls_layout.addWidget(self.preview_fov_slider)

        # checkbox to preview lens distortion correction
        self.preview_stab_toggle_btn = QtWidgets.QCheckBox("Toggle stabilization: ")
        self.preview_stab_toggle_btn.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.preview_stab_toggle_btn.setChecked(True)
        self.preview_stab_toggle_btn.stateChanged.connect(self.update_preview)

        self.calib_controls_layout.addWidget(self.preview_stab_toggle_btn)

        # add control bar to main layout
        self.left_layout.addWidget(self.calib_controls)



        # Right layout: Export settings

        self.main_setting_widget = QtWidgets.QTabWidget()
        self.layout = QtWidgets.QHBoxLayout()
        self.main_setting_widget.setLayout(self.layout)
        self.main_setting_widget.setStyleSheet("font-size: 12px")

        self.right_layout.addWidget(self.main_setting_widget)
        

        self.init_UI()

        self.has_player = True

        # file menu setup
        menubar = self.menuBar()
        filemenu = menubar.addMenu('&File')

        # https://joekuan.wordpress.com/2015/09/23/list-of-qt-icons/

        # Reconnect open video button to display preview
        self.open_vid_button.clicked.disconnect()
        self.open_vid_button.clicked.connect(self.open_video_with_player_func)

        icon = self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon)
        self.open_file = QtWidgets.QAction(icon, 'Open file', self)
        self.open_file.setShortcut("Ctrl+O")
        self.open_file.triggered.connect(self.open_video_with_player_func)
        filemenu.addAction(self.open_file)


        self.recompute_stab_button.clicked.connect(self.update_player_maps)
        #self.sync_correction_button.clicked.connect(self.update_player_maps)

        icon = self.style().standardIcon(QtWidgets.QStyle.SP_FileLinkIcon)
        self.open_preset = QtWidgets.QAction(icon, 'Open calibration preset', self)
        self.open_preset.triggered.connect(self.open_preset_func)
        filemenu.addAction(self.open_preset)

        self.statusBar()
        self.infile_path = ""
        self.show()

        
        self.main_setting_widget.show()

    def open_video_with_player_func(self):
        """Open file using Qt filedialog
        """
        self.video_viewer.reset_map_function()
        if self.open_video_func():
            self.video_viewer.set_video_path(self.infile_path)
            self.video_viewer.next_frame()

    def set_player_video(self):
        self.video_viewer.set_video_path(self.infile_path)
        self.video_viewer.next_frame()

    def update_player_maps(self):
        self.video_viewer.set_map_function(self.stab.map_function)

    def fov_changed(self):
        self.preview_fov_scale = self.preview_fov_slider.value()/10
        #print(self.preview_fov_scale)
        self.preview_fov_text.setText("FOV scale ({}):".format(self.preview_fov_scale))

    def show_error(self, msg):
        QtWidgets.QMessageBox.critical(self, "Something's gone awfully wrong", msg)

        return
        #self.err_window = QtWidgets.QMessageBox(self)
        #self.err_window.setIcon(QtWidgets.QMessageBox.Warning)
        #self.err_window.setText(msg)
        #self.err_window.setWindowTitle("Something's gone awry")
        #self.err_window.exec_()
        #self.err_window.close()

    def show_warning(self, msg):
        QtWidgets.QMessageBox.critical(self, "Something's gone awry", msg)

    #def synchere1(self):
    #    self.sync1_control.setValue(self.video_viewer.get_current_timestamp())
    #    #print(self.video_viewer.get_current_timestamp())

    #def synchere2(self):
    #    self.sync2_control.setValue(self.video_viewer.get_current_timestamp())
    #    #print(self.video_viewer.get_current_timestamp())

    def add_sync_here(self):
        self.video_viewer.stop()
        timestamp = self.video_viewer.get_current_timestamp()
        self.add_sync_at_time(timestamp)
        

    def trimstart(self):
        self.export_starttime.setValue(self.video_viewer.get_current_timestamp())
    
    def trimend(self):
        self.export_stoptime.setValue(self.video_viewer.get_current_timestamp())

    def update_preview(self):
        if self.stab:
            self.stab.set_map_func_scale(self.preview_fov_scale)
        
        self.video_viewer.enable_map_function(self.preview_stab_toggle_btn.isChecked())
        self.video_viewer.update_frame()


    def open_preset_func(self):
        path = QtWidgets.QFileDialog.getOpenFileName(self, "Open preset file", filter="JSON preset (*.json *.JSON)")

        if (len(path[0]) == 0):
            print("No file selected")
            return
        #print(path)
        self.preset_path = path[0]
        self.display_preset_info()


    def check_aspect(self):
        if self.preset_path and self.infile_path:
            # Check if the aspect ratios match
            self.preset_info_dict.get("aspect")
            v_aspect = self.video_info_dict.get("aspect")
            p_aspect = self.preset_info_dict.get("aspect")
            if abs(v_aspect - p_aspect) > 0.01:
                self.aspect_warning_text.setText(f"<h3>Seems like the aspect ratios don't quite match. Video: {v_aspect:.3f}, preset: {p_aspect:.3f}</h3>")
            else:
                self.aspect_warning_text.setText("")

    def closeEvent(self, event):
        print("Closing now")
        self.video_viewer.destroy_thread()
        if self.stab:
            self.stab.release()
        event.accept()



def main():
    QtCore.QLocale.setDefault(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))

    app = QtWidgets.QApplication([])

    widget = Launcher() # Launcher()
    widget.resize(500, 500)


    widget.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
    # Pack to exe using:
    # pyinstaller --icon=media\icon.ico gyroflow.py --add-binary <path-to-python>\Python38\Lib\site-packages\cv2\opencv_videoio_ffmpeg430_64.dll -F
    # in my case:
    # pyside2-rcc images.qrc -o bundled_images.py
    # poetry run pyinstaller -F --icon=media\icon.ico gyroflow.py --add-binary C:\Users\elvin\AppData\Local\Programs\Python\Python38\Lib\site-packages\cv2\opencv_videoio_ffmpeg440_64.dll;.
    # -F == one file, -w == no command window
    # Alternative: pyinstaller gyroflow.py -F --icon=media\icon.ico --add-binary C:\Users\elvin\AppData\Local\Programs\Python\Python39\python39.dll
