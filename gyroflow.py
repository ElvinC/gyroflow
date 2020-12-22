"""Main file containing UI code"""

import sys
import random
import cv2
import numpy as np
from PySide2 import QtCore, QtWidgets, QtGui
from _version import __version__
import calibrate_video
import time
import nonlinear_stretch

from stabilizer import GPMFStabilizer

class Launcher(QtWidgets.QWidget):
    """Main launcher with options to open different utilities
    """
    def __init__(self):

        super().__init__()

        self.setWindowTitle("Gyroflow {} Launcher".format(__version__))

        self.setFixedWidth(450)


        self.calibrator_button = QtWidgets.QPushButton("Camera Calibrator")
        self.calibrator_button.setMinimumSize(300,50)
        self.calibrator_button.setToolTip("Use this to generate camera calibration files")

        self.stabilizer_button = QtWidgets.QPushButton("Video Stabilizer (Doesn't work yet)")
        self.stabilizer_button.setMinimumSize(300,50)

        self.stabilizer_barebone_button = QtWidgets.QPushButton("Video Stabilizer (barebone dev version)")
        self.stabilizer_barebone_button.setMinimumSize(300,50)

        self.stretch_button = QtWidgets.QPushButton("Non-linear Stretch")
        self.stretch_button.setMinimumSize(300,50)

        self.text = QtWidgets.QLabel("<h1>Gyroflow {}</h1>".format(__version__))
        self.text.setAlignment(QtCore.Qt.AlignCenter)

        self.footer = QtWidgets.QLabel('''Developed by Elvin. <a href='https://github.com/ElvinC/gyroflow'>Contribute or support on Github</a>''')
        self.footer.setOpenExternalLinks(True)
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.calibrator_button)
        self.layout.addWidget(self.stabilizer_button)
        self.layout.addWidget(self.stabilizer_barebone_button)
        self.layout.addWidget(self.stretch_button)
        
        self.setLayout(self.layout)

        self.layout.addWidget(self.footer)

        self.calibrator_button.clicked.connect(self.open_calib_util)
        self.stabilizer_button.clicked.connect(self.open_stab_util)
        self.stabilizer_barebone_button.clicked.connect(self.open_stab_util_barebone)
        self.stretch_button.clicked.connect(self.open_stretch_util)

        # Placeholder for utility windows.
        self.calibrator_utility = None
        self.stabilizer_utility = None
        self.stabilizer_utility_barebone = None
        self.stretch_utility = None

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
            if self.calibrator_utility.isVisible():
                return
        
        self.stretch_utility = StretchUtility()
        self.stretch_utility.resize(500, 500)
        self.stretch_utility.show()


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

        # Draw vertical lines at given coords
        self.vert_line_coords = []

        self.cap = None
        self.frame = None

        # used for scaling
        self.max_width = 1280

    def run(self):
        """
        Run the videoplayer using the thread
        """

        self.cap = cv2.VideoCapture()

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

            else:
                time.sleep(1/10)


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

        if rgbImage.shape[1] > self.max_width:
            new_height = int(self.max_width/rgbImage.shape[1] * rgbImage.shape[0])
            rgbImage = cv2.resize(rgbImage, (self.max_width, new_height))

        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        self.changePixmap.emit(convertToQtFormat.copy())

        this_frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if this_frame_num % 5 == 0 and self.frame_pos_update:
            self.frame_pos_update(this_frame_num)



# based on https://robonobodojo.wordpress.com/2018/07/01/automatic-image-sizing-with-pyside/
# and https://stackoverflow.com/questions/44404349/pyqt-showing-video-stream-from-opencv/44404713
class VideoPlayer(QtWidgets.QLabel):
    def __init__(self, img = "placeholder.jpg"):
        super(VideoPlayer, self).__init__()
        self.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        self.pixmap = QtGui.QPixmap(img)
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

        self.control_layout.addWidget(self.play_button)
        self.control_layout.addWidget(self.time_slider)

        self.layout.addWidget(self.control_bar)

        self.frame_width = 1920 # placeholder
        self.frame_height = 1080
        self.num_frames = 0
        

        # initialize thread for video player with frame update function
        self.thread = VideoThread(self, self.set_seek_frame)
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
        self.num_frames = self.thread.cap.get(cv2.CAP_PROP_FRAME_COUNT)

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
        # update slider without triggering valueChange
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(slider_val)
        self.time_slider.blockSignals(False)

    def destroy_thread(self):
        self.thread.terminate()


class CalibratorUtility(QtWidgets.QMainWindow):
    def __init__(self):
        """Qt window containing camera calibration utility
        """
        super().__init__()

        # Initialize UI
        self.setWindowTitle("Gyroflow Calibrator {}".format(__version__))


        self.main_widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QVBoxLayout()
        self.main_widget.setLayout(self.layout)

        # video player with controls
        self.video_viewer = VideoPlayerWidget()
        self.layout.addWidget(self.video_viewer)

        self.setCentralWidget(self.main_widget)

        # control buttons for stretching [Safe area slider] [expo slider] [X]View safe area [recompute maps] [render to file]
        self.calib_controls = QtWidgets.QWidget()
        self.calib_controls_layout = QtWidgets.QHBoxLayout()
        self.calib_controls.setLayout(self.calib_controls_layout)

        self.button_height = 40

        self.calib_msg = ""

        # button for recomputing image stretching maps
        self.add_frame_button = QtWidgets.QPushButton("Add current frame")
        self.add_frame_button.setMinimumHeight(self.button_height)
        self.add_frame_button.clicked.connect(self.add_current_frame)
        
        self.calib_controls_layout.addWidget(self.add_frame_button)

        # button for recomputing image stretching maps
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

        # info text box
        self.info_text = QtWidgets.QLabel("No frames loaded")
        self.calib_controls_layout.addWidget(self.info_text)

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

        # button for recomputing image stretching maps
        self.export_button = QtWidgets.QPushButton("Export preset file")
        self.export_button.setMinimumHeight(self.button_height)
        self.export_button.clicked.connect(self.save_preset_file)
        
        self.calib_controls_layout.addWidget(self.export_button)

        # add control bar to main layout
        self.layout.addWidget(self.calib_controls)

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

        self.chess_window = None

        self.statusBar()

        self.infile_path = ""


        self.show()

        self.main_widget.show()

        # initialize instance of calibrator class
        self.calibrator = calibrate_video.FisheyeCalibrator(chessboard_size=(9,6))


    def open_file_func(self):
        """Open file using Qt filedialog
        """
        path = QtWidgets.QFileDialog.getOpenFileName(self, "Open video file", filter="Video (*.mp4 *.avi *.mov)")
        self.infile_path = path[0]
        self.video_viewer.set_video_path(path[0])

        self.video_viewer.next_frame()

        # reset calibrator and info
        self.calibrator = calibrate_video.FisheyeCalibrator(chessboard_size=(9,6))
        self.update_calib_info()

    def open_preset_func(self):
        """Load in calibration preset
        """
        path = QtWidgets.QFileDialog.getOpenFileName(self, "Open video file", filter="JSON preset (*.json)")

        if (len(path[0]) == 0):
            print("No file selected")
            return

        self.calibrator.load_calibration_json(path[0])

        self.update_calib_info()

    def chessboard_func(self):
        """Function to show the calibration chessboard in a new window
        """
        print("Showing chessboard")

        self.chess_window = QtWidgets.QWidget()
        self.chess_window.setWindowTitle("Calibration target")
        self.chess_window.setStyleSheet("background-color:white;")

        self.chess_layout = QtWidgets.QVBoxLayout()
        self.chess_window.setLayout(self.chess_layout)

        # VideoPlayer class doubles as a auto resizing image viewer
        

        # generate chessboard pattern so no external images are needed
        chess_pic = np.zeros((9,12), np.uint8)

        # Set white squares
        chess_pic[::2,::2] = 255
        chess_pic[1::2,1::2] = 255

        # Borders to white
        chess_pic[0,:] = 255
        chess_pic[-1,:]= 255
        chess_pic[:,0]= 255
        chess_pic[:,-1]= 255

        # double size and reduce borders slightly
        chess_pic = cv2.resize(chess_pic,(12*2, 9*2), interpolation=cv2.INTER_NEAREST)
        chess_pic = chess_pic[1:-1,:]

        # convert to Qt image
        h, w = chess_pic.shape
        convertToQtFormat = QtGui.QImage(chess_pic.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        
        # VideoPlayer doubles as a autoresiznig image viewer
        chess_viewer = VideoPlayer(convertToQtFormat.copy())

        self.chess_layout.addWidget(chess_viewer)

        self.chess_window.resize(500, 500)
        self.chess_window.show()
        

    def closeEvent(self, event):
        print("Closing now")
        self.video_viewer.destroy_thread()
        event.accept()

    def fov_changed(self):
        self.fov_scale = self.fov_slider.value()/10
        self.fov_text.setText("FOV scale ({}):".format(self.fov_scale))


    def save_preset_file(self):
        """save camera preset file
        """
        print("Exporting preset")

        filename = QtWidgets.QFileDialog.getSaveFileName(self, "Export calibration preset", filter="JSON preset (*.json)")
        print(filename[0])

        if len(filename[0]) == 0:
            self.show_error("No output file given")
            return

        self.calibrator.save_calibration_json(filename[0])

    def show_error(self, msg):
        err_window = QtWidgets.QMessageBox(self)
        err_window.setIcon(QtWidgets.QMessageBox.Critical)
        err_window.setText(msg)
        err_window.setWindowTitle("Something's gone awry")
        err_window.show()

    def add_current_frame(self):
        print("Adding frame")

        ret, self.calib_msg, corners = self.calibrator.add_calib_image(self.video_viewer.thread.frame)

        self.update_calib_info()

        if self.calibrator.num_images > 0:
            self.process_frames_btn.setEnabled(True)

    def remove_frame(self):
        """Remove last calibration frame
        """
        self.calibrator.remove_calib_image()

        self.update_calib_info()

        

    def update_calib_info(self):
        """ Update the status text in the utility
        """

        txt = "Good frames: {}\nProcessed frames: {}\nRMS error: {}\n{}".format(self.calibrator.num_images,
                                                                                self.calibrator.num_images_used,
                                                                                self.calibrator.RMS_error,
                                                                                self.calib_msg)
    
        self.info_text.setText(txt)

        # enable/disable buttons
        if self.calibrator.num_images > 0:
            self.process_frames_btn.setEnabled(True)
        else:
            self.process_frames_btn.setEnabled(False)
 
        if self.calibrator.num_images_used > 0:
            self.preview_toggle_btn.setEnabled(True)
        else:
            self.preview_toggle_btn.setChecked(False)
            self.preview_toggle_btn.setEnabled(False)

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

            #map1, map2 = self.calibrator.get_rotation_map()
            #self.video_viewer.add_maps(map1, map2)

        self.video_viewer.update_frame()





class StretchUtility(QtWidgets.QMainWindow):
    def __init__(self):
        """Qt window containing utility for nonlinear stretch
        """
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
        path = QtWidgets.QFileDialog.getOpenFileName(self, "Open video file", filter="Video (*.mp4 *.avi *.mov)")
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
        err_window = QtWidgets.QMessageBox(self)
        err_window.setIcon(QtWidgets.QMessageBox.Critical)
        err_window.setText(msg)
        err_window.setWindowTitle("Something's gone awry")
        err_window.show()



class StabUtility(QtWidgets.QMainWindow):
    def __init__(self):
        """Qt window containing utility for syncing and stabilization
        """
        super().__init__()

        # Initialize UI
        self.setWindowTitle("Gyroflow Stabilizer {}".format(__version__))

        self.main_widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QVBoxLayout()
        self.main_widget.setLayout(self.layout)

        # video player with controls
        self.video_viewer = VideoPlayerWidget()
        self.layout.addWidget(self.video_viewer)

        self.setCentralWidget(self.main_widget)

        # control buttons
        self.main_controls = QtWidgets.QWidget()
        self.main_controls_layout = QtWidgets.QHBoxLayout()
        self.main_controls.setLayout(self.main_controls_layout)

        # button for syncing around current frame
        self.syncpoint_button = QtWidgets.QPushButton("Add sync point")
        self.syncpoint_button.clicked.connect(self.syncpoint_handler)
        self.main_controls_layout.addWidget(self.syncpoint_button)


        # slider for adjusting unwarped area
        self.smooth_text = QtWidgets.QLabel("Smoothness (0%):")
        self.smooth_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.smooth_slider.setMinimum(0)
        self.smooth_slider.setValue(0)
        self.smooth_slider.setMaximum(100)
        self.smooth_slider.setSingleStep(1)
        self.smooth_slider.setTickInterval(1)
        self.smooth_slider.valueChanged.connect(self.smooth_changed)

        self.main_controls_layout.addWidget(self.smooth_text)
        self.main_controls_layout.addWidget(self.smooth_slider)

        # slider for adjusting non linear crop
        self.crop_text = QtWidgets.QLabel("Crop (2.0):")
        self.crop_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.crop_slider.setMinimum(10)
        self.crop_slider.setValue(20)
        self.crop_slider.setMaximum(40)
        self.crop_slider.setSingleStep(1)
        self.crop_slider.setTickInterval(1)
        self.crop_slider.valueChanged.connect(self.crop_changed)


        self.main_controls_layout.addWidget(self.crop_text)
        self.main_controls_layout.addWidget(self.crop_slider)        

        # output size choice
        self.out_size_text = QtWidgets.QLabel("Output size: ")
        self.main_controls_layout.addWidget(self.out_size_text)

        self.out_width_control = QtWidgets.QSpinBox(self)
        self.out_width_control.setMinimum(16)
        self.out_width_control.setMaximum(7680) # 8K max is probably fine
        self.out_width_control.setValue(1920)
        self.out_width_control.valueChanged.connect(self.update_out_size)

        self.main_controls_layout.addWidget(self.out_width_control)       

        # output size choice
        self.out_height_control = QtWidgets.QSpinBox(self)
        self.out_height_control.setMinimum(9)
        self.out_height_control.setMaximum(4320)
        self.out_height_control.setValue(1080)
        self.out_height_control.valueChanged.connect(self.update_out_size)

        self.main_controls_layout.addWidget(self.out_height_control)   

        # button for recomputing image stretching maps
        self.recompute_stab_button = QtWidgets.QPushButton("Apply settings and recompute")
        self.recompute_stab_button.clicked.connect(self.recompute_stab)
        
        self.main_controls_layout.addWidget(self.recompute_stab_button)

        # button for exporting video
        self.export_button = QtWidgets.QPushButton("Export video")
        self.export_button.clicked.connect(self.export_video)
        
        self.main_controls_layout.addWidget(self.export_button)

        # add control bar to main layout
        self.layout.addWidget(self.main_controls)

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

        self.statusBar()

        self.infile_path = ""

        

        self.show()

        self.main_widget.show()

        # non linear setup
        

    def open_file_func(self):
        """Open file using Qt filedialog 
        """
        path = QtWidgets.QFileDialog.getOpenFileName(self, "Open video file", filter="Video (*.mp4 *.avi *.mov)")
        self.infile_path = path[0]
        self.video_viewer.set_video_path(path[0])

        #self.recompute_stab()

        self.video_viewer.next_frame()

    def open_preset_func(self):
        pass
        

    def closeEvent(self, event):
        print("Closing now")
        self.video_viewer.destroy_thread()
        event.accept()

    def smooth_changed(self):
        """Smoothness has changed
        """
        pass


    def crop_changed(self):
        """Nonlinear expo has changed
        """
        crop_val = self.crop_slider.value() / 10
        self.crop_text.setText("Crop ({}):".format(crop_val))

    def syncpoint_handler(self):
        """Add sync point
        """
        pass

    def update_out_size(self):
        """Update export image size
        """
        #print(self.out_width_control.value())
        pass

    def recompute_stab(self):
        """Update sync and stabilization
        """
        pass


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
        
        pass

    def show_error(self, msg):
        err_window = QtWidgets.QMessageBox(self)
        err_window.setIcon(QtWidgets.QMessageBox.Critical)
        err_window.setText(msg)
        err_window.setWindowTitle("Something's gone awry")
        err_window.show()



class StabUtilityBarebone(QtWidgets.QMainWindow):
    def __init__(self):
        """Qt window containing barebone utility for stabilization. (No video viewer)
        """
        super().__init__()

        # Initialize UI
        self.setWindowTitle("Gyroflow Stabilizer Barebone {}".format(__version__))

        self.main_widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QVBoxLayout()
        self.main_widget.setLayout(self.layout)
        self.main_widget.setStyleSheet("font-size: 12px")

        # video player with controls

        self.setCentralWidget(self.main_widget)

        # control buttons
        self.main_controls = QtWidgets.QWidget()
        self.main_controls_layout = QtWidgets.QVBoxLayout()
        self.main_controls.setLayout(self.main_controls_layout)

        self.open_vid_button = QtWidgets.QPushButton("Open video file")
        self.open_vid_button.setMinimumHeight(30)
        self.open_vid_button.clicked.connect(self.open_file_func)
        
        self.main_controls_layout.addWidget(self.open_vid_button)

        # lens preset
        self.open_preset_button = QtWidgets.QPushButton("Open lens preset")
        self.open_preset_button.setMinimumHeight(30)
        self.open_preset_button.clicked.connect(self.open_preset_func)
        self.main_controls_layout.addWidget(self.open_preset_button)


        self.open_bbl_button = QtWidgets.QPushButton("Open BBL file (leave empty for GPMF, not implemented yet)")
        self.open_bbl_button.setMinimumHeight(30)
        self.open_bbl_button.clicked.connect(self.open_bbl_func)
        self.main_controls_layout.addWidget(self.open_bbl_button)

        # slider for adjusting smoothness. 0 = no stabilization. 100 = locked. Scaling is a bit weird still and depends on gyro sample rate.
        self.smooth_text = QtWidgets.QLabel("Smoothness (85%):")
        self.smooth_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.smooth_slider.setMinimum(0)
        self.smooth_slider.setValue(85)
        self.smooth_slider.setMaximum(100)
        self.smooth_slider.setSingleStep(1)
        self.smooth_slider.setTickInterval(1)
        self.smooth_slider.valueChanged.connect(self.smooth_changed)

        self.main_controls_layout.addWidget(self.smooth_text)
        self.main_controls_layout.addWidget(self.smooth_slider)

        explaintext = QtWidgets.QLabel("<b>Note:</b> 0% corresponds to no smoothing and 100% corresponds to a locked camera. " \
        "intermediate values are non-linear and depend on gyro sample rate in current implementation.")
        explaintext.setWordWrap(True)
        explaintext.setMinimumHeight(60)
        self.main_controls_layout.addWidget(explaintext)


        # slider for adjusting non linear crop
        self.fov_text = QtWidgets.QLabel("FOV scale (1.5):")
        self.fov_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.fov_slider.setMinimum(10)
        self.fov_slider.setValue(15)
        self.fov_slider.setMaximum(40)
        self.fov_slider.setSingleStep(1)
        self.fov_slider.setTickInterval(1)
        self.fov_slider.valueChanged.connect(self.fov_scale_changed)


        self.main_controls_layout.addWidget(self.fov_text)
        self.main_controls_layout.addWidget(self.fov_slider)        

        # output size choice
        self.out_size_text = QtWidgets.QLabel("Output crop: ")
        self.main_controls_layout.addWidget(self.out_size_text)

        self.out_width_control = QtWidgets.QSpinBox(self)
        self.out_width_control.setMinimum(16)
        self.out_width_control.setMaximum(7680) # 8K max is probably fine
        self.out_width_control.setValue(1920)
        self.out_width_control.valueChanged.connect(self.update_out_size)

        # output size choice
        self.out_height_control = QtWidgets.QSpinBox(self)
        self.out_height_control.setMinimum(9)
        self.out_height_control.setMaximum(4320)
        self.out_height_control.setValue(1080)
        self.out_height_control.valueChanged.connect(self.update_out_size)

        self.main_controls_layout.addWidget(self.out_height_control)   
        self.main_controls_layout.addWidget(self.out_width_control)


        explaintext = QtWidgets.QLabel("<b>Note:</b> The current code uses two image remappings for lens correction " \
        "and perspective transform, so output must be cropped seperately to avoid black borders. These steps can be combined later. For now fov_scale = 1.5 with appropriate crop depending on resolution works.")
        explaintext.setWordWrap(True)
        explaintext.setMinimumHeight(60)
        self.main_controls_layout.addWidget(explaintext)

        # output size choice
        self.main_controls_layout.addWidget(QtWidgets.QLabel("Initial rough gyro offset in seconds (Sync requires +/- 2 sec. Set to 0 for GPMF):"))

        self.offset_control = QtWidgets.QDoubleSpinBox(self)
        self.offset_control.setMinimum(-100)
        self.offset_control.setMaximum(100)
        self.offset_control.setValue(0)

        self.main_controls_layout.addWidget(self.offset_control)



        self.main_controls_layout.addWidget(QtWidgets.QLabel("Auto sync timestamp 1 (video time in seconds. Shaky parts of video work best)"))
        self.sync1_control = QtWidgets.QDoubleSpinBox(self)
        self.sync1_control.setMinimum(0)
        self.sync1_control.setMaximum(10000)
        self.sync1_control.setValue(5)
        self.main_controls_layout.addWidget(self.sync1_control)

        self.main_controls_layout.addWidget(QtWidgets.QLabel("Auto sync timestamp 2"))
        self.sync2_control = QtWidgets.QDoubleSpinBox(self)
        self.sync2_control.setMinimum(0)
        self.sync2_control.setMaximum(10000)
        self.sync2_control.setValue(30)
        self.main_controls_layout.addWidget(self.sync2_control)

        # How many frames to analyze using optical flow each slice
        self.main_controls_layout.addWidget(QtWidgets.QLabel("Number of frames to analyze per slice using optical flow:"))
        self.OF_frames_control = QtWidgets.QSpinBox(self)
        self.OF_frames_control.setMinimum(10)
        self.OF_frames_control.setMaximum(300)
        self.OF_frames_control.setValue(60)

        self.main_controls_layout.addWidget(self.OF_frames_control)


        self.main_controls_layout.addWidget(QtWidgets.QLabel('Gyro orientation. Write "hero6" or "hero8". Orientation presets to be added later.'))
        self.gyro_control = QtWidgets.QLineEdit(self)
        self.gyro_control.setText("hero6")
        self.main_controls_layout.addWidget(self.gyro_control)
        

        # button for (re)computing sync
        self.recompute_stab_button = QtWidgets.QPushButton("Apply settings and compute sync")
        self.recompute_stab_button.setMinimumHeight(30)
        self.recompute_stab_button.clicked.connect(self.recompute_stab)
        self.main_controls_layout.addWidget(self.recompute_stab_button)

        explaintext = QtWidgets.QLabel("<b>Note:</b> Check console for info after clicking. A number of plots will appear during the" \
                                        "process showing the difference between gyro and optical flow. Just close these after you've done looking at them.")
        explaintext.setWordWrap(True)
        explaintext.setMinimumHeight(60)
        self.main_controls_layout.addWidget(explaintext)


        self.main_controls_layout.addWidget(QtWidgets.QLabel("Video export start and stop (seconds)"))
        self.export_starttime = QtWidgets.QDoubleSpinBox(self)
        self.export_starttime.setMinimum(0)
        self.export_starttime.setMaximum(10000)
        self.export_starttime.setValue(0)
        self.main_controls_layout.addWidget(self.export_starttime)


        self.export_stoptime = QtWidgets.QDoubleSpinBox(self)
        self.export_stoptime.setMinimum(0)
        self.export_stoptime.setMaximum(10000)
        self.export_stoptime.setValue(30)
        self.main_controls_layout.addWidget(self.export_stoptime)

        
        # button for exporting video
        self.export_button = QtWidgets.QPushButton("Export (hopefully) stabilized video")
        self.export_button.setMinimumHeight(30)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_video)
        
        self.main_controls_layout.addWidget(self.export_button)

        # add control bar to main layout
        self.layout.addWidget(self.main_controls)


        self.infile_path = ""
        self.preset_path = ""
        self.BBL_path = ""
        self.stab = None
        self.analyzed = False
        
        self.show()

        self.main_widget.show()

        # non linear setup
        

    def open_file_func(self):
        """Open file using Qt filedialog 
        """
        path = QtWidgets.QFileDialog.getOpenFileName(self, "Open video file", filter="Video (*.mp4 *.avi *.mov)")

        if (len(path[0]) == 0):
            print("No file selected")
            return
        
        self.infile_path = path[0]
        self.open_vid_button.setText("Video file: {}".format(self.infile_path.split("/")[-1]))
        self.open_vid_button.setStyleSheet("font-weight:bold;")




    def open_preset_func(self):
        path = QtWidgets.QFileDialog.getOpenFileName(self, "Open video file", filter="JSON preset (*.json)")

        if (len(path[0]) == 0):
            print("No file selected")
            return
        print(path)
        self.preset_path = path[0]
        self.open_preset_button.setText("Preset file: {}".format(self.preset_path.split("/")[-1]))
        self.open_preset_button.setStyleSheet("font-weight:bold;")

    def open_bbl_func(self):
        pass
        

    def closeEvent(self, event):
        print("Closing now")
        #self.video_viewer.destroy_thread()
        event.accept()

    def smooth_changed(self):
        """Smoothness has changed
        """
        smooth_val = self.smooth_slider.value()
        self.smooth_text.setText("Smoothness ({}%):".format(smooth_val))



    def fov_scale_changed(self):
        """Undistort FOV scale changed
        """
        fov_val = self.fov_slider.value() / 10
        self.fov_text.setText("FOV scale ({}):".format(fov_val))

    def update_out_size(self):
        """Update export image size
        """
        #print(self.out_width_control.value())
        pass

    def recompute_stab(self):
        """Update sync and stabilization
        """
        if self.BBL_path == "":
            # GPMF file
            gyro_orientation_text = self.gyro_control.text().lower().strip()
            if gyro_orientation_text not in ["hero6", "hero8"]:
                self.show_error("{} is not a valid orientation preset, if you can even call it a preset. This will be easier eventually... but you were the one who decided to test alpha software (thanks btw)".format(gyro_orientation_text))
                self.export_button.setEnabled(False)
                return

            is_hero6 = (gyro_orientation_text == "hero6")

            if self.infile_path == "" or self.preset_path == "":
                self.show_error("Hey, looks like you forgot to open a video file and/or camera calibration preset. I guess this button could've been grayed out, but whatever.")
                self.export_button.setEnabled(False)

            # initiate stabilization
            self.stab = GPMFStabilizer(self.infile_path, self.preset_path, hero6=is_hero6) # FPV clip

            smoothness = self.smooth_slider.value() / 100
            fps = self.stab.fps
            num_frames = self.stab.num_frames

            sync1_frame = int(self.sync1_control.value() * fps)
            sync2_frame = int(self.sync2_control.value() * fps)

            OF_slice_length = self.OF_frames_control.value()

            if max(sync1_frame, sync2_frame) + OF_slice_length > num_frames:
                self.show_error("You're trying to analyze frames after the end of video. Video length: {} s, latest allowable sync time: {}".format(num_frames/fps, (num_frames - OF_slice_length-1)/fps))
                return

            print("Starting sync. Smoothness: {}, sync1: {} (frame {}), sync2: {} (frame {}), OF slices of {} frames".format(
                    smoothness, self.sync1_control.value(), sync1_frame, self.sync2_control.value(), sync2_frame, OF_slice_length))

            # Known to work: test_clips/GX016017.MP4", "camera_presets/Hero_7_2.7K_60_4by3_wide.json
            # 5 40

            self.stab.auto_sync_stab(smoothness,sync1_frame, sync2_frame, OF_slice_length)

            self.recompute_stab_button.setText("Recompute sync")

            self.export_button.setEnabled(True)
            self.analyzed = True

        else:
            self.stab = None # TODO: BBL stabilizer


    def export_video(self):
        """Gives save location using filedialog
           and saves video to given location
        """


        
        out_size = (self.out_width_control.value(), self.out_height_control.value())

        if out_size[0] > self.stab.width:
            self.show_error("The given output cropped width ({}) is greater than the video width ({})".format(out_size[0], self.stab.width))
            return
        if out_size[1] > self.stab.height:
            self.show_error("The given output cropped height ({}) is greater than the video height ({})".format(out_size[1], self.stab.height))
            return

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
        filename = QtWidgets.QFileDialog.getSaveFileName(self, "Export video", filter="mp4 (*.mp4);; Quicktime (*.mov)")
        print("Output file: {}".format(filename[0]))

        if len(filename[0]) == 0:
            self.show_error("No output file given")
            return


        self.stab.renderfile(start_time, stop_time, filename[0], out_size = out_size)

        self.stab.release()

    def show_error(self, msg):
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
    # pyinstaller -F gyroflow.py --add-binary C:\Users\elvin\AppData\Local\Programs\Python\Python38\Lib\site-packages\cv2\opencv_videoio_ffmpeg440_64.dll;.
    # -F == one file, -w == no command window