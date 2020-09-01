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

        self.stabilizer_button = QtWidgets.QPushButton("Video Stabilizer")
        self.stabilizer_button.setMinimumSize(300,50)

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
        self.layout.addWidget(self.stretch_button)
        
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
        pass

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

        if (len(path) == 0):
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



def main():
    app = QtWidgets.QApplication([])

    widget = CalibratorUtility()
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