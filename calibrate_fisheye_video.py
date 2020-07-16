import json
from datetime import date
import numpy as np
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename

__version__ = "0.1"

class FisheyeCalibrator:
    """Class for calculating camera matrix and distortion coefficients
       from images or videoframes
       Mostly based on https://stackoverflow.com/a/50876130
       9x6 chessboard by default: https://raw.githubusercontent.com/opencv/opencv/master/doc/pattern.png
    """
    def __init__(self, chessboard_size=(9,6)):

        self.chessboard_size = chessboard_size

        # termination criteria
        self.subpix_criteria = (cv2.TERM_CRITERIA_EPS + 
                                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.calib_criteria = (cv2.TERM_CRITERIA_EPS +
                               cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

        self.calibration_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
                                  cv2.fisheye.CALIB_CHECK_COND +
                                  cv2.fisheye.CALIB_FIX_SKEW)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1],3), np.float32)
        self.objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.

        # num images loaded
        self.num_images = 0

        # num images used in last calibration
        self.num_images_used = 0

        self.first_image_processed = False
        self.calib_dimension = np.array([0, 0])

        # K & D (camera matrix and and distortion coefficients)
        self.K = np.zeros((3, 3))
        self.D = np.zeros((4, 1))

        # RMS error in pixels. Should be <1 after successful calibration
        self.RMS_error = 100

        self.data_from_preset_file = False



    def add_calib_image(self, img):
        """Add chessboard image for calibration

        Args:
            img (np.ndarray): Image or video frame

        Returns:
            (bool, string, np.ndarray): (success, status message, corners)
        """

        if self.data_from_preset_file:
            raise Exception("Preset already loaded from file")


        if not self.first_image_processed:
            # save the dimensions of the first image [width, height]
            self.calib_dimension = img.shape[:2][::-1]


        # check image dimension
        if img.shape[:2][::-1] != self.calib_dimension:
            return (False, "Image dimension doesn't match previous samples", None)


        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

        if not ret:
            return (False, "Failed to detect chessboard", None)


        # If found, add object points, image points (after refining them)
        self.num_images += 1
        self.objpoints.append(self.objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.subpix_criteria)
        self.imgpoints.append(corners2)

        # Draw and display the corners
        #img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
        #scaled = cv2.resize(img, (960,720))
        #cv2.imshow('img',scaled)
        #cv2.waitKey(500)

        return (True, "Image processed and added", corners2)

    def compute_calibration(self):
        """Compute camera calibration from loaded images

        Raises:
            Exception: No calibration frames/data

        Returns:
            float: Calibration RMS pixel error. <1 is great
        """

        if self.num_images == 0:
            raise Exception("No calibration data")

        # recompute only if new images added
        if self.num_images_used == self.num_images:
            return self.RMS_error

        num_corners = self.chessboard_size[0]*self.chessboard_size[1]

        temp_objpoints = np.asarray(self.objpoints,dtype=np.float64)
        temp_objpoints = np.reshape(self.objpoints, (self.num_images, 1, num_corners, 3))

        temp_imgpoints = np.asarray(self.imgpoints,dtype=np.float64)
        temp_imgpoints = np.reshape(self.imgpoints, (self.num_images, 1, num_corners, 2))

        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(self.num_images)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(self.num_images)]

        retval, self.K, self.D, rvecs, tvecs = cv2.fisheye.calibrate(temp_objpoints,
                temp_imgpoints,
                self.calib_dimension,
                self.K,
                self.D,
                rvecs,
                tvecs,
                self.calibration_flags,
                self.calib_criteria)

        self.RMS_error = retval
        self.num_images_used = self.num_images

        return self.RMS_error


    def get_camera_matrix(self):
        """Get camera matrix from calibration

        Returns:
            np.ndarray: Camera matrix (K)
        """
        self.compute_calibration()

        return self.K

    def get_distortion_coefficients(self):
        """Get distortion coefficients from calibration

        Returns:
            np.ndarray: distortion coefficients (D)
        """
        self.compute_calibration()

        return self.D

    def get_rms_error(self):
        """Get the calibration rms error

        Returns:
            float: Calibration RMS pixel error. should be <1.
        """
        return self.compute_calibration()


    def undistort_image(self, img, fov_scale=1.0):
        """Undistort image using the fisheye camera model in OpenCV

        Args:
            img (np.ndarray): Input image
            fov_scale (float, optional): Virtual camera focal length divider. Defaults to 1.

        Returns:
            np.ndarray: Undistorted image
        """

        self.compute_calibration()

        img_dim = img.shape[:2][::-1]

        scaled_K = self.K * img_dim[0] / self.calib_dimension[0]
        scaled_K[2][2] = 1.0

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, self.D,
                img_dim, np.eye(3), fov_scale=fov_scale)

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, self.D, np.eye(3), new_K, img_dim, cv2.CV_16SC2)


        undistorted_image = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)

        return undistorted_image

    def save_calibration_json(self, filename="calibration.json", calib_name="Camera name", note=""):
        """Save camera calibration parameters as JSON file

        Args:
            filename (str, optional): Path and name of file. Defaults to "calibration.json".
            calib_name (str, optional): Calibration name in file. Defaults to "Camera name".
            note (str, optional): Extra note, calibration setup, calibrator name etc.
        """

        self.compute_calibration()
        
        calibration_data = {
            "name": calib_name,
            "note": note,
            "calibrator_version": __version__,
            "date": str(date.today()),
            
            "calib_dimension": {
                "w": self.calib_dimension[0],
                "h": self.calib_dimension[1]
            },

            "num_images": self.num_images_used,

            "use_opencv_fisheye": True,
            "fisheye_params": {
                "RMS_error": self.RMS_error,
                "camera_matrix": self.K.tolist(),
                "distortion_coeffs": self.D.flatten().tolist()
            },
            # For (potential) use with the standard cv2.calibrateCamera
            "use_opencv_standard": False,
            "calib_params": {}
        }

        with open(filename, 'w') as outfile:
            json.dump(
            calibration_data,
            outfile,
            indent=4,
            separators=(',', ': ')
        )


    def load_calibration_json(self, filename, printinfo = False):
        """Load calibration preset from JSON file

        Args:
            filename (string): path and filename to load
            printinfo (bool, optional): Print extra info from preset file. Defaults to False.
        """
        with open(filename, "r") as infile:
            presets = json.load(infile)

            try:
                if not presets["use_opencv_fisheye"]:
                    raise Exception("Preset not for OpenCV fisheye lens model")

                self.data_from_preset_file = True

                if printinfo:
                    print("Preset name: {}".format(presets["name"]))
                    print("Note: {}".format(presets["note"]))
                    print("Made with {} frames using calibrator version {} on date {}"
                        .format(presets["num_images"],
                                presets["calibrator_version"],
                                presets["date"]))

                if presets["calibrator_version"] != __version__:
                    print("Warning: Versions don't match. Calibrator: {}, Preset: {}"
                        .format(__version__, presets["calibrator_version"]))

                self.calib_dimension[0] = presets["calib_dimension"]["w"]
                self.calib_dimension[1] = presets["calib_dimension"]["h"]
                self.num_images = self.num_images_used = presets["num_images"]

                self.RMS_error = presets["fisheye_params"]["RMS_error"]
                self.K = np.array(presets["fisheye_params"]["camera_matrix"])
                self.D = np.array(presets["fisheye_params"]["distortion_coeffs"])
            except KeyError:
                raise KeyError("Error loading preset file")


    def load_calibration_prompt(self, printinfo = False):
        """Trigger file browser to load calibration preset

        Args:
            printinfo (bool, optional): Print extra info from preset file. Defaults to False.
        """

        Tk().withdraw() # hide root window
        # file browser prompt
        filename = askopenfilename(title = "Select calibration preset file",
                                   filetypes = (("JSON files","*.json"),))

        self.load_calibration_json(filename, printinfo)

    def undistort_image_prompt(self, fov_scale=1):
        """Trigger file browser to load and undistort image

        Args:
            fov_scale (float, optional): Virtual camera focal length divider. Defaults to 1.
        """
        self.compute_calibration()

        Tk().withdraw()

        filename = askopenfilename(title = "Select image to undistort",
                                   filetypes = (("jpeg images","*.jpg"),))

        raw_img = cv2.imread(filename)

        undistorted_img = self.undistort_image(raw_img, fov_scale)

        scaled = cv2.resize(undistorted_img, (960,720))
        cv2.imshow('img',scaled)
        cv2.waitKey(0)



            


if __name__ == "__main__":

    # test undistort code using images
    import glob
    #chessboard_size = (9,6)
    #images = glob.glob('calibrationImg/*.jpg')

    

    calibrator = FisheyeCalibrator()
    calibrator.load_calibration_prompt()
    calibrator.undistort_image_prompt()

    #for imagepath in images:
    #    image = cv2.imread(imagepath)

    #    calibrator.add_calib_image(image)


    #calibrator.save_calibration_json("lgg6_wide.json", "LG G6 4:3 wide angle", "Calibrated by yours truly")
