import json
from datetime import date
import numpy as np
import cv2
from _version import __version__
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from scipy.spatial.transform import Rotation

import sys

# https://www.imatest.com/support/docs/pre-5-2/geometric-calibration/projective-camera
def inverse_cam_mtx(K):
    # inverse for zero skew case
    if K.shape != (3,3):
        raise ValueError("Not 3x3 matrix")
    
    fx = K[0,0]
    fy = K[1,1]
    px = K[0,2]
    py = K[1,2]

    Kinv = np.array([[fy, 0,  -px*fy],
                     [0,  fx, -py*fx],
                     [0,  0,  fx*fy]])

    Kinv /= fx * fy

    return Kinv

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

        self.calibration_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + # cv2.fisheye.CALIB_CHECK_COND +
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

    def remove_calib_image(self):
        """Remove last added calibration image
        """
        if self.num_images > 0:
            self.objpoints.pop(-1)
            self.imgpoints.pop(-1)
            self.num_images -= 1

    def compute_calibration(self, center_camera=True):
        """Compute camera calibration from loaded images

        Args:
            center_camera (bool): center camera matrix after calib.

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

        if center_camera:
            self.K[0,2] = self.calib_dimension[0]/2
            self.K[1,2] = self.calib_dimension[1]/2

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

    def get_inverse_camera_matrix(self):
        self.compute_calibration

        return inverse_cam_mtx(self.K)

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

        self.new_K = new_K

        #print("FOV BEFORE: {}".format(scaled_K[0,0]))
        #print("FOV EFTER: {}".format(new_K[0,0]))

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, self.D, np.eye(3), new_K, img_dim, cv2.CV_16SC2)


        undistorted_image = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)

        return undistorted_image

    def get_maps(self, fov_scale = 1.0, new_img_dim = None, update_new_K = True):
        """Get undistortion maps

        Args:
            fov_scale (float, optional): Virtual camera focal length divider. Defaults to 1.
            new_img_dim (tuple, optional): Dimension of new image

        Returns:
            (np.ndarray,np.ndarray): Undistortion maps
        """
        
        img_dim = new_img_dim if new_img_dim else self.calib_dimension

        scaled_K = self.K * img_dim[0] / self.calib_dimension[0]
        scaled_K[2][2] = 1.0

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, self.D,
                img_dim, np.eye(3), fov_scale=fov_scale)

        if update_new_K:
            self.new_K = new_K

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, self.D, np.eye(3), new_K, img_dim, cv2.CV_16SC2)

        return map1, map2


    def undistort_points(self, distorted_points,new_img_dim = None):
        img_dim = new_img_dim if new_img_dim else self.calib_dimension

        scaled_K = self.K * img_dim[0] / self.calib_dimension[0]
        scaled_K[2][2] = 1.0


        return cv2.fisheye.undistortPoints(distorted_points, scaled_K, self.D, None, scaled_K)

    def decompose_homography(self, H, new_img_dim = None):
        img_dim = new_img_dim if new_img_dim else self.calib_dimension

        scaled_K = self.K * img_dim[0] / self.calib_dimension[0]
        scaled_K[2][2] = 1.0
        return cv2.decomposeHomographyMat(H, scaled_K)


    def recover_pose(self, pts1, pts2, new_img_dim = None):
        # https://answers.opencv.org/question/31421/opencv-3-essentialmatrix-and-recoverpose/
        # Find essential matrix from fundamental matrix
        img_dim = new_img_dim if new_img_dim else self.calib_dimension
        scaled_K = self.K * img_dim[0] / self.calib_dimension[0]
        scaled_K[2][2] = 1.0

        E, mask = cv2.findEssentialMat(pts1, pts2, scaled_K, cv2.RANSAC, 0.999, 0.1) # cv2.LMEDS or cv2.RANSAC
        #retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, scaled_K)
        R1, R2, t = cv2.decomposeEssentialMat(E) 

        return R1, R2, t

    def get_rotation_map(self, img, quat):
        """Get maps for doing perspective rotations
        
            WORK IN PROGRESS. Currently for testing
        """

        # https://stackoverflow.com/a/12293128
        # https://en.wikipedia.org/wiki/Homography_(computer_vision)

        rotXval = 0
        rotYval = 0
        rotZval = 0

        rotX = (rotXval)*np.pi/180
        rotY = (rotYval)*np.pi/180
        rotZ = (rotZval)*np.pi/180
        rot_mat = np.eye(4)

        
        #print(Rotation([quat[0,1],quat[0,2],quat[0,3],quat[0,0]]).as_euler('xyz'))
        quat = quat.flatten()
        eul = Rotation([quat[1],quat[2],quat[3],quat[0]]).as_euler('xyz')

        combined_rotation = np.eye(4)
        #combined_rotation[0:3,0:3] = Rotation.from_euler('xyz', [eul[0], eul[1], -eul[2]], degrees=False).as_matrix()
        combined_rotation[0:3,0:3] = Rotation([-quat[1],-quat[2],quat[3],-quat[0]]).as_matrix()
        #eul = Rotation(quat).as_euler('xyz')[0]

        #rot1 = np.eye(4)
        #rot1[0:3,0:3] = Rotation.from_euler('xyz', [0, -eul[1], 0], degrees=False).as_matrix() #

        #rot2 = np.eye(4)
        #rot2[0:3,0:3] = Rotation.from_euler('xyz', [eul[2], 0, 0], degrees=False).as_matrix()

        #rot3 = np.eye(4)
        #rot3[0:3,0:3] = Rotation.from_euler('xyz', [0, 0, eul[0]], degrees=False).as_matrix()
        
        #combined_rotation = np.linalg.multi_dot([rot1, rot2, rot3])
        #combined_rotation = Rotation.from_euler('xyz', [-90, -90, -90], degrees=True) * Rotation(quat)

        rot_mat = combined_rotation

        #rot_mat[0:3,0:3], jac = cv2.Rodrigues(np.array([rotX,rotY,rotZ], dtype=np.float32))
        
        #rot_mat[0,1] = 0
        #rot_mat[1,2] = 0
        #rot_mat[2,2] = 1

        img_dim = img.shape[:2][::-1]

        # Scaled 3x4 camera matrix
        K = np.zeros((3,4))
        K[:3,:3] = self.K

        # should make the rotation match fov change
        # Might not work, idk
        K[0,0] = self.new_K[0,0]
        K[1,1] = self.new_K[1,1]

        #print(K)

        
        K *= img_dim[0] / self.calib_dimension[0]

        K[2][2] = 1.0


        # compute inverse camera matrix using scaled K
        Kinv = np.zeros((4,3))
        Kinv[0:3,0:3] = inverse_cam_mtx(K[:3,:3])
        Kinv[3,:] = [0, 0, 1]

        distX = 0
        distY = 0
        distZ = 0
        
        translation = np.array([[1,0,0,distX],
                                [0,1,0,distY],
                                [0,0,1,distZ],
                                [0,0,0,1]])

        
        H = np.linalg.multi_dot([K, rot_mat, Kinv])

        #trans = rot_mat * translation
        #trans[2,2] += self.calib_dimension[1]/2
        
        #transform = self.K * trans
        outimg = cv2.warpPerspective(img,H,(img.shape[1],img.shape[0]))

        return outimg



    def save_calibration_json(self, filename="calibration.json", calib_name="Camera name", camera_brand="", camera_model="", lens_model="", camera_setting="", note="", calibrated_by=""):
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
            "calibrated_by": calibrated_by,
            "camera_brand": camera_brand,
            "camera_model": camera_model,
            "lens_model": lens_model,
            "camera_setting": camera_setting,
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

                width = presets["calib_dimension"]["w"]
                height = presets["calib_dimension"]["h"]

                self.calib_dimension = (width, height)

                self.num_images = self.num_images_used = presets["num_images"]

                self.RMS_error = presets["fisheye_params"]["RMS_error"]
                self.K = np.array(presets["fisheye_params"]["camera_matrix"])
                self.D = np.array(presets["fisheye_params"]["distortion_coeffs"])

                #if presets["calibrator_version"].split(".")[0:1] != ["0","1"]:
                # version 0.1.x doesn't have cam information
                extra_cam_info = {
                    "name": presets.get("calib_name"),
                    "note": presets.get("note"),
                    "calibrated_by": presets.get("calibrated_by", "N/A"),
                    "camera_brand": presets.get("camera_brand", "N/A"),
                    "camera_model": presets.get("camera_model", "N/A"),
                    "lens_model": presets.get("lens_model", "N/A"),
                    "calibrator_version": presets.get("calibrator_version"),
                    "date": presets.get("date"),
                    "width": width,
                    "height": height,
                    "aspect": width/height,
                    "num_images": self.num_images
                }

                return extra_cam_info

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
                                   filetypes = (("jpeg images","*.jpg"),("png images","*.png")))

        raw_img = cv2.imread(filename)

        undistorted_img = self.undistort_image(raw_img, fov_scale=1)


        for i in range(5):

            rotated_img = self.get_rotation_map(undistorted_img,30)
            scaled = cv2.resize(rotated_img, (960,720))
            cv2.imshow('OpenCV image viewer',scaled)
            cv2.waitKey(500)


            rotated_img = self.get_rotation_map(undistorted_img,0)
            scaled = cv2.resize(rotated_img, (960,720))
            cv2.imshow('OpenCV image viewer',scaled)
            cv2.waitKey(500)


class StandardCalibrator:
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

        self.calibration_flags = (cv2.CALIB_SAME_FOCAL_LENGTH +
                                  cv2.CALIB_RATIONAL_MODEL)

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

    def remove_calib_image(self):
        """Remove last added calibration image
        """
        if self.num_images > 0:
            self.objpoints.pop(-1)
            self.imgpoints.pop(-1)
            self.num_images -= 1

    def compute_calibration(self, center_camera=True):
        """Compute camera calibration from loaded images

        Args:
            center_camera (bool): center camera matrix after calib.

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

        retval, self.K, self.D, rvecs, tvecs = cv2.calibrateCamera(temp_objpoints,
                temp_imgpoints,
                self.calib_dimension,
                self.K,
                self.D,
                rvecs,
                tvecs,
                self.calibration_flags,
                self.calib_criteria)

        if center_camera:
            self.K[0,2] = self.calib_dimension[0]/2
            self.K[1,2] = self.calib_dimension[1]/2

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

    def get_inverse_camera_matrix(self):
        self.compute_calibration

        return inverse_cam_mtx(self.K)

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

        self.new_K = new_K

        #print("FOV BEFORE: {}".format(scaled_K[0,0]))
        #print("FOV EFTER: {}".format(new_K[0,0]))

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, self.D, np.eye(3), new_K, img_dim, cv2.CV_16SC2)


        undistorted_image = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)

        return undistorted_image

    def get_maps(self, fov_scale = 1.0, new_img_dim = None):
        """Get undistortion maps

        Args:
            fov_scale (float, optional): Virtual camera focal length divider. Defaults to 1.
            new_img_dim (tuple, optional): Dimension of new image

        Returns:
            (np.ndarray,np.ndarray): Undistortion maps
        """
        
        img_dim = new_img_dim if new_img_dim else self.calib_dimension

        scaled_K = self.K * img_dim[0] / self.calib_dimension[0]
        scaled_K[2][2] = 1.0

        new_K, _ = cv2.getOptimalNewCameraMatrix(scaled_K, self.D,
                img_dim, 1.3, img_dim)


        self.new_K = new_K
        print(new_K)
        

        map1, map2 = cv2.initUndistortRectifyMap(scaled_K, self.D, np.eye(3), new_K, img_dim, cv2.CV_16SC2)

        return map1, map2


    def undistort_points(self, distorted_points,new_img_dim = None):
        img_dim = new_img_dim if new_img_dim else self.calib_dimension

        scaled_K = self.K * img_dim[0] / self.calib_dimension[0]
        scaled_K[2][2] = 1.0


        return cv2.undistortPoints(distorted_points, scaled_K, self.D, None, scaled_K)

    def decompose_homography(self, H, new_img_dim = None):
        img_dim = new_img_dim if new_img_dim else self.calib_dimension

        scaled_K = self.K * img_dim[0] / self.calib_dimension[0]
        scaled_K[2][2] = 1.0
        return cv2.decomposeHomographyMat(H, scaled_K)


    def recover_pose(self, pts1, pts2, new_img_dim = None):
        # https://answers.opencv.org/question/31421/opencv-3-essentialmatrix-and-recoverpose/
        # Find essential matrix from fundamental matrix
        img_dim = new_img_dim if new_img_dim else self.calib_dimension
        scaled_K = self.K * img_dim[0] / self.calib_dimension[0]
        scaled_K[2][2] = 1.0

        E, mask = cv2.findEssentialMat(pts1, pts2, scaled_K, cv2.RANSAC, 0.999, 0.1) # cv2.LMEDS or cv2.RANSAC
        #retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, scaled_K)
        R1, R2, t = cv2.decomposeEssentialMat(E) 

        return R1, R2, t

    def get_rotation_map(self, img, quat):
        """Get maps for doing perspective rotations
        
            WORK IN PROGRESS. Currently for testing
        """

        # https://stackoverflow.com/a/12293128
        # https://en.wikipedia.org/wiki/Homography_(computer_vision)

        rotXval = 0
        rotYval = 0
        rotZval = 0

        rotX = (rotXval)*np.pi/180
        rotY = (rotYval)*np.pi/180
        rotZ = (rotZval)*np.pi/180
        rot_mat = np.eye(4)

        #print(Rotation([quat[0,1],quat[0,2],quat[0,3],quat[0,0]]).as_euler('xyz'))
        quat = quat.flatten()
        #eul = Rotation([quat[1],quat[2],quat[3],quat[0]]).as_euler('xyz')

        combined_rotation = np.eye(4)
        #combined_rotation[0:3,0:3] = Rotation.from_euler('xyz', [eul[0], eul[1], -eul[2]], degrees=False).as_matrix()
        combined_rotation[0:3,0:3] = Rotation([-quat[1],-quat[2],quat[3],-quat[0]]).as_matrix()
        #eul = Rotation(quat).as_euler('xyz')[0]

        #rot1 = np.eye(4)
        #rot1[0:3,0:3] = Rotation.from_euler('xyz', [0, -eul[1], 0], degrees=False).as_matrix() #

        #rot2 = np.eye(4)
        #rot2[0:3,0:3] = Rotation.from_euler('xyz', [eul[2], 0, 0], degrees=False).as_matrix()

        #rot3 = np.eye(4)
        #rot3[0:3,0:3] = Rotation.from_euler('xyz', [0, 0, eul[0]], degrees=False).as_matrix()
        
        #combined_rotation = np.linalg.multi_dot([rot1, rot2, rot3])
        #combined_rotation = Rotation.from_euler('xyz', [-90, -90, -90], degrees=True) * Rotation(quat)

        rot_mat = combined_rotation

        #rot_mat[0:3,0:3], jac = cv2.Rodrigues(np.array([rotX,rotY,rotZ], dtype=np.float32))
        
        #rot_mat[0,1] = 0
        #rot_mat[1,2] = 0
        #rot_mat[2,2] = 1

        img_dim = img.shape[:2][::-1]

        # Scaled 3x4 camera matrix
        K = np.zeros((3,4))
        K[:3,:3] = self.K

        # should make the rotation match fov change
        K[0,0] = self.new_K[0,0]
        K[1,1] = self.new_K[1,1]

        #print(K)

        
        K *= img_dim[0] / self.calib_dimension[0]

        K[2][2] = 1.0


        # compute inverse camera matrix using scaled K
        Kinv = np.zeros((4,3))
        Kinv[0:3,0:3] = inverse_cam_mtx(K[:3,:3])
        Kinv[3,:] = [0, 0, 1]

        distX = 0
        distY = 0
        distZ = 0
        
        translation = np.array([[1,0,0,distX],
                                [0,1,0,distY],
                                [0,0,1,distZ],
                                [0,0,0,1]])

        
        H = np.linalg.multi_dot([K, rot_mat, Kinv])

        #trans = rot_mat * translation
        #trans[2,2] += self.calib_dimension[1]/2
        
        #transform = self.K * trans
        outimg = cv2.warpPerspective(img,H,(img.shape[1],img.shape[0]))

        return outimg



    def save_calibration_json(self, filename="calibration.json", calib_name="Camera name", camera_brand="", camera_model="", lens_model="",  note="", calibrated_by=""):
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
            "calibrated_by": calibrated_by,
            "camera_brand": camera_brand,
            "camera_model": camera_model,
            "lens_model": lens_model,
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

                width = presets["calib_dimension"]["w"]
                height = presets["calib_dimension"]["h"]

                self.calib_dimension = (width, height)

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
                                   filetypes = (("jpeg images","*.jpg"),("png images","*.png")))

        raw_img = cv2.imread(filename)

        undistorted_img = self.undistort_image(raw_img, fov_scale=1)


        for i in range(5):

            rotated_img = self.get_rotation_map(undistorted_img,30)
            scaled = cv2.resize(rotated_img, (960,720))
            cv2.imshow('OpenCV image viewer',scaled)
            cv2.waitKey(500)


            rotated_img = self.get_rotation_map(undistorted_img,0)
            scaled = cv2.resize(rotated_img, (960,720))
            cv2.imshow('OpenCV image viewer',scaled)
            cv2.waitKey(500)



if __name__ == "__main__":

    # test undistort code using images
    #import glob
    #chessboard_size = (9,6)
    #images = glob.glob('calibrationImg/*.jpg')

    
    CAMERA_DIST_COEFS = [
        0.01945104325838463,
        0.1093842438193295,
        -0.10977045532092518,
        0.037924531473717875
    ]

    DIST = np.array(CAMERA_DIST_COEFS)

    CAMERA_MATRIX = np.array(
    [
                [
                    847.6148226238896,
                    0.0,
                    960.0
                ],
                [
                    0.0,
                    852.8260246970873,
                    720.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ]
    )

    calibrator = FisheyeCalibrator()
    calibrator.load_calibration_json("camera_presets/gopro_calib2.JSON")

    image_points = np.arange(101,121).reshape(1,-1,2).astype('float32')
    undistorted_corners = cv2.fisheye.undistortPoints(image_points, CAMERA_MATRIX, DIST)


    print(undistorted_corners)

    #for imagepath in images:
    #    image = cv2.imread(imagepath)

    #    calibrator.add_calib_image(image)


    #calibrator.save_calibration_json("lgg6_wide.json", "LG G6 4:3 wide angle", "Calibrated by yours truly")
