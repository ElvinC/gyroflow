import json
from datetime import date
import numpy as np
import cv2
import glob
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

        self.num_processed_images = 0

        self.first_image_processed = False
        # Equal if no stretching is applied
        self.orig_dimension = np.array([0, 0])
        self.calib_dimension = np.array([0, 0])

        # K & D (camera matrix and and distortion coefficients)
        self.K = np.zeros((3, 3))
        self.D = np.zeros((4, 1))

        # RMS error in pixels. Should be <1 after successful calibration
        self.RMS_error = 100

        # Horizontal stretching factor
        self.input_horizontal_stretch = 1

        self.data_from_preset_file = False

        # when loading a preset file
        self.extra_cam_info = None

    def new_calibration(self):
        self.objpoints = []
        self.imgpoints = []
        self.num_images = 0
        self.num_images_used = 0
        self.num_processed_images = 0

    def set_horizontal_stretch(self, new_stretch = 1):
        # For handling anamorphic or squeezed footage.
        self.input_horizontal_stretch = new_stretch

    def get_stretched_size_from_dimension(self, new_img_dim):
        return (round(new_img_dim[0] * self.calib_dimension[0] / self.orig_dimension[0]), round(new_img_dim[1] * self.calib_dimension[1] / self.orig_dimension[1]))

    def image_is_stretched(self):
        return self.input_horizontal_stretch != 1

    def get_stretched_size(self, img):
        h, w = img.shape[:2]

        if self.input_horizontal_stretch < 0:
            new_h = round(h / self.input_horizontal_stretch)
            new_w = w
        else:
            new_w = round(w * self.input_horizontal_stretch)
            new_h = h

        return (new_w, new_h)

    def stretch_image(self, img):
        if self.input_horizontal_stretch == 1:
            return img

        # 16:9 to 4:3 gives input_horizontal_stretch of
        # (4/3)/(16/9) = 0.75

        h, w = img.shape[:2]

        if self.input_horizontal_stretch < 0:
            new_h = round(h / self.input_horizontal_stretch)
            new_w = w
        else:
            new_w = round(w * self.input_horizontal_stretch)
            new_h = h

        return cv2.resize(img, (new_w, new_h))


    def add_calib_image(self, img):
        """Add chessboard image for calibration

        Args:
            img (np.ndarray): Image or video frame

        Returns:
            (bool, string, np.ndarray): (success, status message, corners)
        """

        if self.data_from_preset_file:
            raise Exception("Preset already loaded from file")


        if self.num_images == 0:
            # save the dimensions of the first image [width, height]
            self.orig_dimension = img.shape[:2][::-1]
            self.calib_dimension = self.get_stretched_size(img)


        # check image dimension
        if img.shape[:2][::-1] != self.orig_dimension:
            return (False, "Image dimension doesn't match previous samples", None)



        gray = cv2.cvtColor(self.stretch_image(img),cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

        if not ret:
            return (False, "Failed to detect chessboard", None)


        # If found, add object points, image points (after refining them)
        self.num_images += 1
        self.objpoints.append(self.objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.subpix_criteria)
        # print(corners2)


        # horizontal scaling

        self.imgpoints.append(corners2)

        # Draw and display the corners
        #img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
        #scaled = cv2.resize(img, (960,720))
        #cv2.imshow('img',scaled)
        #cv2.waitKey(500)

        corners_orig = np.copy(corners2)

        if self.image_is_stretched():
            # Transform back to original image format
            corners_orig[:,:,0] *= self.orig_dimension[0] / self.calib_dimension[0] # x axis
            corners_orig[:,:,1] *= self.orig_dimension[1] / self.calib_dimension[1] # y axis

        return (True, "Image processed and added", corners_orig)

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
        # if self.num_images_used == self.num_images:
        #     return self.RMS_error

        num_corners = self.chessboard_size[0]*self.chessboard_size[1]

        temp_objpoints = np.asarray(self.objpoints,dtype=np.float64)
        temp_objpoints = np.reshape(self.objpoints, (self.num_images, 1, num_corners, 3))

        temp_imgpoints = np.asarray(self.imgpoints,dtype=np.float64)
        temp_imgpoints = np.reshape(self.imgpoints, (self.num_images, 1, num_corners, 2))

        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(self.num_images)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(self.num_images)]

        try:
            retval, self.K, self.D, rvecs, tvecs = cv2.fisheye.calibrate(temp_objpoints,
                    temp_imgpoints,
                    self.calib_dimension,
                    self.K,
                    self.D,
                    rvecs,
                    tvecs,
                    self.calibration_flags,
                    self.calib_criteria)
        except:
            print("Error computing calibration, remove a frame and try again")
            return 100

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
        self.compute_calibration()

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

    def get_maps(self, fov_scale = 1.0, output_dim = None, new_img_dim = None, update_new_K = True, quat = None, focalCenter = None, original_stretched = True):
        """Get undistortion maps

        Args:
            fov_scale (float, optional): Virtual camera focal length divider. Defaults to 1.
            new_img_dim (tuple, optional): Dimension of new image

        Returns:
            (np.ndarray,np.ndarray): Undistortion maps
        """

        if new_img_dim and self.image_is_stretched() and original_stretched:
            # new_img_dim is dimension of unstretched image
            new_img_dim = (round(new_img_dim[0] * self.calib_dimension[0] / self.orig_dimension[0]), round(new_img_dim[1] * self.calib_dimension[1] / self.orig_dimension[1]))

        img_dim = new_img_dim if new_img_dim else self.calib_dimension
        out_dim = output_dim if output_dim else self.calib_dimension
        focalCenter = focalCenter if focalCenter is not None else np.array([self.calib_dimension[0]/2,self.calib_dimension[1]/2])

        R = np.eye(3)

        if type(quat) != type(None):
            quat = quat.flatten()
            #R = Rotation([-quat[1],-quat[2],quat[3],-quat[0]]).as_matrix()
            R = Rotation([quat[1],quat[2],quat[3],quat[0]]).as_matrix()

            R[[0,0,1,2],[1,2,0,0]] *=-1
            #final_rotation = np.eye(3)
            #final_rotation[0,0] = -1
            #R = np.linalg.multi_dot([np.linalg.inv(final_rotation), R, final_rotation])

        img_dim_ratio = img_dim[0] / self.calib_dimension[0]

        scaled_K = self.K * img_dim_ratio
        scaled_K[2][2] = 1.0

        new_K = np.copy(self.K)
        new_K[0][0] = new_K[0][0] * 1.0/fov_scale
        new_K[1][1] = new_K[1][1] * 1.0/fov_scale
        new_K[0][2] = (self.calib_dimension[0]/2 - focalCenter[0])* img_dim_ratio/fov_scale + out_dim[0]/2
        new_K[1][2] = (self.calib_dimension[1]/2 - focalCenter[1])* img_dim_ratio/fov_scale + out_dim[1]/2

        if update_new_K:
            self.new_K = new_K
        
        if original_stretched and self.image_is_stretched():
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, self.D, R, new_K, out_dim, cv2.CV_32FC1)
            # Rescale input and convert to int for speed
            map1, map2 = cv2.convertMaps(map1 * self.orig_dimension[0] / self.calib_dimension[0], map2 * self.orig_dimension[1] / self.calib_dimension[1], cv2.CV_16SC2)
            
            return map1, map2

        else:
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, self.D, R, new_K, out_dim, cv2.CV_16SC2)
            return map1, map2



    def undistort_points(self, distorted_points,new_img_dim = None):
        img_dim = new_img_dim if new_img_dim else self.calib_dimension

        scaled_K = self.K * img_dim[0] / self.calib_dimension[0]
        scaled_K[2][2] = 1.0

        pts = cv2.fisheye.undistortPoints(distorted_points, scaled_K, self.D, None, scaled_K)

        return pts if type(pts) != type(None) else np.array([])

    def decompose_homography(self, H, new_img_dim = None):
        img_dim = new_img_dim if new_img_dim else self.calib_dimension

        scaled_K = self.K * img_dim[0] / self.calib_dimension[0]
        scaled_K[2][2] = 1.0
        return cv2.decomposeHomographyMat(H, scaled_K)


    def recover_pose(self, pts1, pts2, new_img_dim = None):
        """ Find rotation matrices using epipolar geometry
        
        Args:
            pts1 (np.ndarray): Initial points
            pts2 (np.ndarray): Resulting points
            new_img_dim (tuple, optional): New image dimension. Defaults to None.

        Returns:
            [type]: [description]
        """
        # https://answers.opencv.org/question/31421/opencv-3-essentialmatrix-and-recoverpose/
        img_dim = new_img_dim if new_img_dim else self.calib_dimension
        scaled_K = self.K * img_dim[0] / self.calib_dimension[0]
        scaled_K[2][2] = 1.0

        E, mask = cv2.findEssentialMat(pts1, pts2, scaled_K, cv2.RANSAC, 0.999, 0.1) # cv2.LMEDS or cv2.RANSAC
        #retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, scaled_K)
        try:
            R1, R2, t = cv2.decomposeEssentialMat(E)
        except:
            # Can't figure it out, assume no rotation
            return np.eye(3), np.eye(3), np.array([0,0,0])
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

        rot_mat = combined_rotation
        img_dim = img.shape[:2][::-1]

        # Scaled 3x4 camera matrix
        K = np.zeros((3,4))
        K[:3,:3] = self.K

        # should make the rotation match fov change
        K[0,0] = self.new_K[0,0]
        K[1,1] = self.new_K[1,1]

        K *= img_dim[0] / self.calib_dimension[0]

        K[2][2] = 1.0

        # compute inverse camera matrix using scaled K
        Kinv = np.zeros((4,3))
        Kinv[0:3,0:3] = inverse_cam_mtx(K[:3,:3])
        Kinv[3,:] = [0, 0, 1]

        H = np.linalg.multi_dot([K, rot_mat, Kinv])

        #transform = self.K * trans
        outimg = cv2.warpPerspective(img,H,(img.shape[1],img.shape[0]))
        return outimg

    def get_calibration_data(self):


        calibration_data = {
            "name": self.extra_cam_info.get("name", ""),
            "note": self.extra_cam_info.get("note", ""),
            "calibrated_by": self.extra_cam_info.get("calibrated_by", "N/A"),
            "camera_brand": self.extra_cam_info.get("camera_brand", "N/A"),
            "camera_model": self.extra_cam_info.get("camera_model", "N/A"),
            "lens_model": self.extra_cam_info.get("lens_model", "N/A"),
            "camera_setting": self.extra_cam_info.get("camera_setting", "N/A"),
            "calibrator_version": self.extra_cam_info.get("calibrator_version", "N/A"),
            "date": self.extra_cam_info.get("date", "N/A"),
            "calib_dimension": {
                "w": self.calib_dimension[0],
                "h": self.calib_dimension[1]
            },
            "orig_dimension": {
                "w": self.orig_dimension[0],
                "h": self.orig_dimension[1]
            },
            "input_horizontal_stretch": self.input_horizontal_stretch, # to de-stretch anamorphic/linearly stretched video.
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

        return calibration_data

    def load_calibration_data(self, cal_data, printinfo = False):

        try:
            if not cal_data["use_opencv_fisheye"]:
                raise Exception("Preset not for OpenCV fisheye lens model")

            self.data_from_preset_file = True

            if printinfo:
                print("Preset name: {}".format(cal_data["name"]))
                print("Note: {}".format(cal_data["note"]))
                print("Made with {} frames using calibrator version {} on date {}"
                    .format(cal_data["num_images"],
                            cal_data["calibrator_version"],
                            cal_data["date"]))

            if cal_data["calibrator_version"] != __version__:
                print("Note: Versions don't match. Calibrator: {}, Preset: {}. Should be fine though."
                    .format(__version__, cal_data["calibrator_version"]))

            cal_width = cal_data["calib_dimension"]["w"]
            cal_height = cal_data["calib_dimension"]["h"]

            self.calib_dimension = (cal_width, cal_height)

            # Added in 0.3.0
            if "orig_dimension" in cal_data:
                orig_w = cal_data["orig_dimension"]["w"]
                orig_h = cal_data["orig_dimension"]["h"]
                self.input_horizontal_stretch = cal_data["input_horizontal_stretch"]
                self.orig_dimension = (orig_w, orig_h)
            else:
                self.input_horizontal_stretch = 1
                self.orig_dimension = self.calib_dimension


            self.num_images = self.num_images_used = cal_data["num_images"]

            self.RMS_error = cal_data["fisheye_params"]["RMS_error"]
            self.K = np.array(cal_data["fisheye_params"]["camera_matrix"])
            self.D = np.array(cal_data["fisheye_params"]["distortion_coeffs"])

            #if presets["calibrator_version"].split(".")[0:1] != ["0","1"]:
            # version 0.1.x doesn't have cam information

            fixed_name = " ".join(cal_data.get("name").replace("_", " ").split())

            self.extra_cam_info = {
                "name": fixed_name,
                "note": cal_data.get("note"),
                "calibrated_by": cal_data.get("calibrated_by", "N/A"),
                "camera_brand": cal_data.get("camera_brand", "N/A"),
                "camera_model": cal_data.get("camera_model", "N/A"),
                "camera_setting": cal_data.get("camera_setting", "N/A"),
                "lens_model": cal_data.get("lens_model", "N/A"),
                "calibrator_version": cal_data.get("calibrator_version"),
                "date": cal_data.get("date"),
                "width": self.orig_dimension[0],
                "height": self.orig_dimension[1],
                "aspect": self.orig_dimension[0]/self.orig_dimension[1],
                "num_images": self.num_images
            }

        except ZeroDivisionError:
            raise KeyError("Error loading preset file")
        

        return self.extra_cam_info

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
            "orig_dimension": {
                "w": self.orig_dimension[0],
                "h": self.orig_dimension[1]
            },
            "input_horizontal_stretch": self.input_horizontal_stretch, # to de-stretch anamorphic/linearly stretched video.
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

            return self.load_calibration_data(presets, printinfo)


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
                                  cv2.CALIB_RATIONAL_MODEL +
                                  cv2.CALIB_FIX_PRINCIPAL_POINT +
                                  cv2.CALIB_USE_INTRINSIC_GUESS +
                                  cv2.CALIB_TILTED_MODEL)

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
        # if self.num_images_used == self.num_images:
        #     return self.RMS_error

        num_corners = self.chessboard_size[0]*self.chessboard_size[1]

        temp_objpoints = np.asarray(self.objpoints,dtype=np.float64)
        temp_objpoints = np.reshape(self.objpoints, (self.num_images, 1, num_corners, 3))

        temp_imgpoints = np.asarray(self.imgpoints,dtype=np.float64)
        temp_imgpoints = np.reshape(self.imgpoints, (self.num_images, 1, num_corners, 2))

        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(self.num_images)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(self.num_images)]

        self.K = np.array(
        [
                    [
                        1000,
                        0.0,
                        self.calib_dimension[0]/2
                    ],
                    [
                        0.0,
                        1000,
                        self.calib_dimension[1]/2
                    ],
                    [
                        0.0,
                        0.0,
                        1.0
                    ]
                ]
        )

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

        new_K, _ = cv2.getOptimalNewCameraMatrix(scaled_K, self.D,
                img_dim, 1.3, img_dim)

        self.new_K = new_K

        #print("FOV BEFORE: {}".format(scaled_K[0,0]))
        #print("FOV EFTER: {}".format(new_K[0,0]))

        map1, map2 = self.get_maps()


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

        #translation = np.array([[1,0,0,distX],
        #                        [0,1,0,distY],
        #                        [0,0,1,distZ],
        #                        [0,0,0,1]])


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

            "use_opencv_fisheye": False,
            "fisheye_params": {},
            # For (potential) use with the standard cv2.calibrateCamera
            "use_opencv_standard": True,
            "calib_params": {
                "RMS_error": self.RMS_error,
                "camera_matrix": self.K.tolist(),
                "distortion_coeffs": self.D.flatten().tolist()
            }
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
                    print("Warning: Versions don't match. Calibrator: {}, Preset: {}. (Usually not a problem)"
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

def get_all_preset_paths(preset_folder = "camera_presets"):
    files = glob.glob(preset_folder + '/**/*.json', recursive=True)
    #print(files)
    return [f.replace("\\", "/") for f in files if not f.startswith("Legacy")]

if __name__ == "__main__":
    get_all_preset_paths()
    exit()
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
