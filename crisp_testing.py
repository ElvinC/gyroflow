from GPMF_gyro import Extractor

import os
import sys
import argparse

import numpy as np

import crisp
from crisp.l3g4200d import post_process_L3G4200D_data
import crisp.rotations
from crisp.calibration import PARAM_ORDER
import crisp.videoslice

video_file_path = "test_clips/hero6.mp4"

testgyro = Extractor(video_file_path)

gyro_data = testgyro.get_gyro()
gyro_rate = testgyro.gyro_rate
fps = testgyro.fps
img_size = testgyro.size

print(img_size)

"""
This is an example script that shows how to run the calibrator on our dataset.
The dataset can be found here:
    http://www.cvl.isy.liu.se/research/datasets/gopro-gyro-dataset/
To run, simply point the script to one of the video files in the directory
    $ python gopro_gyro_dataset_example.py /path/to/dataset/video.MP4

    Original code by Hannes Ovrén
"""


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

# Scale to match resolution
CAMERA_MATRIX *= img_size[0] / 1920
CAMERA_MATRIX[2][2] = 1.0


CAMERA_DIST_COEFS = [
            0.01945104325838463,
            0.1093842438193295,
            -0.10977045532092518,
            0.037924531473717875
        ]
CAMERA_FRAME_RATE = fps
CAMERA_IMAGE_SIZE = img_size
CAMERA_READOUT = 0.0316734
GYRO_RATE_GUESS = gyro_rate



class OpenCVFishCameraModel(crisp.CameraModel):
    """OpenCV camera model
    This implements the camera model as defined in OpenCV.
    For details, see the OpenCV documentation.
    """
    def __init__(self, image_size, frame_rate, readout, camera_matrix, dist_coefs):
        """Create camera model
        Parameters
        -------------------
        image_size : tuple (rows, columns)
            The size of the image in pixels
        frame_rate : float
            The frame rate of the camera
        readout : float
            Rolling shutter readout time. Set to 0 for global shutter cameras.
        camera_matrix : (3, 3) ndarray
            The internal camera calibration matrix
        dist_coefs : ndarray
            Distortion coefficients [k1, k2, p1, p2 [,k3 [,k4, k5, k6]] of 4, 5, or 8 elements.
            Can be set to None to use zero parameters
        """
        super(OpenCVFishCameraModel, self).__init__(image_size, frame_rate, readout)
        self.camera_matrix = camera_matrix
        self.inv_camera_matrix = np.linalg.inv(self.camera_matrix)
        self.dist_coefs = dist_coefs

    def project(self, points):
        """Project 3D points to image coordinates.
        This projects 3D points expressed in the camera coordinate system to image points.
        Parameters
        --------------------
        points : (3, N) ndarray
            3D points
        Returns
        --------------------
        image_points : (2, N) ndarray
            The world points projected to the image plane
        """
        rvec = tvec = np.zeros(3)
        image_points, jac = cv2.fisheye.projectPoints(points.T.reshape(-1,1,3), rvec, tvec, self.camera_matrix, self.dist_coefs)
        return image_points.reshape(-1,2).T

    def unproject(self, image_points):
        """Find (up to scale) 3D coordinate of an image point
        This is the inverse of the `project` function.
        The resulting 3D points are only valid up to an unknown scale.
        Parameters
        ----------------------
        image_points : (2, N) ndarray
            Image points
        Returns
        ----------------------
        points : (3, N) ndarray
            3D coordinates (valid up to scale)
        """
        undist_image_points = cv2.fisheye.undistortPoints(image_points.T.reshape(1,-1,2), self.camera_matrix, self.dist_coefs, P=self.camera_matrix)
        world_points = np.dot(self.inv_camera_matrix, to_homogeneous(undist_image_points.reshape(-1,2).T))
        return world_points
    
    @classmethod
    def from_hdf(cls, filepath):
        import h5py
        with h5py.File(filepath, 'r') as f:
            dist_coef = f["dist_coef"].value
            K = f["K"].value
            readout = f["readout"].value
            image_size = f["size"].value
            fps = f["fps"].value
            instance = cls(image_size, fps, readout, K, dist_coef)
            return instance


def to_homogeneous(X):
    if X.ndim == 1:
        return np.append(X, 1)
    else:
        _, N = X.shape
        Y = np.ones((3, N))
        return np.vstack((X, np.ones((N, ))))

def from_homogeneous(X):
    Y = X / X[2]
    return Y[:2]


def to_rot_matrix(r):
    "Convert combined axis angle vector to rotation matrix"
    theta = np.linalg.norm(r)
    v = r/theta
    R = crisp.rotations.axis_angle_to_rotation_matrix(v, theta)
    return R

if __name__ == "__main__":

    camera = OpenCVFishCameraModel(CAMERA_IMAGE_SIZE, CAMERA_FRAME_RATE, CAMERA_READOUT, CAMERA_MATRIX,
                                   CAMERA_DIST_COEFS)

    print('Creating video stream from {}'.format("test_clips/hero5.mp4"))
    video = crisp.VideoStream.from_file(camera, "test_clips/hero5.MP4")

    # Problem with creating videoslices
    slices = crisp.videoslice.Slice.from_stream_randomly(video)
    print(slices)


    print('Creating gyro stream from {}'.format("Gopro gpmf"))
    gyro = crisp.GyroStream.from_data(gyro_data)

    print('Post processing L3G4200D gyroscope data to remove frequency spike noise')
    gyro.data = post_process_L3G4200D_data(gyro.data.T).T

    print('Creating calibrator')
    calibrator = crisp.AutoCalibrator(video, gyro)

    print('Estimating time offset and camera to gyroscope rotation. Guessing gyro rate = {:.2f}'.format(GYRO_RATE_GUESS))
    try:
        calibrator.initialize(gyro_rate=GYRO_RATE_GUESS)
    except crisp.InitializationError as e:
        print('Initialization failed. Reason "{}"'.format(e.message))
        sys.exit(-1)

    print('Running calibration. This can take a few minutes.')
    try:
        calibrator.calibrate()
        calibrator.print_params()
    except crisp.CalibrationError as e:
        print('Calibration failed. Reason "{}"'.format(e.message))
        sys.exit(-2)

