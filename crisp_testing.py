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
import cv2

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

    camera = crisp.OpenCVFisheyeCameraModel(CAMERA_IMAGE_SIZE, CAMERA_FRAME_RATE, CAMERA_READOUT, CAMERA_MATRIX,
                                   CAMERA_DIST_COEFS)

    print('Creating video stream from {}'.format(video_file_path))
    video = crisp.VideoStream.from_file(camera, video_file_path)

    # Problem with creating videoslices
    #slices = crisp.videoslice.Slice.from_stream_randomly(video, step_bounds=(1, 1), length_bounds=(10,10 ), max_start=None, min_distance=1, min_slice_points=10)
    #print(slices)


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

