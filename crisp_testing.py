from GPMF_gyro import Extractor

import os
import sys
import argparse

import numpy as np

import crisp
from crisp.l3g4200d import post_process_L3G4200D_data

# Note: currently uses a modified crisp package with support for the OpenCV Fisheye module
import crisp.rotations
from crisp.calibration import PARAM_ORDER
import crisp.videoslice
import cv2
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


video_file_path = "test_clips/chessboard.mp4"

testgyro = Extractor(video_file_path)

gyro_data = testgyro.get_gyro()
gyro_rate = testgyro.gyro_rate
print("Rate: {}".format(gyro_rate))
fps = testgyro.fps
img_size = testgyro.size


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

CAMERA_DIST_CENTER = (0.00291108,  0.00041897)
CAMERA_DIST_PARAM = 0.8894355
CAMERA_FRAME_RATE = fps
CAMERA_IMAGE_SIZE = img_size
CAMERA_READOUT = 0.01
GYRO_RATE_GUESS = 200 #  201.36990694913803

print(gyro_rate)
Agyro_rate = 397.78157803740754
Atime_offset = 0.33307650572986625
Agbias_x = 0.016650717817938553
Agbias_y = -0.0021448905951827954
Agbias_z = 0.021890920496554694
Arot_x = 0.17130769254650785
Arot_y = -2.209085060212493
Arot_z = 2.0331306863346486



def to_rot_matrix(r):
    "Convert combined axis angle vector to rotation matrix"
    theta = np.linalg.norm(r)
    v = r/theta
    R = crisp.rotations.axis_angle_to_rotation_matrix(v, theta)
    return R

if __name__ == "__main__":

    camera = crisp.OpenCVFisheyeCameraModel(CAMERA_IMAGE_SIZE, CAMERA_FRAME_RATE, CAMERA_READOUT, CAMERA_MATRIX,
                                   CAMERA_DIST_COEFS)

    #camera = crisp.AtanCameraModel(CAMERA_IMAGE_SIZE, CAMERA_FRAME_RATE, CAMERA_READOUT, CAMERA_MATRIX,
    #                               CAMERA_DIST_CENTER, CAMERA_DIST_PARAM)

    print('Creating video stream from {}'.format(video_file_path))
    video = crisp.VideoStream.from_file(camera, video_file_path)


    print('Creating gyro stream from {}'.format("test_clips/walk_gyro.csv"))
    #gyro = crisp.GyroStream.from_csv("test_clips/walk_gyro.csv")
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
