# Gyroflow - Video stabilization using gyroscope data targeting drone footage (WIP)

A program built around Python, OpenCV, and PySide2 for video stabilization using gyroscope data. 

The project consists of three core parts: A utility for the generation of lens undistortion preset, a utility for stabilizing footage using gyro data, and a utility for stretching 4:3 video to 16:9 using non-linear horizontal stretching (similar to GoPro's superview). Only the last part (sorta) works as of right now.

This is very much a work in progress project, but the goal is to use the gyro data logged on drone flight controllers for stabilizing the onboard HD camera. Furthermore, the gyro data embedded in newer GoPro cameras should also be usable for stabilization purposes.

### Status

Working:
* Videoplayer based on OpenCV and Pyside2
* Gyro integration using quaternions
* Non-linear stretch utility
* Basic video import/export
* Camera calibration utility with preset import/export


Not working (yet):
* GoPro/blackbox data import
* Symmetrical quaternion low-pass filter
* Camera rotation perspective transform
* Automatic gyro/video sync
* Stabilization UI