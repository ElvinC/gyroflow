# Gyroflow - Video stabilization using gyroscope data targeting drone footage

A program build around Python, OpenCV, and PySide2 for using gyroscope data for video stabilization of action cameras. 

The project consists of two core parts: A utility for the generation of lens undistortion preset and a utility for stabilizing footage using gyro data.

This is very much a work in progress project, but the goal is to use the gyro data logged on drone flight controllers for stabilizing the onboard HD camera. Furthermore, the gyro data embedded in GoPro cameras should also be usable for stabilization purposes.