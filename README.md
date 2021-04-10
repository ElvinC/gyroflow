# Gyroflow - Video stabilization using gyroscope data targeting drone footage (WIP)

## [In-depth video guide](https://youtu.be/NFsTb_f7y8s)

## [Website with more info](http://gyroflow.xyz/)

A program built around Python, OpenCV, and PySide2 for video stabilization using gyroscope data.

The project consists of three core parts: A utility for the generation of lens undistortion preset, a utility for stabilizing footage using gyro data, and a utility for stretching 4:3 video to 16:9 using non-linear horizontal stretching (similar to GoPro's superview).

The main goal of creating something that can stabilize footage based on blackbox gyro data has mostly been achieved. Moving forwards, more work will be done to improve the quality of the result.

The launcher containing all the utilities is available by executing `gyroflow.py` if all the dependencies are met. Otherwise a binary can be found over in [releases](https://github.com/ElvinC/gyroflow/releases). Also, check out the wiki where there's some more information about the camera calibration and video stabilization process.

## Other things to check out:
* [BlackboxToGPMF](https://github.com/Cleric-K/BlackboxToGPMF/tree/gui) by Cleric-K and Attilafustos. Tool for adding GoPro metadata and blackbox data to non-GoPro cameras for use with Reelsteady GO. Initial discussion [here](https://github.com/ElvinC/gyroflow/issues/1).
* [blackbox2gpmf](https://github.com/jaromeyer/blackbox2gpmf) by Jaromeyer. Tool for adding blackbox gyro data to Hero 7 files for Reelsteady Go.
* [Discord server](https://discord.gg/Rs4GBPm) maintained by [Nicecrash](https://www.youtube.com/channel/UCl3M972T7GbxnEucYHzZ05g) for discussion about gyroflow, BlackboxToGPMF, blackbox2gpmf and other related projects.
* [FPV Stabilization Tools Facebook group](https://www.facebook.com/groups/fpvtools) maintained by Attilafustos.


## Status

**Sample clips:**
* [Handheld Hero 6 + internal gyro](https://youtu.be/ZhVVRnuuMFc) (Clip by Nicecrash)
* [FPV DJI air unit + blackbox data using BlackboxToGPMF](https://youtu.be/veolYMpaNgQ) (Clip by Vitaliy Zaburnuk)
* [FPV Hero 8 + internal gyro](https://youtu.be/MUwERfNBK6U) (Clip by Kyle Li)
* [FPV Session 5 + blackbox data using BlackboxToGPMF](https://youtu.be/5PkTHkl2GsI) (Clip by [iLLjoy Presents](https://www.youtube.com/channel/UCaIqfSaXAFSGEdW1PNbrIjA))

**Working:**
* Videoplayer based on OpenCV and Pyside2
* Gyro integration using quaternions
* Non-linear stretch utility
* Basic video import/export
* Camera calibration utility with preset import/export
* GoPro metadata import
* Symmetrical slerp-based quaternion low-pass filter
* Blackbox data import
* Undistort and rotation perspective transform
* Semi-automatic temporal gyro/video sync.
* Blackbox orientation handling and wide interval sync.
* Stabilization UI without video player
* (Basic) gyro orientation presets for GoPro cameras


**Work in progress:**
* Documentation
* Improved low-pass filter and more stabilization modes (Time-lapse, separate pitch/yaw/roll smoothness control etc.)


**Not working (yet) and _potential_ future additions and ideas:**
* Incorporate acceleration data in orientation estimation for horizon lock (Probably just a complementary filter. Kalman is overkill but could be fun to learn).
* Automatic camera alignment identification with respect to gyro
* Try different calibration pattern? Asymmetric circles/ChArUco?
* Streamlining/optimizing the image processing pipeline (more hardware acceleration etc.)
* Rolling shutter correction (and determination?) + RS-aware gyro sync.
* Automatic determination of required crop
* Integration with external logger hardware
* Native support for other cameras with internal gyro (insta360, sony A7 series?)
* Support for reading and writing professional video formats (video editor plugin?)
* Audio handling