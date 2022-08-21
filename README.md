# Warning: This is the legacy version, find the project here: https://github.com/gyroflow/gyroflow

# Gyroflow - Video stabilization using gyroscope data targeting drone footage

Join the [Gyroflow Discord server](https://discord.gg/BBJ2UVAr2D) for discussion and support.

## [Website with more info](http://gyroflow.xyz/)

## [In-depth video guide for 0.3.x-beta](https://youtu.be/f4YD5pGmnxM)

A program built around Python, OpenCV, and PySide2 for video stabilization using gyroscope data.

The project consists of three core parts: A utility for the generation of lens undistortion preset, a utility for stabilizing footage using gyro data, and a utility for stretching 4:3 video to 16:9 using non-linear horizontal stretching (similar to GoPro's superview).

The main goal of creating something that can stabilize footage based on blackbox gyro data has mostly been achieved. Moving forwards, more work will be done to improve the quality of the result.

The launcher containing all the utilities is available by executing `gyroflow.py` if all the dependencies are met. Otherwise a binary can be found over in [releases](https://github.com/ElvinC/gyroflow/releases). Also, check out the wiki where there's some more information about the camera calibration and video stabilization process.

## Run using python and Poetry:
Note: Try the dev branch for the newest features.

* Install [poetry](https://python-poetry.org/docs/#installation)
* Clone or download the files from this repo
* Navigate to the folder using a commandline and install dependencies using `poetry install`
* Run the application using `poetry run python gyroflow.py`

## Other things to check out:
* [BlackboxToGPMF](https://github.com/Cleric-K/BlackboxToGPMF/tree/gui) by Cleric-K and Attilafustos. Tool for adding GoPro metadata and blackbox data to non-GoPro cameras for use with Reelsteady GO. Initial discussion [here](https://github.com/ElvinC/gyroflow/issues/1).
* [blackbox2gpmf](https://github.com/jaromeyer/blackbox2gpmf) by Jaromeyer. Tool for adding blackbox gyro data to Hero 7 files for Reelsteady Go.
* [Gyroflow Discord server](https://discord.gg/BBJ2UVAr2D)
* [RSGoBlackbox Discord server](https://discord.gg/Rs4GBPm) maintained by [Nicecrash](https://www.youtube.com/channel/UCl3M972T7GbxnEucYHzZ05g) for discussion about gyroflow, BlackboxToGPMF, blackbox2gpmf and other related projects.
* [FPV Stabilization Tools Facebook group](https://www.facebook.com/groups/fpvtools) maintained by Attilafustos.


## General recording tips
* Use the settings that give the widest possible field of view (more data to work with). For a lot of cameras, this is in the 4:3 aspect ratio.
* If using the main drone flight controller for logging, the camera should be hardmounted.
* If using a secondary logger on the camera or internal camera logging, some soft mounting is preferred to isolate vibrations.


## Status

**Sample clips:**
* [0.2.x-alpha results](https://youtu.be/xkVtbYQnH04)
* [In-depth video guide](https://youtu.be/f4YD5pGmnxM)


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
* Frame blending timelapse/hyperlapse
* Automatic determination of required crop
* Native support for insta360 gyro
* Support for high speed video
* GoPro/Insta360 as gyro logger for cinema cameras.
* Basic prores import/export through ffmpeg
* Modular/improved smoothing methods
* Support for non-square pixel aspect ratios
* Incorporate acceleration data in orientation estimation for horizon lock
* Audio handling
* .gyroflow file with synced data
