import cv2
import numpy as np
import time
from tqdm import tqdm
from calibrate_video import FisheyeCalibrator

from scipy.spatial.transform import Rotation


def optical_flow(videofile, lens_preset):
    undistort = FisheyeCalibrator()

    undistort.load_calibration_json(lens_preset, True)
    frame_times = []
    frame_idx = []
    transforms = []
    prev_pts_lst = []
    curr_pts_lst = []

    cap = cv2.VideoCapture(videofile)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    time.sleep(0.05)

    ret, prev = cap.read()

    # if file cant be read return with huge error
    if not ret:
        print("Can't read this part of the file")
        return 0, 999999, [], []

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    height, width, c = prev.shape

    for i in tqdm(range(num_frames), desc=f"Analyzing frames", colour="blue"):
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        succ, curr = cap.read()
        frame_id = i
        frame_time = (cap.get(cv2.CAP_PROP_POS_MSEC)/1000)


        if succ:
            # Only add if succeeded
            frame_idx.append(frame_id)
            frame_times.append(frame_time)


            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

            idx = np.where(status==1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]
            assert prev_pts.shape == curr_pts.shape

            prev_pts_lst.append(prev_pts)
            curr_pts_lst.append(curr_pts)


            # TODO: Try getting undistort + homography working for more accurate rotation estimation
            src_pts = undistort.undistort_points(prev_pts, new_img_dim=(width, height))
            dst_pts = undistort.undistort_points(curr_pts, new_img_dim=(width, height))
            # src_pts = prev_pts_lst
            # dst_pts = curr_pts_lst

            filtered_src = []
            filtered_dst = []
            for i in range(src_pts.shape[0]):
                # if both points are within frame
                if (0 < src_pts[i,0,0] < width) and (0 < dst_pts[i,0,0] < width) and (0 < src_pts[i,0,1] < height) and (0 < dst_pts[i,0,1] < height):
                    filtered_src.append(src_pts[i,:])
                    filtered_dst.append(dst_pts[i,:])

            # rots contains for solutions for the rotation. Get one with smallest magnitude.
            # https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
            # https://en.wikipedia.org/wiki/Essential_matrix#Extracting_rotation_and_translation
            roteul = None
            smallest_mag = 1000

            try:
                R1, R2, t = undistort.recover_pose(np.array(filtered_src), np.array(filtered_dst), new_img_dim=(width, height))

                rot1 = Rotation.from_matrix(R1)
                rot2 = Rotation.from_matrix(R2)

                if rot1.magnitude() < rot2.magnitude():
                    roteul = rot1.as_rotvec() #rot1.as_euler("xyz")
                else:
                    roteul = rot2.as_rotvec() # as_euler("xyz")
            except:
                print("Couldn't recover motion for this frame")
                roteul = np.array([0, 0, 0])

            prev_gray = curr_gray

            transforms.append(list(roteul))


        else:
            print("Skipped frame {}".format(i))

    transforms = np.array(transforms)

    return transforms
import pandas as pd
import os
import gyrolog
import matplotlib.pyplot as plt
video_file = r"D:\Cloud\git\gyroflow\OneR_1inch_gyro_samples\OneR_1inch_gyro_samples\Orentation_display_BACK\PRO_VID_20211102_143001_00_051.mp4"
video_file = r"D:\Cloud\git\gyroflow\OneR_1inch_gyro_samples\OneR_1inch_gyro_samples\Orientation_display_FRONT\PRO_VID_20211102_143237_10_053.mp4"
lens_preset = r"D:\Cloud\git\gyroflow\OneR_1inch_gyro_samples\Insta360_OneR_1inch_PRO_4K_30fps_16by9.json"
transform_file = video_file + ".transform.csv"
if not os.path.isfile(transform_file):
    transforms = optical_flow(video_file, lens_preset)
    df = pd.DataFrame(transforms, columns=['x', 'y', 'z'])
    df.to_csv(transform_file)
else:
    df = pd.read_csv(transform_file, index_col=0)

df = df[df.x.abs() < 0.2]
log_guess, log_type, variant = gyrolog.guess_log_type_from_video(video_file)
print(variant)
log_reader = gyrolog.get_log_reader_by_name(log_type)
log_reader.set_variant("insta360 oner one-inch")
log_reader.set_pre_filter(50)
success = log_reader.extract_log(video_file)
if success:
    gyro = log_reader.standard_gyro
else:
    print("Failed to read gyro!")
fps = 29.97
fig, axes = plt.subplots(3, 2)
axes[0, 0].plot(gyro[:, 0], gyro[:, 1])
axes[1, 0].plot(gyro[:, 0], gyro[:, 2])
axes[2, 0].plot(gyro[:, 0], gyro[:, 3])
# axes[0, 0].plot(df.index / fps, df.x * fps)
# axes[1, 0].plot(df.index / fps, df.y * fps)
# axes[2, 0].plot(df.index / fps, df.z * fps)
axes[0, 1].plot(df.index / fps, df.x * fps)
axes[1, 1].plot(df.index / fps, df.y * fps)
axes[2, 1].plot(df.index / fps, df.z * fps)
plt.show()