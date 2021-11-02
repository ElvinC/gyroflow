import cv2
import numpy as np
import time
import tqdm

from scipy.spatial.transform import Rotation


def optical_flow(videofile):
    frame_times = []
    frame_idx = []
    transforms = []
    prev_pts_lst = []
    curr_pts_lst = []

    cap = cv2.VideoCapture(videofile)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    time.sleep(0.05)

    ret, prev = cap.read()

    # if file cant be read return with huge error
    if not ret:
        print("Can't read this part of the file")
        return 0, 999999, [], []

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
            # src_pts = undistort.undistort_points(prev_pts, new_img_dim=(width, height))
            # dst_pts = undistort.undistort_points(curr_pts, new_img_dim=(width, height))
            src_pts = prev_pts_lst
            dst_pts = curr_pts_lst

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

            transforms.append(list(roteul/num_frames_skipped))


        else:
            print("Skipped frame {}".format(i))

    transforms = np.array(transforms)

    return transforms