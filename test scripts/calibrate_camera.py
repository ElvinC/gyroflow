import numpy as np
import cv2
import glob

chessboard_size = (9,6)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('calibrationImg/*.jpg')

N_imgs = 0

DIM = False 

for i in range(len(images)):
    img = cv2.imread(images[i])

    # dimension of first image
    if i == 0:
        DIM = img.shape[:2][::-1]

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        N_imgs += 1
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
        scaled = cv2.resize(img, (960,720))
        cv2.imshow('img',scaled)
        cv2.waitKey(500)

cv2.destroyAllWindows()


# convert objpoints and imgpoints to np array https://github.com/opencv/opencv/issues/11085
N_OK = len(imgpoints)
objpoints = np.asarray(objpoints,dtype=np.float64)
objpoints = np.reshape(objpoints, (N_OK, 1, chessboard_size[0]*chessboard_size[1], 3))


imgpoints = np.asarray(imgpoints,dtype=np.float64)
imgpoints = np.reshape(imgpoints, (N_OK, 1, chessboard_size[0]*chessboard_size[1], 2))

# calculate K & D (camera matrix and and distortion coefficients)
K = np.zeros((3, 3))
D = np.zeros((4, 1))


rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imgs)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imgs)]

retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

print(K)

balance = 0


# https://stackoverflow.com/questions/50857278/raspicam-fisheye-calibration-with-opencv
img = cv2.imread('calibrationImg/20200714_163753.jpg')
img_dim = img.shape[:2][::-1]

scaled_K = K * img_dim[0] / DIM[0]
scaled_K[2][2] = 1.0  

new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D,
    img_dim, np.eye(3), balance=balance,fov_scale=1.2)

#new_K[0,0]=new_K[0,0]/2
#new_K[1,1]=new_K[1,1]/2

map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, img_dim, cv2.CV_16SC2)

print(map1[0,0,:])
print(map2[0,0])

undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

#newcameramtx, roi=cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(mtx,dist,(w,h),1,(w,h))

# undistort
#mapx,mapy = cv2.fisheye.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
#dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
cv2.imwrite('calibresult.png',undistorted_img)
