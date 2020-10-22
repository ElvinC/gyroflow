# estimate motion from video for syncing with gyro data.


import numpy as np
import cv2
SMOOTHING_RADIUS=50

def movingAverage(curve, radius): 
	window_size = 2 * radius + 1
	# Define the filter 
	f = np.ones(window_size)/window_size 
	# Add padding to the boundaries 
	curve_pad = np.lib.pad(curve, (radius, radius), 'edge') 
	# Apply convolution 
	curve_smoothed = np.convolve(curve_pad, f, mode='same') 
	# Remove padding 
	curve_smoothed = curve_smoothed[radius:-radius]
	# return smoothed curve
	return curve_smoothed 

def smooth(trajectory): 
	smoothed_trajectory = np.copy(trajectory) 
	# Filter the x, y and angle curves
	for i in range(3):
		smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)
 
	return smoothed_trajectory

def fixBorder(frame):
  s = frame.shape
  # Scale the image 4% without moving the center
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
  frame = cv2.warpAffine(frame, T, (s[1], s[0]))
  return frame

# Read input video

filename = '../test_clips/GX016017.MP4'

cp = cv2.VideoCapture(filename)

# To get number of frames
n_frames = int(cp.get(cv2.CAP_PROP_FRAME_COUNT))

# To check the number of frames in the video
print(n_frames)

width = int(cp.get(cv2.CAP_PROP_FRAME_WIDTH)) 
height = int(cp.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("height", width)
print("height", height)

# get the number of frames per second
fps = cp.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
print(fourcc)

# Try doing 2*width
out = cv2.VideoWriter('hero6stab.MP4',0x7634706d, fps, (width, height))

# read the first frame
_, prev = cp.read()

prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
transforms = np.zeros((n_frames-1, 3), np.float32) 

dxlst = []

for i in range(n_frames-2):
	prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

	succ, curr = cp.read()

	if succ:
		print("Works? {}".format(i))

		curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)


		# Track feature points
		# status = 1. if flow points are found
		# err if flow was not find the error is not defined
		# curr_pts = calculated new positions of input features in the second image
		curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)


		idx = np.where(status==1)[0]
		prev_pts = prev_pts[idx]
		curr_pts = curr_pts[idx]
		assert prev_pts.shape == curr_pts.shape 


		# fullAffine= FAlse will set the degree of freedom to only 5 i.e translation, rotation and scaling
		# try fullAffine = True
		m, inliers = cv2.estimateAffine2D(prev_pts, curr_pts) 

		dx = m[0,2]
		dy = m[1,2]

		

		# Extract rotation angle
		da = np.arctan2(m[1,0], m[0,0])
		dxlst.append(da)
			 
		transforms[i] = [dx,dy,da] 

		prev_gray = curr_gray
	

from matplotlib import pyplot as plt


from scipy import fftpack

# sampling rate
f_s = 23.97

#X = fftpack.fft(dxlst)
#freqs = fftpack.fftfreq(len(dxlst)) * f_s


np.savetxt(filename + "opticalflowH6.csv", transforms, delimiter=",")

#plt.plot(freqs, X)
plt.plot(dxlst)
plt.show()

	#print("Frame: " + str(i) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

# Find the cumulative sum of tranform matrix for each dx,dy and da
trajectory = np.cumsum(transforms, axis=0) 

smoothed_trajectory = smooth(trajectory)
difference = smoothed_trajectory - trajectory
transforms_smooth = transforms + difference

# Reset stream to first frame 
cp.set(cv2.CAP_PROP_POS_FRAMES, 0) 
# Write n_frames-1 transformed frames
for i in range(n_frames-2):
	# Read next frame
	success, frame = cp.read() 
	print(success)
	if success:
		# Extract transformations from the new transformation array
		dx = transforms_smooth[i,0]
		dy = transforms_smooth[i,1]
		da = transforms_smooth[i,2]

		# Reconstruct transformation matrix accordingly to new values
		m = np.zeros((2,3), np.float32)
		m[0,0] = np.cos(da)
		m[0,1] = -np.sin(da)
		m[1,0] = np.sin(da)
		m[1,1] = np.cos(da)
		m[0,2] = dx
		m[1,2] = dy

		# Apply affine wrapping to the given frame
		frame_stabilized = cv2.warpAffine(frame, m, (width,height))

		# Fix border artifacts
		frame_stabilized = fixBorder(frame_stabilized) 

		# Write the frame to the file
		frame_out = cv2.hconcat([frame, frame_stabilized])

		# If the image is too big, resize it.
		if(frame_out.shape[1] > 1920): 
			frame_out = cv2.resize(frame_out, (int(frame_out.shape[1]/2), int(frame_out.shape[0]/2)));
		
		cv2.imshow("Before and After", frame_stabilized)
		cv2.waitKey(10)


		out.write(frame_out)

# Release video
cp.release()
out.release()
# Close windows
cv2.destroyAllWindows()