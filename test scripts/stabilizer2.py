import numpy as np
import cv2

from calibrate_video import FisheyeCalibrator
from scipy.spatial.transform import Rotation
from gyro_integrator import GyroIntegrator
from blackbox_extract import BlackboxExtractor
from GPMF_gyro import Extractor
from matplotlib import pyplot as plt

from scipy.fftpack import fft,ifft
from scipy.signal import resample
import time




class GPMFStabilizer:
	def __init__(self, videopath, calibrationfile):
		# General video stuff
		self.cap = cv2.VideoCapture(videopath)
		self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.fps = self.cap.get(cv2.CAP_PROP_FPS)
		self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))


		# Camera undistortion stuff
		self.undistort = FisheyeCalibrator()
		self.undistort.load_calibration_json(calibrationfile, True)
		self.map1, self.map2 = self.undistort.get_maps(1.6,new_img_dim=(self.width,self.height))

		# Get gyro data
		self.gpmf = Extractor(videopath)
		self.gyro_data = self.gpmf.get_gyro(True)

		# Hero 6??
		#self.gyro_data[:,1] = self.gyro_data[:,1]
		#self.gyro_data[:,2] = -self.gyro_data[:,2]
		#self.gyro_data[:,3] = self.gyro_data[:,3]

		# Hero 8??
		self.gyro_data[:,[2, 3]] = self.gyro_data[:,[3, 2]]
		#gyro_data[:,1] = gyro_data[:,1]
		#gyro_data[:,2] = -gyro_data[:,2]
		#gyro_data[:,3] = gyro_data[:,3]

		#gyro_data[:,1:] = -gyro_data[:,1:]

		# Other attributes
		initial_orientation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()

		self.integrator = GyroIntegrator(self.gyro_data,initial_orientation=initial_orientation)
		self.integrator.integrate_all()
		self.times = None
		self.stab_transform = None

	
	def stabilization_settings(self, smooth = 0.95):


		v1 = 20 / self.fps
		v2 = 900 / self.fps
		d1 = 0.042
		d2 = -0.396

		err_slope = (d2-d1)/(v2-v1)
		correction_slope = err_slope + 1
		gyro_start = (d1 - err_slope*v1)

		interval = 1/(correction_slope * self.fps)


		print("Start {}".format(gyro_start))

		print("Interval {}, slope {}".format(interval, correction_slope))

		self.times, self.stab_transform = self.integrator.get_interpolated_stab_transform(smooth=smooth,start=-gyro_start,interval = interval) # 2.2/30 , -1/30

	def auto_sync_stab(self, smooth=0.8, sliceframe1 = 10, sliceframe2 = 1000, slicelength = 50):
		v1 = (sliceframe1 + slicelength/2) / self.fps
		v2 = (sliceframe2 + slicelength/2) / self.fps
		d1, times1, transforms1 = self.optical_flow_comparison(sliceframe1, slicelength)
		d2, times2, transforms2 = self.optical_flow_comparison(sliceframe2, slicelength)

		print("v1: {}, v2: {}, d1: {}, d2: {}".format(v1, v2, d1, d2))

		err_slope = (d2-d1)/(v2-v1)
		correction_slope = err_slope + 1
		gyro_start = (d1 - err_slope*v1) - 0.4/self.fps

		interval = 1/(correction_slope * self.fps)

		print("Start {}".format(gyro_start))

		print("Interval {}, slope {}".format(interval, correction_slope))



		plt.plot(times1, transforms1[:,2])
		plt.plot(times2, transforms2[:,2])
		plt.plot((self.integrator.get_raw_data("t") + gyro_start)*correction_slope, self.integrator.get_raw_data("z"))
		plt.plot((self.integrator.get_raw_data("t") + d2), self.integrator.get_raw_data("z"))
		plt.plot((self.integrator.get_raw_data("t") + d1), self.integrator.get_raw_data("z"))
		plt.show()

		self.times, self.stab_transform = self.integrator.get_interpolated_stab_transform(smooth=smooth,start=-gyro_start,interval = interval) # 2.2/30 , -1/30

	def optical_flow_comparison(self, start_frame=0, analyze_length = 50):
		frame_times = []
		frame_idx = []
		transforms = []

		self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

		# Read first frame
		_, prev = self.cap.read()
		prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

		for i in range(analyze_length):
			prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)



			succ, curr = self.cap.read()

			frame_id = (int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
			frame_time = (self.cap.get(cv2.CAP_PROP_POS_MSEC)/1000)

			if i % 10 == 0:
				print("Analyzing frame: {}/{}".format(i,analyze_length))

			if succ:
				# Only add if succeeded
				frame_idx.append(frame_id)
				frame_times.append(frame_time)
				print(frame_time)

				curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
				# Estimate transform using optical flow
				curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

				idx = np.where(status==1)[0]
				prev_pts = prev_pts[idx]
				curr_pts = curr_pts[idx]
				assert prev_pts.shape == curr_pts.shape


				# TODO: Try getting undistort + homography working for more accurate rotation estimation
				#src_pts = undistort.undistort_points(prev_pts, new_img_dim=(int(width),int(height)))
				#dst_pts = undistort.undistort_points(curr_pts, new_img_dim=(int(width),int(height)))
				#H, mask = cv2.findHomography(src_pts, dst_pts)
				#retval, rots, trans, norms = undistort.decompose_homography(H, new_img_dim=(int(width),int(height)))

				m, inliers = cv2.estimateAffine2D(prev_pts, curr_pts) 

				dx = m[0,2]
				dy = m[1,2]
				
				# Extract rotation angle
				da = np.arctan2(m[1,0], m[0,0])
				transforms.append([dx,dy,da]) 
				prev_gray = curr_gray

			else:
				print("Frame {}".format(i))
		
		transforms = np.array(transforms) * self.fps
		estimated_offset = self.estimate_gyro_offset(frame_times, transforms)
		return estimated_offset, frame_times, transforms

		# Test stuff 
		v1 = 20 / self.fps
		v2 = 1300 / self.fps
		d1 = 0.042
		d2 = -0.604

		err_slope = (d2-d1)/(v2-v1)
		correction_slope = err_slope + 1
		gyro_start = (d1 - err_slope*v1)

		interval = correction_slope * 1/self.fps

		plt.plot(frame_times, transforms[:,2])
		plt.plot((self.integrator.get_raw_data("t") + gyro_start)* correction_slope, self.integrator.get_raw_data("z"))
		plt.show()

	def estimate_gyro_offset(self, OF_times, OF_transforms):
		# Estimate offset between small optical flow slice and gyro data

		gyro_times = self.integrator.get_raw_data("t")
		gyro_roll = self.integrator.get_raw_data("z")
		OF_roll = OF_transforms[:,2]

		costs = []
		offsets = []

		for i in range(800):
			offset = 0.8 - i/500
			cost = self.gyro_cost_func(OF_times, OF_roll, gyro_times + offset, gyro_roll)
			offsets.append(offset)
			costs.append(cost)

		final_offset = offsets[np.argmin(costs)]

		print("Estimated offset: {}".format(final_offset))


		plt.plot(offsets, costs)
		plt.show()

		return final_offset

	def gyro_cost_func(self, OF_times, OF_roll, gyro_times, gyro_roll):
		sum_squared_diff = 0
		gyro_idx = 0

		for OF_idx in range(len(OF_times)):
			while gyro_times[gyro_idx] < OF_times[OF_idx]:
				gyro_idx += 1

			diff = gyro_roll[gyro_idx] - OF_roll[OF_idx]
			sum_squared_diff += diff ** 2
			#print("Gyro {}, OF {}".format(gyro_times[gyro_idx], OF_times[OF_idx]))

		#print("DIFF^2: {}".format(sum_squared_diff))

		#plt.plot(OF_times, OF_roll)
		#plt.plot(gyro_times, gyro_roll)
		#plt.show()
		return sum_squared_diff

	def renderfile(self, outpath = "Stabilized.mp4", out_size = (1920,1080)):

		out = cv2.VideoWriter(outpath, -1, 29.97, (1920*2,1080))
		crop = (int((self.width-out_size[0])/2), int((self.height-out_size[1])/2))

		self.cap.set(cv2.CAP_PROP_POS_FRAMES, 21*30)

		i = 0
		while(True):
			# Read next frame
			success, frame = self.cap.read() 

			frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
			print("FRAME: {}, IDX: {}".format(frame_num, i))

			if success:
				i +=1

			if i > 1000:
				break

			if success and i > 0:

				frame_undistort = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR,
		                                      borderMode=cv2.BORDER_CONSTANT)


				frame_out = self.undistort.get_rotation_map(frame_undistort, self.stab_transform[frame_num-1])

				# Fix border artifacts
				frame_out = frame_out[crop[1]:crop[1]+out_size[1], crop[0]:crop[0]+out_size[0]]
				frame_undistort = frame_undistort[crop[1]:crop[1]+out_size[1], crop[0]:crop[0]+out_size[0]]


				#out.write(frame_out)
				#print(frame_out.shape)

				# If the image is too big, resize it.
			#%if(frame_out.shape[1] > 1920): 
			#		frame_out = cv2.resize(frame_out, (int(frame_out.shape[1]/2), int(frame_out.shape[0]/2)));
				
				size = np.array(frame_out.shape)
				frame_out = cv2.resize(frame_out, (int(size[1]), int(size[0])))

				frame = cv2.resize(frame_undistort, ((int(size[1]), int(size[0]))))
				concatted = cv2.resize(cv2.hconcat([frame_out,frame],2), (1920*2,1080))
				out.write(concatted)
				cv2.imshow("Before and After", concatted)
				cv2.waitKey(5)

		# When everything done, release the capture
		out.release()

	def release(self):
		self.cap.release()



if __name__ == "__main__":
	#stab = GPMFStabilizer("test_clips/GX016017.MP4", "camera_presets/Hero_7_2.7K_60_4by3_wide.json")
	stab = GPMFStabilizer("test_clips/GX010010.MP4", "camera_presets/Hero_7_2.7K_60_4by3_wide.json")
	#stab.stabilization_settings(smooth = 0.8)
	stab.auto_sync_stab(0.86,24*30, 120 * 30, 70)
	#stab.optical_flow_comparison(start_frame=1300, analyze_length = 50)
	stab.renderfile("parkinglot_stab.mp4",out_size = (2560,1440))
	stab.release()

	# 20 / self.fps: 0.042
	# 200 / self.fps: -0.048


exit()



# Other testing code, plz ignore:

cap = cv2.VideoCapture("test_clips/GX016017.MP4")


width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

outpath = "hero6testing3.mp4"


out_size = (1920,1080)

crop_start = (int((width-out_size[0])/2), int((height-out_size[1])/2))


out = cv2.VideoWriter(outpath, -1, 29.97, (1920*2,1080) )


undistort = FisheyeCalibrator()

undistort.load_calibration_json("camera_presets/Hero_7_2.7K_60_4by3_wide.json", True)


map1, map2 = undistort.get_maps(1.6,new_img_dim=(int(width),int(height)))
# Here 

gpmf = Extractor("test_clips/GX016017.MP4")

#bb = BlackboxExtractor("test_clips/GX015563.MP4_emuf_004.bbl")

#gyro_data = bb.get_gyro_data()
gyro_data = gpmf.get_gyro(True)





#gyro_data[:,[2, 3]] = gyro_data[:,[3, 2]]
gyro_data[:,1] = gyro_data[:,1]
gyro_data[:,2] = -gyro_data[:,2]
gyro_data[:,3] = gyro_data[:,3]

#gyro_data[:,1:] = -gyro_data[:,1:]




initial_orientation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()

integrator = GyroIntegrator(gyro_data,initial_orientation=initial_orientation)
integrator.integrate_all()

v1 = 3.4
v2 = 6.179
g1 = 3.4
g2 = 6.263

gyroslope = (g2-g1)/(v2-v1)

gStart = g1 - gyroslope*v1

interval = gyroslope * 1/59.94

print("Start {}".format(gStart))

print("Interval {}, slope {}".format(interval, gyroslope))

times, stab_transform = integrator.get_interpolated_stab_transform(smooth=0.999,start=-3.2/60,interval = interval) # 2.2/30 , -1/30



#cap = cv2.VideoCapture(inpath)


# read the first frame
_, prev = cap.read()

prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

NUM = 0

transforms = np.zeros((NUM, 3), np.float32) 

dxlst = []
time_lst = []

for i in range(NUM):
	prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)


	succ, curr = cap.read()

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
		src_pts = undistort.undistort_points(prev_pts, new_img_dim=(int(width),int(height)))
		dst_pts = undistort.undistort_points(curr_pts, new_img_dim=(int(width),int(height)))
		H, mask = cv2.findHomography(src_pts, dst_pts) 

		# https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga7f60bdff78833d1e3fd6d9d0fd538d92

		retval, rots, trans, norms = undistort.decompose_homography(H, new_img_dim=(int(width),int(height)))

		# rots contains for solutions for the rotation. Get one with smallest magnitude. Idk
		roteul = None
		smallest_mag = 1000
		for rot in rots:
			thisrot = Rotation.from_matrix(rots[0])
			if thisrot.magnitude() < smallest_mag and thisrot.magnitude() < 0.3:
				roteul = Rotation.from_matrix(rot).as_euler("xyz") * 59.94
				smallest_mag = thisrot.magnitude()

		print("STOPPP")
		

		# Extract rotation angle
		#da = np.arctan2(m[1,0], m[0,0])
		#dxlst.append(da)
			 
		m, inliers = cv2.estimateAffine2D(prev_pts, curr_pts) 

		dx = m[0,2]
		dy = m[1,2]

		

		# Extract rotation angle
		da = np.arctan2(m[1,0], m[0,0])
		dxlst.append(da)
			 
		transforms[i] = [dx,dy,da] 

		#if smallest_mag < 0.3:
		#	transforms[i] = [roteul[0],roteul[1],roteul[2]]
		#else:
		#	transforms[i] = [0,0,0] 

		prev_gray = curr_gray


#plt.plot(np.arange(transforms.shape[0]) * 1/29.97,transforms[:,0])
#plt.plot(np.arange(transforms.shape[0]) * 1/29.97,transforms[:,1])
#plt.plot(np.arange(transforms.shape[0]) * 1/29.97,transforms[:,2])

transforms = np.genfromtxt('test_clips/GX016017.MP4' + "opticalflowH6.csv", delimiter=',')

mysample = transforms[0:100,2] * 59.94

N = 0
resampled = resample(mysample, 400)
sampleB = integrator.get_raw_data("z")[N:400+N]


testcorr = np.correlate(sampleB, resampled, "full")
plt.plot(testcorr)
print(np.argmax(testcorr) - 400)

maxcorr = np.argmax(testcorr) - 400

#plt.plot(resampled)
#plt.plot(integrator.get_raw_data("z")[0:400])

plt.plot(np.arange(mysample.shape[0])*interval -3.1/60,mysample)

#plt.figure()

#plt.plot(integrator.get_raw_data("t"), integrator.get_raw_data("z"))
plt.show()


i = 0
while(True):
	# Read next frame
	success, frame = cap.read() 
	if success:
		i +=1

	if i > 1300:
		break

	if success and i > 0:

		frame_undistort = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)



		# Apply affine wrapping to the given frame
		frame_out = undistort.get_rotation_map(frame_undistort, stab_transform[i])

		# Fix border artifacts
		frame_out = frame_out[crop_start[1]:crop_start[1]+out_size[1], crop_start[0]:crop_start[0]+out_size[0]]
		#frame_out = frame_stabilized
		frame_undistort = frame_undistort[crop_start[1]:crop_start[1]+out_size[1], crop_start[0]:crop_start[0]+out_size[0]]


		#out.write(frame_out)
		#print(frame_out.shape)

		# If the image is too big, resize it.
	#%if(frame_out.shape[1] > 1920): 
	#		frame_out = cv2.resize(frame_out, (int(frame_out.shape[1]/2), int(frame_out.shape[0]/2)));
		
		size = np.array(frame_out.shape)
		frame_out = cv2.resize(frame_out, (int(size[1]), int(size[0])))

		frame = cv2.resize(frame_undistort, ((int(size[1]), int(size[0]))))
		concatted = cv2.resize(cv2.hconcat([frame_out,frame],2), (1920*2,1080))
		out.write(concatted)
		cv2.imshow("Before and After", concatted)
		cv2.waitKey(10)

# When everything done, release the capture
cap.release()
out.release()

cv2.destroyAllWindows()




