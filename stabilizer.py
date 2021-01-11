import numpy as np
import cv2
import csv

from calibrate_video import FisheyeCalibrator
from scipy.spatial.transform import Rotation
from gyro_integrator import GyroIntegrator, FrameRotationIntegrator
from blackbox_extract import BlackboxExtractor
from GPMF_gyro import Extractor
from matplotlib import pyplot as plt


from scipy.fftpack import fft,ifft
from scipy.signal import resample
import time


class Stabilizer:
    def auto_sync_stab(self, smooth=0.8, sliceframe1 = 10, sliceframe2 = 1000, slicelength = 50):
        v1 = (sliceframe1 + slicelength/2) / self.fps
        v2 = (sliceframe2 + slicelength/2) / self.fps
        d1, times1, transforms1 = self.optical_flow_comparison(sliceframe1, slicelength)
        #self.initial_offset = d1
        d2, times2, transforms2 = self.optical_flow_comparison(sliceframe2, slicelength)

        self.times1 = times1
        self.times2 = times2
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.v1 = v1
        self.v2 = v2
        self.d1 = d1
        self.d2 = d2


        print("v1: {}, v2: {}, d1: {}, d2: {}".format(v1, v2, d1, d2))

        err_slope = (d2-d1)/(v2-v1)
        correction_slope = err_slope + 1
        gyro_start = (d1 - err_slope*v1)#  + 1.5/self.fps

        interval = 1/(correction_slope * self.fps)

        print("Start {}".format(gyro_start))

        print("Interval {}, slope {}".format(interval, correction_slope))



        plt.plot(times1, transforms1[:,2] * self.fps)
        plt.plot(times2, transforms2[:,2] * self.fps)
        plt.plot((self.integrator.get_raw_data("t") + gyro_start)*correction_slope, self.integrator.get_raw_data("z"))
        #plt.plot((self.integrator.get_raw_data("t") + d2), self.integrator.get_raw_data("z"))
        #plt.plot((self.integrator.get_raw_data("t") + d1), self.integrator.get_raw_data("z"))
        plt.figure()
        plt.plot(times1, -transforms1[:,0] * self.fps)
        plt.plot(times2, -transforms2[:,0] * self.fps)
        plt.plot((self.integrator.get_raw_data("t") + gyro_start)*correction_slope, self.integrator.get_raw_data("x"))
        
        plt.figure()
        
        plt.plot(times1, -transforms1[:,1] * self.fps)
        plt.plot(times2, -transforms2[:,1] * self.fps)
        plt.plot((self.integrator.get_raw_data("t") + gyro_start)*correction_slope, self.integrator.get_raw_data("y"))

        plt.show()

        # Temp new integrator with corrected time scale

        initial_orientation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()

        new_gyro_data = self.gyro_data

        # Correct time scale
        new_gyro_data[:,0] = (new_gyro_data[:,0]+gyro_start) *correction_slope

        new_integrator = GyroIntegrator(new_gyro_data,zero_out_time=True, initial_orientation=initial_orientation)
        new_integrator.integrate_all()

        # Doesn't work for BBL for some reason. TODO: Figure out why
        #self.times, self.stab_transform = new_integrator.get_interpolated_stab_transform(smooth=smooth,start=0,interval = 1/self.fps)

        self.times, self.stab_transform = self.integrator.get_interpolated_stab_transform(smooth=smooth,start=-gyro_start,interval = interval)

    def manual_sync_correction(self, d1, d2, smooth=0.8):
        v1 = self.v1
        v2 = self.v2

        transforms1 = self.transforms1
        transforms2 = self.transforms2
        times1 = self.times1
        times2 = self.times2

        print("v1: {}, v2: {}, d1: {}, d2: {}".format(v1, v2, d1, d2))



        err_slope = (d2-d1)/(v2-v1)
        correction_slope = err_slope + 1
        gyro_start = (d1 - err_slope*v1)#  + 1.5/self.fps

        interval = 1/(correction_slope * self.fps)

        print("Start {}".format(gyro_start))

        print("Interval {}, slope {}".format(interval, correction_slope))



        plt.plot(times1, transforms1[:,2] * self.fps)
        plt.plot(times2, transforms2[:,2] * self.fps)
        plt.plot((self.integrator.get_raw_data("t") + gyro_start)*correction_slope, self.integrator.get_raw_data("z"))
        #plt.plot((self.integrator.get_raw_data("t") + d2), self.integrator.get_raw_data("z"))
        #plt.plot((self.integrator.get_raw_data("t") + d1), self.integrator.get_raw_data("z"))
        plt.figure()
        plt.plot(times1, -transforms1[:,0] * self.fps)
        plt.plot(times2, -transforms2[:,0] * self.fps)
        plt.plot((self.integrator.get_raw_data("t") + gyro_start)*correction_slope, self.integrator.get_raw_data("x"))
        
        plt.figure()
        
        plt.plot(times1, -transforms1[:,1] * self.fps)
        plt.plot(times2, -transforms2[:,1] * self.fps)
        plt.plot((self.integrator.get_raw_data("t") + gyro_start)*correction_slope, self.integrator.get_raw_data("y"))

        plt.show()

        # Temp new integrator with corrected time scale

        initial_orientation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()

        new_gyro_data = self.gyro_data

        # Correct time scale
        new_gyro_data[:,0] = (new_gyro_data[:,0]+gyro_start) *correction_slope

        new_integrator = GyroIntegrator(new_gyro_data,zero_out_time=False, initial_orientation=initial_orientation)
        new_integrator.integrate_all()

        # Doesn't work for BBL for some reason. TODO: Figure out why
        #self.times, self.stab_transform = new_integrator.get_interpolated_stab_transform(smooth=smooth,start=0,interval = 1/self.fps)

        self.times, self.stab_transform = self.integrator.get_interpolated_stab_transform(smooth=smooth,start=-gyro_start,interval = interval)




    def optical_flow_comparison(self, start_frame=0, analyze_length = 50):
        frame_times = []
        frame_idx = []
        transforms = []
        prev_pts_lst = []
        curr_pts_lst = []

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        time.sleep(0.05)

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


                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                # Estimate transform using optical flow
                curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

                idx = np.where(status==1)[0]
                prev_pts = prev_pts[idx]
                curr_pts = curr_pts[idx]
                assert prev_pts.shape == curr_pts.shape

                prev_pts_lst.append(prev_pts)
                curr_pts_lst.append(curr_pts)


                # TODO: Try getting undistort + homography working for more accurate rotation estimation
                src_pts = self.undistort.undistort_points(prev_pts, new_img_dim=(self.width,self.height))
                dst_pts = self.undistort.undistort_points(curr_pts, new_img_dim=(self.width,self.height))

                filtered_src = []
                filtered_dst = []

                for i in range(src_pts.shape[0]):
                    # if both points are within frame
                    if (0 < src_pts[i,0,0] < self.width) and (0 < dst_pts[i,0,0] < self.width) and (0 < src_pts[i,0,1] < self.height) and (0 < dst_pts[i,0,1] < self.height):
                        filtered_src.append(src_pts[i,:])
                        filtered_dst.append(dst_pts[i,:])



                H, mask = cv2.findHomography(np.array(filtered_src), np.array(filtered_dst))
                retval, rots, trans, norms = self.undistort.decompose_homography(H, new_img_dim=(self.width,self.height))


                # rots contains for solutions for the rotation. Get one with smallest magnitude. Idk
                # TODO: Implement rotation determination using essential matrix instead:
                # https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
                # https://en.wikipedia.org/wiki/Essential_matrix#Extracting_rotation_and_translation
                roteul = None
                smallest_mag = 1000
                for rot in rots:
                    thisrot = Rotation.from_matrix(rots[0]) # First one?
                    #thisrot = Rotation.from_matrix(rot)
                    if thisrot.magnitude() < smallest_mag and thisrot.magnitude() < 0.6:
                        # For some reason some camera calibrations lead to super high rotation magnitudes... Still testing.
                        roteul = Rotation.from_matrix(rot).as_euler("xyz")
                        smallest_mag = thisrot.magnitude()

                if type(roteul) == type(None):
                    print("Optical flow rotation determination failed")
                    roteul = [0, 0, 0]

                # Compute fundamental matrix
                #F, mask = cv2.findFundamentalMat(np.array(filtered_src), np.array(filtered_dst),cv2.FM_LMEDS)
                # Compute essential matrix

                # https://answers.opencv.org/question/206817/extract-rotation-and-translation-from-fundamental-matrix/
                #E = self.undistort.find_essential_matrix(F, new_img_dim=(self.width,self.height))

                self.use_essential_matrix = True

                if self.use_essential_matrix:
                    R1, R2, t = self.undistort.recover_pose(np.array(filtered_src), np.array(filtered_dst), new_img_dim=(self.width,self.height))
                
                    rot1 = Rotation.from_matrix(R1)
                    rot2 = Rotation.from_matrix(R2)

                    if rot1.magnitude() < rot2.magnitude():
                        roteul = rot1.as_euler("xyz")
                    else: 
                        roteul = rot2.as_euler("xyz")


                #w, u, vt = cv2.SVDecomp(E) # , flag = cv2.SVD.FULL_UV
                
                #W = np.array([[0, -1.0, 0],[1.0, 0, 0],[0, 0, 1.0]])

                #U_W_Vt = np.linalg.multi_dot([u, W, vt])
                #U_Wt_Vt = np.linalg.multi_dot([u, W.transpose(), vt]) # Rotation matrix?
                #
                

                #points_drawn = curr

                #for point in curr_pts:
                #    print(point)
                #    cv2.circle(points_drawn,tuple(point[0]),1,(0,0,255))

                #for point in dst_pts:
                #    #print(point)
                #    cv2.circle(points_drawn,tuple(point[0]),1,(255,0,0))

                #cv2.imshow("Dot test", points_drawn)
                #cv2.waitKey(300)

                m, inliers = cv2.estimateAffine2D(src_pts, dst_pts) 

                dx = m[0,2]
                dy = m[1,2]
                
                # Extract rotation angle
                da = np.arctan2(m[1,0], m[0,0])
                #transforms.append([dx,dy,da]) 
                transforms.append(list(roteul))
                prev_gray = curr_gray

            else:
                print("Frame {}".format(i))
        
        transforms = np.array(transforms)
        estimated_offset = self.estimate_gyro_offset(frame_times, transforms, prev_pts_lst, curr_pts_lst)
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

    def estimate_gyro_offset(self, OF_times, OF_transforms, prev_pts_list, curr_pts_list):
        #print(prev_pts_list)
        # Estimate offset between small optical flow slice and gyro data

        gyro_times = self.integrator.get_raw_data("t")
        gyro_data = self.integrator.get_raw_data("xyz")
        #print(gyro_data)

        # quick low pass filter
        self.frame_lowpass = True

        if self.frame_lowpass:
            params = [0.3,0.4,0.3] # weights. last frame, current frame, next frame
            new_OF_transforms = np.copy(OF_transforms)
            for i in range(1,new_OF_transforms.shape[0]-1):
                new_OF_transforms[i,:] = new_OF_transforms[i-1,:] * params[0] + new_OF_transforms[i,:]*params[1] + new_OF_transforms[i+1,:] * params[2]

            OF_transforms = new_OF_transforms

        costs = []
        offsets = []

        N = 600
        dt = 4

        for i in range(N):
            offset = dt/2 - i * (dt/N) + self.initial_offset
            cost = self.better_gyro_cost_func(OF_times, OF_transforms, gyro_times + offset, gyro_data)
            offsets.append(offset)
            costs.append(cost)

        slice_length = len(OF_times)
        cutting_ratio = 1
        new_slice_length = int(slice_length*cutting_ratio)

        start_idx = int((slice_length - new_slice_length)/2)

        OF_times = OF_times[start_idx:start_idx + new_slice_length]
        OF_transforms = OF_transforms[start_idx:start_idx + new_slice_length,:]

        rough_offset = offsets[np.argmin(costs)]

        print("Estimated offset: {}".format(rough_offset))


        plt.plot(offsets, costs)
        plt.show()

        costs = []
        offsets = []

        N = 800
        dt = 0.15

        for i in range(N):
            offset = dt/2 - i * (dt/N) + rough_offset
            cost = self.better_gyro_cost_func(OF_times, OF_transforms, gyro_times + offset, gyro_data)
            offsets.append(offset)
            costs.append(cost)

        better_offset = offsets[np.argmin(costs)]

        print("Better offset: {}".format(better_offset))

        plt.plot(offsets, costs)
        plt.show()

        return better_offset

    def gyro_cost_func(self, OF_times, OF_transforms, gyro_times, gyro_data):

        # Estimate time delay using only roll direction

        gyro_roll = gyro_data[:,2] * self.fps
        OF_roll = OF_transforms[:,2]



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

    def better_gyro_cost_func(self, OF_times, OF_transforms, gyro_times, gyro_data):


        new_OF_transforms = np.copy(OF_transforms) * self.fps
        new_OF_transforms[:,0] = -new_OF_transforms[:,0]
        new_OF_transforms[:,1] = -new_OF_transforms[:,1]

        #gyro_x = gyro_data[:,0]
        #OF_x = -OF_transforms[:,0]

        #gyro_y = gyro_data[:,1]
        #OF_y = -OF_transforms[:,1]

        #gyro_z = gyro_data[:,2]
        #OF_z = OF_transforms[:,2]

        axes_weight = np.array([0.5,1,1]) #np.array([0.5,0.5,1]) # Weight of the xyz in the cost function. More weight to roll

        sum_squared_diff = 0
        gyro_idx = 1

        next_gyro_snip = np.array([0, 0, 0], dtype=np.float64)
        next_cumulative_time = 0

        while gyro_times[gyro_idx + 1] < OF_times[0]:
            gyro_idx += 1

        for OF_idx in range(len(OF_times)):			
            cumulative = next_gyro_snip
            cumulative_time =  next_cumulative_time

            while gyro_times[gyro_idx] < OF_times[OF_idx]:
                delta_time = gyro_times[gyro_idx] - gyro_times[gyro_idx-1]
                cumulative_time += delta_time

                cumulative += gyro_data[gyro_idx,:] * delta_time
                gyro_idx += 1

            time_delta = OF_times[OF_idx] - gyro_times[gyro_idx-2]
            time_weight = time_delta / (gyro_times[gyro_idx] - gyro_times[gyro_idx-1])
            cumulative += gyro_data[gyro_idx-1,:] * time_delta
            cumulative_time  += time_delta

            time_delta = gyro_times[gyro_idx-1] - OF_times[OF_idx]
            next_gyro_snip = gyro_data[gyro_idx-1,:] * time_delta
            next_cumulative_time = time_delta

            cumulative /= cumulative_time

            diff = cumulative - new_OF_transforms[OF_idx,:]
            sum_squared_diff += np.sum(np.multiply(diff ** 2, axes_weight))
            #print("Gyro {}, OF {}".format(gyro_times[gyro_idx], OF_times[OF_idx]))

        #print("DIFF^2: {}".format(sum_squared_diff))

        #plt.plot(OF_times, OF_roll)
        #plt.plot(gyro_times, gyro_roll)
        #plt.show()
        return sum_squared_diff

    def renderfile(self, starttime, stoptime, outpath = "Stabilized.mp4", out_size = (1920,1080), split_screen = True):

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(outpath, fourcc, self.fps, (out_size[0]*2 if split_screen else out_size[0], out_size[1]))
        crop = (int((self.width-out_size[0])/2), int((self.height-out_size[1])/2))


        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(starttime * self.fps))
        time.sleep(0.1)

        num_frames = int((stoptime - starttime) * self.fps) 


        i = 0
        while(True):
            # Read next frame
            frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            success, frame = self.cap.read() 

            
            print("FRAME: {}, IDX: {}".format(frame_num, i))

            if success:
                i +=1

            if i > num_frames or i == len(self.stab_transform):
                break

            if success and i > 0:
                


                frame_undistort = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT)


                #print(self.stab_transform[frame_num])
                frame_out = self.undistort.get_rotation_map(frame_undistort, self.stab_transform[frame_num])

                #frame_out = self.undistort.get_rotation_map(frame, self.stab_transform[frame_num])


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
                if split_screen:
                    concatted = cv2.resize(cv2.hconcat([frame_out,frame],2), (out_size[0]*2,out_size[1]))
                    out.write(concatted)
                    cv2.imshow("Before and After", concatted)
                else:
                    out.write(frame_out)
                    cv2.imshow("Stabilized?", frame_out)
                cv2.waitKey(2)

        # When everything done, release the capture
        out.release()
        cv2.destroyAllWindows()

    def release(self):
        self.cap.release()


class GPMFStabilizer(Stabilizer):
    def __init__(self, videopath, calibrationfile, hero = 8, fov_scale = 1.6):
        # General video stuff
        self.undistort_fov_scale = fov_scale
        self.cap = cv2.VideoCapture(videopath)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))


        # Camera undistortion stuff
        self.undistort = FisheyeCalibrator()
        self.undistort.load_calibration_json(calibrationfile, True)
        self.map1, self.map2 = self.undistort.get_maps(self.undistort_fov_scale,new_img_dim=(self.width,self.height))

        # Get gyro data
        self.gpmf = Extractor(videopath)
        self.gyro_data = self.gpmf.get_gyro(True)

        # Hero 6??
        if hero == 6:
            self.gyro_data[:,1] = self.gyro_data[:,1]
            self.gyro_data[:,2] = -self.gyro_data[:,2]
            self.gyro_data[:,3] = self.gyro_data[:,3]
        elif hero == 5:
            self.gyro_data[:,1] = -self.gyro_data[:,1]
            self.gyro_data[:,2] = self.gyro_data[:,2]
            self.gyro_data[:,3] = -self.gyro_data[:,3]
            self.gyro_data[:,[2, 3]] = self.gyro_data[:,[3, 2]]

        elif hero == 8:
            # Hero 8??
            self.gyro_data[:,[2, 3]] = self.gyro_data[:,[3, 2]]

        

        #gyro_data[:,1] = gyro_data[:,1]
        #gyro_data[:,2] = -gyro_data[:,2]
        #gyro_data[:,3] = gyro_data[:,3]

        #gyro_data[:,1:] = -gyro_data[:,1:]

        #points_test = np.array([[[0,0],[100,100],[200,300],[400,400]]], dtype = np.float32)
        #result = self.undistort.undistort_points(points_test, new_img_dim=(self.width,self.height))
        #print(result)
        #exit()

        # Other attributes
        initial_orientation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()

        self.integrator = GyroIntegrator(self.gyro_data,initial_orientation=initial_orientation)
        self.integrator.integrate_all()
        self.times = None
        self.stab_transform = None


        self.initial_offset = 0

    
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





class InstaStabilizer(Stabilizer):
    def __init__(self, videopath, calibrationfile, gyrocsv, fov_scale = 1.6):
        # General video stuff
        self.undistort_fov_scale = fov_scale
        self.cap = cv2.VideoCapture(videopath)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))


        # Camera undistortion stuff
        self.undistort = FisheyeCalibrator()
        self.undistort.load_calibration_json(calibrationfile, True)
        self.map1, self.map2 = self.undistort.get_maps(self.undistort_fov_scale,new_img_dim=(self.width,self.height))

        # Get gyro data

        self.gyro_data = self.instaCSVGyro(gyrocsv)
        hero = 6

        # Hero 6??
        if hero == 6:
            self.gyro_data[:,1] = self.gyro_data[:,1]
            self.gyro_data[:,2] = -self.gyro_data[:,2]
            self.gyro_data[:,3] = self.gyro_data[:,3]
        elif hero == 5:
            self.gyro_data[:,1] = -self.gyro_data[:,1]
            self.gyro_data[:,2] = self.gyro_data[:,2]
            self.gyro_data[:,3] = -self.gyro_data[:,3]
            self.gyro_data[:,[2, 3]] = self.gyro_data[:,[3, 2]]

        elif hero == 8:
            # Hero 8??
            self.gyro_data[:,[2, 3]] = self.gyro_data[:,[3, 2]]

        

        #gyro_data[:,1] = gyro_data[:,1]
        #gyro_data[:,2] = -gyro_data[:,2]
        #gyro_data[:,3] = gyro_data[:,3]

        #gyro_data[:,1:] = -gyro_data[:,1:]

        #points_test = np.array([[[0,0],[100,100],[200,300],[400,400]]], dtype = np.float32)
        #result = self.undistort.undistort_points(points_test, new_img_dim=(self.width,self.height))
        #print(result)
        #exit()

        # Other attributes
        initial_orientation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()

        self.integrator = GyroIntegrator(self.gyro_data,zero_out_time=False,initial_orientation=initial_orientation)
        self.integrator.integrate_all()
        self.times = None
        self.stab_transform = None


        self.initial_offset = 0

    def instaCSVGyro(self, csvfile):
        gyrodata = []
        with open(csvfile) as f:
            reader = csv.reader(f, delimiter=",", quotechar='"')
            next(reader, None)
            for row in reader:
                gyro = [float(row[0])] + [float(val) for val in row[2].split(" ")] # Time + gyro
                gyrodata.append(gyro)

        gyrodata = np.array(gyrodata)
        print(gyrodata)
        return gyrodata
    
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




class BBLStabilizer(Stabilizer):
    def __init__(self, videopath, calibrationfile, bblpath, cam_angle_degrees=0, initial_offset=0):
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
        self.bbe = BlackboxExtractor("test_clips/GX015563.MP4_emuf_004.bbl")
        self.gyro_data = self.bbe.get_gyro_data(cam_angle_degrees=cam_angle_degrees)

        # This seems to make the orientation match. Implement auto match later
        self.gyro_data[:,[2, 3]] = self.gyro_data[:,[3, 2]]
        self.gyro_data[:,2] = -self.gyro_data[:,2]

        # Other attributes
        initial_orientation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()

        self.integrator = GyroIntegrator(self.gyro_data,initial_orientation=initial_orientation)
        self.integrator.integrate_all()
        self.times = None
        self.stab_transform = None

        self.initial_offset = initial_offset

    
    def stabilization_settings(self, smooth = 0.99):


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

        self.times, self.stab_transform = self.integrator.get_interpolated_stab_transform(smooth=0.985,start=2.56+0.07,interval = 1/59.94)


        #self.times, self.stab_transform = self.integrator.get_interpolated_stab_transform(smooth=smooth,start=-gyro_start,interval = interval) # 2.2/30 , -1/30




class OpticalStabilizer:
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

        # Other attributes
        self.times = None
        self.stab_transform = None

    
    def stabilization_settings(self, smooth = 0.65):

        frame_idx, transforms = self.optical_flow_comparison(0, 902)

        # Match "standard" coordinate system
        #transforms[0] = transforms[0]
        #transforms[1] = transforms[1]

        transforms[:,0] = -transforms[:,0]
        transforms[:,1] = -transforms[:,1]
        transforms[:,2] = transforms[:,2]

        stacked_data = np.hstack([np.atleast_2d(frame_idx).T,transforms])
        

        initial_orientation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()

        self.integrator = FrameRotationIntegrator(stacked_data,initial_orientation=initial_orientation)
        self.integrator.integrate_all()

        self.times, self.stab_transform = self.integrator.get_stabilize_transform(smooth=smooth)


        self.stab_transform_array = np.zeros((self.num_frames, 4))
        self.stab_transform_array[:,0] = 1

        for i in range(len(self.times)):
            self.stab_transform_array[round(self.times[i])] = self.stab_transform[i,:]


        print(self.stab_transform_array)


    def optical_flow_comparison(self, start_frame=0, analyze_length = 50):
        frame_times = []
        frame_idx = []
        transforms = []
        prev_pts_lst = []
        curr_pts_lst = []

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Read first frame
        _, prev = self.cap.read()
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        for i in range(analyze_length):
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)


            
            frame_id = (int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
            frame_time = (self.cap.get(cv2.CAP_PROP_POS_MSEC)/1000)

            succ, curr = self.cap.read()


            if i % 10 == 0:
                print("Analyzing frame: {}/{}".format(i,analyze_length))

            if succ:
                # Only add if succeeded
                frame_idx.append(frame_id)
                frame_times.append(frame_time)

                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                # Estimate transform using optical flow
                curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

                idx = np.where(status==1)[0]
                prev_pts = prev_pts[idx]
                curr_pts = curr_pts[idx]
                assert prev_pts.shape == curr_pts.shape

                prev_pts_lst.append(prev_pts)
                curr_pts_lst.append(curr_pts)


                # TODO: Try getting undistort + homography working for more accurate rotation estimation
                src_pts = self.undistort.undistort_points(prev_pts, new_img_dim=(self.width,self.height))
                dst_pts = self.undistort.undistort_points(curr_pts, new_img_dim=(self.width,self.height))
                H, mask = cv2.findHomography(src_pts, dst_pts)
                retval, rots, trans, norms = self.undistort.decompose_homography(H, new_img_dim=(self.width,self.height))


                # rots contains for solutions for the rotation. Get one with smallest magnitude. Idk
                roteul = None
                smallest_mag = 1000
                for rot in rots:
                    thisrot = Rotation.from_matrix(rots[0]) # first one?
                    
                    if thisrot.magnitude() < smallest_mag and thisrot.magnitude() < 0.3:
                        roteul = Rotation.from_matrix(rot).as_euler("xyz")
                        smallest_mag = thisrot.magnitude()




                m, inliers = cv2.estimateAffine2D(src_pts, dst_pts) 

                dx = m[0,2]
                dy = m[1,2]
                
                # Extract rotation angle
                da = np.arctan2(m[1,0], m[0,0])
                #transforms.append([dx,dy,da]) 
                transforms.append(list(roteul))
                prev_gray = curr_gray

            else:
                print("Frame {}".format(i))
        
        transforms = np.array(transforms)
        return frame_idx, transforms


    def renderfile(self, starttime, stoptime, outpath = "Stabilized.mp4", out_size = (1920,1080)):

        out = cv2.VideoWriter(outpath, -1, 30, (1920*2,1080))
        crop = (int((self.width-out_size[0])/2), int((self.height-out_size[1])/2))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(starttime * self.fps))

        num_frames = int((stoptime - starttime) * self.fps) 

        i = 0
        while(True):
            frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Read next frame
            success, frame = self.cap.read() 

            
            print("FRAME: {}, IDX: {}".format(frame_num, i))

            if success:
                i +=1

            if i > num_frames:
                break

            if success and i > 0:

                frame_undistort = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT)


                frame_out = self.undistort.get_rotation_map(frame_undistort, self.stab_transform_array[frame_num, :])

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
    """
    #stab = GPMFStabilizer("test_clips/GX016017.MP4", "camera_presets/Hero_7_2.7K_60_4by3_wide.json")
    stab = OpticalStabilizer("test_clips/GX016017.MP4", "camera_presets/gopro_calib2.JSON")
    stab.stabilization_settings(smooth = 0.7)
    #stab.optical_flow_comparison(start_frame=1300, analyze_length = 50)
    

    # Camera undistortion stuff
    stab.undistort = FisheyeCalibrator()
    stab.undistort.load_calibration_json("camera_presets/Hero_7_2.7K_60_4by3_wide.json", True)
    stab.map1, stab.map2 = stab.undistort.get_maps(2.2,new_img_dim=(stab.width,stab.height))


    stab.renderfile("GX016017_2_stab_optical.mp4",out_size = (1920,1080))
    stab.release()
    """


    # insta360 test
    
    stab = InstaStabilizer("test_clips/insta360.mp4", "camera_presets/gopro_calib2.JSON", gyrocsv="test_clips/insta360_gyro.csv")
    stab.auto_sync_stab(0.985,30 *24, 66 * 24, 170)
    #stab.renderfile(19, 40, "insta360test.mp4",out_size = (2560,1440))

    exit()
    #stab = GPMFStabilizer("test_clips/GX016017.MP4", "camera_presets/Hero_7_2.7K_60_4by3_wide.json") # Walk
    #stab = GPMFStabilizer("test_clips/GX016015.MP4", "camera_presets/gopro_calib2.JSON", ) # Rotate around
    #stab = GPMFStabilizer("test_clips/GX010010.MP4", "camera_presets/gopro_calib2.JSON", hero6=False) # Parking lot

    stab = BBLStabilizer("test_clips/GX015563.MP4", "camera_presets/gopro_calib2.JSON", "test_clips/GX015563.MP4_emuf_004.bbl", initial_offset=-2) # FPV clip

    #stab.stabilization_settings(smooth = 0.8)
    # stab.auto_sync_stab(0.89,25*30, (2 * 60 + 22) * 30, 50) Gopro clips

    stab.auto_sync_stab(0.985,5*24, 18 * 60, 70) # FPV clip
    #stab.stabilization_settings()


    # Camera undistortion stuff
    #stab.undistort = FisheyeCalibrator()
    #stab.undistort.load_calibration_json("camera_presets/Hero_7_2.7K_60_4by3_wide.json", True)
    #stab.map1, stab.map2 = stab.undistort.get_maps(2.6,new_img_dim=(stab.width,stab.height))


    #stab.renderfile(24, 63, "parkinglot_stab_3.mp4",out_size = (1920,1080))
    stab.renderfile(4, 26, "FPV_stab.mp4",out_size = (1920,1080))
    stab.release()

    # 20 / self.fps: 0.042
    # 200 / self.fps: -0.048
