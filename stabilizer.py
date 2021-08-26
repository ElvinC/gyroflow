#from quaternion import quaternion
import numpy as np
from datetime import date
import cv2
import csv
import os

for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        print("deleting" + os.environ[k])
        del os.environ[k]

import platform
from tqdm import tqdm

from freqAnalysis import FreqAnalysis
from calibrate_video import FisheyeCalibrator, StandardCalibrator
from scipy.spatial.transform import Rotation
from gyro_integrator import GyroIntegrator, FrameRotationIntegrator
from adaptive_zoom import AdaptiveZoom
from blackbox_extract import BlackboxExtractor
from GPMF_gyro import Extractor
from matplotlib import pyplot as plt
from matplotlib import colors
from vidgear.gears import WriteGear
from vidgear.gears import helper as vidgearHelper
import gyrolog
import json
import multiprocessing as mp
from _version import __version__

from scipy import signal, interpolate

import time
import insta360_utility as insta360_util
import smoothing_algos

VIDGEAR_LOGGING = False
BLOCKING_PLOTS = True

if platform.system() == "Darwin":
    BLOCKING_PLOTS = False

def impute_gyro_data(input_data):


    input_data = np.copy(input_data)
    # Check for corrupted/out of order timestamps
    time_order_check = input_data[:-1,0] > input_data[1:,0]
    if np.any(time_order_check):
        print("Truncated bad gyro data")
        input_data = input_data[0:np.argmax(time_order_check)+1,:]

    frame_durations = input_data[1:,0] - input_data[:-1,0]
    min_frame_duration = frame_durations.min()
    max_frame_duration = np.percentile(frame_durations, 10) * 1.5
    average_frame_duration = frame_durations[(frame_durations >= min_frame_duration) & (frame_durations <= max_frame_duration)].mean()
    print(f'average_frame_duration: {average_frame_duration}')
    max_allowed_frame_duration = average_frame_duration * 2

    missing_runs = []

    last_ts = input_data[0,0]
    for ix, ts in np.ndenumerate(input_data[1:,0]):
        if ts - last_ts > max_allowed_frame_duration:
            missing_runs.append((ix[0], round((ts - last_ts) / average_frame_duration) - 1))
        last_ts = ts
    print(f'missing_runs: {missing_runs}')

    last_ix = 0
    arrays_to_concat = []
    if len(missing_runs) > 0:
        for start, length in missing_runs:
            print(f'Appending {input_data[last_ix, 0]}..={input_data[start, 0]}')
            arrays_to_concat.append(input_data[last_ix:start+1,:])
            prev_row = input_data[start]
            next_row = input_data[start+1]
            filled_data = np.linspace(prev_row, next_row, length + 1, endpoint=False)[1:]
            print(f'Appending {filled_data[0, 0]}..={filled_data[-1, 0]} (filled)')
            arrays_to_concat.append(filled_data)
            last_ix = start + 1
    print(f'Appending {input_data[last_ix, 0]}..={input_data[-1, 0]}')
    arrays_to_concat.append(input_data[last_ix:,:])

    return np.concatenate(arrays_to_concat)


class Stabilizer:
    def __init__(self, videopath, calibrationfile=None, gyro_path=None, fov_scale = 1.6, gyro_lpf_cutoff = -1, video_rotation = -1, gyroflow_file=None):

        ### Define all important variables

        self.initial_offset = 0
        self.rough_sync_search_interval = 10
        self.better_sync_search_interval = 0.2
        self.gyro_lpf_cutoff = gyro_lpf_cutoff
        self.do_video_rotation = False
        self.num_frames_skipped = 1

        self.use_gyroflow_data_file = False

        # General video stuff
        self.cap = 0
        self.width = 0
        self.height = 0
        self.fps = 0
        self.num_frames = 0

        # Camera undistortion stuff
        self.undistort = None #FisheyeCalibrator()
        self.map1 = None
        self.map2 = None

        self.map_func_scale = 0.9

        self.integrator = None #GyroIntegrator(self.gyro_data,initial_orientation=initial_orientation)
        self.new_integrator = None
        self.times = None
        self.stab_transform = None
        self.smoothing_algo = None

        self.initial_orientation = Rotation.from_euler('zxy', [0,0,np.pi/2]).as_quat()
        self.initial_orientation[[0,1,2,3]] = self.initial_orientation[[3,0,1,2]]

        # self.raw_gyro_data = None
        self.gyro_data = None # self.bbe.get_gyro_data(cam_angle_degrees=cam_angle_degrees)
        self.acc_data = None

        # time lapse features
        self.hyperlapse_multiplier = 1
        self.hyperlapse_num_blended_frames = 1 # must be equal or less than hyperlapse_multiplier
        self.hyperlapse_skipped_frames = 0

        ## Combined from individual classes

        # Save info
        self.videopath = videopath
        self.calibrationfile = calibrationfile
        self.gyro_path = gyro_path

        # General video stuff
        self.undistort_fov_scale = fov_scale
        self.cap = cv2.VideoCapture(videopath)
        orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.orig_dimension = (orig_w,orig_h) #Dimension of input file

        self.undistort = FisheyeCalibrator()

        self.video_rotate_code = video_rotation

        if type(gyroflow_file) != type(None):
            success = self.import_gyroflow_file(gyroflow_file)
            if not success:
                raise RuntimeError

        else:
            self.undistort.load_calibration_json(calibrationfile, True)

        self.do_video_rotation = self.video_rotate_code != -1
        if self.video_rotate_code == cv2.ROTATE_90_CLOCKWISE or self.video_rotate_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
            orig_w, orig_h = orig_w, orig_h

        self.map1, self.map2 = self.undistort.get_maps(self.undistort_fov_scale,new_img_dim=(orig_w,orig_h))

        self.process_dimension = self.undistort.get_stretched_size_from_dimension(self.orig_dimension) # Dimension after any stretch corrections
        self.width, self.height = self.process_dimension

        # Sync stuff

        # No longer used:
        self.d1 = 0
        self.d2 = 0

        # Syncpoints
        self.transform_times = []
        self.transforms = []
        self.sync_inputs = [] # consists of pairs of (frame_start, slice_length)
        self.sync_vtimes = []
        self.sync_delays = []
        self.sync_costs = []

    def set_initial_offset(self, initial_offset):
        self.initial_offset = initial_offset

    def set_rough_search(self, interval = 10):
        self.rough_sync_search_interval = interval

    def set_gyro_lpf(self, cutoff_frequency = -1):
        self.gyro_lpf_cutoff = cutoff_frequency

    def set_num_frames_skipped(self, num=1):
        self.num_frames_skipped = num

    def filter_gyro(self):

        # Replaces self.gyrodata and should only be used once
        num_data_points = self.gyro_data.shape[0]
        gyro_sample_rate = num_data_points / (self.gyro_data[-1,0] - self.gyro_data[0,0])

        # Nyquist frequency
        if (gyro_sample_rate / 2) <= self.gyro_lpf_cutoff:
            self.gyro_lpf_cutoff = gyro_sample_rate / 2 - 1

        # Tweak with filter order
        sosgyro = signal.butter(1, self.gyro_lpf_cutoff, "lowpass", fs=gyro_sample_rate, output="sos")

        self.gyro_data[:,1:4] = signal.sosfiltfilt(sosgyro, self.gyro_data[:,1:4], 0) # Filter along "vertical" time axis

    def filter_acc(self):
        # rather aggressive filtering is applied here
        if type(self.acc_data) != type(None):
            acc_cutoff = 1

            num_data_points = self.acc_data.shape[0]
            acc_sample_rate = num_data_points / (self.acc_data[-1,0] - self.acc_data[0,0])

            # Nyquist frequency
            if (acc_sample_rate / 2) <= acc_cutoff:
                self.gyro_lpf_cutoff = acc_sample_rate / 2 - 1

            # First order filters to avoid overshoot

            # Get rid of high freq.
            sosgyro = signal.butter(1, 40, "lowpass", fs=acc_sample_rate, output="sos")
            self.acc_data[:,1:4] = signal.sosfiltfilt(sosgyro, self.acc_data[:,1:4], 0) # Filter along "vertical" time axis

            sosacc = signal.butter(1, acc_cutoff, "lowpass", fs=acc_sample_rate, output="sos")

            self.acc_data[:,1:4] = signal.sosfiltfilt(sosacc, self.acc_data[:,1:4], 0) # Filter along "vertical" time axis


            sosgyro = signal.butter(1, 0.5, "lowpass", fs=acc_sample_rate, output="sos")
            self.acc_data[:,1:4] = signal.sosfiltfilt(sosgyro, self.acc_data[:,1:4], 0) # Filter along "vertical" time axis

    def set_hyperlapse(self, hyperlapse_multiplier = 1, hyperlapse_num_blended_frames = 1):

        # Orig frames:
        # |0|1|2|3|4|5|6|7|8|9|10|11|12|13
        # mult=4, num_blend=2
        # |0+1|4+5|8+9|12+13|

        self.hyperlapse_multiplier = hyperlapse_multiplier
        self.hyperlapse_num_blended_frames = min(hyperlapse_multiplier, hyperlapse_num_blended_frames) # Ensure no overlapping frames
        self.hyperlapse_skipped_frames = self.hyperlapse_multiplier - self.hyperlapse_num_blended_frames

    def set_smoothing_algo(self, algo = None):
        if not algo:
            algo = smoothing_algos.PlainSlerp() # Default
        else:
            self.smoothing_algo = algo

    def update_smoothing(self):

        if type(self.new_integrator) != type(None):
            self.new_integrator.integrate_all(use_acc=self.smoothing_algo.require_acceleration)
            self.new_integrator.set_smoothing_algo(self.smoothing_algo)
            self.times, self.stab_transform = self.new_integrator.get_interpolated_stab_transform(start=0,interval = 1/self.fps)

            return True


        print("Orientations not calculated yet")
        return False

    def auto_sync_stab(self, sliceframe1 = 10, sliceframe2 = 1000, slicelength = 50, debug_plots = True):
        if debug_plots:
            FreqAnalysis(self.integrator).sampleFrequencyAnalysis()

        if self.use_gyroflow_data_file:
            self.update_smoothing()
            return

        v1 = (sliceframe1 + slicelength/2) / self.fps
        v2 = (sliceframe2 + slicelength/2) / self.fps
        d1, cost1, times1, transforms1 = self.optical_flow_comparison(sliceframe1, slicelength, debug_plots = debug_plots)
        #self.initial_offset = d1
        d2, cost2, times2, transforms2 = self.optical_flow_comparison(sliceframe2, slicelength, debug_plots = debug_plots)

        self.transform_times = [times1, times2]
        self.transforms = [transforms1, transforms2]
        self.v1 = v1
        self.v2 = v2
        self.d1 = d1
        self.d2 = d2


        print("v1: {}, v2: {}, d1: {}, d2: {}".format(v1, v2, d1, d2))

        err_slope = (d2-d1)/(v2-v1)
        correction_slope = err_slope + 1
        gyro_start = (d1 - err_slope*v1)

        interval = 1/(correction_slope * self.fps)

        g1 = v1 - d1
        g2 = v2 - d2
        slope = (v2 - v1) / (g2 - g1)
        corrected_times = slope * (self.integrator.get_raw_data("t") - g1) + v1

        #print("Start {}".format(gyro_start))

        print("Gyro correction slope {}".format(slope))

        self.plot_sync(corrected_times, slicelength, show=True)



        oldplot = True
        if oldplot and debug_plots:
            plt.figure()
            xplot = plt.subplot(311)

            plt.plot(times1, -transforms1[:,0] * self.fps)
            plt.plot(times2, -transforms2[:,0] * self.fps)
            plt.plot(corrected_times, self.integrator.get_raw_data("x"))
            plt.ylabel("omega x [rad/s]")

            plt.subplot(312, sharex=xplot)

            plt.plot(times1, transforms1[:,1] * self.fps)
            plt.plot(times2, transforms2[:,1] * self.fps)
            plt.plot(corrected_times, self.integrator.get_raw_data("y"))
            plt.ylabel("omega y [rad/s]")

            plt.subplot(313, sharex=xplot)

            plt.plot(times1, transforms1[:,2] * self.fps)
            plt.plot(times2, transforms2[:,2] * self.fps)
            plt.plot(corrected_times, self.integrator.get_raw_data("z"))
            #plt.plot(self.integrator.get_raw_data("t") + d2, self.integrator.get_raw_data("z"))
            plt.xlabel("time [s]")
            plt.ylabel("omega z [rad/s]")

            plt.show(block=BLOCKING_PLOTS)

        # Temp new integrator with corrected time scale

        initial_orientation = Rotation.from_euler('zxy', [0,0,np.pi/2]).as_quat()
        initial_orientation[[0,1,2,3]] = initial_orientation[[3,0,1,2]]

        new_gyro_data = np.copy(self.gyro_data)


        # Correct time scale
        new_gyro_data[:,0] = slope * (self.integrator.get_raw_data("t") - g1) + v1 # (new_gyro_data[:,0]+gyro_start) *correction_slope


        if type(self.acc_data) != type(None):
            new_acc_data = np.copy(self.acc_data)
            new_acc_data[:,0] = new_gyro_data[:,0]
        else:
            new_acc_data = None

        if not self.smoothing_algo:
            self.smoothing_algo = smoothing_algos.PlainSlerp()

        self.new_integrator = GyroIntegrator(new_gyro_data,zero_out_time=False, initial_orientation=initial_orientation, acc_data=new_acc_data)
        if self.smoothing_algo.require_acceleration and type(new_acc_data) == type(None):
            print("No acceleration data available. Horizon reference doesn't work without it.")
        self.new_integrator.integrate_all(use_acc=self.smoothing_algo.require_acceleration)
        #self.last_smooth = smooth

        self.new_integrator.set_smoothing_algo(self.smoothing_algo)
        self.times, self.stab_transform = self.new_integrator.get_interpolated_stab_transform(start=0,interval = 1/self.fps)

        #self.times, self.stab_transform = self.integrator.get_interpolated_stab_transform(smooth=smooth,start=-gyro_start,interval = interval)

    def multi_sync_init(self):
        self.transform_times = []
        self.transforms = []
        self.sync_inputs = [] # consists of pairs of (frame_start, slice_length)
        self.sync_vtimes = []
        self.sync_delays = []
        self.sync_costs = []


    def multi_sync_add_slice(self, slice_frame_start, slicelength = 50, debug_plots = True):
        v1 = (slice_frame_start + slicelength/2) / self.fps
        d1, cost1, times1, transforms1 = self.optical_flow_comparison(slice_frame_start, slicelength, debug_plots = debug_plots)
        N = len(self.sync_inputs)
        # Find where to insert
        idx = 0
        if N == 0:
            pass
        elif slice_frame_start > self.sync_inputs[-1][0]:
            idx = N
        else:
            for i in range(len(self.sync_inputs)):
                if self.sync_inputs[i][0] > slice_frame_start:
                    idx = i
                    break

        self.sync_inputs.insert(idx, (slice_frame_start, slicelength))
        self.transform_times.insert(idx, times1)
        self.transforms.insert(idx, transforms1)
        self.sync_vtimes.insert(idx, v1)
        self.sync_delays.insert(idx, d1)
        self.sync_costs.insert(idx, cost1)

        return cost1
    
    def multi_sync_delete_slice(self, idx):
        if len(self.transform_times) > idx:
            del self.transform_times[idx]
            del self.transforms[idx]
            del self.sync_inputs[idx]
            del self.sync_vtimes[idx]
            del self.sync_delays[idx]
            del self.sync_costs[idx]

            return True

        return False

    def multi_sync_change_offset(self, idx, newoffset=0):
        if len(self.transform_times) > idx:
            self.sync_delays[idx] = newoffset
            return True

        return False

    def multi_sync_compute(self, max_cost = 5, max_fitting_error = 0.02, piecewise_correction = False, debug_plots = True):

        assert len(self.transform_times) == len(self.transforms) == len(self.sync_vtimes) == len(self.sync_delays) == len(self.sync_costs)

        if piecewise_correction:
            print("Not implemented yet")

        N = len(self.transform_times)
        if N == 0:
            # no change
            print("No valid syncpoints")
            self.new_integrator = GyroIntegrator(self.gyro_data,zero_out_time=False, initial_orientation=self.initial_orientation, acc_data=self.acc_data)
        
        elif N == 1:

            new_gyro_data = np.copy(self.gyro_data)

            # Shift the time
            new_gyro_data[:,0] = self.integrator.get_raw_data("t") + self.sync_delays[0] # (new_gyro_data[:,0]+gyro_start) *correction_slope

            if type(self.acc_data) != type(None):
                new_acc_data = np.copy(self.acc_data)
                new_acc_data[:,0] = new_gyro_data[:,0]
            else:
                new_acc_data = None
            
            self.new_integrator = GyroIntegrator(new_gyro_data,zero_out_time=False, initial_orientation=self.initial_orientation, acc_data=new_acc_data)
        else:
            # N is two or above, use the weird non-random RANSAC fitting
            times = np.array(self.sync_vtimes)
            delays = np.array(self.sync_delays)
            sync_costs = np.array(self.sync_costs)

            chosen_indices = {}
            num_chosen = 0
            rsquared_best = 1000
            chosen_coefs = None

            #max_sync_cost = 6 # > 6 is nogo.

            for i in range(N):
                for j in range(i, N):
                    if i != j:
                        
                        del_i = delays[i]
                        del_j = delays[j]

                        t_i = times[i]
                        t_j = times[j]

                        slope = (del_j - del_i) / (t_j - t_i)
                        intersect = del_i - t_i * slope

                        within_error = []
                        est_curve = times * slope + intersect
                        within_error = np.where(np.abs(est_curve - delays) < max_fitting_error)[0]

                        

                        if within_error.shape[0] >= num_chosen and set(within_error) != chosen_indices:
                            #print(times[within_error])
                            fit = np.polyfit(times[within_error], delays[within_error], 1, full=True)
                            coefs = fit[0]

                            close_constant = -0.1 < coefs[0] < 0.1

                            if within_error.shape[0] > 2 and close_constant:
                                rsquared = fit[1]

                                if rsquared < rsquared_best:
                                    rsquared_best = rsquared
                                    chosen_coefs = coefs
                                    num_chosen = within_error.shape[0]
                                    chosen_indices = set(within_error)
                            elif close_constant: # close to linear
                                chosen_coefs = coefs
                                num_chosen = within_error.shape[0]
                                chosen_indices = set(within_error)

                            
            if type(chosen_coefs) == type(None):
                return False

            print(chosen_coefs)
            print(chosen_indices)

            new_gyro_data = np.copy(self.gyro_data)
            new_gyro_data[:,0] = (self.integrator.get_raw_data("t") + chosen_coefs[1])/(1- chosen_coefs[0])

            if debug_plots:
                est_curve = times * chosen_coefs[0] + chosen_coefs[1]

                self.plot_sync(new_gyro_data[:,0], 60, True)
                plt.figure()
                plt.scatter(times, delays)
                plt.plot(times, est_curve)
                plt.show(block=BLOCKING_PLOTS)

            if type(self.acc_data) != type(None):
                new_acc_data = np.copy(self.acc_data)
                new_acc_data[:,0] = new_gyro_data[:,0]
            else:
                new_acc_data = None

            self.new_integrator = GyroIntegrator(new_gyro_data,zero_out_time=False, initial_orientation=self.initial_orientation, acc_data=new_acc_data)
            


        if not self.smoothing_algo:
            self.smoothing_algo = smoothing_algos.PlainSlerp()

        if self.smoothing_algo.require_acceleration and type(new_acc_data) == type(None):
            print("No acceleration data available. Horizon reference doesn't work without it.")
        self.new_integrator.integrate_all(use_acc=self.smoothing_algo.require_acceleration)

        self.new_integrator.set_smoothing_algo(self.smoothing_algo)
        self.times, self.stab_transform = self.new_integrator.get_interpolated_stab_transform(start=0,interval = 1/self.fps)
        return True

    def get_recommended_syncpoints(self, num_frames_analyze, max_points=9):
        syncpoints = []

        num_frames_offset = int(num_frames_analyze / 2)
        end_delay = 3 # seconds buffer zone
        end_frames = end_delay * self.fps # buffer zone

        num_frames = self.num_frames
        vid_length = num_frames / self.fps

        inter_delay = 13 # second between syncs
        inter_delay_frames = int(inter_delay * self.fps)

        min_slices = 4

        max_slices = max_points

        if vid_length < 4: # only one sync
            syncpoints.append([5, max(60, int(num_frames-5-self.fps)) ])

            num_syncs = 1

        elif vid_length < 10: # two points
            first_index = 30
            last_index = num_frames - 30 - num_frames_analyze
            syncpoints.append([first_index, num_frames_analyze])
            syncpoints.append([last_index, num_frames_analyze])

            num_syncs = 2

        else:
            # Analysis starts at first frame, so take this into account
            # Add also motion analysis from logs here
            first_index = end_frames - num_frames_offset
            last_index = num_frames - end_frames - num_frames_offset

            num_syncs = max(min(round((last_index - first_index)/inter_delay_frames), max_slices), min_slices)
            inter_frames_actual = (last_index - first_index) / num_syncs

            for i in range(num_syncs):
                syncpoints.append([round(first_index + i * inter_frames_actual), num_frames_analyze])

        return syncpoints

    def full_auto_sync(self, max_fitting_error = 0.02, max_points=9, debug_plots=True):

        if self.use_gyroflow_data_file:
            self.update_smoothing()
            return

        self.multi_sync_init()

        max_sync_cost_tot = 10 # > 10 is nogo.
        num_frames_analyze = 30
        syncpoints = self.get_recommended_syncpoints(num_frames_analyze, max_points = max_points)
         # save where to analyze. list of [frameindex, num_analysis_frames]
        
        max_sync_cost = max_sync_cost_tot / 30 * num_frames_analyze
        
        # Analyze these slices
        num_syncs = len(syncpoints)

        print(f"Analyzing {num_syncs} slices")

        for frame_index, n_frames in syncpoints:
            self.multi_sync_add_slice(frame_index, n_frames, False)

            if self.sync_costs[-1] > max_sync_cost:
                print("Removing slice due to large error")
                self.multi_sync_delete_slice(-1)

            elif np.sum( (np.abs(self.transforms[-1] * self.fps) < 0.05) ) >= (0.95 * self.transforms[-1].size):
                print("Removing slice due to lack of movement")
                self.multi_sync_delete_slice(-1) # if more than 95% of the slice doesn't have significant movement (<3 deg/s)


        success = self.multi_sync_compute(max_fitting_error = max_fitting_error, debug_plots=debug_plots)

        if not success:
            success = self.multi_sync_compute(max_fitting_error = max_fitting_error * 2, debug_plots=debug_plots) # larger bound

        if success:
            print("Auto sync complete")
            return True
        else:
            print("Auto sync failed to converge. Sorry about that")
            return False


    def full_auto_sync_parallel(self, max_fitting_error = 0.02, debug_plots = True):
        # TODO: Figure out why this fails
        
        if self.use_gyroflow_data_file:
            self.update_smoothing()
            return

        self.multi_sync_init()

        max_sync_cost_tot = 10 # > 10 is nogo.
        num_frames_analyze = 30
        syncpoints = self.get_recommended_syncpoints(num_frames_analyze)
         # save where to analyze. list of [frameindex, num_analysis_frames]
        
        max_sync_cost = max_sync_cost_tot / 30 * num_frames_analyze
        
        # Analyze these slices
        num_syncs = len(syncpoints)

        print(f"Analyzing {num_syncs} slices in parallel")

        # Analyze in parallel

        n_proc = num_syncs # max about 10, should be fine

        with mp.Pool(processes=n_proc) as pool:
            # starts the sub-processes without blocking
            # pass the chunk to each worker process
            proc_results = [pool.apply_async(self.optical_flow_comparison_parallel,
                                            args=(spoint[0],spoint[1],))
                            for spoint in syncpoints]
            # blocks until all results are fetched
            result_chunks = [r.get() for r in proc_results]


        print(result_chunks)



    def plot_sync(self, corrected_times, slicelength, show=False):
        n = len(self.transform_times)
        fig, axes = plt.subplots(3, n, sharey=True)
        fig.set_size_inches(4 * n, 6)
        for j in range(n):
            mask = ((corrected_times > self.transform_times[j][0] - .2 * slicelength / self.fps) & (corrected_times < self.transform_times[j][-1] + .2 * slicelength / self.fps))
            axes[0][j].set(title=f"Syncpoint {j + 1}")
            for i, r in enumerate(['x', 'y', 'z']):
                axes[i][j].plot(corrected_times[mask], self.integrator.get_raw_data(r)[mask], alpha=.8)
                if r == 'x':
                    axes[i][j].plot(self.transform_times[j], -self.transforms[j][:, i] * self.fps, alpha=.8)
                else:
                    axes[i][j].plot(self.transform_times[j], self.transforms[j][:, i] * self.fps, alpha=.8)

        axes[0][0].set(ylabel="omega x [rad/s]")    
        axes[1][0].set(ylabel="omega y [rad/s]")
        axes[2][0].set(ylabel="omega z [rad/s]")
        for i in range(n):
            axes[2][i].set(xlabel="time [s]")
        plt.tight_layout()
        if show:
            plt.show(block=BLOCKING_PLOTS)
            
        return fig, axes

    def manual_sync_correction(self, d1, d2):


        if self.use_gyroflow_data_file:
            self.update_smoothing()
            return

        v1 = self.v1
        v2 = self.v2

        print("v1: {}, v2: {}, d1: {}, d2: {}".format(v1, v2, d1, d2))
        g1 = v1 - d1
        g2 = v2 - d2
        if g1==g2:
            slope = 1
        else:
            slope = (v2 - v1) / (g2 - g1)
        corrected_times = slope * (self.integrator.get_raw_data("t") - g1) + v1
        print("Gyro correction slope {}".format(slope))

        self.plot_sync(corrected_times, slicelength=50, show=True)

        # Temp new integrator with corrected time scale

        initial_orientation = Rotation.from_euler('zxy', [0,0,np.pi/2]).as_quat()
        initial_orientation[[0,1,2,3]] = initial_orientation[[3,0,1,2]]

        new_gyro_data = np.copy(self.gyro_data)

        # Correct time scale
        new_gyro_data[:,0] = slope * (self.integrator.get_raw_data("t") - g1) + v1 # (new_gyro_data[:,0]+gyro_start) *correction_slope

        if type(self.acc_data) != type(None):
            new_acc_data = np.copy(self.acc_data)
            new_acc_data[:,0] = new_gyro_data[:,0]
        else:
            new_acc_data = None


        if not self.smoothing_algo:
            self.smoothing_algo = smoothing_algos.PlainSlerp()

        self.new_integrator = GyroIntegrator(new_gyro_data,zero_out_time=False, initial_orientation=initial_orientation,acc_data=new_acc_data)
        self.new_integrator.integrate_all(use_acc=self.smoothing_algo.require_acceleration)
        #self.last_smooth = smooth



        self.new_integrator.set_smoothing_algo(self.smoothing_algo)
        self.times, self.stab_transform = self.new_integrator.get_interpolated_stab_transform(start=0,interval = 1/self.fps)

    def optical_flow_comparison(self, start_frame=0, analyze_length = 50, debug_plots = True):
        frame_times = []
        frame_idx = []
        transforms = []
        prev_pts_lst = []
        curr_pts_lst = []

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        time.sleep(0.05)

        # Read first frame
        _, prev = self.cap.read()
        if self.do_video_rotation:
            prev = cv2.rotate(prev, self.video_rotate_code)
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        if self.undistort.image_is_stretched():
            prev_gray = cv2.resize(prev_gray, self.process_dimension)

        for i in tqdm(range(analyze_length), desc="Analyzing frame", colour="blue"):
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)



            succ, curr = self.cap.read()
            if self.do_video_rotation:
                curr = cv2.rotate(curr, self.video_rotate_code)

            frame_id = (int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
            frame_time = (self.cap.get(cv2.CAP_PROP_POS_MSEC)/1000)

            #if i % 10 == 0:
            #    print("Analyzing frame: {}/{}".format(i,analyze_length))

            if succ and i % self.num_frames_skipped == 0:
                # Only add if succeeded
                frame_idx.append(frame_id)
                frame_times.append(frame_time)


                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                if self.undistort.image_is_stretched():
                    curr_gray = cv2.resize(curr_gray, self.process_dimension)
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

                # rots contains for solutions for the rotation. Get one with smallest magnitude.
                # https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
                # https://en.wikipedia.org/wiki/Essential_matrix#Extracting_rotation_and_translation
                roteul = None
                smallest_mag = 1000

                try:
                    R1, R2, t = self.undistort.recover_pose(np.array(filtered_src), np.array(filtered_dst), new_img_dim=(self.width,self.height))

                    rot1 = Rotation.from_matrix(R1)
                    rot2 = Rotation.from_matrix(R2)

                    if rot1.magnitude() < rot2.magnitude():
                        roteul = rot1.as_rotvec() #rot1.as_euler("xyz")
                    else:
                        roteul = rot2.as_rotvec() # as_euler("xyz")
                except:
                    print("Couldn't recover motion for this frame")
                    roteul = np.array([0,0,0])

                transforms.append(list(roteul/self.num_frames_skipped))

                prev_gray = curr_gray

            else:
                print("Frame {}".format(i))

        transforms = np.array(transforms)
        estimated_offset, cost = self.estimate_gyro_offset(frame_times, transforms, prev_pts_lst, curr_pts_lst, debug_plots = debug_plots)
        return estimated_offset, cost, frame_times, transforms


    def optical_flow_comparison_parallel(self, start_frame=0, analyze_length = 50, debug_plots = False):
        frame_times = []
        frame_idx = []
        transforms = []
        prev_pts_lst = []
        curr_pts_lst = []

        cap = cv2.VideoCapture(self.videofile)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        time.sleep(0.05)


        # Read first frame
        _, prev = cap.read()
        if self.do_video_rotation:
            prev = cv2.rotate(prev, self.video_rotate_code)
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        if self.undistort.image_is_stretched():
            prev_gray = cv2.resize(prev_gray, self.process_dimension)

        for i in range(analyze_length):
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

            succ, curr = cap.read()
            if self.do_video_rotation:
                curr = cv2.rotate(curr, self.video_rotate_code)

            frame_id = (int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            frame_time = (cap.get(cv2.CAP_PROP_POS_MSEC)/1000)

            #if i % 10 == 0:
            #    print("Analyzing frame: {}/{}".format(i,analyze_length))

            if succ and i % self.num_frames_skipped == 0:
                # Only add if succeeded
                frame_idx.append(frame_id)
                frame_times.append(frame_time)


                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                if self.undistort.image_is_stretched():
                    curr_gray = cv2.resize(curr_gray, self.process_dimension)
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

                # rots contains for solutions for the rotation. Get one with smallest magnitude.
                # https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
                # https://en.wikipedia.org/wiki/Essential_matrix#Extracting_rotation_and_translation
                roteul = None

                try:
                    R1, R2, t = self.undistort.recover_pose(np.array(filtered_src), np.array(filtered_dst), new_img_dim=(self.width,self.height))

                    rot1 = Rotation.from_matrix(R1)
                    rot2 = Rotation.from_matrix(R2)

                    if rot1.magnitude() < rot2.magnitude():
                        roteul = rot1.as_rotvec() #rot1.as_euler("xyz")
                    else:
                        roteul = rot2.as_rotvec() # as_euler("xyz")
                except:
                    print("Couldn't recover motion for this frame")
                    roteul = np.array([0,0,0])

                transforms.append(list(roteul/self.num_frames_skipped))

                prev_gray = curr_gray

            else:
                print("Frame {}".format(i))

        cap.release()

        transforms = np.array(transforms)
        estimated_offset, cost = self.estimate_gyro_offset(frame_times, transforms, prev_pts_lst, curr_pts_lst, debug_plots = debug_plots)
        return estimated_offset, cost, frame_times, transforms

    def estimate_gyro_offset(self, OF_times, OF_transforms, prev_pts_list, curr_pts_list, debug_plots = True):
        #print(prev_pts_list)
        # Estimate offset between small optical flow slice and gyro data

        gyro_times = self.integrator.get_raw_data("t")
        gyro_data = self.integrator.get_raw_data("xyz")
        #print(gyro_data)

        # quick low pass filter
        self.frame_lowpass = False

        if self.frame_lowpass:
            params = [0.3,0.4,0.3] # weights. last frame, current frame, next frame
            new_OF_transforms = np.copy(OF_transforms)
            for i in range(1,new_OF_transforms.shape[0]-1):
                new_OF_transforms[i,:] = new_OF_transforms[i-1,:] * params[0] + new_OF_transforms[i,:]*params[1] + new_OF_transforms[i+1,:] * params[2]

            OF_transforms = new_OF_transforms

        costs = []
        offsets = []

        dt = self.rough_sync_search_interval # Search +/- 3 seconds
        N = int(dt * 100) # 1/100 of a second in rough sync

        #dt = self.better_sync_search_interval
        #N = int(dt * 5000)

        for i in range(N):
            offset = dt/2 - i * (dt/N) + self.initial_offset
            cost = self.fast_gyro_cost_func(OF_times, OF_transforms, gyro_times + offset, gyro_data) #fast_gyro_cost_func(OF_times, OF_transforms, gyro_times + offset, gyro_data)
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


        if debug_plots:
            plt.figure()
            plt.plot(offsets, costs)
        #    plt.show()
        costs = []
        offsets = []

        # Find better sync with smaller search space
        dt = self.better_sync_search_interval
        N = int(dt * 5000)
        do_hpf = False

        # run both gyro and video through high pass filter
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html

        if do_hpf:
            filterorder = 10
            filterfreq = 4 # hz
            sosgyro = signal.butter(filterorder, filterfreq, "highpass", fs=self.integrator.gyro_sample_rate, output="sos")
            sosvideo = signal.butter(filterorder, filterfreq, "highpass", fs=self.fps, output="sos")

            gyro_data = signal.sosfilt(sosgyro, gyro_data, 0) # Filter along "vertical" time axis
            OF_transforms = signal.sosfilt(sosvideo, OF_transforms, 0)

        #plt.plot(gyro_times, gyro_data[:,0])
        #plt.plot(gyro_times, filtered_gyro_data[:,0])

        for i in range(N):
            offset = dt/2 - i * (dt/N) + rough_offset
            #cost = self.fast_gyro_cost_func(OF_times, OF_transforms, gyro_times + offset, gyro_data)
            cost = self.better_gyro_cost_func(OF_times, OF_transforms, gyro_times + offset, gyro_data)
            offsets.append(offset)
            costs.append(cost)

        better_offset = offsets[np.argmin(costs)]
        cost = min(costs)

        print("Better offset: {}, cost: {}".format(better_offset, cost))

        if debug_plots:
            #plt.figure()
            plt.plot(offsets, costs)
            plt.xlabel("Offset [s]")
            plt.ylabel("Cost")
            plt.title(f"Syncpoint Offset Estimation\nCosts: {min(costs):.4f}, Offset: {better_offset:.4f}")

            #plt.show()

        return better_offset, cost

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

        new_OF_times = np.array(OF_times)

        # Shift by one frame to patch timing
        #new_OF_times += np.mean(new_OF_times[1:] - new_OF_times[:-1]) # / 2

        if new_OF_times[0] < gyro_times[0]:
            return 100

        if new_OF_times[-1] > gyro_times[-1]:
            return 100

        new_OF_transforms = np.copy(OF_transforms) * self.fps
        # Optical flow movements gives pixel movement, not camera movement
        new_OF_transforms[:,0] = -new_OF_transforms[:,0]
        #new_OF_transforms[:,1] = -new_OF_transforms[:,1]

        axes_weight = np.array([0.7,0.7,1]) #np.array([0.5,0.5,1]) # Weight of the xyz in the cost function. pitch, yaw, roll. More weight to roll

        sum_squared_diff = 0
        gyro_idx = 1

        next_gyro_snip = np.array([0, 0, 0], dtype=np.float64)
        next_cumulative_time = 0

        # Start close to match
        mask = gyro_times > (new_OF_times[0] - 0.5)
        first_idx = np.argmax(mask)
        if gyro_times[first_idx] > (new_OF_times[0] - 0.5):
            gyro_idx = first_idx
        else:
            return 100

        while gyro_times[gyro_idx + 1] < new_OF_times[0] and gyro_idx + 2 < len(gyro_times):
            gyro_idx += 1


        for OF_idx in range(len(new_OF_times)):
            cumulative = next_gyro_snip
            cumulative_time =  next_cumulative_time

            # if near edge of gyro track
            if gyro_idx + 100 > len(gyro_times):
                #print("Outside of gyro range")
                return 100

            while gyro_times[gyro_idx] < new_OF_times[OF_idx]:
                delta_time = gyro_times[gyro_idx] - gyro_times[gyro_idx-1]
                cumulative_time += delta_time

                cumulative += gyro_data[gyro_idx,:] * delta_time
                gyro_idx += 1

            time_delta = new_OF_times[OF_idx] - gyro_times[gyro_idx-2]
            time_weight = time_delta / (gyro_times[gyro_idx] - gyro_times[gyro_idx-1])
            cumulative += gyro_data[gyro_idx-1,:] * time_delta
            cumulative_time  += time_delta

            time_delta = gyro_times[gyro_idx-1] - new_OF_times[OF_idx]
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

    def fast_gyro_cost_func(self, OF_times, OF_transforms, gyro_times, gyro_data):


        if OF_times[0] < gyro_times[0]:
            return 100

        if OF_times[-1] > gyro_times[-1]:
            return 100

        new_OF_transforms = np.copy(OF_transforms) * self.fps
        # Optical flow movements gives pixel movement, not camera movement
        new_OF_transforms[:,0] = -new_OF_transforms[:,0]
        #new_OF_transforms[:,1] = -new_OF_transforms[:,1]


        axes_weight = np.array([0.7,0.7,1]) #np.array([0.5,0.5,1]) # Weight of the xyz in the cost function. pitch, yaw, roll. More weight to roll


        #t1 = OF_times[0]
        #t2 = OF_times[-1]

        #mask = ((t1 <= gyro_times) & (gyro_times <= t2))

        #sliced_gyro_data = gyro_data[mask,:]
        #sliced_gyro_times = gyro_times[mask]

        nearest = interpolate.interp1d(gyro_times, gyro_data, kind='nearest', assume_sorted=True, axis = 0, fill_value=np.array([0,0,0]), bounds_error=False)
        gyro_dat_resampled = nearest(OF_times)

        squared_diff = (gyro_dat_resampled - new_OF_transforms)**2
        sum_squared_diff = (squared_diff.sum(0) * axes_weight).sum()

        return sum_squared_diff

    def export_stabilization(self, outpath = "Stabilized.csv"):
        basename = os.path.splitext(outpath)[0]
        with open(outpath, 'w') as f, open(basename + ".w.spl", 'w') as w, open(basename + ".x.spl", 'w') as x, open(basename + ".y.spl", 'w') as y, open(basename + ".z.spl", 'w') as z:
            csv_writer = csv.writer(f, delimiter=",", quotechar='"')
            csv_writer.writerow(["frame", "w", "x", "y", "z"])

            w.write("DFSP\n")
            x.write("DFSP\n")
            y.write("DFSP\n")
            z.write("DFSP\n")

            for ix in range(len(self.times)):
                q = self.stab_transform[ix]

                if type(q) != type(None):
                    q = q.flatten()

                csv_writer.writerow([ix, q[0], q[1], q[2], q[3]])
                w.write(f"{ix} {q[0]}\n")
                x.write(f"{ix} {q[1]}\n")
                y.write(f"{ix} {q[2]}\n")
                z.write(f"{ix} {q[3]}\n")

    def map_function(self, frame_num, out_size = (1280,720), regenerate = False):
        if frame_num >= len(self.stab_transform):
            frame_num = len(self.stab_transform)-1
            print("No more stabilization data. Using last frame")

        return self.undistort.get_maps(self.map_func_scale,
            new_img_dim=(self.orig_dimension[0], self.orig_dimension[1]),
            output_dim=out_size,
            update_new_K = False, quat = self.stab_transform[frame_num],
            focalCenter = None,
            original_stretched = True)

    def set_map_func_scale(self, map_scale = 0.9):
        self.map_func_scale = map_scale

    def renderfile(self, starttime, stoptime, outpath = "Stabilized.mp4", out_size = (1920,1080), split_screen = False,
                   bitrate_mbits = 20, display_preview = False, scale=1, vcodec = "libx264", vprofile="main", pix_fmt = "",
                   debug_text = False, custom_ffmpeg = "", smoothingFocus=4.0, fov_scale=1.0, bg_color="#000000", audio=True, viewer_thread = None):
        if outpath == self.videopath:
            outpath = outpath.lower().replace(".mp4", "_gyroflow.mp4", )
        (out_width, out_height) = out_size

        #export_out_size = (int(out_size[0]*2*scale) if split_screen else int(out_size[0]*scale), int(out_size[1]*scale))

        # Should cover the valid ones even if not supported https://en.wikipedia.org/wiki/Video_file_format
        if outpath.split(".")[-1].lower() not in ["webm", "flv", "vob", "ogv", "ogg", "drc", "gif", "mng", "avi", "mts", "m2ts", "ts", "mov", "qt", "yuv", "rmvb", "viv", "asf", "amv", "mp4", "m4p", "m4v", "mpg",
                                                  "mpeg", "m2v", "svi", "3gp", "3g2", "mxf", "roq", "nsv", "flv"]:
            print(f"{outpath} does not have a valid file extension")
            return

        borderMode = 0
        borderValue = 0

        if bg_color == "REPLICATE":
            borderMode = cv2.BORDER_REPLICATE
        elif bg_color == "HISTORY":
            borderMode = cv2.BORDER_TRANSPARENT
        else:
            borderMode = cv2.BORDER_CONSTANT
            borderValue = [round(x*255) for x in colors.to_rgb(bg_color)][::-1]


        if vcodec == "libx264":
            output_params = {
                "-input_framerate": self.fps,
                "-vcodec": "libx264",
                "-profile:v": vprofile,
                "-crf": "1",  # Can't use 0 as it triggers "lossless" which does not allow  -maxrate
                "-maxrate": "%sM" % bitrate_mbits,
                "-bufsize": "%sM" % int(bitrate_mbits * 1.2),
                "-pix_fmt": "yuv420p",
            }
        elif vcodec == "h264_nvenc":
            output_params = {
                "-input_framerate": self.fps,
                "-vcodec": "h264_nvenc",
                "-profile:v": vprofile,
                "-rc:v": "cbr",
                "-b:v": "%sM" % bitrate_mbits,
                "-bufsize:v": "%sM" % int(bitrate_mbits * 2),
            }
        elif vcodec == "h264_amf":
            output_params = {
                "-input_framerate": self.fps,
                "-vcodec": "h264_amf",
                "-profile:v": vprofile,
                "-rc:v": "cbr",
                "-b:v": "%sM" % bitrate_mbits,
                "-bufsize:v": "%sM" % int(bitrate_mbits * 2),
            }
        elif vcodec == "h264_vaapi":
            output_params = {
                "-input_framerate": self.fps,
                "-vcodec": "h264_vaapi",
                "-vaapi_device": "/dev/dri/renderD128",
                "-profile:v": vprofile,
                "-b:v": "%sM" % bitrate_mbits,
            }
        elif vcodec == "h264_videotoolbox":
            output_params = {
                "-input_framerate": self.fps,
                "-vcodec": "h264_videotoolbox",
                "-profile:v": vprofile,
                "-b:v": "%sM" % bitrate_mbits,
                }
        elif vcodec == "prores_ks":
            output_params = {
                "-input_framerate": self.fps,
                "-vcodec": "prores_ks",
                "-profile:v": vprofile,
            }
        elif vcodec == "v210":
            output_params = {
                "-input_framerate": self.fps,
                "-vcodec": "v210"
            }
        else:
            output_params = {}

        if pix_fmt:
            output_params["-pix_fmt"] = pix_fmt  # override pix_fmt if user needs to

        if custom_ffmpeg:
            output_params = eval(custom_ffmpeg)
            output_params["-input_framerate"] = self.fps

        # Find locally installed ffmpeg from PATH and use it
        if platform.system() == "Windows":
            ffmpeg_exe_path = os.popen("WHERE ffmpeg").read()
            if ffmpeg_exe_path:
                # Only first line
                ffmpeg_local_path = os.path.dirname(ffmpeg_exe_path).split("\n")[0]
                output_params["custom_ffmpeg"] = ffmpeg_local_path
                print(f"Using ffmpeg path {ffmpeg_local_path}")
            else:
                print("No FFmpeg detected in the windows PATH")

        # non compression fallback fps
        #output_params["-fps"] = self.fps


        out = WriteGear(output_filename=outpath, logging=debug_text, **output_params)

        num_frames = int((stoptime - starttime) * self.fps)

        tstart = int(starttime * self.fps)
        tend = tstart + num_frames

        #tempmap1 = cv2.resize(self.map1, (int(self.map1.shape[1]*scale), int(self.map1.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
        #tempmap2 = cv2.resize(self.map2, (int(self.map2.shape[1]*scale), int(self.map2.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)


        print("Starting to compute optimal Fov")
        adaptZ = AdaptiveZoom(fisheyeCalibrator=self.undistort)
        fcorr, focalCenter = adaptZ.compute(quaternions=self.stab_transform, output_dim=out_size, fps=self.fps,
                                                        smoothingFocus=smoothingFocus,
                                                        tstart = tstart, tend = tend,
                                                        debug_plots=False, plot_blocking = BLOCKING_PLOTS)
        print("Done computing optimal Fov")

        #new_img_dim=(int(self.width * scale),int(self.height*scale))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(starttime * self.fps))
        time.sleep(0.1)

        # Generate initial black frame
        success, frame = self.cap.read()
        if self.do_video_rotation:
            frame = cv2.rotate(frame, self.video_rotate_code)
        frame_out = cv2.resize(frame, out_size, interpolation=cv2.INTER_LINEAR) * 0.0

        # temporary float frame
        frame_temp = (cv2.resize(frame, out_size, interpolation=cv2.INTER_LINEAR) * 0.0).astype(np.float64)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(starttime * self.fps))
        time.sleep(0.1)

        i = 0

        starttime_render = time.time()

        # Double press q to stop render
        quit_button = False

        num_not_success = 0
        num_not_success_lim = 5 # stop after 5 failures to read frame

        old_map_enable_setting = False
        if type(viewer_thread) != type(None) and display_preview:
            old_map_enable_setting = viewer_thread.map_function_enable
            viewer_thread.map_function_enable = False


        for i in tqdm(range(1, num_frames), desc="Rendering", colour="blue"):

            try:
                # Read next frame
                frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                success, frame = self.cap.read()

                if self.do_video_rotation:
                    frame = cv2.rotate(frame, self.video_rotate_code)
                # Getting frame_num _before_ cap.read gives index of the read frame.

                if i % 5 == 0:
                    fraction_done = i/num_frames
                    elapsed_time = time.time() - starttime_render  # in seconds
                    est_remain = (elapsed_time) * (1/max(fraction_done, 0.00001) - 1)
                    #print("frame: {}, {}/{} ({}%), ~{} s remaining".format(frame_num, i, num_frames, round(100 * fraction_done,1), round(est_remain)))


                if success:
                    #i +=1
                    num_not_success = 0
                elif num_not_success >= num_not_success_lim:
                    # If unable to read multiple frames in a row
                    print("Unable to read multiple frames")
                    break
                else:
                    num_not_success += 1

                #
                if i > num_frames:
                    break   # This condition will never happen
                elif frame_num >= len(self.stab_transform):
                    print("No more stabilization data. Stopping render here.")
                    break

                if success and i > 0:

                    #if scale != 1:
                    #    frame = cv2.resize(frame, (int(self.width * scale),int(self.height*scale)), interpolation=cv2.INTER_LINEAR)

                    #frame_undistort = cv2.remap(frame, tempmap1, tempmap2, interpolation=cv2.INTER_LINEAR, # INTER_CUBIC
                    #                              borderMode=cv2.BORDER_CONSTANT)

                    fac = fov_scale

                    tmap1, tmap2 = self.undistort.get_maps((fac)*fcorr[frame_num],
                                                            new_img_dim=(self.orig_dimension[0], self.orig_dimension[1]),
                                                            output_dim=out_size,
                                                            update_new_K = False, quat = self.stab_transform[frame_num],
                                                            focalCenter = focalCenter[frame_num])

                    #tmap1, tmap2 = self.undistort.get_maps(self.undistort_fov_scale,new_img_dim=(int(self.width * scale),int(self.height*scale)), update_new_K = False, quat = self.stab_transform[frame_num])

                    if (i-1) % self.hyperlapse_multiplier == 0 and self.hyperlapse_num_blended_frames > 1:
                        # Reset frame at beginning of hyperlapse range
                        print("reset")
                        frame_temp = frame_temp * 0.0


                    #frame = cv2.resize(frame, (int(self.width * scale),int(self.height*scale)), interpolation=cv2.INTER_LINEAR)
                    if (i-1) % self.hyperlapse_multiplier < self.hyperlapse_num_blended_frames:
                        #print(f"adding frame {i}")

                        # Process using integers for speed
                        frame_out = cv2.remap(frame, tmap1, tmap2, interpolation=cv2.INTER_LINEAR, dst=frame_out, # INTER_CUBIC
                                                    borderMode=borderMode, borderValue=borderValue)

                        if self.hyperlapse_num_blended_frames > 1:
                            # process using floats
                            frame_temp += 1/(self.hyperlapse_num_blended_frames) * frame_out.astype(np.float64)


                    if debug_text and ((i-1) - self.hyperlapse_num_blended_frames + 1) % self.hyperlapse_multiplier == 0:
                        # Add debug text to last frame only
                        topleft = ( int(out_width/2*(1-fac)), int(out_height/2*(1-fac)) )
                        bottomright = ( int(out_width/2*(1+fac)), int(out_height/2*(1+fac)) )
                        frame_out = cv2.rectangle(frame_out, topleft,
                                                            bottomright, (255,0,0), 3)

                        frame_out = cv2.putText(frame_out, "{} | {:0.1f} s ({})".format(__version__, frame_num/self.fps, frame_num),
                                                (5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(200,200,200),2)


                    size = np.array(frame_out.shape)

                    # if last frame
                    if ((i-1) - self.hyperlapse_num_blended_frames + 1) % self.hyperlapse_multiplier == 0:

                        # Convert to int
                        if self.hyperlapse_num_blended_frames > 1:
                            frame_out = frame_temp.astype(np.uint8)

                        if split_screen and False: # Disable for now

                            # Fix border artifacts
                            frame_undistort = frame_undistort[crop[1]:crop[1]+out_size[1]* scale, crop[0]:crop[0]+out_size[0]* scale]
                            frame = cv2.resize(frame_undistort, ((int(size[1]), int(size[0]))))
                            concatted = cv2.resize(cv2.hconcat([frame_out,frame],2), (int(out_size[0]*2*scale),int(out_size[1]*scale)))

                            out.write(concatted)
                            if display_preview:
                                # Resize if preview is huge
                                if concatted.shape[1] > 1280:
                                    concatted = cv2.resize(concatted, (1280, int(concatted.shape[0] * 1280 / concatted.shape[1])), interpolation=cv2.INTER_NEAREST)
                                cv2.imshow("Before and After", concatted)
                                cv2.waitKey(2)
                        else:

                            try:
                                out.write(frame_out)
                            except Exception as e:
                                print("Failed to write frame. Aborting render")
                                print(e)
                                break

                            if display_preview:
                                if frame_out.shape[1] > 1280:
                                    frame_preview = cv2.resize(frame_out, (1280, int(frame_out.shape[0] * 1280 / frame_out.shape[1])), interpolation=cv2.INTER_NEAREST)
                                else:
                                    frame_preview = frame_out

                                if type(viewer_thread) == type(None):
                                    cv2.imshow("Stabilized? Double press Q to stop render", frame_preview)
                                else:
                                    try:
                                        viewer_thread.frame = frame_preview
                                        viewer_thread.update_once = True
                                    except Exception as e:
                                        print("Failed to display preview")
                                        print(e)
                                
                                key = cv2.waitKey(1)
                            
                                # Double press Q to exit
                                if key == 113 and quit_button:
                                    break
                                elif key == 113:
                                    time.sleep(0.3)
                                    quit_button = True
                                else:
                                    quit_button = False
            
            except KeyboardInterrupt:
                print("terminating render")
                break
        
        # When everything done, release the capture
        #out.release()
        print("Render finished")
        cv2.destroyAllWindows()
        out.close()
    
        if type(viewer_thread) != type(None):
            viewer_thread.map_function_enable = old_map_enable_setting


        if audio:
            time.sleep(.5)
            ffmpeg_command = [
                "-y",
                "-i",
                self.videopath,
                "-ss",
                str(tstart / self.fps),
                "-to",
                str(tend / self.fps),
                "-vn",
                "-acodec",
                "copy",
                "audio.mp4"
            ]
            out.execute_ffmpeg_cmd(ffmpeg_command)
            ffmpeg_command = [
                "-y",
                "-i",
                outpath,
                "-i",
                "audio.mp4",
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                outpath + "_a.mp4",
            ]  # `-y` parameter is to overwrite outputfile if exists

            # execute FFmpeg command
            out.execute_ffmpeg_cmd(ffmpeg_command)
            os.replace(outpath + "_a.mp4", outpath)
            os.remove("audio.mp4")

            print("Audio exported")

    def export_gyroflow_file(self, filename=None, out_size = (1920,1080), smoothingFocus=2.0, zoom=1.0):

        print("Exporting gyroflow data file")

        if type(self.undistort) == type(None) or type(self.integrator) == type(None) or type(self.new_integrator) == type(None) or type(self.stab_transform) == type(None):
            print("Unable to export a data file")

        gyroflow_data = {}
        gyroflow_data["title"] = "Gyroflow data file"
        gyroflow_data["videofile"] = os.path.split(self.videopath)[-1]
        gyroflow_data["calibration_data"] = self.undistort.get_calibration_data()
        gyroflow_data["video_rotate_code"] = self.video_rotate_code
        gyroflow_data["gyro_lpf_cutoff"] = self.gyro_lpf_cutoff
        gyroflow_data["date"] = str(date.today())

        video_info = {
            "orig_w": self.orig_dimension[0],
            "orig_h": self.orig_dimension[1],
            "process_w": self.process_dimension[0],
            "process_h": self.process_dimension[0],
            "fps": self.fps,
            "num_frames": self.num_frames
        }

        gyroflow_data["video_info"] = video_info


        gyroflow_data["sensor_DOF"] = 6 if type(self.acc_data) != type(None) else 3 # 6 DOF if acc is available
        raw_imu = np.round(self.new_integrator.get_raw_gyro_acc(), 5)
        gyroflow_data["raw_imu"] = raw_imu.tolist() # time is already corrected

        gyroflow_data["stab_summary"] = self.smoothing_algo.get_summary()

        #get_interpolated_orientations
        interpolated_times, interpolated_orientations =  self.new_integrator.get_interpolated_orientations(start=0,interval = 1/self.fps)

        # [index, time, w, x, y, z]
        time_orientation = np.hstack([np.arange(interpolated_orientations.shape[0])[...,None],
                                                np.array(interpolated_times)[...,None],
                                                interpolated_orientations ])

        gyroflow_data["frame_orientation"] = time_orientation.tolist()

        if self.new_integrator.acc_available:
            print("Exporting complementary filter orientations")
            self.new_integrator.integrate_all(use_acc=True)

            interpolated_times, interpolated_orientations =  self.new_integrator.get_interpolated_orientations(start=0,interval = 1/self.fps)

            time_orientation = np.hstack([np.arange(interpolated_orientations.shape[0])[...,None],
                                                np.array(interpolated_times)[...,None],
                                                interpolated_orientations ])

            gyroflow_data["frame_orientation_filtered"] = time_orientation.tolist()
        else:
            gyroflow_data["frame_orientation_filtered"] = []



        stab_transform = np.array(self.stab_transform)

        adaptZ = AdaptiveZoom(fisheyeCalibrator=self.undistort)

        fcorr, focalCenter = adaptZ.compute(quaternions=self.stab_transform, output_dim=out_size, fps=self.fps,
                                                        smoothingFocus=smoothingFocus, debug_plots=False)

        gyroflow_data["min_fcorr"] = np.min(fcorr)

        #print(self.stab_transform)
        #print(stab_transform)
        # [index, time, smoothed_fov, w, x, y, z]
        time_stab_transform = np.hstack([np.arange(stab_transform.shape[0])[...,None],
                                         np.array(self.times)[...,None],
                                         fcorr[...,None],
                                         stab_transform])

        time_stab_transform = np.round(time_stab_transform, 5)

        gyroflow_data["stab_transform"] = time_stab_transform.tolist()

        # General format description:
        # Contains the following:
        # * Gyroflow version info
        # * General info, video filename, camera name
        # * Camera parameters (Json from calibration utility)
        # * raw gyro/acc data
        # * Processed orientation data
        # * Per frame
        # Try to make
        if not filename:
            filename = self.videopath + ".gyroflow"

        with open(filename, 'w') as outfile:
            json.dump(
            gyroflow_data,
            outfile,
            indent=1,
            separators=(',', ': ')
        )

        print("Finished exporting")

    def import_gyroflow_file(self, filename="file.gyroflow"):
        print("Loading Gyroflow data file")
        # Load absolutely everything

        with open(filename, "r") as infile:


            try:
                gyroflow_data = json.load(infile)
                calibration_data = gyroflow_data["calibration_data"]
                self.undistort.load_calibration_data(calibration_data,True)

                self.video_rotate_code = gyroflow_data["video_rotate_code"]

                initial_orientation = Rotation.from_euler('zxy', [0,0,np.pi/2]).as_quat()
                initial_orientation[[0,1,2,3]] = initial_orientation[[3,0,1,2]]

                motion_dof = gyroflow_data["sensor_DOF"]
                raw_imu = gyroflow_data["raw_imu"]
                raw_imu = np.array(raw_imu)
                self.gyro_data = raw_imu[:,[0,1,2,3]]
                if motion_dof >= 6:
                    self.acc_data = raw_imu[:,[0,4,5,6]]

                self.integrator = GyroIntegrator(self.gyro_data, zero_out_time=False, initial_orientation=initial_orientation, acc_data=self.acc_data)
                self.new_integrator = self.integrator

                self.new_integrator.integrate_all(use_acc=False)

                #self.times = None
                #self.stab_transform = None
                self.use_gyroflow_data_file = True

            except KeyError: # TODO change?
                print("Couldn't load gyroflow data file")
                return False
            return True

    def release(self):
        self.cap.release()


class OnlyUndistort:
    def __init__(self, videopath, calibrationfile, fov_scale = 1.5):
        self.undistort_fov_scale = fov_scale
        self.cap = cv2.VideoCapture(videopath)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.undistort = FisheyeCalibrator()
        self.undistort.load_calibration_json(calibrationfile, True)
        self.map1, self.map2 = self.undistort.get_maps(self.undistort_fov_scale,new_img_dim=(self.width,self.height))

    def renderfile(self, starttime, stoptime, outpath = "Stabilized.mp4", out_size = (1920,1080), split_screen = False,
                   bitrate_mbits = 20, display_preview = False, scale=1, vcodec = "libx264", vprofile="main", pix_fmt = "",
                   debug_text = False, custom_ffmpeg = ""):

        export_out_size = (int(out_size[0]*scale), int(out_size[1]*scale))

        if vcodec == "libx264":
            output_params = {
                "-input_framerate": self.fps,
                "-vcodec": "libx264",
                "-profile:v": vprofile,
                "-crf": "1",  # Can't use 0 as it triggers "lossless" which does not allow  -maxrate
                "-maxrate": "%sM" % bitrate_mbits,
                "-bufsize": "%sM" % int(bitrate_mbits * 1.2),
                "-pix_fmt": "yuv420p",
            }
        elif vcodec == "h264_nvenc":
            output_params = {
                "-input_framerate": self.fps,
                "-vcodec": "h264_nvenc",
                "-profile:v": vprofile,
                "-rc:v": "cbr",
                "-b:v": "%sM" % bitrate_mbits,
                "-bufsize:v": "%sM" % int(bitrate_mbits * 2),
            }
        elif vcodec == "h264_vaapi":
            output_params = {
                "-input_framerate": self.fps,
                "-vcodec": "h264_vaapi",
                "-vaapi_device": "/dev/dri/renderD128",
                "-profile:v": vprofile,
                "-b:v": "%sM" % bitrate_mbits,
            }
        elif vcodec == "h264_videotoolbox":
            output_params = {
                "-input_framerate": self.fps,
                "-vcodec": "h264_videotoolbox",
                "-profile:v": vprofile,
                "-b:v": "%sM" % bitrate_mbits,
                }
        elif vcodec == "prores_ks":
            output_params = {
                "-input_framerate": self.fps,
                "-vcodec": "prores_ks",
                "-profile:v": vprofile,
            }
        else:
            output_params = {}

        if pix_fmt:
            output_params["-pix_fmt"] = pix_fmt  # override pix_fmt if user needs to

        if custom_ffmpeg:
            output_params = eval(custom_ffmpeg)
            output_params["-input_framerate"] = self.fps

        # non compression fallback fps
        output_params["-fps"] = self.fps

        out = WriteGear(output_filename=outpath, **output_params)
        output_params["custom_ffmpeg"] = vidgearHelper.get_valid_ffmpeg_path()
        crop = (int(scale*(self.width-out_size[0])/2), int(scale*(self.height-out_size[1])/2))


        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(starttime * self.fps))
        time.sleep(0.1)

        num_frames = int((stoptime - starttime) * self.fps)

        tempmap1 = cv2.resize(self.map1, (int(self.map1.shape[1]*scale), int(self.map1.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
        tempmap2 = cv2.resize(self.map2, (int(self.map2.shape[1]*scale), int(self.map2.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)

        i = 0
        while(True):
            # Read next frame
            frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            success, frame = self.cap.read()

            # Getting frame_num _before_ cap.read gives index of the read frame.

            if i % 5 == 0:
                print("frame: {}, {}/{} ({}%)".format(frame_num, i, num_frames, round(100 * i/num_frames,1)))

            if success:
                i +=1

            if i > num_frames:
                break

            if success and i > 0:

                if scale != 1:
                    frame = cv2.resize(frame, (int(self.width * scale),int(self.height*scale)), interpolation=cv2.INTER_LINEAR)

                frame_out = cv2.remap(frame, tempmap1, tempmap2, interpolation=cv2.INTER_LINEAR, # INTER_CUBIC
                                              borderMode=cv2.BORDER_CONSTANT)

                frame_out = frame_out[crop[1]:crop[1]+out_size[1] * scale, crop[0]:crop[0]+out_size[0]* scale]

                size = np.array(frame_out.shape)

                out.write(frame_out)
                if display_preview:
                    if frame_out.shape[1] > 1280:
                        frame_out = cv2.resize(frame_out, (1280, int(frame_out.shape[0] * 1280 / frame_out.shape[1])), interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("Stabilized?", frame_out)
                    cv2.waitKey(2)

        # When everything done, release the capture
        #out.release()
        cv2.destroyAllWindows()
        out.close()

        self.cap.release()


class GPMFStabilizer(Stabilizer):
    def __init__(self, videopath, calibrationfile, gyro_path, hero = 8, fov_scale = 1.6, gyro_lpf_cutoff = -1, video_rotation = -1):

        super().__init__(videopath, calibrationfile, gyro_path, fov_scale = fov_scale, gyro_lpf_cutoff = gyro_lpf_cutoff, video_rotation = video_rotation)

        # Get gyro data
        self.gpmf = Extractor(gyro_path)
        self.gyro_data = self.gpmf.get_gyro(True)

        # Hero 6??
        if hero == 6:
            self.gyro_data[:,1] = self.gyro_data[:,1]
            self.gyro_data[:,2] = self.gyro_data[:,2]
            self.gyro_data[:,3] = self.gyro_data[:,3]
        if hero == 7:
            self.gyro_data[:,1] = self.gyro_data[:,1]
            self.gyro_data[:,2] = self.gyro_data[:,2]
            self.gyro_data[:,3] = self.gyro_data[:,3]
        elif hero == 5:
            self.gyro_data[:,1] = -self.gyro_data[:,1]
            self.gyro_data[:,2] = self.gyro_data[:,2]
            self.gyro_data[:,3] = self.gyro_data[:,3]
            self.gyro_data[:,[2, 3]] = self.gyro_data[:,[3, 2]]

        elif hero == 8:
            # Hero 8??
            self.gyro_data[:,[2, 3]] = self.gyro_data[:,[3, 2]]
            self.gyro_data[:,2] = -self.gyro_data[:,2]
        elif hero == 9:
            self.gyro_data[:,1] = -self.gyro_data[:,1]
            self.gyro_data[:,2] = self.gyro_data[:,2]
            self.gyro_data[:,3] = self.gyro_data[:,3]
            self.gyro_data[:,[2, 3]] = self.gyro_data[:,[3, 2]]


        if self.gyro_lpf_cutoff > 0:
            self.filter_gyro()

        # Other attributes
        initial_orientation = Rotation.from_euler('xyz', [0, 0, 180], degrees=True).as_quat()

        self.integrator = GyroIntegrator(self.gyro_data,initial_orientation=initial_orientation)
        self.integrator.integrate_all(use_acc=False)
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

        self.times, self.stab_transform = self.integrator.get_interpolated_stab_transform(start=-gyro_start,interval = interval) # 2.2/30 , -1/30


class InstaStabilizer(Stabilizer):
    def __init__(self, videopath, calibrationfile, gyro_path, fov_scale = 1.6, gyro_lpf_cutoff = -1, video_rotation = -1, InstaType=""):

        super().__init__(videopath, calibrationfile, gyro_path, fov_scale = fov_scale, gyro_lpf_cutoff = gyro_lpf_cutoff, video_rotation = video_rotation)

        # Get gyro data
        if InstaType=="smo4k":
            gyro_data_input, self.acc_data = insta360_util.get_insta360_gyro_data(videopath, filterArray=[])
        elif InstaType=="insta360 oner":
            gyro_data_input, self.acc_data = insta360_util.get_insta360_gyro_data(videopath, filterArray=[], revertIMU=False)
        else:
            # Assume SMO4K - For no real reason....
            gyro_data_input, self.acc_data = insta360_util.get_insta360_gyro_data(videopath, filterArray=[])

        # Coverting gyro to XYZ to -Z,-X,Y
        self.gyro_data = np.empty([len(gyro_data_input), 4])
        self.gyro_data[:,0] = gyro_data_input[:,0][:]
        self.gyro_data[:,1] = gyro_data_input[:,2][:] * -1
        self.gyro_data[:,2] = gyro_data_input[:,3][:]
        self.gyro_data[:,3] = gyro_data_input[:,1][:] * -1

        hero = 0


        if self.gyro_lpf_cutoff > 0:
            self.filter_gyro()


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

        self.times, self.stab_transform = self.integrator.get_interpolated_stab_transform(start=-gyro_start,interval = interval) # 2.2/30 , -1/30


class MultiStabilizer(Stabilizer):
    def __init__(self, videopath, calibrationfile, logpath, fov_scale = 1.6, cam_angle_degrees=0, initial_offset=0, gyro_lpf_cutoff = 100, logtype="Gyroflow IMU log", logvariant="", video_rotation = -1):

        super().__init__(videopath, calibrationfile, logpath, fov_scale = fov_scale, gyro_lpf_cutoff = gyro_lpf_cutoff, video_rotation = video_rotation)

        # Get gyro data
        print(logpath)

        # quick fix
        # TODO: integrate with gyrolog.py for modularity

        self.log_reader = gyrolog.get_log_reader_by_name(logtype)
        if not self.log_reader:
            print(f"Failed to initiate log reader of type {logtype}")
            return

        self.log_reader.set_variant(logvariant)
        self.log_reader.set_cam_up_angle(cam_angle_degrees, degrees=True)

        extracted = self.log_reader.extract_log(logpath)

        if not extracted:
            print("Failed to extract and parse motion data")
            return

        self.gyro_data = self.log_reader.get_transformed_gyro()

        # If no data:
        if type(self.gyro_data) == type(None):
            print("No valid gyro data")
            return
        elif self.gyro_data.shape[0] < 2:
            print("No valid gyro data")
            return
        self.gyro_data = impute_gyro_data(self.gyro_data)
        self.acc_data = self.log_reader.get_transformed_acc()
        if type(self.acc_data) != type(None):
            self.acc_data = impute_gyro_data(self.acc_data)



        if self.gyro_lpf_cutoff > 0:
            self.filter_gyro()

        # Other attributes
        initial_orientation = Rotation.from_euler('zxy', [0,0,np.pi/2]).as_quat()
        initial_orientation[[0,1,2,3]] = initial_orientation[[3,0,1,2]]


        self.integrator = GyroIntegrator(self.gyro_data,initial_orientation=initial_orientation, acc_data=self.acc_data)
        self.integrator.integrate_all(use_acc=False)
        self.times = None
        self.stab_transform = None

        self.initial_offset = initial_offset


class BBLStabilizer(Stabilizer):
    def __init__(self, videopath, calibrationfile, bblpath, fov_scale = 1.6, cam_angle_degrees=0, initial_offset=0, use_csv=False, gyro_lpf_cutoff = 200, logtype="", video_rotation = -1, use_raw_gyro_data=False):

        super().__init__(videopath, calibrationfile, bblpath, fov_scale = fov_scale, gyro_lpf_cutoff = gyro_lpf_cutoff, video_rotation = video_rotation)

        # Get gyro data
        print(bblpath)

        # quick fix
        cam_angle_degrees = -cam_angle_degrees

        # TODO: integrate with gyrolog.py for modularity

        if use_csv:
            with open(bblpath) as bblcsv:
                gyro_index = None

                csv_reader = csv.reader(bblcsv)
                for i, row in enumerate(csv_reader):
                    #print(row)

                    stripped_row = [field.strip() for field in row]
                    if stripped_row[0] == "loopIteration":
                        if use_raw_gyro_data:
                            gyro_index = stripped_row.index('debug[0]')
                            print('Using raw gyro data')
                        else:
                            gyro_index = stripped_row.index('gyroADC[0]')
                            print('Using filtered gyro data')

                        break

                data_list = []
                gyroscale = np.pi/180
                r  = Rotation.from_euler('x', cam_angle_degrees, degrees=True)
                for row in csv_reader:

                    gx = float(row[gyro_index+1])* gyroscale
                    gy = float(row[gyro_index+2])* gyroscale
                    gz = float(row[gyro_index]) * gyroscale

                    to_rotate = [-(gx),
                                    (gy),
                                    -(gz)]

                    rotated = r.apply(to_rotate)

                    f = [float(row[1]) / 1000000,
                            rotated[0],
                            rotated[1],
                            rotated[2]]

                    data_list.append(f)

                self.gyro_data = np.array(data_list)

        elif logtype == "gyroflow":
            with open(bblpath) as csvfile:
                next(csvfile)
                lines = csvfile.readlines()

                data_list = []
                gyroscale = 0.070 * np.pi/180 # plus minus 2000 dps 16 bit two's complement. 70 mdps/LSB per datasheet.
                #gyroscale = 0.070/4 * np.pi/180 # 500 dps
                r  = Rotation.from_euler('x', cam_angle_degrees, degrees=True)

                for line in lines:
                    splitdata = [float(x) for x in line.split(",")]
                    t = splitdata[0]/1000
                    gx = splitdata[1] * gyroscale
                    gy = splitdata[2] * gyroscale
                    gz = splitdata[3] * gyroscale
                    # Z: roll
                    # X: yaw
                    # y: pitch

                    data_list.append([t, gx, gy, gz])
                #from scipy.signal import resample
                #gyro_arr = np.array(data_list)
                #x, t = resample(gyro_arr[:,1:], 22 * 200,gyro_arr[:,0])
                #self.gyro_data = np.column_stack((t,x))
                self.gyro_data = np.array(data_list)
                print(self.gyro_data)
        elif logtype == "runcam":
            with open(bblpath) as csvfile:
                next(csvfile)

                lines = csvfile.readlines()

                data_list = []
                #gyroscale = 0.070 * np.pi/180 # plus minus 2000 dps 16 bit two's complement. 70 mdps/LSB per datasheet.
                gyroscale = 500 / 2**15 * np.pi/180 # 500 dps
                r  = Rotation.from_euler('x', cam_angle_degrees, degrees=True)

                for line in lines:
                    splitdata = [float(x) for x in line.split(",")]
                    t = splitdata[0]/1000

                    # RC5
                    gx = splitdata[3] * gyroscale
                    gy = -splitdata[1] * gyroscale
                    gz = splitdata[2] * gyroscale

                    # Z: roll
                    # X: yaw
                    # y: pitch

                    data_list.append([t, gx, gy, gz])
                #from scipy.signal import resample
                #gyro_arr = np.array(data_list)
                #x, t = resample(gyro_arr[:,1:], 22 * 200,gyro_arr[:,0])
                #self.gyro_data = np.column_stack((t,x))
                self.gyro_data = np.array(data_list)
                print(self.gyro_data)

        elif logtype == "gocam":
            with open(bblpath) as csvfile:
                next(csvfile)
                lines = csvfile.readlines()
                data_list = []
                #gyroscale = 0.070 * np.pi/180 # plus minus 2000 dps 16 bit two's complement. 70 mdps/LSB per datasheet.
                gyroscale = 500 / 2**15 * np.pi/180 # 500 dps
                r  = Rotation.from_euler('x', cam_angle_degrees, degrees=True)

                for line in lines:
                    splitdata = [float(x) for x in line.split(",")]
                    t = splitdata[0]/1000
                    # RC/IF test
                    gx = -splitdata[3] * gyroscale
                    gy = -splitdata[1] * gyroscale
                    gz = -splitdata[2] * gyroscale

                    data_list.append([t, gx, gy, gz])

                self.gyro_data = np.array(data_list)
                print(self.gyro_data)
        else:
            try:
                self.bbe = BlackboxExtractor(bblpath)
                self.gyro_data = self.bbe.get_gyro_data(cam_angle_degrees=cam_angle_degrees)
            except ValueError:
                print("Error reading raw blackbox file. Try converting to CSV in blackbox explorer")

        # This seems to make the orientation match. Implement auto match later
        #self.gyro_data[:,[2, 3]] = self.gyro_data[:,[3, 2]]
        #self.gyro_data[:,2] = -self.gyro_data[:,2]

        #self.gyro_data[:,[2, 3]] = self.gyro_data[:,[3, 2]]
        #self.gyro_data[:,2] = self.gyro_data[:,2]
        #self.gyro_data[:,3] = -self.gyro_data[:,3]

        if self.gyro_lpf_cutoff > 0:
            self.filter_gyro()

        # Other attributes
        initial_orientation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()

        self.gyro_data = impute_gyro_data(self.gyro_data)

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

        self.times, self.stab_transform = self.integrator.get_interpolated_stab_transform(start=2.56+0.07,interval = 1/59.94)


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
        self.undistort = StandardCalibrator() #FisheyeCalibrator()
        self.undistort.load_calibration_json(calibrationfile, True)
        self.map1, self.map2 = self.undistort.get_maps(1.6,new_img_dim=(self.width,self.height))

        # Other attributes
        self.times = None
        self.stab_transform = None


    def stabilization_settings(self, smooth = 0.65):

        frame_idx, transforms = self.optical_flow_comparison(112 * 30, 29 * 30)

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

        self.times, self.stab_transform = self.integrator.get_stabilize_transform()


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

        for i in tqdm(range(analyze_length), desc="Analyzing frame", colour="blue"):
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)



            frame_id = (int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
            frame_time = (self.cap.get(cv2.CAP_PROP_POS_MSEC)/1000)

            succ, curr = self.cap.read()


            #if i % 10 == 0:
            #    print("Analyzing frame: {}/{}".format(i,analyze_length))

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
                src_pts = prev_pts #self.undistort.undistort_points(prev_pts, new_img_dim=(self.width,self.height))
                dst_pts = curr_pts #self.undistort.undistort_points(curr_pts, new_img_dim=(self.width,self.height))
                #H, mask = cv2.findHomography(src_pts, dst_pts)
                #retval, rots, trans, norms = self.undistort.decompose_homography(H, new_img_dim=(self.width,self.height))


                # rots contains for solutions for the rotation. Get one with smallest magnitude. Idk
                roteul = None
                #smallest_mag = 1000
                #for rot in rots:
                #    thisrot = Rotation.from_matrix(rots[0]) # first one?
                #
                #    if thisrot.magnitude() < smallest_mag and thisrot.magnitude() < 0.3:
                #        roteul = Rotation.from_matrix(rot).as_euler("xyz")
                #        smallest_mag = thisrot.magnitude()

                filtered_src = []
                filtered_dst = []

                for i in range(src_pts.shape[0]):
                    # if both points are within frame
                    if (0 < src_pts[i,0,0] < self.width) and (0 < dst_pts[i,0,0] < self.width) and (0 < src_pts[i,0,1] < self.height) and (0 < dst_pts[i,0,1] < self.height):
                        filtered_src.append(src_pts[i,:])
                        filtered_dst.append(dst_pts[i,:])


                self.use_essential_matrix = True

                if self.use_essential_matrix:
                    R1, R2, t = self.undistort.recover_pose(np.array(filtered_src), np.array(filtered_dst), new_img_dim=(self.width,self.height))

                    rot1 = Rotation.from_matrix(R1)
                    rot2 = Rotation.from_matrix(R2)

                    if rot1.magnitude() < rot2.magnitude():
                        roteul = rot1.as_euler("xyz")
                    else:
                        roteul = rot2.as_euler("xyz")


                #m, inliers = cv2.estimateAffine2D(src_pts, dst_pts)

                #dx = m[0,2]
                #dy = m[1,2]

                # Extract rotation angle
                #da = np.arctan2(m[1,0], m[0,0])
                #transforms.append([dx,dy,da])
                transforms.append(list(roteul))
                prev_gray = curr_gray

            else:
                print("Frame {}".format(i))

        transforms = np.array(transforms)
        return frame_idx, transforms


    def renderfile(self, starttime, stoptime, outpath = "Stabilized.mp4", out_size = (1920,1080)):

        out = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'mp4v'), 30, (out_size[0]*2,out_size[1]))
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
                concatted = cv2.resize(cv2.hconcat([frame_out,frame],2), (out_size[0]*2,out_size[1]))
                out.write(concatted)
                cv2.imshow("Before and After", concatted)
                cv2.waitKey(5)

        # When everything done, release the capture
        out.release()
        cv2.destroyAllWindows()

    def release(self):
        self.cap.release()


def find_gyroflow_data_file(videofile="in.mp4"):
    if os.path.isfile(videofile + ".gyroflow"):
        return videofile + ".gyroflow"

    return ""


if __name__ == "__main__":
    infile_path = "test_clips/Runcam/RC_0036_filtered.MP4"
    log_guess, log_type, variant = gyrolog.guess_log_type_from_video(infile_path)
    if not log_guess:
        print("Can't guess log")
        exit()

    stab = MultiStabilizer(infile_path, "camera_presets/RunCam/DEV_Runcam_5_Orange_4K_30FPS_XV_16by9_stretched.json", log_guess, gyro_lpf_cutoff = 50, logtype=log_type, logvariant=variant)

    stab.full_auto_sync_parallel()

    stab.export_gyroflow_file()

    stab.renderfile(30, 70, "autosync_1.mp4", out_size = (1920,1080))

    # insta360 test

    #stab = InstaStabilizer("test_clips/insta360.mp4", "camera_presets/SMO4K_4K_Wide43.json", gyrocsv="test_clips/insta360_gyro.csv")
    #stab.auto_sync_stab(0.985,100 *24, 119 * 24, 70)
    #stab.renderfile(100, 125, "insta360test4split.mp4",out_size = (2560,1440), split_screen=False, scale=0.5)

    #exit()
    #stab = GPMFStabilizer("test_clips/GX016017.MP4", "camera_presets/Hero_7_2.7K_60_4by3_wide.json") # Walk
    #stab = GPMFStabilizer("test_clips/GX016015.MP4", "camera_presets/gopro_calib2.JSON", ) # Rotate around
    #stab = GPMFStabilizer("test_clips/GX010010.MP4", "camera_presets/gopro_calib2.JSON", hero6=False) # Parking lot

    #stab = BBLStabilizer("test_clips/MasterTim17_caddx.mp4", "camera_presets/Nikon/Nikon_D5100_Nikkor_35mm_F_1_8_1280x720.json", "test_clips/starling.csv", use_csv=False, logtype = "gyroflow")

    #undistortTest = OnlyUndistort("test_clips/MasterTim17_caddx.mp4", "camera_presets/Nikon/Nikon_D5100_Nikkor_35mm_F_1_8_1280x720.json",fov_scale=1)
    #undistortTest.renderfile(0, 5, "mastertim_out.mp4",out_size = (1920,1080), split_screen = False, scale=1, display_preview = True)
    exit()
    #stab.stabilization_settings(smooth = 0.8)
    # stab.auto_sync_stab(0.89,25*30, (2 * 60 + 22) * 30, 50) Gopro clips

    #stab.auto_sync_stab(0.21,1*30, 25 * 30, 50) # FPV clip
    #stab.stabilization_settings()

    # Visual stabilizer test
    # stab = OpticalStabilizer("test_clips/P1000004nurk.MP4", "camera_presets/BGH1_test.json")


    # Camera undistortion stuff
    #stab.undistort = FisheyeCalibrator()
    #stab.undistort.load_calibration_json("camera_presets/Hero_7_2.7K_60_4by3_wide.json", True)
    #stab.map1, stab.map2 = stab.undistort.get_maps(2.6,new_img_dim=(stab.width,stab.height))


    #stab.renderfile(24, 63, "parkinglot_stab_3.mp4",out_size = (1920,1080))
    #stab.renderfile(0, 25, "mastertim_out.mp4",out_size = (1920,1080), split_screen = False, scale=1, display_preview = True)
    #stab.stabilization_settings(smooth=0.6)
    #stab.renderfile(113, 130, "nurk_stabi3.mp4",out_size = (3072,1728))

    #stab.release()

    # 20 / self.fps: 0.042
    # 200 / self.fps: -0.048
