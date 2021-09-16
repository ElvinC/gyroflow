import cv2
import numpy as np
import multiprocessing as mp
import time
from calibrate_video import FisheyeCalibrator
import gyrolog
from stabilizer import MultiStabilizer, fast_gyro_cost_func, better_gyro_cost_func, optical_flow, estimate_gyro_offset


from tqdm import tqdm
from scipy.spatial.transform import Rotation
from scipy import signal, interpolate
import matplotlib.pyplot as plt

class ParallelSync:
    def __init__(self, stab, num_frames_analyze, debug_plots=True):
        self.videofile = stab.videopath
        self.offset = 0

        self.is_adding = False
        self.index_list = []
        self.OF_list = []
        self.undistort = stab.undistort
        self.height = 2028
        self.width = 2704
        self.num_frames_skipped = stab.num_frames_skipped
        self.num_frames_analyze = num_frames_analyze

        self.gyro_times = stab.integrator.get_raw_data("t")
        self.gyro_data = stab.integrator.get_raw_data("xyz")
        self.rough_sync_search_interval = stab.rough_sync_search_interval
        self.initial_offset = stab.initial_offset
        self.gyro_rate = stab.integrator.gyro_sample_rate
        self.fps = stab.fps
        self.debug_plots = debug_plots
        self.sync_points = stab.get_recommended_syncpoints(self.num_frames_analyze)

        self.process_dimension = stab.process_dimension
        self.video_rotate_code = stab.video_rotate_code


    def optical_flow(self, start_frame, analyze_length):
        return optical_flow(
            self.videofile,
            start_frame,
            self.num_frames_skipped,
            self.video_rotate_code,
            self.process_dimension,
            self.width,
            self.height,
            self.undistort,
            self.gyro_times,
            self.gyro_data,
            self.rough_sync_search_interval,
            self.initial_offset,
            self.gyro_rate,
            self.fps,
            analyze_length=analyze_length,
            debug_plots=True)


    def begin_sync_parallel(self):
        n_proc = min(mp.cpu_count(), len(self.sync_points))
        print(n_proc)
        pool = mp.Pool(n_proc)
        print("Starting parallel auto sync")
        proc_results = pool.starmap(self.optical_flow, self.sync_points)
        return proc_results


if __name__ == "__main__":

    infile_path = r"D:\git\FPV\videos\GH011145.MP4"
    log_guess, log_type, variant = gyrolog.guess_log_type_from_video(infile_path)
    if not log_guess:
        print("Can't guess log")
        exit()

    stab = MultiStabilizer(infile_path, r"D:\git\FPV\GoPro_Hero6_2160p_43.json", log_guess, gyro_lpf_cutoff = -1, logtype=log_type, logvariant=variant)

    start = time.time()
    ps = ParallelSync(stab, 30)
    ps.begin_sync_parallel()
    print(f"time needed for parallel auto sync: {time.time() - start:.2f} s")