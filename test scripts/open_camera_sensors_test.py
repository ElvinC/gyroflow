
import stabilizer
import smoothing_algos
import matplotlib.pyplot as plt
from gyro_integrator import GyroIntegrator
import cv2
import gyrolog

path = r"D:\git\FPV\videos\opencamera\VID_20210827_211112.mp4"
path = r"D:\git\FPV\videos\opencamera\VID_20210802_120804.mp4"
path = r"D:\git\FPV\videos\opencamera\VID_20210828_131205.mp4"
path = r"D:\git\FPV\videos\opencamera\VID_20210828_155108.mp4"
path = r"D:\git\FPV\videos\opencamera\VID_20210828_155424.mp4"
path = r"D:\git\FPV\videos\opencamera\VID_20210828_160247.mp4"

lens_calibration = r"Google_Pixel_4a_HD.json"
log_reader = gyrolog.GyroflowGyroLog()
log_guess, log_type, variant = gyrolog.guess_log_type_from_video(path)
print(log_guess, log_type, variant)



# stab = stabilizer.OpenCameraSensors(path, lens_calibration, gyro_lpf_cutoff=50, video_rotation=cv2.ROTATE_90_COUNTERCLOCKWISE)
# stab.set_initial_offset(0)
# stab.set_rough_search(0.1)
# stab.set_num_frames_skipped(1)
#
# stab.set_smoothing_algo()
#
# max_fitting_error = 1000
#
# plt.plot(stab.gyro_data[:, 0], stab.gyro_data[:, 1:])
# plt.plot(stab.acc_data[:, 0], stab.acc_data[:, 1:])
# plt.legend(['gx', 'gy', 'gz', 'ax', 'ay', 'az'])
# plt.show()

# success = stab.full_auto_sync(max_fitting_error)
# if not stab.smoothing_algo:
#     stab.smoothing_algo = smoothing_algos.PlainSlerp()
# stab.gyro_data[:, 0] = stab.gyro_data[:, 0] + stab.fps * 2
# stab.acc_data[:, 0] = stab.acc_data[:, 0] + stab.fps * 2
# stab.new_integrator = GyroIntegrator(stab.gyro_data[:, 0] + stab.fps * 2, zero_out_time=False,
#                                      initial_orientation=stab.initial_orientation,
#                                      acc_data=stab.acc_data)
#
# stab.new_integrator.integrate_all(use_acc=stab.smoothing_algo.require_acceleration)
#
# stab.new_integrator.set_smoothing_algo(stab.smoothing_algo)
# stab.times, stab.stab_transform = stab.new_integrator.get_interpolated_stab_transform(start=0, interval=1 / stab.fps)

# stab.renderfile(0, 10, out_size=(1500, 750), zoom=.5)