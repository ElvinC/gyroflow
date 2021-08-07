import stabilizer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import itertools

file = r"D:\git\FPV\videos\GH011162.MP4"
cam_preset = r"camera_presets\GoPro\GoPro_Hero6_2160p_43.json"

minimum_duration = 3
still_threshold = .05
flippy_threshold = 1
trim_offset = 1

stab = stabilizer.GPMFStabilizer(file, cam_preset, file, hero=6, fov_scale = 1.6, gyro_lpf_cutoff = -1)
x = stab.integrator.get_raw_data('x')
y = stab.integrator.get_raw_data('y')
z = stab.integrator.get_raw_data('z')
t = stab.integrator.get_raw_data('t')
tot = (x**2 + y**2 + z**2)**.5

# masking for the 3 categories still, smooth and flippy
still_mask = tot <= still_threshold
smooth_mask = (still_threshold < tot) & (tot < flippy_threshold)
flippy_mask = tot > flippy_threshold

gyro_rate = len(t) / (t[-1] - t[0])

# get still parts of gyro data
still = np.zeros(len(t))
still[still_mask] = 1
end_points = np.where(np.diff(still) == -1)[0]
start_points = np.where(np.diff(still) == 1)[0]
duration = np.array([sum(1 for _ in group) for key, group in itertools.groupby(still_mask)])[-(int(still[0])-1)::2]

long_still = np.where(duration > gyro_rate * minimum_duration)[0]
trim_start = 0
trim_end = t[-1]

# suggest trim start and end
if len(long_still) == 1:
    if end_points[long_still[0]] > len(t) / 2:
        trim_end = start_points[long_still[-1]] / gyro_rate
    else:
        trim_start = end_points[long_still[0]] / gyro_rate
else:
    trim_start = end_points[long_still[0]] / gyro_rate
    trim_end = start_points[long_still[-1]] / gyro_rate
video_end = int(t[-1])
trim_start = max(trim_start - trim_offset, 0)
trim_end = min(trim_end + trim_offset, int(video_end))

print("Trim start: ", trim_start)
print("Trim end: ", trim_end)

# suggest good sync points. Not working for seperate gyro data
smooth = np.zeros(len(t))
smooth[smooth_mask] = 1
start_points = np.where(np.diff(smooth) == 1)[0]
duration = np.array([sum(1 for _ in group) for key, group in itertools.groupby(smooth_mask)])[-(int(smooth[0])-1)::2]
long_smooth = np.where(duration > gyro_rate * 1)[0]
fps = 29.97
print("\nStart Duration")
for pt in long_smooth:
    print(f"{start_points[pt - 1] / gyro_rate:.2f} {duration[pt] / gyro_rate:.2f}")

# plot
fig, ax = plt.subplots(1, 1, sharey=True, sharex=True)
alpha = .02
ax.plot(t[still_mask], len(t[still_mask])*[-1], '.k', markersize=1, alpha=alpha)
ax.plot(t[smooth_mask], len(t[smooth_mask])*[-.75], marker='.', color='lime', markersize=1, alpha=alpha)
ax.plot(t[flippy_mask], len(t[flippy_mask])*[-.5], '.r', markersize=1, alpha=alpha)
ax.plot(t, tot, 'b')
ax.set(xlabel="time [s]", ylabel="omega_total [rad/s]")
plt.grid()

# create legend
black_patch = mpatches.Patch(color='black', label='still')
green_patch = mpatches.Patch(color='lime', label='smooth')
red_patch = mpatches.Patch(color='red', label='flippy')
orange_line = mlines.Line2D([], [], color='orange', label='trim start')
teal_line = mlines.Line2D([], [], color='darkviolet', label='trim end')
blue_line = mlines.Line2D([], [], color='blue', label='gyro omega_total')
plt.legend(handles=[blue_line, red_patch, green_patch, black_patch, orange_line, teal_line])

# plot trim
ax.axvline(trim_start, color='orange')
ax.axvline(trim_end, color='darkviolet')

plt.show()


