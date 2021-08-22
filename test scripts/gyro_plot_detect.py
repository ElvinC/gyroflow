import stabilizer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas as pd
file = r"D:\git\FPV\videos\GH011160.MP4"

stab = stabilizer.GPMFStabilizer(file, r"D:\git\FPV\GoPro_Hero6_2160p_43.json", file, hero=6, fov_scale = 1.6, gyro_lpf_cutoff = -1)
x = stab.integrator.get_raw_data('x')
y = stab.integrator.get_raw_data('y')
z = stab.integrator.get_raw_data('z')
t = stab.integrator.get_raw_data('t')
tot = (x**2 + y**2 + z**2)**.5
time = 3
still_threshold = .02
trim_offset = 1
still_mask = tot <= still_threshold
flippy_mask = tot > 1
smooth_mask = (still_threshold < tot) & (tot < 1)
rate = len(t) / (t[-1] - t[0])
print(rate)
fig, ax = plt.subplots(1, 1, sharey=True, sharex=True)
# for idx, gyro in enumerate([x, y, z]):
#     idx = idx + 1
#     axes[idx].plot(t, gyro)
#     ylim = axes[idx].get_ylim()
#     mask_flippy = gyro > .5
#     mask_smooth = (gyro <=.5) & (gyro > .02)
#     flippy[mask_flippy] = 1
#     axes[3].plot(t[mask_smooth], len(t[mask_smooth])*[3 * idx], '.g', markersize=1, alpha=.1)
#     axes[3].plot(t[mask_flippy], len(t[mask_flippy])*[3 * idx], '.r', markersize=1, alpha=.1)
    # for i, v in enumerate(gyro):
    #     if abs(v) > .5:
    #         axes[idx].vlines(t[i], *ylim, 'r', alpha=0.007)
    #     elif .5 >= abs(v) > 0.05 :
    #         axes[idx].vlines(t[i], *ylim, 'g', alpha=0.007)
alpha = .02
ax.plot(t[still_mask], len(t[still_mask])*[-1], '.k', markersize=1, alpha=alpha)
ax.plot(t[smooth_mask], len(t[smooth_mask])*[-.75], marker='.', color='lime', markersize=1, alpha=alpha)
ax.plot(t[flippy_mask], len(t[flippy_mask])*[-.5], '.r', markersize=1, alpha=alpha)
ax.plot(t, tot, 'b')
ax.set(xlabel="time [s]", ylabel="omega_total [rad/s]")
plt.grid()
black_patch = mpatches.Patch(color='black', label='still')
green_patch = mpatches.Patch(color='lime', label='smooth')
red_patch = mpatches.Patch(color='red', label='flippy')
orange_line = mlines.Line2D([], [], color='orange', label='trim start')
teal_line = mlines.Line2D([], [], color='darkviolet', label='trim end')
blue_line = mlines.Line2D([], [], color='blue', label='gyro omega_total')
plt.legend(handles=[blue_line, red_patch, green_patch, black_patch, orange_line, teal_line])
# plt.legend()
import itertools
# df = pd.DataFrame(data=[t, x, y, z, tot], )
still = np.zeros(len(t))
still[still_mask] = 1
end_points = np.where(np.diff(still) == -1)[0]
start_points = np.where(np.diff(still) == 1)[0]
duration = np.array([sum(1 for _ in group) for key, group in itertools.groupby(still_mask)])[-(int(still[0])-1)::2]
rate = 195.87
long_still = np.where(duration > rate * time)[0]
trim_start = 0
trim_end = t[-1]
if len(long_still) == 1:
    if end_points[long_still[0]] > len(t) / 2:
        trim_end = start_points[long_still[-1]] / rate
    else:
        trim_start = end_points[long_still[0]] / rate

else:
    trim_start = end_points[long_still[0]] / rate
    trim_end = start_points[long_still[-1]] / rate
video_end = int(t[-1])
trim_start = max(trim_start - trim_offset, 0)
trim_end = min(trim_end + trim_offset, int(video_end))

print("Trim start: ", trim_start)
print("Trim end: ", trim_end)

ax.axvline(trim_start, color='orange')
ax.axvline(trim_end, color='darkviolet')

smooth = np.zeros(len(t))
smooth[smooth_mask] = 1
start_points = np.where(np.diff(smooth) == 1)[0]
duration = np.array([sum(1 for _ in group) for key, group in itertools.groupby(smooth_mask)])[-(int(smooth[0])-1)::2]
long_smooth = np.where(duration > rate * 1)[0]
fps = 29.97
print("\nStart Duration")
for pt in long_smooth:
    print(f"{start_points[pt - 1] / rate:.2f} {duration[pt] / rate:.2f}")
# ind = np.argpartition(a, -4)[-4:]
plt.show()


