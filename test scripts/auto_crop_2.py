import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter,sosfilt,sosfiltfilt,savgol_filter

###
# - Apply Savitzky–Golay to filtered values to make them smoother
# - Detect max difference between savgol curve and zoom values
# - Add the difference to all values
def softenfiltered(zooms, filtered, smoothing_value):
        window_size = (int(len(zooms)/smoothing_value) - 1 ) // 2 * 2 + 1
        newzooms_soft_y = savgol_filter(filtered, window_size, 3)

        newzooms_soft = np.zeros(len(zooms))
        differences = newzooms_soft_y - zooms
        for i in range(len(zooms)):
                newzooms_soft[i] = newzooms_soft_y[i] - np.amin(differences)

        return newzooms_soft
###

# zooms
N = 1000
butterfilter = butter(2,0.05,output='sos')
zooms = sosfiltfilt(butterfilter, np.random.random(N + 100) * 100)[100:]

# Calculate zoom
windowsize = 50
newzooms = np.zeros(N)
for i in range(len(newzooms)):
	l = int(i - windowsize/2)
	h = int(i + windowsize/2)
	if l < 0:
		l = 0
	if h > N-1:
		h = N-1

	maxval = np.max(zooms[l:h])
	newzooms[i] = maxval

butterfilter2 = butter(2,0.05,output='sos')
newzooms = sosfiltfilt(butterfilter2, newzooms)

plt.plot(zooms)
plt.plot(newzooms)
plt.plot(softenfiltered(zooms, newzooms, 4))
plt.show()
