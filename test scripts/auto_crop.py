import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter,sosfilt

butterfilter = butter(2,0.05,output='sos')

zooms = sosfilt(butterfilter, np.random.random(1000) * 100)[100:]

filtered = np.zeros(zooms.shape[0])

smoothness = 0.99

currentval = zooms[0]
for i in range(zooms.shape[0]):
	filtered[i] = currentval
	currentval = currentval * smoothness + (1-smoothness) * zooms[i]
	currentval = max(currentval, zooms[i])

filtered2 = np.zeros(zooms.shape[0])
currentval = filtered[-1]
for i in range(zooms.shape[0]-1, -1, -1):
	filtered2[i] = currentval
	currentval = currentval * smoothness + (1-smoothness) * filtered[i]
	currentval = max(currentval, filtered[i])


final = (filtered + filtered2)/2

plt.plot(zooms)	
#plt.plot(filtered)
plt.plot(filtered2)
#plt.plot(final)
plt.show()