import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter,sosfilt

butterfilter = butter(2,0.05,output='sos')

N = 1000

zooms = sosfilt(butterfilter, np.random.random(N + 100) * 100)[100:]

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




# Alternative idea
windowsize = 10
newzooms = np.zeros(N)

for i in range(len(newzooms)):
	l = int(i - windowsize/2)
	h = int(i + windowsize/2)
	if l < 0:
		l = 0
	if h > N-1:
		h = N-1

	maxval = np.max(filtered[l:h])
	newzooms[i] = maxval

#newzooms = sosfilt(butterfilter, newzooms)

plt.plot(zooms)	
#plt.plot(filtered)
plt.plot(newzooms)
#plt.plot(final)
plt.show()