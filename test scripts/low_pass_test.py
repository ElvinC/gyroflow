# File for testing single-variable low pass filtering.

import numpy as np
from matplotlib import pyplot as plt

N = 1000


time_constant = 2 # s

dt = 0.2

#Smoothness = 1-np.exp(-1/Smoothness_input) # Smothness_input**(1/6)

Smoothness = 1 - np.exp(-dt / time_constant)

# https://en.wikipedia.org/wiki/Exponential_smoothing
# https://dsp.stackexchange.com/questions/28308/exponential-weighted-moving-average-time-constant


times = np.arange(0,N) * dt
raw_clean = np.sin(times)

raw = raw_clean #+ np.random.normal(0, 0.3, N)

#raw[500:600] += 2

#raw[100:] = 1

#raw = np.ones(N) * -0.5
#raw[500:] = 0.5


processed = np.zeros(N)

value = raw[0]


for i in range(raw.shape[0]):
    value = value * (1-Smoothness) + raw[i] * Smoothness
    processed[i] = value



processed2 = np.zeros(N)

value2 = processed[-1]

for i in range(raw.shape[0]-1, -1, -1):
    value2 = value2 * (1-Smoothness) + processed[i] * Smoothness
    processed2[i] = value2

#final = (processed + processed2)/2

plt.plot(times, raw)
#plt.plot(raw_clean)
plt.plot(times, processed2)
plt.show()