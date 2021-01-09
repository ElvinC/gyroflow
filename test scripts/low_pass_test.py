# File for testing single-variable low pass filtering.

import numpy as np
from matplotlib import pyplot as plt

N = 1000


Smothness_input = 0.8

Smothness = Smothness_input**(1/6)



raw_clean = np.sin(np.arange(0,N)/30)

raw = raw_clean+ np.random.normal(0, 0.3, N)

#raw[500:600] += 2

#raw[100:] = 1

#raw = np.ones(N) * -0.5
#raw[500:] = 0.5


processed = np.zeros(N)

value = raw[0]


for i in range(raw.shape[0]):
    value = value * Smothness + raw[i] * (1-Smothness)
    processed[i] = value



processed2 = np.zeros(N)

value2 = processed[-1]

for i in range(raw.shape[0]-1, -1, -1):
    value2 = value2 * Smothness + processed[i] * (1-Smothness)
    processed2[i] = value2

#final = (processed + processed2)/2

plt.plot(raw)
#plt.plot(raw_clean)
plt.plot(processed2)
plt.show()