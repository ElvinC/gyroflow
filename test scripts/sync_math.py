import numpy as np
from matplotlib import pyplot as plt

video_times = np.arange(0,100)

video_data = np.zeros(100)
video_data[10] = 1
video_data[90] = 2


gyro_times = np.arange(0,100) * 0.99 + 1

gyro_data = np.zeros(100)
gyro_data[10] = 1
gyro_data[90] = 2


plt.plot(video_times, video_data)
#plt.plot(gyro_times, gyro_data)
#plt.show()


v1 = 10
v2 = 90

g1 = gyro_times[10]
g2 = gyro_times[90]

d1 = v1 - g1
d2 = v2 - g2



# Find slope

# point slope v - v1 = slope * (g - g1) => v = slope * (g - g1) + v1

#err_slope = (d2-d1)/(v2-v1)
#correction_slope = err_slope* + 1
#gyro_start = (d1 - err_slope*v1)#  + 1.5/self.fps

#corrected_times = (gyro_times )*correction_slope + gyro_start

g1 = v1 - d1
g2 = v2 - d2
slope =  (v2 - v1) / (g2 - g1)
corrected_times = slope * (gyro_times - g1) + v1

plt.plot(corrected_times, gyro_data)


err_slope = (d2-d1)/(v2-v1)
correction_slope = err_slope + 1
gyro_start = (d1 - err_slope*v1)#  + 1.5/self.fps

wrong_times = (gyro_times + gyro_start)*correction_slope
plt.plot(wrong_times, gyro_data)
plt.show()