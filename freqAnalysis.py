"""
FreqAnalysis

This is just to check if sampling frequency is consistent
"""


import numpy as np
import quaternion as quat
from matplotlib import pyplot as plt
from scipy import stats


class FreqAnalysis:
    def __init__(self, gyroInt):
        self.gyroInt = gyroInt

    def sampleFrequencyAnalysis(self, show_plots = False):
        timestamps = self.gyroInt.get_raw_data("t")
        gyro_data = self.gyroInt.get_raw_data("xyz")
        interarrival = np.diff(timestamps, n=1)
        w = int(self.gyroInt.gyro_sample_rate/100.0) # aggregate over 1%, e.g., 9 gyro samples for 900Hz/900 samples per second
        interarrival = np.convolve(interarrival, np.ones(w), 'valid') / w   # moving average
        freqs = 1.0/interarrival

        median = np.median(freqs)
        mad = stats.median_abs_deviation(freqs)
        mad_normal = stats.median_abs_deviation(freqs, scale='normal')
        std = np.std(freqs)

        print('Computed sample rate is {}'.format(self.gyroInt.gyro_sample_rate))
        print('Median freq is {}'.format(median))
        print('Mean freq is {}'.format(np.mean(freqs)))
        print('Stdev of freqs is {}'.format(std))
        print('MAD of freqs is {}'.format(mad))
        print('MAD (normal) of freqs is {}'.format(mad_normal))
        print('Max inter sample delay is {}'.format(np.max(interarrival)))



        if show_plots:
            thresh = mad_normal if mad_normal > std else std
            thresh = 6*thresh #corresponds to 100% of observations when following normal distribution
            outlierMask = median - freqs > thresh
            plt.plot(timestamps, gyro_data)
            plt.plot(timestamps[:-1-(w-1)], outlierMask*2)
            plt.show()

            plt.hist(freqs, bins=300)
            plt.yscale("log")
            plt.axvline(x=median+thresh, color="red")
            plt.axvline(x=median-thresh, color="red")
            plt.axvline(x=median, color='green')
            plt.axvline(x=np.mean(freqs), color='orange')
            plt.show()
