# Script to extract gopro metadata into a useful format.
# Uses python-gpmf by  from https://github.com/rambo/python-gpmf

import gpmf.parse as gpmf_parse
from gpmf.extract import get_gpmf_payloads_from_file
import sys
import numpy as np
from matplotlib import pyplot as plt

class Extractor:
    def __init__(self, videopath = "hero5.mp4"):
        self.videopath = videopath

        payloads, parser = get_gpmf_payloads_from_file(videopath)

        self.parsed = []

        for gpmf_data, timestamps in payloads:
            self.parsed.append(gpmf_parse.parse_dict(gpmf_data))

            

                

    def get_gyro(self):
        self.gyro = []
        self.scal = 0
        for frame in self.parsed:
            for stream in frame["DEVC"]["STRM"]:
                if "GYRO" in stream:
                    self.gyro += stream["GYRO"]
                    self.scal = stream["SCAL"]
        
        
        
        omega = np.array(self.gyro) / self.scal

        plt.plot(omega[:,0])
        plt.show()

    def get_accl(self):
        return 1

    def get_video_length(self):
        return 1


if __name__ == "__main__":
    testing = Extractor()
    testing.get_gyro()