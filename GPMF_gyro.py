# Script to extract gopro metadata into a useful format.
# Uses a modified python-gpmf from https://github.com/rambo/python-gpmf

import gpmf.parse as gpmf_parse
from gpmf.extract import get_gpmf_payloads_from_file
import sys
import numpy as np
from matplotlib import pyplot as plt
import cv2

class Extractor:
    def __init__(self, videopath = "hero5.mp4"):
        self.videopath = videopath

        payloads, parser = get_gpmf_payloads_from_file(videopath)

        self.parsed = []
        print(payloads)

        for gpmf_data, timestamps in payloads:
            self.parsed.append(gpmf_parse.parse_dict(gpmf_data))


        self.video_length = 0 # video length in seconds
        self.fps = 0
        self.find_video_length()

        # Parsed gyro samples
        self.gyro = [] 
        self.gyro_scal = 0
        self.num_gyro_samples = 0
        self.gyro_rate = 0 # gyro rate in Hz
        self.parsed_gyro = np.zeros((1,4)) # placeholder
        self.parse_gyro()

    def find_video_length(self):
        
        #find video length using openCV
        video = cv2.VideoCapture(self.videopath)
        num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = video.get(cv2.CAP_PROP_FPS)
        self.video_length =  num_frames / self.fps
        print("Video length: {} s, framerate: {} FPS".format(self.video_length,self.fps))

        self.size = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video.release()

    def parse_gyro(self):
        for frame in self.parsed:
            for stream in frame["DEVC"]["STRM"]:
                if "GYRO" in stream:
                    #print(stream["STNM"]) # print stream name
                    self.gyro += stream["GYRO"]
                    
                    # Calibration scale shouldn't change
                    self.gyro_scal = stream["SCAL"]
                    #print(self.gyro_scal)
        
        
        # Convert to angular vel. vector in rad/s
        omega = np.array(self.gyro) / self.gyro_scal
        self.num_gyro_samples = omega.shape[0]


        self.gyro_rate = self.num_gyro_samples / self.video_length 
        print("Gyro rate: {} Hz, should be close to 200 or 400 Hz".format(self.gyro_rate))


        self.parsed_gyro = np.zeros((self.num_gyro_samples, 4))
        self.parsed_gyro[:,0] = np.arange(self.num_gyro_samples) * 1/self.gyro_rate

        # Data order for gopro gyro is (z,x,y)
        self.parsed_gyro[:,3] = omega[:,0] # z
        self.parsed_gyro[:,1] = omega[:,1] # x
        self.parsed_gyro[:,2] = omega[:,2] # y
        
    def parse_accl(self):
        for frame in self.parsed:
            for stream in frame["DEVC"]["STRM"]:
                if "ACCL" in stream:
                    #print(stream["STNM"]) # print stream name
                    self.accl += stream["ACCL"]
                    
                    # Calibration scale shouldn't change
                    self.accl_scal = stream["SCAL"]
                    #print(self.accl_scal)
        
        
        # Convert to angular vel. vector in rad/s ??
        omega = np.array(self.accl) / self.accl_scal
        self.num_accl_samples = omega.shape[0]

        self.accl_rate = self.num_accl_samples / self.video_length 
        print("Accl rate: {} Hz, should be close to 200 or 400 Hz".format(self.accl_rate))


        self.parsed_accl = np.zeros((self.num_accl_samples, 4))
        self.parsed_accl[:,0] = np.arange(self.num_accl_samples) * 1/self.accl_rate

        # Data order for gopro gyro is (z,x,y)
        self.parsed_accl[:,3] = omega[:,0] # z
        self.parsed_accl[:,1] = omega[:,1] # x
        self.parsed_accl[:,2] = omega[:,2] # y

    def get_gyro(self, with_timestamp = False):
        if with_timestamp:
            return self.parsed_gyro
        return self.parsed_gyro[:,1:]
    
    def get_accl(self, with_timestanp = False):
        if with_timestanp:
            return self.parsed_accl
        return self.parsed_accl[:,1:]

    def get_video_length(self):
        return self.video_length


if __name__ == "__main__":
    testing = Extractor()
    testing.get_gyro()