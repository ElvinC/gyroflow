# Script to extract gopro metadata into a useful format.
# Uses python-gpmf by  from https://github.com/rambo/python-gpmf

import gpmf.parse as gpmf_parse
from gpmf.extract import get_gpmf_payloads_from_file
import sys
    

class Extractor:
    def __init__(self, videopath = "hero5.mp4"):
        self.videopath = videopath

        payloads, parser = get_gpmf_payloads_from_file(videopath)

        for gpmf_data, timestamps in payloads:
            for element, parents in gpmf_parse.recursive(gpmf_data):
                try:
                    value = gpmf_parse.parse_value(element)
                except ValueError:
                    value = element.data
                print("{} {} > {}: {}".format(
                    timestamps,
                    ' > '.join([x.decode('ascii') for x in parents]),
                    element.key.decode('ascii'),
                    value
                ))

    def get_gyro(self):
        return 1

    def get_accl(self):
        return 1

    def get_video_length(self):
        return 1


if __name__ == "__main__":
    testing = Extractor()