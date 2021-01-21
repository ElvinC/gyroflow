from orangebox import Parser
from scipy.spatial.transform import Rotation
import math
import numpy as np

class BlackboxExtractor:
    def __init__(self, path):
        print("Opening {}".format(path))
        self.parser = Parser.load(path)
        self.n_of_logs = self.parser.reader.log_count

        self.gyro_scale = self.parser.headers["gyro_scale"] #should be already scaled in the fc
        self.final_gyro_data = []
        self.extracted = False
        self.camera_angle = None

        self.gyro_rate = 0

    def get_gyro_data(self,cam_angle_degrees=0):

        if self.extracted:
            return np.array(self.final_gyro_data)
        
        self.camera_angle = cam_angle_degrees
        r  = Rotation.from_euler('x', self.camera_angle, degrees=True)
        
        for lg in range(1,self.n_of_logs+1):
            self.parser.set_log_index(lg)
            t  = self.parser.field_names.index('time')
            gx = self.parser.field_names.index('gyroADC[1]')
            gy = self.parser.field_names.index('gyroADC[2]')
            gz = self.parser.field_names.index('gyroADC[0]')
            data_frames = []
            
            for frame in self.parser.frames():
                to_rotate = [-math.radians(frame.data[gx]),
                             math.radians(frame.data[gy]),
                             -math.radians(frame.data[gz])]
                
                rotated = r.apply(to_rotate)
                
                f = [frame.data[t]/1000000,
                     rotated[0],
                     rotated[1],
                     rotated[2]]
                #f = [frame.data[t]/1000000,
                #     math.radians(frame.data[gx]),
                #     math.radians(frame.data[gz]),
                #     -math.radians(frame.data[gy])]
                data_frames.append(f)
                
            self.final_gyro_data.extend(data_frames)


        self.final_gyro_data = np.array(self.final_gyro_data)


        # rough gyro rate assumed to be constant
        self.gyro_rate = self.final_gyro_data.shape[0]/(self.final_gyro_data[-1,0] - self.final_gyro_data[0,0])


        self.extracted = True

        return self.final_gyro_data





#testing
if __name__ == "__main__":


    with open("test_clips/GX010010.MP4.bbl") as f:
        #print(f.readline())
        pass


    #exit()

    bbe = BlackboxExtractor("test_clips/GX010010.MP4.bbl") # btfl_all.bbl
    gyro_data = bbe.get_gyro_data()
    print(gyro_data)
    print(bbe.n_of_logs)
