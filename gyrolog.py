import os
import numpy as np
import matplotlib.pyplot as plt


class GyrologReader:
    def __init__(self):
        self.gyro = None # N*4 array with each column containing [t, gx, gy, gz]
        self.acc = None # N*4 array with each column containing [t, ax, ay, az]
        self.extracted = False
        # Assume same time reference and orientation used for both

    def check_log_type(self, filename):
        # method to check if a data or video file is a certain log type
        return False

    def extract_log(self, filename):
        # Return fully formatted data

        # arbitrary convention used in gyroflow for no reason
        # x axis: points to the right. positive equals pitch up (objects in frame move down)
        # y axis: points up. positive equals pan left (objects move right)
        # z axis: points away from lens. positive equals CCW rotation (objects moves CW)

        # note that measured gravity vector points upwards when stationary due to equivalence to upwards acceleration

        return self.gyro, self.acc

    def apply_rotation(self, rotmat):
        if self.extracted:
            self.gyro[:,1:] = (rotmat * self.gyro[:,1:].transpose()).transpose

    def plot_gyro(self):
        xplot = plt.subplot(311)

        plt.plot(self.gyro[:,0], self.gyro[:,1])
        plt.ylabel("omega x [rad/s]")

        plt.subplot(312, sharex=xplot)

        plt.plot(self.gyro[:,0], self.gyro[:,2])
        plt.ylabel("omega y [rad/s]")

        plt.subplot(313, sharex=xplot)

        plt.plot(self.gyro[:,0], self.gyro[:,3])
        #plt.plot(self.integrator.get_raw_data("t") + d2, self.integrator.get_raw_data("z"))
        plt.xlabel("time [s]")
        plt.ylabel("omega z [rad/s]")

        plt.show()

class FakeData(GyrologReader):
    def __init__(self):
        super().__init__()


    def check_log_type(self, filename):
        return True

    def extract_log(self, filename):

        if filename == "rollpitchyaw":

            N = 1000

            self.gyro = np.zeros((N,4))
            self.gyro[:,0] = np.arange(N)/100 # 100 Hz data

            # t=2 to 3: positive roll
            self.gyro[200:300,3] = 1 # rad/s

            # t=4 to 5: positive pitch
            self.gyro[400:500,1] = 1

            # t=6 to 7
            self.gyro[600:700,2] = 1

            self.acc = np.zeros((N,4))
            self.acc[:,0] = np.arange(N)/100 # 100 Hz data

        else:
            np.random.seed(sum([ord(c) for c in filename]))

            N = 1000

            self.gyro = np.random.random((N,4))
            self.gyro[:,0] = np.arange(N)/100 # 100 Hz data

            self.acc = np.random.random((N,4))
            self.acc[:,0] = np.arange(N)/100 # 100 Hz data


        return self.gyro, self.acc





if __name__ == "__main__":
    reader = FakeData()
    g, a = reader.extract_log("rollpitchyaw")
    reader.plot_gyro()
    print(g)
    print(a)
