"""
gyrointegrator

This module uses gyroscope data to compute quaternion orientations over time
"""

import numpy as np

def quaternion(q0,q1,q2,q3):
    return np.array([q0,q1,q2,q3])

def vector(x,y,z):
    return np.array([x,y,z])

def normalize_quaternion(q):
    return q/np.sqrt(q.dot(q)) # q/|q|

# https://stackoverflow.com/questions/39000758/how-to-multiply-two-quaternions-by-python-or-numpy
def quaternion_multiply(Q1, Q2):
    w0, x0, y0, z0 = Q2
    w1, x1, y1, z1 = Q1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0])



class gyroIntegrator:
    def __init__(self, input_data, time_scaling=1, gyro_scaling=1, zero_out_time=True, initial_orientation=None):
        """Initialize instance of gyroIntegrator for getting orientation from gyro data

        Args:
            input_data (numpy.ndarray): Nx4 array, where each row is [time, gyroX,gyroY,gyroZ]
            time_scaling (int, optional): time * time_scaling should give time in second. Defaults to 1.
            gyro_scaling (int, optional): gyro<xyz> * gyro_scaling should give angular velocity in rad/s. Defaults to 1.
            zero_out_time (bool, optional): Always start time at 0 in the output data. Defaults to True.
            initial_orientation (float[4]): Quaternion representing the starting orientation, Defaults to [1, 0.0001, 0.0001, 0.0001].
        """

    
        self.data = np.copy(input_data)
        # scale input data
        self.data[:,0] *= time_scaling
        self.data[:,1:4] *= gyro_scaling

        # zero out timestamps
        if zero_out_time:
            self.data[:,0] -= self.data[0,0]

        self.num_data_points = self.data.shape[0]

        # initial orientation quaternion
        if initial_orientation:
            self.orientation = np.array(initial_orientation)
        else:
            self.orientation = np.array([1, 0.0001, 0.0001, 0.0001])

        # Variables to save integration data
        self.orientation_list = None
        self.time_list = None

        # IMU reference vectors
        self.imuRefX = vector(1,0,0)
        self.imuRefY = vector(0,1,0)
        self.imuRefY = vector(0,0,1)

        self.already_integrated = False


    def integrate_all(self):
        """go through each gyro sample and integrate to find orientation

        Returns:
            (np.ndarray, np.ndarray): tuple (time_list, quaternion orientation array)
        """

        if self.already_integrated:
            return (self.time_list, self.orientation_list)


        # temp lists to save data
        temp_orientation_list = []
        temp_time_list = []

        for i in range(self.num_data_points):

            # angular velocity vector
            omega = self.data[i][1:]

            # get current and adjecent times
            last_time = self.data[i-1][0] if i > 0 else self.data[i][0]
            this_time = self.data[i][0]
            next_time = self.data[i+1][0] if i < self.num_data_points - 1 else self.data[i][0]

            # symmetrical dt calculation. Should give slightly better results when missing data
            delta_time = (next_time - last_time)/2

            # Only calculate if angular velocity is present
            if np.any(omega):
                # calculate rotation quaternion
                delta_q = self.rate_to_quart(omega, delta_time)

                # rotate orientation by this quaternion
                self.orientation = quaternion_multiply(self.orientation, delta_q)

                self.orientation = normalize_quaternion(self.orientation)

            temp_orientation_list.append(np.copy(self.orientation))
            temp_time_list.append(this_time)

        self.orientation_list = np.array(temp_orientation_list)
        self.time_list = np.array(temp_time_list)

        self.already_integrated = True

        return (self.time_list, self.orientation_list)


    def get_orientations(self):
        """Get the processed quaternion orientations

        Returns:
            (np.ndarray, np.ndarray): tuple (time_list, quaternion orientation array)
        """
        if self.already_integrated:
            return (self.time_list, self.orientation_list)

        return None


    def get_raw_data(self, axis):
        """get a column of the raw data. Either time or gyro.

        Args:
            axis (string|int): Column index or keyword(t,x,y,z)

        Returns:
            numpy.ndarray: The selected column as numpy matrix.
        """


        idx = axis if type(axis) == int else {
            "t": 0,
            "x": 1,
            "y": 2,
            "z": 3
        }[axis]

        return np.copy(self.data[:,idx])




    def rate_to_quart(self, omega, dt):
        """Rotation quaternion from gyroscope sample

        Args:
            omega (numpy.ndarray): angular velocity vector [x,y,z]. Same as scaled gyro sample in rad/s.
            dt (float): Time delta between gyro samples for angle integration.

        Returns:
            numpy.ndarray: Rotation quaternion corresponding to orientation change
        """

        # https://stackoverflow.com/questions/24197182/efficient-quaternion-angular-velocity/24201879#24201879
        # no idea how it fully works, but it does
        ha = omega * dt * 0.5
        l = np.sqrt(ha.dot(ha))

        ha *= np.sin(l) / l

        q0 = np.cos(l)
        q1 = ha[0]
        q2 = ha[1]
        q3 = ha[2]

        return quaternion(q0,q1,q2,q3)