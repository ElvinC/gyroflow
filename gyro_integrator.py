"""
gyrointegrator

This module uses gyroscope data to compute quaternion orientations over time
"""


import numpy as np
import quaternion as quart

class GyroIntegrator:
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
        if type(initial_orientation) != type(None):
            self.orientation = np.array(initial_orientation)
        else:
            self.orientation = np.array([1, 0.0001, 0.0001, 0.0001])

        # Variables to save integration data
        self.orientation_list = None
        self.time_list = None

        # IMU reference vectors
        self.imuRefX = quart.vector(1,0,0)
        self.imuRefY = quart.vector(0,1,0)
        self.imuRefY = quart.vector(0,0,1)

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
                self.orientation = quart.quaternion_multiply(self.orientation, delta_q) # Maybe change order

                self.orientation = quart.normalize(self.orientation)

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


    def get_smoothed_orientation(self, smooth = 0.94):

        smothness = smooth**(1/6)


        smoothed_orientation = np.zeros(self.orientation_list.shape)

        value = self.orientation_list[0,:]


        for i in range(self.num_data_points):
            value = quart.slerp(value, self.orientation_list[i,:],[1-smothness])[0]
            smoothed_orientation[i] = value

        # reverse pass
        smoothed_orientation2 = np.zeros(self.orientation_list.shape)

        value2 = smoothed_orientation[-1,:]

        for i in range(self.num_data_points-1, -1, -1):
            value2 = quart.slerp(value2, smoothed_orientation[i,:],[(1-smothness)*1.9])[0]
            smoothed_orientation2[i] = value2

        return (self.time_list, smoothed_orientation2)

        #

    def get_stabilize_transform(self,smooth=0.94):
        time_list, smoothed_orientation = self.get_smoothed_orientation(smooth)


        # rotations that'll stabilize the camera
        stab_rotations = np.zeros(self.orientation_list.shape)

        for i in range(self.num_data_points):
            # rotation quaternion from smooth motion -> raw motion to counteract it
            stab_rotations[i,:] = quart.rot_between(smoothed_orientation[i],self.orientation_list[i])

        return (self.time_list, stab_rotations) 

        
    def get_interpolated_stab_transform(self,smooth, start=0, interval=1/29.97):
        time_list, smoothed_orientation = self.get_stabilize_transform(smooth)

        time = start

        out_times = []
        slerped_rotations = []

        while time_list[0] >= time:
            slerped_rotations.append(smoothed_orientation[0])
            out_times.append(time)
            time += interval


        for i in range(len(time_list)-1):
            if time_list[i] <= time < time_list[i+1]:

                # interpolate between two quaternions
                weight = (time - time_list[i])/(time_list[i+1]-time_list[i])
                slerped_rotations.append(quart.slerp(smoothed_orientation[i],smoothed_orientation[i+1],[weight]))
                out_times.append(time)

                time += interval

        return (out_times, slerped_rotations)

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

        if l > 1.0e-12:

            ha *= np.sin(l) / l

            q0 = np.cos(l)
            q1 = ha[0]
            q2 = ha[1]
            q3 = ha[2]

            return quart.normalize(quart.quaternion(q0,q1,q2,q3))

        else:
            return quart.quaternion(1,0,0,0)