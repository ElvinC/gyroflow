"""
gyrointegrator

This module uses gyroscope data to compute quaternion orientations over time
"""


import numpy as np
import quaternion as quat

class GyroIntegrator:
    def __init__(self, input_data, time_scaling=1, gyro_scaling=1, zero_out_time=True, initial_orientation=None, acc_data=None):
        """Initialize instance of gyroIntegrator for getting orientation from gyro data

        Args:
            input_data (numpy.ndarray): Nx4 array, where each row is [time, gyroX,gyroY,gyroZ]
            time_scaling (int, optional): time * time_scaling should give time in second. Defaults to 1.
            gyro_scaling (int, optional): gyro<xyz> * gyro_scaling should give angular velocity in rad/s. Defaults to 1.
            zero_out_time (bool, optional): Always start time at 0 in the output data. Defaults to True.
            initial_orientation (float[4]): Quaternion representing the starting orientation, Defaults to [1, 0.0001, 0.0001, 0.0001].
            acc_data (numpy.ndarray): Nx4 array, where each row is [time, accX, accY, accZ]. TODO: Use this in orientation determination
        """


    
        self.data = np.copy(input_data)
        # scale input data
        self.data[:,0] *= time_scaling
        self.data[:,1:4] *= gyro_scaling

        # Make sure input data is right handed. Final virtual camera rotation is left-handed
        # while image rotation is right-handed.
        self.data[:,2] *= -1

        # zero out timestamps
        if zero_out_time:
            self.data[:,0] -= self.data[0,0]

        self.num_data_points = self.data.shape[0]

        self.gyro_sample_rate = self.num_data_points / (self.data[-1,0] - self.data[0,0])

        # initial orientation quaternion
        if type(initial_orientation) != type(None):
            self.orientation = np.array(initial_orientation)
        else:
            self.orientation = np.array([1, 0.0001, 0.0001, 0.0001])

        # Variables to save integration data
        self.orientation_list = None
        self.time_list = None

        # IMU reference vectors
        self.imuRefX = quat.vector(1,0,0)
        self.imuRefY = quat.vector(0,1,0)
        self.imuRefY = quat.vector(0,0,1)

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
                delta_q = self.rate_to_quat(omega, delta_time)

                # rotate orientation by this quaternion
                self.orientation = quat.quaternion_multiply(self.orientation, delta_q) # Maybe change order

                self.orientation = quat.normalize(self.orientation)

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
        # https://en.wikipedia.org/wiki/Exponential_smoothing
        # the smooth value corresponds to the time constant

        alpha = 1 - np.exp(-(1 / self.gyro_sample_rate) /smooth)

        smoothed_orientation = np.zeros(self.orientation_list.shape)

        value = self.orientation_list[0,:]


        for i in range(self.num_data_points):
            value = quat.slerp(value, self.orientation_list[i,:],[alpha])[0]
            smoothed_orientation[i] = value

        # reverse pass
        smoothed_orientation2 = np.zeros(self.orientation_list.shape)

        value2 = smoothed_orientation[-1,:]

        for i in range(self.num_data_points-1, -1, -1):
            value2 = quat.slerp(value2, smoothed_orientation[i,:],[alpha])[0]
            smoothed_orientation2[i] = value2

        # Test rotation lock (doesn't work)
        #if test:
        #    from scipy.spatial.transform import Rotation
        #    for i in range(self.num_data_points):
        #        quat = smoothed_orientation2[i,:]
        #        eul = Rotation([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz")
        #        new_quat = Rotation.from_euler('xyz', [eul[0], eul[1], np.pi]).as_quat()
        #        smoothed_orientation2[i,:] = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]

        return (self.time_list, smoothed_orientation2)


    def get_stabilize_transform(self,smooth=0.94):
        time_list, smoothed_orientation = self.get_smoothed_orientation(smooth)


        # rotations that'll stabilize the camera
        stab_rotations = np.zeros(self.orientation_list.shape)

        for i in range(self.num_data_points):
            # rotation quaternion from smooth motion -> raw motion to counteract it
            stab_rotations[i,:] = quat.rot_between(smoothed_orientation[i],self.orientation_list[i])

        return (self.time_list, stab_rotations) 

        
    def get_interpolated_stab_transform(self,smooth, start=0, interval=1/29.97):
        time_list, smoothed_orientation = self.get_stabilize_transform(smooth)

        time = start

        out_times = []
        slerped_rotations = []

        while time < 0:
            slerped_rotations.append(smoothed_orientation[0])
            out_times.append(time)
            time += interval

        while time_list[0] >= time:
            slerped_rotations.append(smoothed_orientation[0])
            out_times.append(time)
            time += interval


        for i in range(len(time_list)-1):
            if time_list[i] <= time < time_list[i+1]:

                # interpolate between two quaternions
                weight = (time - time_list[i])/(time_list[i+1]-time_list[i])
                slerped_rotations.append(quat.slerp(smoothed_orientation[i],smoothed_orientation[i+1],[weight]))
                out_times.append(time)

                time += interval

            elif time < time_list[i]:
                # continue even if missing gyro data
                slerped_rotations.append(smoothed_orientation[i])
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
            "z": 3,
            "xyz": slice(1,4)
        }[axis]

        return np.copy(self.data[:,idx])




    def rate_to_quat(self, omega, dt):
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

            return quat.normalize(quat.quaternion(q0,q1,q2,q3))

        else:
            return quat.quaternion(1,0,0,0)


class FrameRotationIntegrator(GyroIntegrator):
    def __init__(self, input_data, initial_orientation=None):
        """Initialize instance of FrameRotationIntegrator for getting orientation from frame change data

        Args:
            input_data (numpy.ndarray): Nx4 array, where each row is [frame num, gyroX,gyroY,gyroZ]
            initial_orientation (float[4]): Quaternion representing the starting orientation, Defaults to [1, 0.0001, 0.0001, 0.0001].
        """

            
        self.data = np.copy(input_data)

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
        self.imuRefX = quat.vector(1,0,0)
        self.imuRefY = quat.vector(0,1,0)
        self.imuRefY = quat.vector(0,0,1)

        self.already_integrated = False


    def integrate_all(self):
        """go through each sample and integrate to find orientation. Assumes sample N contains change between N and N-1

        Returns:
            (np.ndarray, np.ndarray): tuple (time_list, quaternion orientation array)
        """

        if self.already_integrated:
            return (self.time_list, self.orientation_list)


        # temp lists to save data
        temp_orientation_list = []
        temp_time_list = []
        

        temp_orientation_list.append(np.copy(self.orientation))
        temp_time_list.append(self.data[0][0] - 1)


        for i in range(self.num_data_points):

            # angular velocity vector
            omega = self.data[i][1:]

            # get current time
            this_time = self.data[i][0]
            # symmetrical dt calculation. Should give slightly better results when missing data
            delta_time = 1 # frame

            # Only calculate if angular velocity is present
            if np.any(omega):
                # calculate rotation quaternion
                delta_q = self.rate_to_quat(omega, delta_time)

                # rotate orientation by this quaternion
                self.orientation = quat.quaternion_multiply(self.orientation, delta_q) # Maybe change order

                self.orientation = quat.normalize(self.orientation)

            temp_orientation_list.append(np.copy(self.orientation))
            temp_time_list.append(this_time)

        self.orientation_list = np.array(temp_orientation_list)
        self.time_list = np.array(temp_time_list)

        self.already_integrated = True

        return (self.time_list, self.orientation_list)

    def integrate_complementary(self):
        """
        TODO: Implement this
        """
        # Useful ressource: https://josephmalloch.wordpress.com/portfolio/imu-sensor-fusion/


class EulerIntegrator:
    def __init__(self, input_data, time_scaling=1, gyro_scaling=1, zero_out_time=True, acc_data=None):
        """Initialize instance of eulerintegrator for getting a faux orientation from gyro data (not true orientation) easier xyz stabilization

        Args:
            input_data (numpy.ndarray): Nx4 array, where each row is [time, gyroX,gyroY,gyroZ]
            time_scaling (int, optional): time * time_scaling should give time in second. Defaults to 1.
            gyro_scaling (int, optional): gyro<xyz> * gyro_scaling should give angular velocity in rad/s. Defaults to 1.
            zero_out_time (bool, optional): Always start time at 0 in the output data. Defaults to True.
            initial_orientation (float[4]): Quaternion representing the starting orientation, Defaults to [1, 0.0001, 0.0001, 0.0001].
            acc_data (numpy.ndarray): Nx4 array, where each row is [time, accX, accY, accZ]. TODO: Use this in orientation determination
        """

    
        self.data = np.copy(input_data)
        # scale input data
        self.data[:,0] *= time_scaling
        self.data[:,1:4] *= gyro_scaling

        # zero out timestamps
        if zero_out_time:
            self.data[:,0] -= self.data[0,0]

        self.num_data_points = self.data.shape[0]

        # Variables to save integration data
        self.euler_orientation_list = None
        self.time_list = None

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

        euler_orientation = np.array([0, 0, 0])

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
                    euler_orientation += omega * delta_time

                temp_orientation_list.append(np.copy(euler_orientation))
                temp_time_list.append(this_time)



        self.euler_orientation_list = np.array(temp_orientation_list)
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
            value = quat.slerp(value, self.orientation_list[i,:],[1-smothness])[0]
            smoothed_orientation[i] = value

        # reverse pass
        smoothed_orientation2 = np.zeros(self.orientation_list.shape)

        value2 = smoothed_orientation[-1,:]

        for i in range(self.num_data_points-1, -1, -1):
            value2 = quat.slerp(value2, smoothed_orientation[i,:],[(1-smothness)])[0]
            smoothed_orientation2[i] = value2

        # Test rotation lock (doesn't work)
        #if test:
        #    from scipy.spatial.transform import Rotation
        #    for i in range(self.num_data_points):
        #        quat = smoothed_orientation2[i,:]
        #        eul = Rotation([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz")
        #        new_quat = Rotation.from_euler('xyz', [eul[0], eul[1], np.pi]).as_quat()
        #        smoothed_orientation2[i,:] = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]

        return (self.time_list, smoothed_orientation2)


    def get_stabilize_transform(self,smooth=0.94):
        time_list, smoothed_orientation = self.get_smoothed_orientation(smooth)


        # rotations that'll stabilize the camera
        stab_rotations = np.zeros(self.orientation_list.shape)

        for i in range(self.num_data_points):
            # rotation quaternion from smooth motion -> raw motion to counteract it
            stab_rotations[i,:] = quat.rot_between(smoothed_orientation[i],self.orientation_list[i])

        return (self.time_list, stab_rotations) 

        
    def get_interpolated_stab_transform(self,smooth, start=0, interval=1/29.97):
        time_list, smoothed_orientation = self.get_stabilize_transform(smooth)

        time = start

        out_times = []
        slerped_rotations = []

        while time < 0:
            slerped_rotations.append(smoothed_orientation[0])
            out_times.append(time)
            time += interval

        while time_list[0] >= time:
            slerped_rotations.append(smoothed_orientation[0])
            out_times.append(time)
            time += interval


        for i in range(len(time_list)-1):
            if time_list[i] <= time < time_list[i+1]:

                # interpolate between two quaternions
                weight = (time - time_list[i])/(time_list[i+1]-time_list[i])
                slerped_rotations.append(quat.slerp(smoothed_orientation[i],smoothed_orientation[i+1],[weight]))
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
            "z": 3,
            "xyz": slice(1,4)
        }[axis]

        return np.copy(self.data[:,idx])




    def rate_to_quat(self, omega, dt):
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

            return quat.normalize(quat.quaternion(q0,q1,q2,q3))

        else:
            return quat.quaternion(1,0,0,0)

if __name__ == "__main__":
    from scipy.spatial.transform import Rotation

    fake_gyro_data = np.random.random((1000,4))
    fake_gyro_data[:,0] = np.arange(1000)/10
    print(fake_gyro_data)

    integrator = GyroIntegrator(fake_gyro_data, time_scaling=1, gyro_scaling=1, zero_out_time=True, initial_orientation=None, acc_data=None)
    integrator.integrate_all()
    stabtransforms =integrator.get_interpolated_stab_transform(0.5)[1]
    orig = stabtransforms[50]
 
    # Hero 6 as reference
    fake_gyro_data[:,2] = -fake_gyro_data[:,2]
    integrator = GyroIntegrator(fake_gyro_data, time_scaling=1, gyro_scaling=1, zero_out_time=True, initial_orientation=None, acc_data=None)
    integrator.integrate_all()
    stabtransforms =integrator.get_interpolated_stab_transform(0.5)[1]
    weird = stabtransforms[50]
    
    print(weird)    
    print(orig)


    combined_rotation[0:3,0:3] = Rotation([-quart[1],-quart[2],quart[3],-quart[0]]).as_matrix()