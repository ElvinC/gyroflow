"""
gyrointegrator

This module uses gyroscope data to compute quaternion orientations over time
"""


import numpy as np
import quaternion as quat
import smoothing_algos

class GyroIntegrator:
    def __init__(self, gyro_data, time_scaling=1, gyro_scaling=1, zero_out_time=True, initial_orientation=None, acc_data=None, acc_scaling=1):
        """Initialize instance of gyroIntegrator for getting orientation from gyro data

        Args:
            gyro_data (numpy.ndarray): Nx4 array, where each row is [time, gyroX,gyroY,gyroZ]
            time_scaling (int, optional): time * time_scaling should give time in second. Defaults to 1.
            gyro_scaling (int, optional): gyro<xyz> * gyro_scaling should give angular velocity in rad/s. Defaults to 1.
            zero_out_time (bool, optional): Always start time at 0 in the output data. Defaults to True.
            initial_orientation (float[4]): Quaternion representing the starting orientation, Defaults to [1, 0.0001, 0.0001, 0.0001].
            acc_data (numpy.ndarray): Nx4 array, where each row is [time, accX, accY, accZ]. TODO: Use this in orientation determination
            acc_scaling (float): Scaling to give the acceleration in g
        """

        # data is only the gyro
        self.gyro = np.copy(gyro_data)
        self.acc = None

        self.last_used_acc = False

        self.acc_cutoff = 1 # Hz, low cutoff
        self.acc_available = False
        if type(acc_data) != type(None):
            # resample if they don't already match
            if self.gyro.shape[0] == acc_data.shape[0]:
                self.acc_available = True

                self.acc = np.copy(acc_data)
                self.gyro[:,0] *= time_scaling
                self.gyro[:,1:4] *= acc_scaling
            else:
                print("Gyro and acceleration data don't line up")
                self.acc_available = False
        #if self.acc_available:
            #print(self.acc.shape)
        # Check for corrupted/out of order timestamps
        time_order_check = self.gyro[:-1,0] > self.gyro[1:,0]
        if np.any(time_order_check):
            print("Truncated bad gyro data")
            self.gyro = self.gyro[0:np.argmax(time_order_check)+1,:]
            if self.acc_available:
                self.acc = self.acc[0:np.argmax(time_order_check)+1,:]

        # scale input data
        self.gyro[:,0] *= time_scaling
        self.gyro[:,1:4] *= gyro_scaling

        # Make sure input data is right handed. Final virtual camera rotation is left-handed
        # while image rotation is right-handed. Improve this later
        #self.gyro[:,2] *= -1 # y axis

        # zero out timestamps
        if zero_out_time:
            self.gyro[:,0] -= self.gyro[0,0]
            if self.acc_available:
                self.acc[:,0] -= self.acc[0,0]

        self.num_data_points = self.gyro.shape[0]

        self.gyro_sample_rate = self.num_data_points / (self.gyro[-1,0] - self.gyro[0,0])

        # initial orientation quaternion
        if type(initial_orientation) != type(None):
            self.initial_orientation = np.array(initial_orientation)
            self.orientation = np.array(initial_orientation)
        else:
            self.initial_orientation = np.array([1, 0.0001, 0.0001, 0.0001])
            self.orientation = np.array([1, 0.0001, 0.0001, 0.0001])

        # Variables to save integration data
        self.orientation_list = None
        self.time_list = None

        # IMU reference vectors
        self.imuRefX = quat.vector(1,0,0)
        self.imuRefY = quat.vector(0,1,0)
        self.imuRefY = quat.vector(0,0,1)

        # Gravity vector
        # points upwards, since it's equivalent to an upwards acceleration at rest
        self.grav_vec = np.array([0,1,0]) # Per convention it's upwards

        self.already_integrated = False

        self.smoothing_algo = None

        self.interp_times = None
        self.interp_orientations = None


    def integrate_all(self, use_acc = False):
        """go through each gyro sample and integrate to find orientation

        Returns:
            (np.ndarray, np.ndarray): tuple (time_list, quaternion orientation array)
        """

        if self.already_integrated and (use_acc == self.last_used_acc or not self.acc_available):
            return (self.time_list, self.orientation_list)

        apply_complementary = self.acc_available and use_acc

        self.last_used_acc = use_acc

        if apply_complementary:
            # find valid accelation data points
            #print(self.acc)
            #print(self.acc.shape)
            asquared = np.sum(self.acc[:,1:]**2,1)
            # between 0.9 and 1.1 g
            complementary_mask = np.logical_and(0.81<asquared,asquared<1.21)


        self.orientation = np.copy(self.initial_orientation)

        # temp lists to save data
        temp_orientation_list = []
        temp_time_list = []

        start_time = self.gyro[0][0] # seconds

        for i in range(self.num_data_points):

            # angular velocity vector
            omega = self.gyro[i][1:]

            # get current and adjecent times
            last_time = self.gyro[i-1][0] if i > 0 else self.gyro[i][0]
            this_time = self.gyro[i][0]
            next_time = self.gyro[i+1][0] if i < self.num_data_points - 1 else self.gyro[i][0]

            # symmetrical dt calculation. Should give slightly better results when missing data
            delta_time = (next_time - last_time)/2

            # Only calculate if angular velocity is present
            if np.any(omega) or apply_complementary:
                # complementary filter
                if apply_complementary:
                    if complementary_mask[i]:
                        avec = self.acc[i][1:]
                        avec /= np.linalg.norm(avec)

                        accWorldVec = quat.rotate_vector_fast(self.orientation, avec)
                        correctionWorld = np.cross(accWorldVec, self.grav_vec)

                        # high weight for first two seconds to "lock" it, then 
                        weight = 10 if this_time - start_time < 1.5 else 0.6
                        correctionBody = weight * quat.rotate_vector_fast(quat.conjugate(self.orientation), correctionWorld)
                        omega = omega + correctionBody


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

        return None, None

    def set_smoothing_algo(self, algo):
        if not algo:
            algo = smoothing_algos.PlainSlerp() # Default
        else:
            self.smoothing_algo = algo

    def get_smoothed_orientation(self):
        # https://en.wikipedia.org/wiki/Exponential_smoothing
        # the smooth value corresponds to the time constant

        if not self.smoothing_algo:
            self.smoothing_algo = smoothing_algos.PlainSlerp()

        return self.smoothing_algo.get_smooth_orientations(self.time_list, self.orientation_list)

        # Old code:

        alpha = 1
        if smooth > 0:
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


    def get_stabilize_transform(self):
        time_list, smoothed_orientation = self.get_smoothed_orientation()


        # rotations that'll stabilize the camera
        stab_rotations = np.zeros(self.orientation_list.shape)

        for i in range(self.num_data_points):
            # rotation quaternion from smooth motion -> raw motion to counteract it
            stab_rotations[i,:] = quat.rot_between(smoothed_orientation[i],self.orientation_list[i])

        return (self.time_list, stab_rotations) 

    def get_interpolated_orientations(self, start=0, interval=1/29.97):

        time_list, smoothed_orientation = self.get_orientations()
        
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
            while time_list[i] <= time < time_list[i+1]:

                # interpolate between two quaternions
                weight = (time - time_list[i])/(time_list[i+1]-time_list[i])
                slerped_rotations.append(quat.single_slerp(smoothed_orientation[i],smoothed_orientation[i+1],weight))
                out_times.append(time)

                time += interval

            if time < time_list[i]:
                # continue even if missing gyro data
                slerped_rotations.append(smoothed_orientation[i])
                out_times.append(time)
                time += interval

        self.interp_times = np.array(out_times)
        self.interp_orientations = np.array(slerped_rotations)

        return (self.interp_times, self.interp_orientations)
    

    def get_interpolated_stab_transform(self, start=0, interval=1/29.97):
        if self.smoothing_algo:
            if self.smoothing_algo.bypass_external_processing:
                print("Bypassing quaternion orientation integration")
                return self.get_interpolated_stab_transform_old(start=start, interval=interval)
                #time_list, smoothed_orientation = self.smoothing_algo.get_stabilize_transform(self.gyro)
        else:
            self.smoothing_algo = smoothing_algos.PlainSlerp()


        time_list, interp_orientations = self.get_interpolated_orientations(start=start, interval=interval)
        time_list = np.array(time_list)
        interp_orientations = np.array(interp_orientations)

        _, smoothed_orientations = self.smoothing_algo.get_smooth_orientations(time_list, interp_orientations)
        smoothed_orientations = np.array(smoothed_orientations)

        stab_rotations = np.zeros(smoothed_orientations.shape)

        for i in range(smoothed_orientations.shape[0]):
            # rotation quaternion from smooth motion -> raw motion to counteract it
            stab_rotations[i,:] = quat.rot_between(smoothed_orientations[i],interp_orientations[i])

        return time_list, stab_rotations

    def get_interpolated_stab_transform_old(self, start=0, interval=1/29.97):
        
        if self.smoothing_algo:
            if self.smoothing_algo.bypass_external_processing:
                print("Bypassing quaternion orientation integration")
                time_list, smoothed_orientation = self.smoothing_algo.get_stabilize_transform(self.gyro)
            else:
                time_list, smoothed_orientation = self.get_stabilize_transform()
        else:
            time_list, smoothed_orientation = self.get_stabilize_transform()

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
            while time_list[i] <= time < time_list[i+1]:

                # interpolate between two quaternions
                weight = (time - time_list[i])/(time_list[i+1]-time_list[i])
                slerped_rotations.append(quat.single_slerp(smoothed_orientation[i],smoothed_orientation[i+1],weight))
                out_times.append(time)

                time += interval

            if time < time_list[i]:
                # continue even if missing gyro data
                slerped_rotations.append(smoothed_orientation[i])
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

        return np.copy(self.gyro[:,idx])

    def get_raw_gyro_acc(self):
        if self.acc_available:
            return np.hstack([self.gyro, self.acc[:,1:]])
        return np.copy(self.gyro)

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
    def __init__(self, gyro_data, initial_orientation=None):
        """Initialize instance of FrameRotationIntegrator for getting orientation from frame change data

        Args:
            gyro_data (numpy.ndarray): Nx4 array, where each row is [frame num, gyroX,gyroY,gyroZ]
            initial_orientation (float[4]): Quaternion representing the starting orientation, Defaults to [1, 0.0001, 0.0001, 0.0001].
        """

            
        self.gyro = np.copy(gyro_data)

        self.num_data_points = self.gyro.shape[0]

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
        temp_time_list.append(self.gyro[0][0] - 1)


        for i in range(self.num_data_points):

            # angular velocity vector
            omega = self.gyro[i][1:]

            # get current time
            this_time = self.gyro[i][0]
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


class EulerIntegrator:
    def __init__(self, gyro_data, time_scaling=1, gyro_scaling=1, zero_out_time=True, acc_data=None):
        """Initialize instance of eulerintegrator for getting a faux orientation from gyro data (not true orientation) easier xyz stabilization

        Args:
            gyro_data (numpy.ndarray): Nx4 array, where each row is [time, gyroX,gyroY,gyroZ]
            time_scaling (int, optional): time * time_scaling should give time in second. Defaults to 1.
            gyro_scaling (int, optional): gyro<xyz> * gyro_scaling should give angular velocity in rad/s. Defaults to 1.
            zero_out_time (bool, optional): Always start time at 0 in the output data. Defaults to True.
            initial_orientation (float[4]): Quaternion representing the starting orientation, Defaults to [1, 0.0001, 0.0001, 0.0001].
            acc_data (numpy.ndarray): Nx4 array, where each row is [time, accX, accY, accZ]. TODO: Use this in orientation determination
        """

    
        self.gyro = np.copy(gyro_data)
        # scale input data
        self.gyro[:,0] *= time_scaling
        self.gyro[:,1:4] *= gyro_scaling

        # zero out timestamps
        if zero_out_time:
            self.gyro[:,0] -= self.gyro[0,0]

        self.num_data_points = self.gyro.shape[0]

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
                omega = self.gyro[i][1:]

                # get current and adjecent times
                last_time = self.gyro[i-1][0] if i > 0 else self.gyro[i][0]
                this_time = self.gyro[i][0]
                next_time = self.gyro[i+1][0] if i < self.num_data_points - 1 else self.gyro[i][0]

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

        return np.copy(self.gyro[:,idx])




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
    np.random.seed(1234)
    fake_gyro_data = np.random.random((1000,4))
    fake_gyro_data[:,0] = np.arange(1000)/10
    #print(fake_gyro_data)

    integrator = GyroIntegrator(fake_gyro_data, time_scaling=1, gyro_scaling=4, zero_out_time=True, initial_orientation=None, acc_data=None)
    integrator.integrate_all()
    stabtransforms =integrator.get_interpolated_stab_transform(0.5)[1]
    #print("\n".join([str(q) for q in stabtransforms]))
    
    q = stabtransforms[-1].flatten()

    rotmat = np.array([[1,0,0],
                       [0,0,0],
                       [0,0,0]])
    rot = Rotation([q[1],q[2],q[3],q[0]]).as_matrix()

    final_rotation = np.eye(3)
    final_rotation[0,0] = -1

    #combined_rotation[0:3,0:3] = np.linalg.multi_dot([final_rotation, np.linalg.inv(combined_rotation[0:3,0:3]), np.linalg.inv(final_rotation)])
    #rot = Rotation([-q[1],-q[2],q[3],-q[0]]).as_matrix()
    print(rot)