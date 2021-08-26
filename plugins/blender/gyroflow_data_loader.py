import json
import bpy

FILE_NAME = bpy.path.abspath("//GX016015.MP4.gyroflow")

processed_list = [] # store in [uSecond, rotX, rotY, rotZ]


with open(FILE_NAME, "r") as infile:
    try:
        gyroflow_data = json.load(infile)
        calibration_data = gyroflow_data["calibration_data"]
        #self.undistort.load_calibration_data(calibration_data,True)

        #self.video_rotate_code = gyroflow_data["video_rotate_code"]

        #motion_dof = gyroflow_data["sensor_DOF"]
        #raw_imu = gyroflow_data["raw_imu"]

        frame_orientations = gyroflow_data["frame_orientation"]
        #self.gyro_data = raw_imu[:,[0,1,2,3]]
        #if motion_dof >= 6:
        #    self.acc_data = raw_imu[:,[0,4,5,6]]

    except KeyError: # TODO change?
        print("Couldn't load gyroflow data file")
        exit()
        #return False

N = len(frame_orientations)

for i in range(N):
    
    quat = frame_orientations[i][1:]  
    bpy.data.objects["Plane"].rotation_quaternion[0] = quat[0] # x = pitch
    bpy.data.objects["Plane"].rotation_quaternion[1] = quat[1]# y = roll
    bpy.data.objects["Plane"].rotation_quaternion[2] = quat[2] # z = yaw
    bpy.data.objects["Plane"].rotation_quaternion[3] = quat[3]

    bpy.data.objects["Plane"].keyframe_insert(data_path="rotation_quaternion",frame=i)