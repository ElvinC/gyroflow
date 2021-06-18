import numpy as np
import cv2
import serial
import serial.tools.list_ports
import os,sys,inspect

# move to "main" folder
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from calibrate_video import FisheyeCalibrator



print('Search...')
ports = serial.tools.list_ports.comports(include_links=False)
for port in ports :
    print('Find port '+ port.device)

ser = serial.Serial(port.device)
if ser.isOpen():
    ser.close()

ser = serial.Serial(port.device, 9600, timeout=1)
ser.flushInput()
ser.flushOutput()
print('Connect ' + ser.name)

preset_path = "../camera_presets/Caddx/Caddx_Ratel_2_1mm_4by3.json"

undistort = FisheyeCalibrator()
undistort.load_calibration_json(preset_path)

cap = cv2.VideoCapture(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

quat = np.array([1,0,0,0])
smoothed_quat = np.array([1,0,0,0])
correction_quat = np.array([1,0,0,0])

def quaternion(w,x,y,z):
    return np.array([w,x,y,z])

def quaternion_multiply(Q1, Q2):
    w0, x0, y0, z0 = Q2
    w1, x1, y1, z1 = Q1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0])

def slerp(v0, v1, t_array):
    """Spherical linear interpolation."""
    # >>> slerp([1,0,0,0], [0,0,0,1], np.arange(0, 1, 0.001))
    t_array = np.array(t_array)
    v0 = np.array(v0)
    v1 = np.array(v1)
    dot = np.sum(v0 * v1)

    if dot < 0.0:
        v1 = -v1
        dot = -dot
    
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = v0[np.newaxis,:] + t_array[:,np.newaxis] * (v1 - v0)[np.newaxis,:]
        return (result.T / np.linalg.norm(result, axis=1)).T
    
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t_array
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0[:,np.newaxis] * v0[np.newaxis,:]) + (s1[:,np.newaxis] * v1[np.newaxis,:])

def inverse(q):
    # negate imaginary components to get inverse of unit quat
    return quaternion(q[0],-q[1],-q[2],-q[3])

def rot_between(q1, q2):
    """Compute rotation quaternion from q1 to q2"""

    # https://www.gamedev.net/forums/topic/423462-rotation-difference-between-two-quaternions/
    return quaternion_multiply(inverse(q1), q2)

while(True):
    # Capture frame-by-frame
    while ser.in_waiting:
        line = ser.readline()
        print(line)

        parsed = [float(x) for x in line.decode().strip().split(":")[1].split(",")]

        quat = np.array(quaternion_multiply([parsed[0], -parsed[1], -parsed[3], -parsed[2]], [1,0,0,0]))
        smoothed_quat = slerp(smoothed_quat, quat, [0.03])[0]

    correction_quat = rot_between(quat, smoothed_quat)

    ret, frame = cap.read()

    tmap1, tmap2 = undistort.get_maps(1,
                                            new_img_dim=(width,height),
                                            output_dim=(1280,960),
                                            update_new_K = False, quat = correction_quat)

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    stabbed_frame = cv2.remap(frame, tmap1, tmap2, interpolation=cv2.INTER_LINEAR)
    

    # Display the resulting frame
    cv2.imshow('frame',stabbed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
ser.close()