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

while(True):
    # Capture frame-by-frame
    while ser.in_waiting:
        line = ser.readline()
        print(line)

        quat = np.array([float(x) for x in line.decode().strip().split(":")[1].split(",")])

    ret, frame = cap.read()

    tmap1, tmap2 = undistort.get_maps(1,
                                            new_img_dim=(width,height),
                                            output_dim=(1280,960),
                                            update_new_K = False, quat = quat)

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