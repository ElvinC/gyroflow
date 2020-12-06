import numpy as np
import cv2

from calibrate_video import FisheyeCalibrator
from scipy.spatial.transform import Rotation
from gyro_integrator import GyroIntegrator
from blackbox_extract import BlackboxExtractor

cap = cv2.VideoCapture("test_clips/GX015563.MP4")


width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

outpath = "newsmooth92-crop.mp4"

out_size = (1920,1080)

crop_start = (int((width-out_size[0])/2), int((height-out_size[1])/2))

out = cv2.VideoWriter(outpath, -1, 59.94, out_size )


undistort = FisheyeCalibrator()

undistort.load_calibration_json("camera_presets/gopro_calib2.JSON", True)


map1, map2 = undistort.get_maps(2)


bb = BlackboxExtractor("test_clips/GX015563.MP4_emuf_004.bbl")

gyro_data = bb.get_gyro_data(cam_angle_degrees=-2)




gyro_data[:,[2, 3]] = gyro_data[:,[3, 2]]


gyro_data[:,2] = -gyro_data[:,2]

#gyro_data[:,1:] = -gyro_data[:,1:]

initial_orientation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()

integrator = GyroIntegrator(gyro_data,initial_orientation=initial_orientation)
integrator.integrate_all()


times, stab_transform = integrator.get_interpolated_stab_transform(smooth=0.985,start=2.56+0.07,interval = 1/59.94)



#cap = cv2.VideoCapture(inpath)





i = 0
while(True):
    # Read next frame
    success, frame = cap.read() 
    if success:
        i +=1

    if i > 2000:
        break

    if success and i > 300:

        frame_undistort = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)



        # Apply affine wrapping to the given frame
        frame_stabilized = undistort.get_rotation_map(frame_undistort, stab_transform[i])
        
        # crop edges


        frame_out = crop_img = frame_stabilized[crop_start[1]:crop_start[1]+out_size[1], crop_start[0]:crop_start[0]+out_size[0]]
        out.write(frame_out)
        #print(frame_out.shape)

        # If the image is too big, resize it.
    #%if(frame_out.shape[1] > 1920): 
    #		frame_out = cv2.resize(frame_out, (int(frame_out.shape[1]/2), int(frame_out.shape[0]/2)));
        
        if(frame_out.shape[1] > 1920): 
            frame_out = cv2.resize(frame_out, (int(frame_out.shape[1]/2), int(frame_out.shape[0]/2)));

        cv2.imshow("Before and After", frame_out)
        cv2.waitKey(10)

# When everything done, release the capture
cap.release()
out.release()

cv2.destroyAllWindows()




