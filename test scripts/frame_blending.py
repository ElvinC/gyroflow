import cv2
import numpy as np
import os
   
# Create a VideoCapture object and read from input file
f = 'D:/DCIM/100RUNCAM/RC_0005_210823153931.MP4'
if os.path.isfile(f):
    print("IS FILE")
else:
    print("NOT FILE")
cap = cv2.VideoCapture(f,cv2.CAP_FFMPEG)
#cap = cv2.VideoCapture('C:/Users/elvin/Downloads/IF-RC01_0000.MP4')
#cap.set(cv2.CAP_PROP_POS_FRAMES, 60 * 10)

ret, frame_out = cap.read()
frame_out = cv2.resize(frame_out, (1920,1080))

frame_out = (frame_out * 0).astype(np.float64)

out = cv2.VideoWriter('outpy5.mp4',-1, 30, (1920,1080))

mult = 60
num_blend = 14

diff = mult - num_blend

i = 1

# Read until video is completed
while(cap.isOpened()):
        
    # Capture frame-by-frame
    ret, frame = cap.read()
    #print(frame.shape)
    
    if ret == True:
        frame = cv2.resize(frame, (1920,1080), interpolation=cv2.INTER_NEAREST)
        
        # Display the resulting frame
        


        if (i-1) % mult == 0:
            # Reset frame at beginning of hyperlapse range
            print("reset")
            frame_out = frame_out * 0.0

        if (i-1) % mult < num_blend:
            print(f"adding {i}")
            frame_out += 1/(num_blend) * frame.astype(np.float64)
        elif (i-1) % mult == num_blend:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_POS_FRAMES) + diff-1) )
            i += diff - 1

        if ((i-1) - num_blend + 1) % mult == 0:
            #cv2.imshow('Frame', frame_out.astype(np.uint8))
            out.write(frame_out.astype(np.uint8))

            #cv2.waitKey(5)


        i += 1
        # Press Q on keyboard to  exit
        if 0xFF == ord('q'):
            break
    else: 
        break

cap.release()
cv2.destroyAllWindows()