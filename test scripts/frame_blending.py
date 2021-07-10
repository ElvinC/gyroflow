import cv2
import numpy as np
   
# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('../test_clips/DJIG0043wiebe.mp4')

cap.set(cv2.CAP_PROP_POS_FRAMES, 60 * 10)

ret, frame_out = cap.read()

frame_out = (frame_out * 0).astype(np.float64)

mult = 10
num_blend = 3

i = 0

# Read until video is completed
while(cap.isOpened()):
        
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        
        # Display the resulting frame
        


        if i % mult == 0:
            # Reset frame at beginning of hyperlapse range
            print("reset")
            frame_out = frame_out * 0.0

        if i % mult < num_blend:
            print(f"adding {i}")
            frame_out += 1/(num_blend) * frame.astype(np.float64)


        if (i - num_blend + 1) % mult == 0:
            cv2.imshow('Frame', frame_out.astype(np.uint8))

            cv2.waitKey(5)


        i += 1
        # Press Q on keyboard to  exit
        if 0xFF == ord('q'):
            break
    else: 
        break

cap.release()
cv2.destroyAllWindows()