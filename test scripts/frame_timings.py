import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('D:/DCIM/100RUNCAM/RC_0036_210723214223.MP4', cv2.CAP_FFMPEG)

print("Start reading presentation time stamps of video frames")
totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
time.sleep(0.05)

fps = float(cap.get(cv2.CAP_PROP_FPS))
print(fps)

pts = []
for i in range(totalFrames):
    if i % int(totalFrames/10) == 0:
        print("{}%".format(int(100*(i+1)/totalFrames)))
    cap.grab()
    frame_time = (cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
    pts.append(frame_time)
pts = np.array(pts)
print("Done reading presentation time stamps of video frames")
#np.savez(npzFile, frameTS=pts)
#print("Save frame TS of video {} to {}".format(videopath,npzFile))
diff = (pts[1:] - pts[:-1])

perfect_pts = np.arange(pts.shape[0]) * 1/fps

error = pts - perfect_pts
print(np.max(error))
print(np.min(error))
exit()
plt.plot(pts)
plt.plot(perfect_pts)
plt.plot(pts - perfect_pts)
plt.show()