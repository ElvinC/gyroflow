import matplotlib.pyplot as plt
import cv2
import stabilizer
import pandas as pd
path = r"D:\git\FPV\videos\opencamera\VID_20210802_120804.mp4"
path = r"D:\git\FPV\videos\opencamera\VID_20210827_211112.mp4"
cap = cv2.VideoCapture(path)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)

lens_calibration = r"Google_Pixel_4a_HD.json"
stab = stabilizer.OpenCameraSensors(path, lens_calibration)
df = pd.DataFrame(stab.timestamps)
diff = df.diff()[1:]
plt.scatter(df[:-1], diff)
plt.ylim([0.03, 0.04])
print(diff.describe())
print(diff)
plt.show()
