
import stabilizer
path = r"D:\git\FPV\videos\opencamera\VID_20210802_104622.mp4"
lens_calibration = r"Google_Pixel_4a_HD.json"
stab = stabilizer.OpenCameraSensors(path, lens_calibration)
