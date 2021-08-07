import stabilizer
import os

file = r"D:\git\FPV\videos\GH011162.MP4"

stab = stabilizer.GPMFStabilizer(file, r"camera_presets\GoPro\GoPro_Hero6_2160p_43.json", file, hero=6, fov_scale = 1.6, gyro_lpf_cutoff = -1)
stab.stabilization_settings()
stab.renderfile(starttime=30, stoptime=33, outpath="render_test.mp4", out_size=(400, 300))
os.startfile("render_test.mp4")
