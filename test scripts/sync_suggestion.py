import stabilizer

file = r"D:\git\FPV\videos\GH011155.MP4"

stab = stabilizer.GPMFStabilizer(file, r"D:\git\FPV\GoPro_Hero6_2160p_43.json", file, hero=6, fov_scale = 1.6, gyro_lpf_cutoff = -1)

stab.full_auto_sync()

