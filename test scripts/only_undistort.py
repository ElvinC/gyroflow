import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import stabilizer

undistortTest = stabilizer.OnlyUndistort("../test_clips/GX010010.MP4", "../camera_presets/GoPro/GoPro_Hero7_2_7K_4by3_wide_V2.json",fov_scale=1)
undistortTest.renderfile(0, 5, "undistort_out.mp4",out_size = (2560,1440), split_screen = False, scale=1, display_preview = True)