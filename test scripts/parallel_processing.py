import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt 
import time
import cv2

from calibrate_video import FisheyeCalibrator, StandardCalibrator


def compute_cost(a,b,offset):
    c = (a + offset)* (b -offset) * -1 * a
    return sum(c)

def process_chunk(a,b,chunk):
    N = len(chunk)

    costs = np.zeros(N)
    for i in range(N):
        costs[i] = compute_cost(a,b,chunk[i])

    return costs


def main():
    a = np.random.random(20000)
    b = np.random.random(20000)

    N = 2000

    offsets = [x for x in range(N)]


    start = time.time()
    n_proc = 4 # mp.cpu_count()
    chunksize = N // n_proc

    proc_chunks = []
    for i_proc in range(n_proc):
        chunkstart = i_proc * chunksize
        # make sure to include the division remainder for the last process
        chunkend = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else None

        proc_chunks.append(offsets[chunkstart:chunkend])

    assert sum(map(len, proc_chunks)) == N

    with mp.Pool(processes=n_proc) as pool:
        # starts the sub-processes without blocking
        # pass the chunk to each worker process
        proc_results = [pool.apply_async(process_chunk,
                                        args=(a,b,chunk,))
                        for chunk in proc_chunks]
        # blocks until all results are fetched
        result_chunks = [r.get() for r in proc_results]

    costs = np.hstack(result_chunks)
    tot = time.time() - start
    print(f"Multi: {tot}s")

    start = time.time()
    process_chunk(a,b,offsets)
    tot = time.time() - start
    print(f"Single: {tot}s")

    

    plt.plot(costs)
    plt.show()


def video_speed():
    new_dim = (2560,1440)

    undistort = FisheyeCalibrator()
    undistort.load_calibration_json("../camera_presets\RunCam\DEV_Runcam_5_Orange_4K_30FPS_XV_16by9_stretched.json", True)


    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture('../test_clips/Runcam/RC_0036_filtered.MP4')
    #cap = cv2.VideoCapture('C:/Users/elvin/Downloads/IF-RC01_0000.MP4')
    #cap.set(cv2.CAP_PROP_POS_FRAMES, 60 * 10)

    ret, frame_out = cap.read()

    frame_out = (frame_out * 0).astype(np.float64)

    mult = 3
    num_blend = 3

    i = 1

    tstart = time.time()

    # Read until video is completed
    while(cap.isOpened()):
            
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            
            # Display the resulting frame
            
            if True:
                print(i)
                # Some random processing steps
                #prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                map1, map2 = undistort.get_maps(1,
                                        output_dim=new_dim,
                                        update_new_K = False, quat = np.array([1,0,0,0]))
                frame_out = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
                #cv2.imshow('Frame', frame_out)
                #cv2.waitKey(1)
            else:

                if (i-1) % mult == 0:
                    # Reset frame at beginning of hyperlapse range
                    print("reset")
                    frame_out = frame_out * 0.0

                if (i-1) % mult < num_blend:
                    print(f"adding {i}")
                    frame_out += 1/(num_blend) * frame.astype(np.float64)


                if ((i-1) - num_blend + 1) % mult == 0:
                    cv2.imshow('Frame', frame_out.astype(np.uint8))

                    cv2.waitKey(5)


            i += 1
            # Press Q on keyboard to  exit
            if 0xFF == ord('q'):
                break

        else: 
            break
        if i == 200:
            break
        

    tstop = time.time()
    dtime = tstop - tstart
    process_fps = 200 / dtime
    print(f"elapsed: {dtime}, fps: {process_fps}")

    cap.release()
    cv2.destroyAllWindows()


def get_map_placeholder(undistort, indices):
    new_dim = (2560,1440)
    for i in indices:
        print(i)
        map1, map2 = undistort.get_maps(1,
                            output_dim=new_dim,
                            update_new_K = False, quat = np.array([1,0,0,0]))
    
    
    return 2

def render_queue():

    new_dim = (2560,1440)

    undistort = FisheyeCalibrator()
    undistort.load_calibration_json("../camera_presets\RunCam\DEV_Runcam_5_Orange_4K_30FPS_XV_16by9_stretched.json", True)

    N = 64

    start = time.time()
    
    for i in range(N):
        print(i)
        map1, map2 = undistort.get_maps(1,
                    output_dim=new_dim,
                    update_new_K = False, quat = np.array([1,0,0,0]))

    tot = time.time() - start
    print(f"Single: {tot}s")
    
    start = time.time()
    n_proc = 4 # mp.cpu_count()
    chunksize = N // n_proc

    indices = [x for x in range(N)]

    proc_chunks = []
    for i_proc in range(n_proc):
        chunkstart = i_proc * chunksize
        # make sure to include the division remainder for the last process
        chunkend = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else None


        proc_chunks.append(indices[chunkstart:chunkend])

    assert sum(map(len, proc_chunks)) == N

    with mp.Pool(processes=n_proc) as pool:
        # starts the sub-processes without blocking
        # pass the chunk to each worker process
        proc_results = [pool.apply_async(get_map_placeholder,
                                        args=(undistort,chunk,))
                        for chunk in proc_chunks]
        # blocks until all results are fetched
        result_chunks = [r.get() for r in proc_results]

    tot = time.time() - start
    print(f"Multi: {tot}s")
    print(result_chunks)

if __name__ == "__main__":

    render_queue()