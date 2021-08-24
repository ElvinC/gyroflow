import cv2
import numpy as np
import multiprocessing as mp
import time

class ParallelSync:
    def __init__(self, videofile="../test_clips/Runcam/RC_0036_filtered.MP4"):
        self.videofile = videofile
        self.offset = 10
        self.fps = 30

        self.is_adding = False
        self.index_list = []
        self.OF_list = []

    def opticalflow(self, id, start_frame, analyze_length):
        frame_times = []
        frame_idx = []
        transforms = []
        prev_pts_lst = []
        curr_pts_lst = []

        cap = cv2.VideoCapture(self.videofile)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        time.sleep(0.05)

        _, prev = cap.read()
        print(prev)
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        for i in range(analyze_length):
            
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

            succ, curr = cap.read()
            print(id, succ)
            frame_id = (int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            frame_time = (cap.get(cv2.CAP_PROP_POS_MSEC)/1000)

            #if i % 10 == 0:
            #    print("Analyzing frame: {}/{}".format(i,analyze_length))

            if succ:
                # Only add if succeeded
                frame_idx.append(frame_id)
                frame_times.append(frame_time)


                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                # Estimate transform using optical flow
                curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

                idx = np.where(status==1)[0]
                prev_pts = prev_pts[idx]
                curr_pts = curr_pts[idx]
                assert prev_pts.shape == curr_pts.shape

                prev_pts_lst.append(prev_pts)
                curr_pts_lst.append(curr_pts)


                transforms.append([1,2,3])

                prev_gray = curr_gray

            else:
                print("Frame {}".format(i))

        transforms = np.array(transforms)
        estimated_offset, cost = self.offset, 3
        return estimated_offset, cost, frame_times, transforms

    def begin_sync(self, timelist, slicelength):
        frame_list = [round(t) for t in timelist]

        n_proc = len(frame_list) # max about 10, should be fine

        with mp.Pool(processes=n_proc) as pool:
            # starts the sub-processes without blocking
            # pass the chunk to each worker process
            proc_results = [pool.apply_async(self.opticalflow,
                                            args=(i, frame_list[i],slicelength,))
                            for i in range(n_proc)]
            # blocks until all results are fetched
            result_chunks = [r.get() for r in proc_results]

        print(result_chunks)



if __name__ == "__main__":
    ps = ParallelSync()
    start = time.time()
    ps.begin_sync([2,10,20,30,40,50,60,80,90], 30)
    print(time.time() - start)