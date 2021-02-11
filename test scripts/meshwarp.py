import cv2
import numpy as np
import timeit
from scipy import signal
from skimage.transform import resize

# Literature:
# https://en.wikipedia.org/wiki/Bilinear_interpolation
# https://graphics.stanford.edu/papers/stabilization/karpenko_gyro.pdf
# https://www.mathworks.com/help/vision/ug/interpolation-methods.html
# https://github.com/subeeshvasu/Awesome-Image-Distortion-Correction

def test1():
    #Read input image, and create output image
    width = 2000
    height = 2000
    src = cv2.imread('pattern.png')
    src = cv2.resize(src,(width,height))
    dst = np.zeros_like(src)
    map1 = np.broadcast_to(np.arange(width)[..., None], (width, height)).T # X values  #np.zeros_like(src)
    map2 = np.broadcast_to(np.arange(height)[..., None], (height, width)) #np.zeros_like(src)
    map1 = map1.astype('float32')
    map2 = map2.astype('float32')

    src_pts = np.array([[0,0],[width,0],[width, height],[0,height]], dtype='float32')
    dst_pts = np.array([[0,0],[width,0-250],[width, height-250],[0,height]], dtype='float32')

    perspective = cv2.getPerspectiveTransform(src_pts,dst_pts)

    #print(perspective)

    #map2 = map2 - (map1 * 0.5) * (2 - 2 * map1 / width)

    dist_perspective = cv2.warpPerspective(src, perspective, (width, height))


    # Try bilinear interpolation
    # Dist
    # (x0,y0) -- (x1,y1)
    #  |          |
    # (x3,y3) -- (x2,y2)
    dst_pts = np.array([[0,0],[width,0],[width, height],[0,height]], dtype='float32')
    src_pts = np.array([[-100,-100],[width + 50,-50],[width, height],[0,height]], dtype='float32')

    map1 = np.broadcast_to(np.arange(width)[..., None], (width, height)).T # X values  #np.zeros_like(src)
    map2 = np.broadcast_to(np.arange(height)[..., None], (height, width)) #np.zeros_like(src)
    map1 = map1.astype('float32')
    map2 = map2.astype('float32')

    start = timeit.default_timer()

    # handle map1
    fxy1 = (dst_pts[1,0] - map1)/(dst_pts[1,0] - dst_pts[0,0]) * src_pts[0,0] + \
           (map1 - dst_pts[0,0])/(dst_pts[1,0] - dst_pts[0,0]) * src_pts[1,0]    

    fxy2 = (dst_pts[1,0] - map1)/(dst_pts[1,0] - dst_pts[0,0]) * src_pts[3,0] + \
           (map1 - dst_pts[0,0])/(dst_pts[1,0] - dst_pts[0,0]) * src_pts[2,0]

    fxy = (dst_pts[2,1] - map2)/(dst_pts[2,1] - dst_pts[1,1]) * fxy1 + \
           (map2 - dst_pts[1,1])/(dst_pts[2,1] - dst_pts[1,1]) * fxy2
    
    # handle map2
    fxy1y = (dst_pts[1,0] - map1)/(dst_pts[1,0] - dst_pts[0,0]) * src_pts[0,1] + \
           (map1 - dst_pts[0,0])/(dst_pts[1,0] - dst_pts[0,0]) * src_pts[1,1]

    fxy2y = (dst_pts[1,0] - map1)/(dst_pts[1,0] - dst_pts[0,0]) * src_pts[3,1] + \
           (map1 - dst_pts[0,0])/(dst_pts[1,0] - dst_pts[0,0]) * src_pts[2,1]

    fxyy = (dst_pts[2,1] - map2)/(dst_pts[2,1] - dst_pts[1,1]) * fxy1y + \
           (map2 - dst_pts[1,1])/(dst_pts[2,1] - dst_pts[1,1]) * fxy2y
    stop = timeit.default_timer()
    #print(fxyy)
    map1 = np.copy(fxy)
    map2 = np.copy(fxyy)

    
    dist = cv2.remap(src, map1, map2, interpolation=cv2.INTER_LINEAR)

    #signal.bilinear

    print("Time {}".format(stop - start))
    cv2.imshow("img", dist)
    k = cv2.waitKey()

def scale_test():
    meshw = 4
    meshh = 1

    for i in range(1, 100):

        outw = meshw * i
        outh = meshh * 1

        a = np.arange(meshw * meshh).reshape((meshh,meshw)) * 10
        a = np.array(a, dtype="float64")
        start = timeit.default_timer()
        b = cv2.resize(a, (outw,outh), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        #b = resize(a, (outh,outw), order=1, mode='edge')
        stop = timeit.default_timer()

        #print(stop - start)
        #print(a)
        #print(b)

        p = np.sum(b == 0) - 1
        c = (i -1)/2

        print("s {}, p: {}, c {}, outw: {}".format(i, p, c, outw))

        fw = outw - 2 * c

        print(f"final: {fw}")


def mwarp():
    # size of mesh points
    meshw = 10
    meshh = 10

    width = 1300
    height = 1000
    print(np.linspace(0, width, meshw)[..., None])
    start = timeit.default_timer()
    meshx = np.broadcast_to(np.linspace(0, width, meshw)[..., None], (meshw, meshh)).T.copy()
    meshy = np.broadcast_to(np.linspace(0, height, meshh)[..., None], (meshh, meshw)).copy()

    #print(meshx)
    #print(meshy)
    #meshx[1,1] = meshx[1,1]
    #meshy[1,1] = meshy[1,1]
    meshx += np.random.random((meshh,meshw)) * 100
    meshy += np.random.random((meshh,meshw)) * 100
    
    meshx = meshx.astype('float64')
    meshy = meshy.astype('float64')

    src = cv2.imread('test.jpg')
    src = cv2.resize(src,(width,height))
    dst = np.zeros_like(src)
    meshx.shape += (1,)
    meshy.shape+= (1,)
    combined = np.concatenate((meshx, meshy), axis=2)
    map1 = cv2.resize(combined , (width, height), interpolation=cv2.INTER_CUBIC)
    map2 = map1 # cv2.resize(meshy, (width, height), interpolation=cv2.INTER_CUBIC)
    map1 = map1.astype('float32')
    map2 = map2.astype('float32')
    stop = timeit.default_timer()
    dst = cv2.remap(src, map1, np.array([]), interpolation=cv2.INTER_LINEAR)
    print(f"time {start-stop}")

    # draw points
    for i in range(meshw):
        for j in range(meshh):
            pass
            #cv2.circle(dst,(int(meshx[j,i]),int(meshy[j,i])), 3, (255,100,20), thickness=5)

    cv2.imshow("img", dst)
    k = cv2.waitKey()

if __name__ == '__main__':
    mwarp()