import cv2
import numpy as np
import timeit
from scipy import signal

# Literature:
# https://en.wikipedia.org/wiki/Bilinear_interpolation
# https://graphics.stanford.edu/papers/stabilization/karpenko_gyro.pdf
# https://www.mathworks.com/help/vision/ug/interpolation-methods.html
# https://github.com/subeeshvasu/Awesome-Image-Distortion-Correction

if __name__ == '__main__':

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