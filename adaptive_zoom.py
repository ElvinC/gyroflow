import numpy as np
import numpy.linalg
import cv2
import csv
import platform
import math

from calibrate_video import FisheyeCalibrator, StandardCalibrator
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
from gyro_integrator import GyroIntegrator, FrameRotationIntegrator
from matplotlib import pyplot as plt

from _version import __version__

from scipy import signal, interpolate
import time


class AdaptiveZoom:
    def __init__(self, fisheyeCalibrator):
        self.calibrator = fisheyeCalibrator
        self.calib_dimension = fisheyeCalibrator.calib_dimension
        self.K = np.copy(fisheyeCalibrator.K)
        self.D = np.copy(fisheyeCalibrator.D)

    def min_rolling(self, a, window,axis =1):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        return np.min(rolling,axis=axis)

    def findFcorr(self, center, polygon, output_dim):
        (output_width, output_height) = output_dim
        angle_output = np.arctan2( output_height/2, output_width/2 )

        #fig, ax = plt.subplots()

        polygon = polygon - center
        #ax.scatter(polygon[:,0], polygon[:,1])

        distP = numpy.linalg.norm(polygon, axis=1)
        angles = np.abs( np.arctan2(polygon[:,1], polygon[:,0]) )

        #angles = angles[0:2]
        #distP = distP[0:2]
        mask = (angle_output <= np.abs(angles)) & (np.abs(angles) < (np.pi - angle_output))

        #ax.plot(distP*np.cos(angles), distP*np.sin(angles), 'ro')
        #ax.plot(distP[mask]*np.cos(angles[mask]), distP[mask]*np.sin(angles[mask]), 'yo')
        #ax.add_patch(matplotlib.patches.Rectangle((-output_width/2,-output_height/2), output_width, output_height,color="yellow"))
        dWidth = np.abs( (output_width/2)/np.cos(angles) )
        dHeight = np.abs( (output_height/2)/np.sin(angles) )


        ffactor = dWidth/distP
        ffactor[mask] = dHeight[mask]/distP[mask]

        fcorr = np.max( ffactor )
        idx = np.argmax( ffactor )

        return fcorr, idx

    def findFov(self, center, polygon, output_dim, numIntPoints=20):
        #(original_width, original_height) = self.calib_dimension
        fcorr, idx = self.findFcorr(center, polygon, output_dim)
        nP = (polygon.shape)[0]
        relevantP = polygon[ ((idx-1)%nP,idx,(idx+1)%nP),:]

        distance = np.cumsum( np.sqrt(np.sum( np.diff(relevantP, axis=0)**2, axis=1 )) )
        distance = np.insert(distance, 0, 0)/distance[-1]

        #interpolations_methods = ['slinear', 'quadratic', 'cubic']
        alpha = np.linspace(0, 1, numIntPoints)
        interpolator =  interp1d(distance, relevantP, kind='quadratic', axis=0)
        interpolated_points = interpolator(alpha)

        fcorrI, _ = self.findFcorr(center, interpolated_points, output_dim)

        fcorr = np.max((fcorr, fcorrI))
        #plt.plot(polygon[:,0], polygon[:,1], 'ro')
        #plt.plot(relevantP[:,0], relevantP[:,1], 'bo')
        #plt.plot(interpolated_points[:,0], interpolated_points[:,1], 'yo')
        #plt.show()

        return 1/fcorr #np.min([xminDist/output_width, yminDist/output_height])

    def compute(self, quaternions, output_dim, fps, smoothingFocus=2.0, tstart = False, tend = False, debug_plots=False, plot_blocking = False):

        # if smoothingFocus == -1: Totally disable
        # if smoothingFocus == -2: Find minimum sufficient crop

        #print(locals())
        #smoothingNumFrames = int(smoothingCenter * fps)
        #if smoothingNumFrames % 2 == 0:
        #    smoothingNumFrames = smoothingNumFrames+1



        smoothingFocusFrames = int(smoothingFocus * fps)
        if smoothingFocusFrames % 2 == 0:
            smoothingFocusFrames = smoothingFocusFrames+1

        boundaryPolygons = [self.boundingPolygon(quat=q) for q in quaternions]
        #focusWindows = [self.findFocalCenter(box, output_dim=output_dim) for box in boundaryBoxes]

        #focusWindows = np.array(focusWindows)

        # TODO: implement smoothing of position of crop, s.t. cropping area can "move" anywhere within bounding polygon
        cropCenterPositions = np.array([(self.calib_dimension[0]/2,self.calib_dimension[1]/2) for q in quaternions])

        #if smoothingCenter > 0:
        #    focusWindowsPad = np.pad(focusWindows, ( (int(smoothingNumFrames/2), int(smoothingNumFrames/2)), (0,0) ), mode='edge')
        #    filterCoeff = signal.gaussian(smoothingNumFrames,smoothingNumFrames/6)
        #    filterCoeff = filterCoeff / np.sum(filterCoeff)
        #    smoothXpos = np.convolve(focusWindowsPad[:,0], filterCoeff, 'valid')
        #    smoothYpos = np.convolve(focusWindowsPad[:,1], filterCoeff, 'valid')
        #    plt.plot(focusWindows)
        #    plt.plot(smoothXpos)
        #    plt.plot(smoothYpos)
        #    plt.show()
        #    focusWindows = np.stack((smoothXpos, smoothYpos), axis=-1)
        fovValues = [self.findFov(center,polygon,output_dim) for center, polygon in zip(cropCenterPositions,boundaryPolygons)]



        fovValues = np.array(fovValues)
        

        if tend != False:
            # Only within render range.
            max_fov = np.max(fovValues)
            fovValues[:max(tstart,0)] = max_fov
            fovValues[tend:] = max_fov

        if smoothingFocus > 0:
            filterCoeffFocus = signal.gaussian(smoothingFocusFrames,smoothingFocusFrames/6)
            filterCoeffFocus = filterCoeffFocus / np.sum(filterCoeffFocus)
            fovValuesPad = np.pad(fovValues, ( (int(smoothingFocusFrames/2), int(smoothingFocusFrames/2)) ), mode='edge')
            fovMin = self.min_rolling(fovValuesPad, window=smoothingFocusFrames)
            fovSmooth = np.convolve(np.pad(fovMin, ( (int(smoothingFocusFrames/2), int(smoothingFocusFrames/2)) ), mode='edge'),
                                            filterCoeffFocus, 'valid')
            if debug_plots:
                plt.plot(fovValues)
                plt.plot(fovMin)
                plt.plot(fovSmooth)
                plt.show(block=plot_blocking)
            fovValues = fovSmooth
        elif smoothingFocus == -1: #disabled
            maxF = np.min(fovValues)
            fovValues = np.repeat(maxF, fovValues.size )
        elif smoothingFocus == -2: # apply nothing
            fovValues = np.repeat(1, fovValues.size )

        return fovValues, cropCenterPositions


    def findFocalCenter(self, box, output_dim):
        (mleft,mright,mtop,mbottom) = box
        (output_width, output_height) = output_dim
        (calib_width, calib_height) = self.calib_dimension
        (window_width, window_height) = output_dim

        maxX = mright-mleft
        maxY = mbottom-mtop

        ratio = maxX/maxY
        output_ratio = float(output_width)/float(output_height)

        fX = 0
        fY = 0
        if maxX/output_ratio < maxY:
            window_width = maxX
            window_height = maxX/output_ratio
            fX = mleft + window_width/2
            fY = calib_height/2
            if fY+window_height/2 > mbottom:
                fY = mbottom - window_height/2
            elif fY-window_height/2 < mtop:
                fY = mtop + window_height/2
        else:
            window_height = maxY
            window_width = maxY*output_ratio
            fY = mtop + window_height/2
            fX = calib_width/2
            if fX+window_width/2 > mright:
                fX = mright - window_width/2
            elif fX-window_width/2 < mleft:
                fX = mleft + window_width/2
        return (fX,fY) #, window_width, window_height)


    def boundingPolygon(self, quat, numPoints = 9):
        (original_width, original_height) = self.calib_dimension

        R = np.eye(3)
        if type(quat) != type(None):
            quat = quat.flatten()
            #R = Rotation([-quat[1],-quat[2],quat[3],-quat[0]]).as_matrix()
            R = Rotation([quat[1],quat[2],quat[3],quat[0]]).as_matrix()

            R[[0,0,1,2],[1,2,0,0]] *=-1

        distorted_points = []
        for i in range(numPoints-1):
            distorted_points.append( (i*(original_width/(numPoints-1)), 0) )
        for i in range(numPoints-1):
            distorted_points.append( (original_width, i*(original_height/(numPoints-1)) ) )
        for i in range(numPoints-1):
            p = numPoints-1 - i
            distorted_points.append( (p*(original_width/(numPoints-1)), original_height) )
        for i in range(numPoints-1):
            p = numPoints-1 - i
            distorted_points.append( (0, p*(original_height/(numPoints-1)) ) )


        distorted_points = np.array(distorted_points, np.float64)
        distorted_points = np.expand_dims(distorted_points, axis=0) #add extra dimension so opencv accepts points

        undistorted_points = cv2.fisheye.undistortPoints(distorted_points, self.K, self.D, R=R, P=self.K)
        undistorted_points = undistorted_points[0,:,:] #remove extra dimension

        #mtop = np.max(undistorted_points[:(numPoints-1),1])
        #mbottom = np.min(undistorted_points[numPoints:(2*numPoints-1),1])
        #mleft = np.max(undistorted_points[(2*numPoints):(3*numPoints-1),0])
        #mright = np.min(undistorted_points[(3*numPoints):,0])

        return undistorted_points
