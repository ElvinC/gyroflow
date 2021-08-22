import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import gyrolog
from scipy.spatial.transform import Rotation
import numpy as np

#print(gyrolog.ORIENTATIONS)

example_in = np.array([1,2,3])
gx, gy, gz = example_in
out = np.array([-(gy), gz, -(gx)])

#camera_gyro_angle = 45

#r  = Rotation.from_euler('x', camera_gyro_angle, degrees=True)
#out = r.apply(out)

def generate_uptilt_mat(angle, degrees=False):
    angle = angle * np.pi / 180 if degrees else angle
    angle = -angle

    rotmat = np.array([[1,0,0],
                       [0,np.cos(angle),-np.sin(angle)],
                       [0,np.sin(angle),np.cos(angle)]])
    return rotmat

for i, rotmat in enumerate(gyrolog.ORIENTATIONS):
    #result = np.linalg.multi_dot([generate_uptilt_mat(45, True),rotmat, example_in])
    result = np.linalg.multi_dot([rotmat, example_in])
    #print(result)
    if np.allclose(result,out):
        print(i)
        print(rotmat)
        print(result)
print()
print(out)



a = np.random.random((4,3))
print(a)

rotmat = np.eye(3)
rotmat[2,2] = 2 # double last value

print(a.dot(rotmat.T))
print((rotmat * a.transpose()).transpose())

