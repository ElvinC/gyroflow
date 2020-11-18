import numpy as np
from scipy.spatial.transform import Rotation

def quaternion(w,x,y,z):
    return np.array([w,x,y,z])

def vector(x,y,z):
    return np.array([x,y,z])

def normalize(q):
    return q/np.sqrt(q.dot(q)) # q/|q|

# https://stackoverflow.com/questions/39000758/how-to-multiply-two-quaternions-by-python-or-numpy
def quaternion_multiply(Q1, Q2):
    w0, x0, y0, z0 = Q2
    w1, x1, y1, z1 = Q1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0])


# https://www.mathworks.com/help/aeroblks/quaternioninverse.html
def inverse(q):
    # negate imaginary components to get inverse of unit quat
    return quaternion(q[0],-q[1],-q[2],-q[3])

def rot_between(q1, q2):
    """Compute rotation quaternion from q1 to q2"""

    # https://www.gamedev.net/forums/topic/423462-rotation-difference-between-two-quaternions/
    return quaternion_multiply(inverse(q1), q2)

# https://en.wikipedia.org/wiki/Slerp
def slerp(v0, v1, t_array):
    """Spherical linear interpolation."""
    # >>> slerp([1,0,0,0], [0,0,0,1], np.arange(0, 1, 0.001))
    t_array = np.array(t_array)
    v0 = np.array(v0)
    v1 = np.array(v1)
    dot = np.sum(v0 * v1)

    if dot < 0.0:
        v1 = -v1
        dot = -dot
    
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = v0[np.newaxis,:] + t_array[:,np.newaxis] * (v1 - v0)[np.newaxis,:]
        return (result.T / np.linalg.norm(result, axis=1)).T
    
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t_array
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0[:,np.newaxis] * v0[np.newaxis,:]) + (s1[:,np.newaxis] * v1[np.newaxis,:])