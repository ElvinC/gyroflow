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

def quat_mult_nnp(Q1,Q2):
    w0, x0, y0, z0 = Q2
    w1, x1, y1, z1 = Q1
    return [-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0]

# https://www.mathworks.com/help/aeroblks/quaternioninverse.html
def inverse(q):
    # negate imaginary components to get inverse of unit quat
    return quaternion(q[0],-q[1],-q[2],-q[3])

def conjugate(q):
    # negate imaginary components to get inverse of unit quat
    return quaternion(q[0],-q[1],-q[2],-q[3])

def rotate_vector(q, v):
    q2 = [0, v[0],v[1],v[2]]
    return quaternion_multiply(quaternion_multiply(q,q2), conjugate(q))[1:]

def rotate_vector_fast(q,v):

    q2 = [0, v[0],v[1],v[2]]
    return np.array(quat_mult_nnp(quat_mult_nnp(q,q2), [q[0],-q[1],-q[2],-q[3]])[1:])

def rotate_vector_standalone(q,v):
    wxyz = q ** 2
    wx,wy,wz = q[0] * q[1:]

    xy = q[1] * q[2]
    xz = q[1] * q[3]
    yz = q[2] * q[3]

    # Formula from http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/transforms/index.htm
    # p2.x = w*w*p1.x + 2*y*w*p1.z - 2*z*w*p1.y + x*x*p1.x + 2*y*x*p1.y + 2*z*x*p1.z - z*z*p1.x - y*y*p1.x;
    # p2.y = 2*x*y*p1.x + y*y*p1.y + 2*z*y*p1.z + 2*w*z*p1.x - z*z*p1.y + w*w*p1.y - 2*x*w*p1.z - x*x*p1.y;
    # p2.z = 2*x*z*p1.x + 2*y*z*p1.y + z*z*p1.z - 2*w*y*p1.x - y*y*p1.z + 2*w*x*p1.y - x*x*p1.z + w*w*p1.z;

    r1 = wxyz[0]*v[0] + 2*wy*v[2] - 2*wz*v[1] + \
                wxyz[1]*v[0] + 2*xy*v[1] + 2*xz*v[2] - \
                wxyz[3]*v[0] - wxyz[2]*v[0]
    r2 = 2*xy*v[0] + wxyz[2]*v[1] + 2*yz*v[2] + \
                2*wz*v[0] - wxyz[3]*v[1] + wxyz[0]*v[1] - \
                2*wx*v[2] - wxyz[1]*v[1]
    r3 = 2*xz*v[0] + 2*yz*v[1] + wxyz[3]*v[2] - \
                2*wy*v[0] - wxyz[2]*v[2] + 2*wx*v[1] - \
                wxyz[1]*v[2] + wxyz[0]*v[2]

    return np.array([r1,r2,r3])

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

def single_slerp(v0, v1, t):
    v0 = np.array(v0)
    v1 = np.array(v1)
    dot = np.sum(v0 * v1)

    if dot < 0.0:
        v1 = -v1
        dot = -dot
    
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = v0 + t * (v1 - v0)
        return result / np.linalg.norm(result)
    
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (s0 * v0) + (s1 * v1)


def angle_between(q1, q2):
    z = quaternion_multiply(inverse(q1), q2)
    angle = 2 * np.arccos(min(z[0], 1))
    return angle

if __name__ == "__main__":
    import time


    q = np.array([0.5,0.7,0.5,0.5])
    v = np.array([1,2,3])
    start = time.time()

    for i in range(100000):
        rotate_vector(q,v)

    stop = time.time()
    print((stop-start) * 1000)

    start = time.time()

    for i in range(100000):
        rotate_vector_fast(q,v)

    stop = time.time()
    print((stop-start) * 1000)
    

    print(rotate_vector_fast(q,v))

    exit()

    a = pyquaternion.Quaternion([1,0,0,0])
    b = pyquaternion.Quaternion([0,1,0,0])
    start = time.time()
    for i in range(10000):
        c = pyquaternion.Quaternion.slerp(a,b,0.5)
    stop = time.time()

    print((stop - start) * 1000)

    a = quaternion(1, 0, 0, 0)
    b = quaternion(0, 1, 0, 0)
    
    start = time.time()
    for i in range(10000):
        c = slerp(a,b,[0.5])[0]
    stop = time.time()
    print((stop - start) * 1000)

    
    a = quaternion(1, 0, 0, 0)
    b = quaternion(0, 1, 0, 0)
    
    start = time.time()
    for i in range(10000):
        c = single_slerp(a,b,0.5)
    stop = time.time()
    print((stop - start) * 1000)

    for i in range(1):
        a = np.random.random(4)
        b = np.random.random(4)

        c = single_slerp(a,b,0.5)
        d = slerp(a,b,[0.5])[0]
        print(d-c)
