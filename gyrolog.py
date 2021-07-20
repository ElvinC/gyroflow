import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# Generate 24 different (right handed) orientations using cross products
def generate_rotmats():
    basis = [[1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1]] # Six different unit vectors
    basis = [np.array(v) for v in basis]
    ORIENTATIONS = []


    for i in range(len(basis)):
        for j in range(len(basis)):
            if i != j and (i + 3) % 6 != j:
                ivec = basis[i]
                jvec = basis[j]
                kvec = np.cross(ivec,jvec)
                mat = np.vstack([ivec, jvec, kvec]).transpose()
                ORIENTATIONS.append(mat)

# 24 different (right handed) rotation matrices
ORIENTATIONS = [[[1, 0, 0], # 0 = identity
                [0, 1, 0],
                [0, 0, 1]],

                [[ 1,  0,  0],
                [ 0,  0, -1],
                [ 0,  1,  0]],

                [[ 1,  0,  0],
                [ 0, -1,  0],
                [ 0,  0, -1]],

                [[ 1,  0,  0],
                [ 0,  0,  1],
                [ 0, -1,  0]],

                [[ 0,  1,  0],
                [ 1,  0,  0],
                [ 0,  0, -1]],

                [[0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]],

                [[ 0, -1,  0],
                [ 1,  0,  0],
                [ 0,  0,  1]],

                [[ 0,  0, -1],
                [ 1,  0,  0],
                [ 0, -1,  0]],

                [[0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]],

                [[ 0,  0, -1],
                [ 0,  1,  0],
                [ 1,  0,  0]],

                [[ 0, -1,  0],
                [ 0,  0, -1],
                [ 1,  0,  0]],

                [[ 0,  0,  1],
                [ 0, -1,  0],
                [ 1,  0,  0]],

                [[-1,  0,  0],
                [ 0,  1,  0],
                [ 0,  0, -1]],

                [[-1,  0,  0],
                [ 0,  0,  1],
                [ 0,  1,  0]],

                [[-1,  0,  0],
                [ 0, -1,  0],
                [ 0,  0,  1]],

                [[-1,  0,  0],
                [ 0,  0, -1],
                [ 0, -1,  0]],

                [[ 0,  1,  0],
                [-1,  0,  0],
                [ 0,  0,  1]],

                [[ 0,  0, -1],
                [-1,  0,  0],
                [ 0,  1,  0]],

                [[ 0, -1,  0],
                [-1,  0,  0],
                [ 0,  0, -1]],

                [[ 0,  0,  1],
                [-1,  0,  0],
                [ 0, -1,  0]],

                [[ 0,  1,  0],
                [ 0,  0, -1],
                [-1,  0,  0]],

                [[ 0,  0,  1],
                [ 0,  1,  0],
                [-1,  0,  0]],

                [[ 0, -1,  0],
                [ 0,  0,  1],
                [-1,  0,  0]],

                [[ 0,  0, -1],
                [ 0, -1,  0],
                [-1,  0,  0]]]

ORIENTATIONS = [np.array(mat) for mat in ORIENTATIONS]

print(ORIENTATIONS)

def show_orientation(rotmat):
    orig_lw = 4
    sensor_lw = 2

    rotmat = np.array(rotmat)
    ivec = np.array([1,0,0]) # points to the "right". positive equals pitch up (objects in frame move down)
    jvec = np.array([0,1,0]) # points up.
    kvec = np.array([0,0,1]) # points away from lens

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    v = np.array([[-0.9, -0.7, -1], [0.9, -0.7, -1], [0.9, 0.7, -1],  [-0.9, 0.7, -1], [0, 0, 0]])
    ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])

    # "Standard" orientation
    ax.quiver([0],[0],[0], [1], [0], [0], color = 'red', alpha = .6, lw = orig_lw,)
    ax.quiver([0],[0],[0], [0], [1], [0], color = 'green', alpha = .6, lw = orig_lw,)
    ax.quiver([0],[0],[0], [0], [0], [1], color = 'blue', alpha = .6, lw = orig_lw,)

    sensor_i = rotmat * ivec * 1.6
    sensor_j = rotmat * jvec * 1.6
    sensor_k = rotmat * kvec * 1.6

    ax.quiver([0],[0],[0], sensor_i[0], sensor_i[1], sensor_i[2], color = 'red', alpha = .8, lw = sensor_lw,)
    ax.quiver([0],[0],[0], sensor_j[0], sensor_j[1], sensor_j[2], color = 'green', alpha = .8, lw = sensor_lw,)
    ax.quiver([0],[0],[0], sensor_k[0], sensor_k[1], sensor_k[2], color = 'blue', alpha = .8, lw = sensor_lw,)

    ax.set_xlim3d(-1.8, 1.8)
    ax.set_ylim3d(-1.8, 1.8)
    ax.set_zlim3d(-1.8, 1.8)



    # based on https://stackoverflow.com/questions/39408794/python-3d-pyramid
    verts = [ [v[0],v[1],v[4]], [v[0],v[3],v[4]],
    [v[2],v[1],v[4]], [v[2],v[3],v[4]], [v[0],v[1],v[2],v[3]]]

    ax.add_collection3d(Poly3DCollection(verts, facecolors='red', linewidths=1, edgecolors='red', alpha=0.1))

    plt.show()


class GyrologReader:
    def __init__(self):
        self.gyro = None # N*4 array with each column containing [t, gx, gy, gz]
        self.acc = None # N*4 array with each column containing [t, ax, ay, az]
        self.extracted = False
        # Assume same time reference and orientation used for both

        self.orientation_presets = []
        self.current_orientation_preset = ""

    def add_orientation_preset(self, orientation_name, correction_mat):
        self.orientation_presets.append([len(self.orientation_presets),orientation_name, correction_mat])


    def check_log_type(self, filename):
        # method to check if a data or video file is a certain log type
        return False

    def extract_log(self, filename):
        # Return fully formatted data

        # arbitrary convention used in gyroflow for no reason
        # x axis: points to the right. positive equals pitch up (objects in frame move down)
        # y axis: points up. positive equals pan left (objects move right)
        # z axis: points away from lens. positive equals CCW rotation (objects moves CW)

        # note that measured gravity vector points upwards when stationary due to equivalence to upwards acceleration

        return self.gyro, self.acc

    def apply_rotation(self, rotmat):
        if self.extracted:
            self.gyro[:,1:] = (rotmat * self.gyro[:,1:].transpose()).transpose

    def apply_inverse_rotation(self, rotmat):
        mat = np.linalg.inv(rotmat)
        pass

    def plot_gyro(self):
        xplot = plt.subplot(311)

        plt.plot(self.gyro[:,0], self.gyro[:,1])
        plt.ylabel("omega x [rad/s]")

        plt.subplot(312, sharex=xplot)

        plt.plot(self.gyro[:,0], self.gyro[:,2])
        plt.ylabel("omega y [rad/s]")

        plt.subplot(313, sharex=xplot)

        plt.plot(self.gyro[:,0], self.gyro[:,3])
        #plt.plot(self.integrator.get_raw_data("t") + d2, self.integrator.get_raw_data("z"))
        plt.xlabel("time [s]")
        plt.ylabel("omega z [rad/s]")

        plt.show()

class FakeData(GyrologReader):
    def __init__(self):
        super().__init__()


    def check_log_type(self, filename):
        return True

    def extract_log(self, filename):

        if filename == "rollpitchyaw":

            N = 1000

            self.gyro = np.zeros((N,4))
            self.gyro[:,0] = np.arange(N)/100 # 100 Hz data

            # t=2 to 3: positive roll
            self.gyro[200:300,3] = 1 # rad/s

            # t=4 to 5: positive pitch
            self.gyro[400:500,1] = 1

            # t=6 to 7
            self.gyro[600:700,2] = 1

            self.acc = np.zeros((N,4))
            self.acc[:,0] = np.arange(N)/100 # 100 Hz data

        else:
            np.random.seed(sum([ord(c) for c in filename]))

            N = 1000

            self.gyro = np.random.random((N,4))
            self.gyro[:,0] = np.arange(N)/100 # 100 Hz data

            self.acc = np.random.random((N,4))
            self.acc[:,0] = np.arange(N)/100 # 100 Hz data


        return self.gyro, self.acc





if __name__ == "__main__":
    show_orientation(ORIENTATIONS[15])
    exit()
    reader = FakeData()
    g, a = reader.extract_log("rollpitchyaw")
    reader.plot_gyro()
    print(g)
    print(a)
