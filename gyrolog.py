import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial.transform import Rotation
import csv
import re
import time
import sys, inspect
import logging
from scipy import signal, interpolate
from scipy.fft import fft, fftfreq

import insta360_utility as insta360_util
from blackbox_extract import BlackboxExtractor
from GPMF_gyro import Extractor as GPMFExtractor


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

def get_rotmat_from_id(id):
    return ORIENTATIONS[id]

def generate_uptilt_mat(angle, degrees=False):
    # Positive angle equals tilting camera up (gyro tilts down)
    angle = angle * np.pi / 180 if degrees else angle
    angle = -angle

    rotmat = np.array([[1,0,0],
                       [0,np.cos(angle),-np.sin(angle)],
                       [0,np.sin(angle),np.cos(angle)]])
    return rotmat

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
    def __init__(self, name="gyrolog"):

        self.name = name

        # The scaled data read from the file
        self.gyro = None # N*4 array with each column containing [t, gx, gy, gz]
        self.acc = None # N*4 array with each column containing [t, ax, ay, az]

        # The transformed data according to the gyroflow convention
        self.standard_gyro = None
        self.standard_acc = None

        self.extracted = False
        self.has_acc = False
        # Assume same time reference and orientation used for both

        self.default_filter = -1
        self.default_search_size = 10 # expected range of gyro/video offset

        self.pre_filter = -1

        self.filename = ""

        # Extra settings
        self.angle_setting = 0

        # Slightly different log formats
        self.variants = {
            "default": [0], # dict entry with correction matrix ID from ORIENTATIONS
            "default": [-1, [[1,0,0],[0,1,0],[0,0,1]]], # Alternatively -1 with second entry being a rotation matrix
        }
        self.variant = "default"

        self.orientation_presets = []
        self.current_orientation_preset = ""

        self.filename_pattern = ""

    def set_pre_filter(self, cutoff = -1):
        # Filter is applied before orientation transformation
        self.pre_filter = cutoff

    def post_init(self):
        # Run after init
        assert self.variant in self.variants


    def set_cam_up_angle(self,angle=0,degrees=False):
        self.angle_setting = angle * np.pi / 180 if degrees else angle

    def get_variants(self):
        return list(self.variants)
    
    def set_variant(self, variant=None):
        if variant in self.variants:
            self.variant = variant

    def get_variant_rotmat(self):
        info = self.variants[self.variant]
        if info[0] == -1 and len(info) == 2:
            return np.array(info[1])
        else:
            return get_rotmat_from_id(info[0])

    def filename_matches(self, filename):
        pattern = re.compile(self.filename_pattern)
        if pattern.match(filename):
            return True
        return False

    def add_orientation_preset(self, orientation_name, correction_mat):
        self.orientation_presets.append([len(self.orientation_presets),orientation_name, correction_mat])

    def guess_log_from_videofile(self, videofile):
        return ""
        return videofile

    def load_log_from_videofile(self, videofile):
        # Detect corresponding gyro log to a video file and loads it if available

        path = self.guess_log_from_videofile(videofile)
        if path:
            # detected valid path
            return self.extract_log(path)
        
        return False

    def check_log_type(self, filename):
        # method to check if a data or video file is a certain log type
        return False

    def extract_log_internal(self, filename):
        # To be overloaded
        # Return fully formatted data

        # arbitrary convention used in gyroflow for no reason
        # x axis: points to the right. positive equals pitch up (objects in frame move down)
        # y axis: points up. positive equals pan left (objects move right)
        # z axis: points away from lens. positive equals CCW rotation (objects moves CW)

        # note that measured gravity vector points upwards when stationary due to equivalence to upwards acceleration

        # These are the "raw, untransformed" values
        self.gyro = None
        self.acc = None

        self.num_data_points = 0
        self.gyro_sample_rate = 1

        # True if successful
        return True

    def extract_log(self, filename, check_file_exist= True):


        if os.path.isfile(filename) or (not check_file_exist):
            self.extracted = self.extract_log_internal(filename)

            if self.extracted:
                
                self.filename = filename
                if type(self.gyro) != type(None):

                    self.num_data_points = self.gyro.shape[0]
                    if self.num_data_points < 20:
                        print("Not enough datapoints")
                        return False

                    self.gyro_sample_rate = self.num_data_points / (self.gyro[-1,0] - self.gyro[0,0])

                    self.standard_gyro = np.copy(self.gyro)

                    if self.pre_filter > 0:
                        sosgyro = signal.butter(1, self.pre_filter, "lowpass", fs=self.gyro_sample_rate, output="sos")
                        self.standard_gyro[:,1:4] = signal.sosfiltfilt(sosgyro, self.gyro[:,1:4], 0) # Filter along "vertical" time axis

                    self.apply_variant_rotation_in_place(self.standard_gyro)



                if type(self.acc) != type(None):
                    self.standard_acc = np.copy(self.acc)

                    self.apply_variant_rotation_in_place(self.standard_acc)


                    # valid range: 0.9 to 1.1 g


            return self.extracted

        else:
            logging.error("Gyro file doesn't exist")
            return False

    def get_transformed_gyro(self):
        if self.extracted:
            return self.standard_gyro
        return None

    def get_transformed_acc(self):
        if self.extracted:
            return self.standard_acc
        return None

    def get_gyro(self):
        if self.extracted:
            return self.gyro
        return None

    def get_acc(self):
        if self.extracted and self.has_acc:
            return self.acc
        return None

    def apply_rotation(self, rotmat, time_data):
        # Applies in place
        time_data[:,1:] = time_data[:,1:].dot(rotmat.T)

    def apply_variant_rotation_in_place(self, time_data):

        # Transform to standard first
        print(self.name)
        if self.variants[self.variant][0] == 0:
            pass # identity
        else:
            # apply in place
            self.apply_rotation(self.get_variant_rotmat(), time_data)

        # handle tilt
        if self.angle_setting: # not zero
            self.apply_rotation(generate_uptilt_mat(self.angle_setting), time_data)

    def apply_inverse_rotation(self, rotmat):
        mat = np.linalg.inv(rotmat)
        pass

    def plot_gyro(self, blocking=False):
        
        
        xplot = plt.subplot(321)

        plt.plot(self.standard_gyro[:,0], self.standard_gyro[:,1])
        plt.ylabel("omega x [rad/s]")
        plt.grid()

        plt.subplot(323, sharex=xplot)

        plt.plot(self.standard_gyro[:,0], self.standard_gyro[:,2])
        plt.ylabel("omega y [rad/s]")
        plt.grid()

        plt.subplot(325, sharex=xplot)

        plt.plot(self.standard_gyro[:,0], self.standard_gyro[:,3])
        #plt.plot(self.integrator.get_raw_data("t") + d2, self.integrator.get_raw_data("z"))
        plt.xlabel("time [s]")
        plt.ylabel("omega z [rad/s]")
        plt.grid()

        #plt.show(block=blocking)

        #plt.figure()
        xplot = plt.subplot(222)

        N = self.standard_gyro.shape[0]
        T = (self.standard_gyro[-1,0] - self.standard_gyro[0,0]) /  N
        freq = 1/T

        x = self.standard_gyro[:,0]
        y = self.standard_gyro[:,1]
        
        yf = fft(y)
        xf = fftfreq(N, T)[:N//2]

        alpha = 0.7

        f, Pxx_den = signal.welch(y, freq, nperseg=1024)
        plt.plot(f, Pxx_den)
        plt.legend("x")

        y = self.standard_gyro[:,2]
        yf = fft(y)
        xf = fftfreq(N, T)[:N//2]
        
        f, Pxx_den = signal.welch(y, freq, nperseg=1024)
        plt.plot(f, Pxx_den)
        plt.legend("y")

        y = self.standard_gyro[:,3]
        yf = fft(y)
        xf = fftfreq(N, T)[:N//2]
        
        f, Pxx_den = signal.welch(y, freq, nperseg=1024)
        plt.plot(f, Pxx_den)
        plt.legend(["x", "y", "z"])
        plt.grid()
        plt.ylabel("Power density")

        plt.subplot(224, sharex=xplot)
        y = self.standard_gyro[:,1]
        
        yf = fft(y)
        xf = fftfreq(N, T)[:N//2]

        plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]), alpha=alpha)
        plt.legend("x")

        y = self.standard_gyro[:,2]
        yf = fft(y)
        xf = fftfreq(N, T)[:N//2]
        plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]), alpha=alpha)
        plt.legend("y")

        y = self.standard_gyro[:,3]
        yf = fft(y)
        xf = fftfreq(N, T)[:N//2]
        plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]), alpha=alpha)
        plt.legend(["x", "y", "z"])

        plt.grid()
        plt.ylabel("FFT")
        plt.xlabel("Frequency [Hz]")
        plt.show(block=blocking)

    def plot_acc(self):
        if type(self.acc) != type(None):
            xplot = plt.subplot(411)

            plt.plot(self.standard_acc[:,0], self.standard_acc[:,1])
            plt.ylabel("acc x [g]")

            plt.subplot(412, sharex=xplot)

            plt.plot(self.standard_acc[:,0], self.standard_acc[:,2])
            plt.ylabel("acc y [g]")

            plt.subplot(413, sharex=xplot)

            plt.plot(self.standard_acc[:,0], self.standard_acc[:,3])
            #plt.plot(self.integrator.get_raw_data("t") + d2, self.integrator.get_raw_data("z"))
            plt.xlabel("time [s]")
            plt.ylabel("acc z [g]")

            plt.subplot(414, sharex=xplot)

            plt.plot(self.standard_acc[:,0], np.sqrt(np.sum(self.standard_acc[:,1:]**2,1)))
            plt.plot([0, self.standard_acc[-1,0]], [1.1,1.1])
            plt.plot([0, self.standard_acc[-1,0]], [0.9,0.9])
            #plt.plot(self.integrator.get_raw_data("t") + d2, self.integrator.get_raw_data("z"))
            plt.xlabel("time [s]")
            plt.ylabel("mag [g]")

            plt.show()


    def save_gyroflow_format(self, filename=False):
        if not filename:
            filename = self.filename + ".gcsv"
        
        has_gyro = type(self.gyro) != type(None)
        has_acc = type(self.acc) != type(None)

        if not has_gyro:
            return False

        if has_acc:
            if self.gyro.shape != self.acc.shape:
                print(self.gyro.shape, self.acc.shape)
                print("Gyro and acc are not the same shape")
                return False

        with open(filename, "w") as f:
            # GYROFLOW IMU LOG
            # tscale,0.001
            # gscale,0.0002663161
            # ascale,0.00059875488
            # t,gx,gy,gz,ax,ay,az
            # 0,39,86,183,-1137,-15689,-29
            f.write("GYROFLOW IMU LOG\n")
            f.write("tscale,1\n") # time in seconds
            f.write("gscale,1\n") # gyro in rad/s
            f.write("ascale,1\n") # acceleration in g
            f.write("t,gx,gy,gz,ax,ay,az" if has_acc else "t,gx,gy,gz\n")

            for i in range(self.gyro.shape[0]):
                line = list(self.gyro[i,1:])
                if has_acc:
                    line += list(self.acc[i,1:]) # don't add time
                # round time to tenth of millisecond
                # 4 significant digits in data
                line = [str(round(self.gyro[i,0], 4))] + [f"{n:.4g}" for n in line] 
                f.write(",".join(line) + "\n")
            

        return True
        
class BlackboxCSVData(GyrologReader):
    def __init__(self):
        super().__init__("Blackbox CSV file")
        self.filename_pattern = "(?i).*\.csv"
        self.angle_setting = 0

        self.variants = {
            "default": [12], # dict entry with correction matrix ID from ORIENTATIONS
            "Raw gyro (debug_mode = GYRO_SCALED)": [12],
            "iNav/blackbox-tools": [12]
        }

        self.variant = "default"
        self.default_search_size = 10

        self.post_init()

    def check_log_type(self, filename):
        fname = os.path.split(filename)[-1]
        if self.filename_matches(fname):
            # open and check first line
            with open(filename, "r") as f:
                firstline = f.readline().strip()
                if firstline == '"Product","Blackbox flight data recorder by Nicholas Sherlock"':
                    return True
                elif firstline.startswith('loopIteration,time (us),'):
                    # File generated by https://github.com/iNavFlight/blackbox-tools
                    self.set_variant("iNav/blackbox-tools")
                    return True

        return False
        
    def guess_log_from_videofile(self, videofile):
        no_suffix = os.path.splitext(videofile)[0]
        #path, fname = os.path.split(videofile)

        log_suffixes = [".bbl.csv", ".bfl.csv", ".csv"]
        log_suffixes += [ex.upper() for ex in log_suffixes]
        for suffix in log_suffixes:
            if os.path.isfile(no_suffix + suffix):
                logpath = no_suffix + suffix
                #print("Automatically detected gyro log file: {}".format(logpath.split("/")[-1]))

                if self.check_log_type(logpath):
                    return logpath

        return False

    def extract_log_internal(self, filename):

        with open(filename) as bblcsv:
            gyro_index = None
            acc_index = None
            max_index = 0

            csv_reader = csv.reader(bblcsv)
            for i, row in enumerate(csv_reader):
                #print(row)

                stripped_row = [field.strip() for field in row]
                if stripped_row[0] == "loopIteration":
                    if self.variant == "Raw gyro (debug_mode = GYRO_SCALED)" and 'debug[0]' in stripped_row:
                        gyro_index = stripped_row.index('debug[0]')
                        print('Using raw gyro data')
                    else:
                        gyro_index = stripped_row.index('gyroADC[0]')
                        #print('Using filtered gyro data')

                    max_index = gyro_index + 2

                    if "accSmooth[0]" in stripped_row:
                        acc_index = stripped_row.index("accSmooth[0]")
                        max_index = acc_index + 2

                    break

            data_list = []
            acc_list = []
            gyroscale = np.pi/180
            acc_scale = 1/2048

            last_t = 0
            self.max_data_gab = 10
            for row in csv_reader:
                t = float(row[1])
                if max_index<len(row) and (((0 < (t - last_t) < 1000000 * self.max_data_gab) or (last_t == 0))) :

                    gx = float(row[gyro_index+1])
                    gy = float(row[gyro_index+2])
                    gz = float(row[gyro_index])
                    last_t = t

                    #data_list.append(f)
                    data_list.append([t / 1000000, gx, gy, gz])
                    if acc_index:
                        ax = float(row[acc_index+1])
                        ay = float(row[acc_index+2])
                        az = float(row[acc_index])

                        acc_list.append([t / 1000000, ax, ay, az])

            self.gyro = np.array(data_list)
            self.gyro[:,1:] *= gyroscale

            if acc_index:
                self.acc = np.array(acc_list)
                self.acc[:,1:] *= acc_scale


        return True

class BlackboxRawData(GyrologReader):
    def __init__(self):
        super().__init__("Blackbox raw file")
        self.filename_pattern = "(?i).*\.(?:bbl|bfl|txt)"
        self.angle_setting = 0

        self.variants = {
            "default": [12] # dict entry with correction matrix ID from ORIENTATIONS
        }

        self.variant = "default"

        self.default_filter = -1
        self.default_search_size = 10

        self.post_init()

    def check_log_type(self, filename):
        fname = os.path.split(filename)[-1]
        if self.filename_matches(fname):
            # open and check first line
            with open(filename, "rb") as f:
                firstline = f.read(64)
                try:
                    firstline = firstline.decode('ascii').split("\n")[0]

                    if "Product:Blackbox flight data recorder by Nicholas Sherlock" in firstline:
                        return True
                    return False
                except:
                    return False

        return False
        
    def guess_log_from_videofile(self, videofile):

        no_suffix = os.path.splitext(videofile)[0]
        #path, fname = os.path.split(videofile)

        log_suffixes = [".bbl", ".bfl", ".txt"] # txt is inav blackbox
        log_suffixes += [ex.upper() for ex in log_suffixes]
        for suffix in log_suffixes:
            if os.path.isfile(no_suffix + suffix):
                logpath = no_suffix + suffix
                #print("Automatically detected gyro log file: {}".format(logpath.split("/")[-1]))

                if self.check_log_type(logpath):
                    return logpath

        return False

    def extract_log_internal(self, filename):

        try:
            bbe = BlackboxExtractor(filename)
            self.gyro, self.acc = bbe.get_untransformed_imu_data()

            return True
        except: # TODO: change
            print("Error reading raw blackbox file. Try converting to CSV in blackbox explorer")
            return False

class RuncamData(GyrologReader):
    def __init__(self):
        super().__init__("Runcam CSV log")
        self.filename_pattern = ".*\.csv"

        self.variants = {
            "Runcam 5 Orange": [0],
            "iFlight GOCam GR": [0]
        }
        self.variant = "Runcam 5 Orange"

        self.default_filter = 70
        self.default_search_size = 4 # usually within +/- 1 second

        self.post_init()


    def check_log_type(self, filename):
        fname = os.path.split(filename)[-1]
        firstlines = ["time,x,y,z,ax,ay,az", "time,rx,ry,rz,ax,ay,az", "time,x,y,z"] # Different firmware versions
        if self.filename_matches(fname):
            # open and check first line
            with open(filename, "r") as f:
                firstline = f.readline().strip()
                #print(firstline)
                if firstline in firstlines:
                    return True

        return False

    def guess_log_from_videofile(self, videofile):
        path, fname = os.path.split(videofile)

        # Runcam 5 Orange
        rc5pattern = re.compile("RC_(\d{4})_.*\..*") # example: RC_0030_210719221659.MP4
        gocampattern = re.compile("IF-RC01_(\d{4})\..*") # example: IF-RC01_0011.MP4
        
        if rc5pattern.match(fname):
            self.variant = "Runcam 5 Orange"
            counter = int(rc5pattern.match(fname).group(1))
        
        # Gocam
        elif gocampattern.match(fname):
            self.variant = "iFlight GOCam GR"
            counter = int(gocampattern.match(fname).group(1))

        else:
            return False

        lognames = [f"RC_GyroData{counter:04d}.csv", f"gyroDate{counter:04d}.csv"] # different firmwares
        for logname in lognames:
            logpath = videofile.rstrip(fname) + logname
            print(logpath)
            if os.path.isfile(logpath):
                if self.check_log_type(logpath):
                    return logpath
        
        return False


    def extract_log_internal(self, filename):

        with open(filename) as csvfile:
            next(csvfile)

            lines = csvfile.readlines()

            has_acc = len(lines[0].split(",")) == 7

            data_list = []
            acc_list = []
            #gyroscale = 0.070 * np.pi/180 # plus minus 2000 dps 16 bit two's complement. 70 mdps/LSB per datasheet.
            gyroscale = 500 / 2**15 * np.pi/180 # 500 dps
            acc_scale = 2 / 2**15 # +/- 2 g

            

            for line in lines:
                splitdata = [float(x) for x in line.split(",")]
                t = splitdata[0]/1000

                # RC5
                if self.variant=="Runcam 5 Orange":
                    gx = splitdata[3] * gyroscale
                    gy = -splitdata[1] * gyroscale
                    gz = splitdata[2] * gyroscale
                elif self.variant == "iFlight GOCam GR":
                    gx = -splitdata[3] * gyroscale
                    gy = -splitdata[1] * gyroscale
                    gz = -splitdata[2] * gyroscale
                
                if has_acc:
                    if self.variant=="Runcam 5 Orange":
                        ax = -splitdata[4] * acc_scale
                        ay = -splitdata[5] * acc_scale
                        az = splitdata[6] * acc_scale
                    elif self.variant == "iFlight GOCam GR":
                        ax = -splitdata[4] * acc_scale
                        ay = splitdata[5] * acc_scale
                        az = -splitdata[6] * acc_scale

                    acc_list.append([t, ax, ay, az])

                # accelerometer

                # Z: roll
                # X: yaw
                # y: pitch

                data_list.append([t, gx, gy, gz])

        self.gyro = np.array(data_list)
        #sosgyro = signal.butter(1, 8, "lowpass", fs=500, output="sos")

        #self.gyro[:,1:4] = signal.sosfiltfilt(sosgyro, self.gyro[:,1:4], 0) # Filter along "vertical" time axis

        if has_acc:
            self.acc = np.array(acc_list)

        return True


class Insta360Log(GyrologReader):
    def __init__(self):
        super().__init__("Insta360 IMU metadata")
        self.filename_pattern = "(?i).*\.mp4"

        self.variants = {
            "smo4k": [22],
            "insta360 oner": [22]
        }

        self.variant = "smo4k"

        self.default_filter = 50
        self.default_search_size = 10

        self.post_init()

    def check_log_type(self, filename):
        if self.filename_matches(filename):
            return insta360_util.isInsta360Video(filename)
        return False


    def guess_log_from_videofile(self, videofile):

        if self.check_log_type(videofile):
            return videofile
        else:
            return False


    def extract_log_internal(self, filename):

        if self.variant=="smo4k":
            self.gyro, self.acc = insta360_util.get_insta360_gyro_data(filename, filterArray=[])
        elif self.variant=="insta360 oner":
            self.gyro, self.acc = insta360_util.get_insta360_gyro_data(filename, filterArray=[], revertIMU=False)
        else:
            # Assume SMO4K - For no real reason....
            self.gyro, self.acc = insta360_util.get_insta360_gyro_data(filename, filterArray=[])

        # Coverting gyro to XYZ to -Z,-X,Y
        #self.gyro = np.empty([len(gyro_data_input), 4])
        #self.gyro[:,0] = gyro_data_input[:,0][:]
        #self.gyro[:,1] = gyro_data_input[:,2][:] * -1
        #self.gyro[:,2] = gyro_data_input[:,3][:]
        #self.gyro[:,3] = gyro_data_input[:,1][:] * -1

        return True

class GPMFLog(GyrologReader):
    def __init__(self):
        super().__init__("GoPro GPMF metadata")
        self.filename_pattern = "(?i).*\.mp4"

        self.variants =  {
            "hero5": [13],
            "hero6": [0],
            "hero7": [0],
            "hero8": [1],
            "hero9": [13]
        }

        self.variant = "hero6"

        self.default_filter = -1
        self.default_search_size = 4

        self.gpmf = None

        self.post_init()

    def check_log_type(self, filename):

        
        #gyro_data = gpmf.get_gyro(True)

        if self.filename_matches(filename):
            try:
                self.gpmf = GPMFExtractor(filename)
                return True
            except:
                # Error if it doesn't contain GPMF data
                return False
        return False


    def guess_log_from_videofile(self, videofile):

        if self.check_log_type(videofile):
            return videofile
        else:
            return False


    def extract_log_internal(self, filename):

        try:
            if self.gpmf:
                if self.gpmf.videopath == filename:
                    pass
                else:
                    self.gpmf = GPMFExtractor(filename)
            else:
                self.gpmf = GPMFExtractor(filename)

            self.gyro = self.gpmf.get_gyro(True)
            self.gpmf.parse_accl()
            self.acc = self.gpmf.get_accl(True)
            
            minlength = min(self.gyro.shape[0], self.acc.shape[0])
            maxlength = max(self.gyro.shape[0], self.acc.shape[0])
            # Make sure they match
            if maxlength - minlength == 0: #
                pass
            elif maxlength - minlength < 10:
                # probably just some missing datapoints
                self.gyro = self.gyro[0:minlength]
                self.acc = self.acc[0:minlength]
                self.acc[:,0] = self.gyro[:,0] # same timescale, acceleration less time-sensitive
            else:
                to_interp = interpolate.interp1d(self.acc[:,0], self.acc[:,1:], axis=0,fill_value=np.array([0,1,0]), bounds_error=False)
                new_acc = np.copy(self.gyro)
                new_acc[:,1:] = to_interp(self.gyro[:,0])
                self.acc = new_acc
                # resample acc to gyro timescale

        except Exception as e:
            print(e)
            print("Failed to extract GPMF gyro")
            return False

        hero = int(self.variant.lstrip("hero"))

        # Hero 6??
        if hero == 6:
            pass
            # Identity
            #self.gyro[:,1] = self.gyro[:,1]
            #self.gyro[:,2] = self.gyro[:,2]
            #self.gyro[:,3] = self.gyro[:,3]
        elif hero == 7:
            pass
            #self.gyro[:,1] = self.gyro[:,1]
            #self.gyro[:,2] = self.gyro[:,2]
            #self.gyro[:,3] = self.gyro[:,3]
        elif hero == 5:
            pass
            # equivalent to matrix index 13 
            #self.gyro[:,1] = -self.gyro[:,1]
            #self.gyro[:,2] = self.gyro[:,2]
            #self.gyro[:,3] = self.gyro[:,3]
            #self.gyro[:,[2, 3]] = self.gyro[:,[3, 2]]

        elif hero == 8:
            pass
            # Hero 8??
            # equal matrix index 1
            #self.gyro[:,[2, 3]] = self.gyro[:,[3, 2]]
            #self.gyro[:,2] = -self.gyro[:,2]
        elif hero == 9:
            pass
            #self.gyro[:,1] = -self.gyro[:,1]
            #self.gyro[:,2] = self.gyro[:,2]
            #self.gyro[:,3] = self.gyro[:,3]
            #self.gyro[:,[2, 3]] = self.gyro[:,[3, 2]]

        return True

class GyroflowGyroLog(GyrologReader):
    def __init__(self):
        super().__init__("Gyroflow IMU log")
        # Eh, gcsv for gyro csv, that works I guess.
        self.filename_pattern = ".*\.gcsv"

        self.firstlines = ["GYROFLOW IMU LOG", "CAMERA IMU LOG"] #["t,gx,gy,gz,ax,ay,az,mx,my,mz"]

        self.variants =  {
            "default": [0]
        }

        self.variant = "default"

        self.default_filter = -1
        self.default_search_size = 10

        self.post_init()

        # Prelim log format:
        # GYROFLOW IMU LOG
        # tscale,0.001
        # gscale,0.0002663161
        # ascale,0.00059875488
        # t,gx,gy,gz,ax,ay,az
        # 0,39,86,183,-1137,-15689,-2986
        # 1,56,100,202,-1179,-15694,-2887
        # 2,63,108,218,-1247,-15702,-2794
        # 3,71,108,243,-1308,-15675,-2727
        # 4,83,101,268,-1420,-15662,-2661

    def check_log_type(self, filename):
        fname = os.path.split(filename)[-1]
        if self.filename_matches(fname):
            # open and check first line
            with open(filename, "r") as f:
                firstline = f.readline().strip()
                #print(firstline)
                if firstline in self.firstlines:
                    return True

        return False

    def guess_log_from_videofile(self, videofile):
        no_suffix = os.path.splitext(videofile)[0]
        #path, fname = os.path.split(videofile)

        if os.path.isfile(no_suffix + ".gcsv"):
            logpath = no_suffix + ".gcsv"
            #print("Automatically detected gyro log file: {}".format(logpath.split("/")[-1]))

            if self.check_log_type(logpath):
                return logpath
        return False
        


    def extract_log_internal(self, filename):

        tscale = 0.001
        gscale = 1
        ascale = 1
        mscale = 1

        with open(filename) as csvfile:
            firstline = csvfile.readline().strip()

            if firstline not in self.firstlines:
                return False

            line = ""
            while not line.startswith("t,"):
                line = csvfile.readline().strip()
                #print(line)
                if line.startswith("tscale,"):
                    tscale = float(line.split(",")[1])
                elif line.startswith("gscale,"):
                    gscale = float(line.split(",")[1])
                elif line.startswith("ascale,"):
                    ascale = float(line.split(",")[1])
                elif line.startswith("mscale,"):
                    mscale = float(line.split(",")[1])

            #print(tscale, gscale, ascale)


            header = line.split(",")
            header_length = len(header)

            has_gyro = "gx" in header
            has_acc = "ax" in header
            has_mag = "mx" in header


            # Get data
            lines = csvfile.readlines()

            data_list = []

            for line in lines:
                splitdata = [float(x) for x in line.split(",")]
                if len(splitdata) == header_length: # make sure no missing fields
                    t = splitdata[0] * tscale

                    gx = splitdata[1] * gscale
                    gy = splitdata[2] * gscale
                    gz = splitdata[3] * gscale

                    data_list.append([t, gx, gy, gz])

        self.gyro = np.array(data_list)

        return True


class FakeData(GyrologReader):
    def __init__(self):
        super().__init__()


    def check_log_type(self, filename):
        return True

    def extract_log_internal(self, filename):

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


        return True


log_reader_classes = [GyroflowGyroLog,
                      BlackboxCSVData,
                      BlackboxRawData,
                      RuncamData,
                      Insta360Log,
                      GPMFLog]

print("Available log types")
for alg in log_reader_classes:
    print(alg().name)
log_reader_names = [alg().name for alg in log_reader_classes]

def get_log_reader_names():
    """List of available control algorithms in plaintext
    """
    return log_reader_names

def get_all_log_reader_instances():
    return [alg() for alg in log_reader_classes]

log_reader_instances = get_all_log_reader_instances()

def get_log_reader_by_name(name="nothing"):
    """Get an instance of a log reader class from name
    """
    if name in log_reader_names:
        return log_reader_classes[log_reader_names.index(name)]()
    else:
        return None

def get_variants_by_log_type(logtype="Gyroflow IMU log"):
    reader = get_log_reader_by_name(logtype)
    if reader:
        return reader.get_variants()
    return []

def guess_log_type_from_video(videofile, check_data = False):
    for reader in log_reader_instances:
        guess = reader.guess_log_from_videofile(videofile)
        if guess:
            print(f"{videofile} has log {guess} with type '{reader.name}'")
            

            if check_data:
                if reader.extract_log(guess):
                    N = reader.gyro.shape[0]
                    print(f"{N} samples extracted")

                    reader.plot_gyro()
                    reader.plot_acc()

                    reader.save_gyroflow_format()
                
            return guess, reader.name, reader.variant

    print(f"Couldn't guess log type of {videofile}")
    return False, "", ""

def guess_log_type_from_log(logfile, check_data = False):
    for reader in log_reader_instances:
        check = reader.check_log_type(logfile)
        if check:
            print(f"log {logfile} has type '{reader.name}'")
            

            if check_data:
                if reader.extract_log(logfile):
                    N = reader.gyro.shape[0]
                    print(f"{N} samples extracted")

                    reader.plot_gyro()

                    reader.save_gyroflow_format()
                
            return True, reader.name, reader.variant

    print(f"Couldn't guess log type of {logfile}")
    return False, "", ""


if __name__ == "__main__":

    tests = [
        "test_clips/badbbl.bbl",
        "test_clips/Runcam/gyroDate0006.csv"
        "C:/Users/TUDelftSID/Downloads/20210814 gocam/IF-RC01_0010.bbl",
        "C:/Users/TUDelftSID/Downloads/20210814 gocam/gyroDate0010.csv",
    ]

    #reader = BlackboxRawData()
    #reader.set_cam_up_angle(30,degrees=True)
    #reader.extract_log("C:/Users/TUDelftSID/Downloads/20210814 gocam/IF-RC01_0010.bbl")
    #reader.plot_gyro()

    #reader = RuncamData()
    success, logtype, variant = guess_log_type_from_log(tests[0])
    reader = get_log_reader_by_name(logtype)
    reader.set_variant(variant)
    reader.extract_log(tests[0])
    reader.plot_gyro()
    plt.show()
    exit()

    test_video_clips = [
        #"D:\\DCIM\\100RUNCAM\\RC_0038_210813215250.MP4",
        #"test_clips/PRO_VID_20210111_144304_00_010.mp4",
        #"test_clips/IF-RC01_0026.MP4",
        #"test_clips/RC_0038_210813211513.MP4",
        #"test_clips/RC_0031_210722220523.MP4",
        #"test_clips/Runcam/RC_0036_filtered.MP4",
        "test_clips/DJIG0043wiebe.mp4",
        #"test_clips/GX016015.MP4",
        #"test_clips/nivim_insta360.mp4",
        "C:/Users/TUDelftSID/Downloads/20210814 gocam/IF-RC01_0010.MP4",
        "test_clips/smo4k_calibration.mp4",
        "test_clips/Tiago_Ferreira_5_inch.mp4",
        "test_clips/MasterTim17_caddx.mp4",
        "test_clips/starling2.MOV",
        "test_clips/raw_inav_log.mp4"
    ]
    for clip in test_video_clips[0:2]:
        guess_log_type_from_video(clip,check_data=True)
        

    exit()


    testcases = [[BlackboxCSVData(), "test_clips/btfl_005.bbl.csv"],
                 [BlackboxRawData(), "test_clips/nicecrash_hero7.bbl"],
                 [RuncamData(), "test_clips/Runcam/RC_GyroData0038.csv"],
                 [Insta360Log(), "test_clips/PRO_VID_20210111_144304_00_010.mp4"],
                 [GPMFLog(), "test_clips/GX016015.MP4"],
                 [GyroflowGyroLog(), "test_clips/gyroflow_format_example.gcsv"],
                 [None, "test_clips/inav_log.csv"]]
    
    for reader, path in testcases[6:]:

        print(guess_log_type_from_log(path,check_data=True))
        continue
        print(f"Using {reader.name}")
        check = reader.check_log_type(path)

        print(f"Log type detected: {check}")

        if check:
            start = time.time()
            check = reader.extract_log(path)
            print(f"Extraction success: {check}")
            N = reader.gyro.shape[0]
            tottime = time.time() - start
            print(f"{N} samples extracted in {tottime} seconds")
            
            print()
            

    #print(reader.load_log_from_videofile("test_clips/Runcam/RC_0038_210723215432.MP4"))
    #print(reader.load_log_from_videofile("test_clips/DJIG0043wiebe.mp4"))

    #reader.plot_gyro()
