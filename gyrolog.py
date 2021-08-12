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

        # Extra settings
        self.angle_setting = 0

        # Slightly different log formats
        self.variants = {
            "standard": [0], # dict entry with correction matrix ID from ORIENTATIONS
            "standard1": [-1, [[1,0,0],[0,1,0],[0,0,1]]], # Alternatively -1 with second entry being a rotation matrix
        }
        self.variant = "standard"

        self.orientation_presets = []
        self.current_orientation_preset = ""

        self.filename_pattern = ""

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

        # True if successful
        return True

    def extract_log(self, filename, check_file_exist= True):

        if os.path.isfile(filename) or (not check_file_exist):
            self.extracted = self.extract_log_internal(filename)

            if self.extracted:
                if type(self.gyro) != type(None):
                    self.standard_gyro = np.copy(self.gyro)

                    self.apply_variant_rotation_in_place(self.standard_gyro)

                if type(self.acc) != type(None):
                    self.standard_acc = np.copy(self.acc)

                    self.apply_variant_rotation_in_place(self.standard_acc)

            return self.extracted

        else:
            logging.error("Gyro file doesn't exist")
            return False

    def get_gyro(self):
        if self.extracted:
            return self.gyro

    def get_acc(self):
        if self.extracted and self.has_acc:
            return self.acc

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

class BlackboxCSVData(GyrologReader):
    def __init__(self):
        super().__init__("Blackbox CSV file")
        self.filename_pattern = "(?i).*\.csv"
        self.angle_setting = 0

        self.variants = {
            "standard": [12] # dict entry with correction matrix ID from ORIENTATIONS
        }

        self.variant = "standard"

        self.post_init()

    def check_log_type(self, filename):
        fname = os.path.split(filename)[-1]
        if self.filename_matches(fname):
            # open and check first line
            with open(filename, "r") as f:
                firstline = f.readline().strip()
                if firstline == '"Product","Blackbox flight data recorder by Nicholas Sherlock"':
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

        use_raw_gyro_data = False

        with open(filename) as bblcsv:
            gyro_index = None

            csv_reader = csv.reader(bblcsv)
            for i, row in enumerate(csv_reader):
                #print(row)

                stripped_row = [field.strip() for field in row]
                if stripped_row[0] == "loopIteration":
                    if use_raw_gyro_data:
                        gyro_index = stripped_row.index('debug[0]')
                        print('Using raw gyro data')
                    else:
                        gyro_index = stripped_row.index('gyroADC[0]')
                        #print('Using filtered gyro data')

                    break

            data_list = []
            gyroscale = np.pi/180
            r  = Rotation.from_euler('x', self.angle_setting, degrees=True)
            last_t = 0
            self.max_data_gab = 10
            for row in csv_reader:
                t = float(row[1])
                if ((0 < (t - last_t) < 1000000 * self.max_data_gab) or (last_t == 0)):

                    gx = float(row[gyro_index+1])* gyroscale
                    gy = float(row[gyro_index+2])* gyroscale
                    gz = float(row[gyro_index])* gyroscale
                    last_t = t

                    #to_rotate = [-(gx),
                    #                (gy),
                    #                -(gz)]

                    #rotated = r.apply(to_rotate)

                    #f = [t / 1000000,
                    #        rotated[0],
                    #        rotated[1],
                    #        rotated[2]]

                    #data_list.append(f)
                    data_list.append([t / 1000000, gx, gy, gz])

            self.gyro = np.array(data_list)

        return True

class BlackboxRawData(GyrologReader):
    def __init__(self):
        super().__init__("Blackbox raw file")
        self.filename_pattern = "(?i).*\.(?:bbl|bfl|txt)"
        self.angle_setting = 0

        self.variants = {
            "standard": [12] # dict entry with correction matrix ID from ORIENTATIONS
        }

        self.variant = "standard"

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
            self.gyro = bbe.get_gyro_data(cam_angle_degrees=self.angle_setting)

            return True
        except ZeroDivisionError:
            print("Error reading raw blackbox file. Try converting to CSV in blackbox explorer")
            return False

class RuncamData(GyrologReader):
    def __init__(self):
        super().__init__("Runcam CSV log")
        self.filename_pattern = "RC_GyroData\d{4}\.csv"

        self.variants = {
            "Runcam 5 Orange": [0],
            "iFlight GOCam GR": [0]
        }
        self.variant = "Runcam 5 Orange"

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
            counter = int(rc5pattern.match(fname).group(1))
        
        # Gocam
        elif gocampattern.match(fname):
            counter = int(gocampattern.match(fname).group(1))

        else:
            return False

        logname = f"RC_GyroData{counter:04d}.csv"
        
        logpath = videofile.rstrip(fname) + logname
        
        if not os.path.isfile(logpath):
            return False

        if self.check_log_type(logpath):
            return logpath
        else:
            return False


    def extract_log_internal(self, filename):

        with open(filename) as csvfile:
            next(csvfile)

            lines = csvfile.readlines()

            data_list = []
            #gyroscale = 0.070 * np.pi/180 # plus minus 2000 dps 16 bit two's complement. 70 mdps/LSB per datasheet.
            gyroscale = 500 / 2**15 * np.pi/180 # 500 dps

            for line in lines:
                splitdata = [float(x) for x in line.split(",")]
                t = splitdata[0]/1000

                # RC5
                gx = splitdata[3] * gyroscale
                gy = -splitdata[1] * gyroscale
                gz = splitdata[2] * gyroscale

                # Z: roll
                # X: yaw
                # y: pitch

                data_list.append([t, gx, gy, gz])

        self.gyro = np.array(data_list)

        return True


class Insta360Log(GyrologReader):
    def __init__(self):
        super().__init__("Insta360 IMU metadata")
        self.filename_pattern = "(?i).*\.mp4"

        self.variants = {
            "smo4k": [0],
            "insta360 oner": [0]
        }

        self.variant = "smo4k"

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
            gyro_data_input, self.acc = insta360_util.get_insta360_gyro_data(filename, filterArray=[])
        elif self.variant=="insta360 oner":
            gyro_data_input, self.acc = insta360_util.get_insta360_gyro_data(filename, filterArray=[], revertIMU=False)
        else:
            # Assume SMO4K - For no real reason....
            gyro_data_input, self.acc = insta360_util.get_insta360_gyro_data(filename, filterArray=[])

        # Coverting gyro to XYZ to -Z,-X,Y
        self.gyro = np.empty([len(gyro_data_input), 4])
        self.gyro[:,0] = gyro_data_input[:,0][:]
        self.gyro[:,1] = gyro_data_input[:,2][:] * -1
        self.gyro[:,2] = gyro_data_input[:,3][:]
        self.gyro[:,3] = gyro_data_input[:,1][:] * -1

        return True

class GPMFLog(GyrologReader):
    def __init__(self):
        super().__init__("GoPro GPMF metadata")
        self.filename_pattern = "(?i).*\.mp4"

        self.variants =  {
            "hero5": [0], # Zero since this is handled during extraction, not after
            "hero6": [0],
            "hero7": [0],
            "hero8": [0],
            "hero9": [0]
        }

        self.variant = "hero6"

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
        except:
            print("Failed to extract GPMF gyro")
            return False

        hero = int(self.variant.lstrip("hero"))

        # Hero 6??
        if hero == 6:
            self.gyro[:,1] = self.gyro[:,1]
            self.gyro[:,2] = self.gyro[:,2]
            self.gyro[:,3] = self.gyro[:,3]
        if hero == 7:
            self.gyro[:,1] = self.gyro[:,1]
            self.gyro[:,2] = self.gyro[:,2]
            self.gyro[:,3] = self.gyro[:,3]
        elif hero == 5:
            self.gyro[:,1] = -self.gyro[:,1]
            self.gyro[:,2] = self.gyro[:,2]
            self.gyro[:,3] = self.gyro[:,3]
            self.gyro[:,[2, 3]] = self.gyro[:,[3, 2]]

        elif hero == 8:
            # Hero 8??
            self.gyro[:,[2, 3]] = self.gyro[:,[3, 2]]
            self.gyro[:,2] = -self.gyro[:,2]
        elif hero == 9:
            self.gyro[:,1] = -self.gyro[:,1]
            self.gyro[:,2] = self.gyro[:,2]
            self.gyro[:,3] = self.gyro[:,3]
            self.gyro[:,[2, 3]] = self.gyro[:,[3, 2]]

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

            # Get data
            lines = csvfile.readlines()

            data_list = []

            for line in lines:
                splitdata = [float(x) for x in line.split(",")]
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
    """Get an instance of a smoothing algorithm class from name
    """
    if name in log_reader_names:
        return log_reader_classes[log_reader_names.index(name)]()
    else:
        return None

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
                
            return guess, reader.name

    print(f"Couldn't guess log type of {videofile}")
    return False, ""




if __name__ == "__main__":

    test_video_clips = [
        "test_clips/PRO_VID_20210111_144304_00_010.mp4",
        "test_clips/RC_0031_210722220523.MP4",
        "test_clips/GX016015.MP4",
        "test_clips/nivim_insta360.mp4",
        "test_clips/Tiago_Ferreira_5_inch.mp4",
        "test_clips/MasterTim17_caddx.mp4",
        "test_clips/starling2.MOV",
        "test_clips/raw_inav_log.mp4"
    ]
    for clip in test_video_clips:
        guess_log_type_from_video(clip,check_data=False)

    #exit()


    testcases = [[BlackboxCSVData(), "test_clips/btfl_005.bbl.csv"],
                 [BlackboxRawData(), "test_clips/nicecrash_hero7.bbl"],
                 [RuncamData(), "test_clips/Runcam/RC_GyroData0038.csv"],
                 [Insta360Log(), "test_clips/PRO_VID_20210111_144304_00_010.mp4"],
                 [GPMFLog(), "test_clips/GX016015.MP4"],
                 [GyroflowGyroLog(), "test_clips/gyroflow_format_example.gcsv"]]
    
    for reader, path in testcases:

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
