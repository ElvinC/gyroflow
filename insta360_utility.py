from orangebox import Parser
from scipy.spatial.transform import Rotation
import math

from scipy import signal
import numpy as np
import struct

def get_insta360_gyro_data(path, filterArray=[[1, 0.35], [1,0.04]], revertIMU=True):
    fin = open(str(path), "rb")
    gyro_data, exposure_timestamps, acc_data = _extract_metadata(fin)
    gyro_data, acc_data = _truncate_data(gyro_data, exposure_timestamps, acc_data)
    gyro_data = _set_time_from_zero(gyro_data)
    gyro_data = _filtering(gyro_data, filterArray)
    if revertIMU is not True:
        gyro_data[:,1] = gyro_data[:,1]*-1
        gyro_data[:,3] = gyro_data[:,3]*-1
        acc_data[:,1] = acc_data[:,1]*-1
        acc_data[:,3] = acc_data[:,3]*-1

    fin.close()
    return gyro_data, acc_data


# Extract insta360 accelerometer data.
# read from the end to get each insta360 record size.
def _extract_metadata(fin):
    dlen = 56
    time_at_arm = None
    final_gyro_data = []
    # then trailer length: 
    fin.seek(-(32+42+4),2)
    buf = fin.read(38+4)
    trailer_len = struct.unpack('<38xL', buf)[0]

    offset = -78
    while offset > -trailer_len:
        # Iterates over each insta360 trailer records from the end.
        # Record ID and size are written after the actual data.
        final_gyro_data = []
        fin.seek(offset,2)
        buf = fin.read(2+4)
        id, size = struct.unpack('<HL', buf)
        or_size = size
        hid = hex(id)
        fin.seek(offset-size,2)
        if hid == '0x300': # Acelerometer data.
            d_times, gyro_data, acc_data = __parseAccRecord(fin, size)
        if hid == '0x400': # exposure data
            expo_timestamps = __parseExpRecord(fin, size)
        offset = offset - size - 4 - 2
    return gyro_data, expo_timestamps , acc_data

def _truncate_data(gyro_data, exposure_timestamps, acc_data):
    # Removes the gyro data based on the exprosure timestamps, also sorts the time to be from zero.
    expo_time = exposure_timestamps[1]
    for i in range(0, len(gyro_data[:,0]), 1):
        if (gyro_data[:,0][i]-expo_time)>0:
            gyro_sync_item = i+400
            break
    gyro_data = np.delete(gyro_data, range(0, gyro_sync_item, 1), axis=0)
    acc_data = np.delete(acc_data, range(0, gyro_sync_item, 1), axis=0)
    return gyro_data, acc_data

def _set_time_from_zero(gyro_data):
    start_time = gyro_data[:,0][0]
    for j in range(0, len(gyro_data), 1):
        gyro_data[:,0][j] = float((gyro_data[:,0][j] - start_time))
    return gyro_data

def _filtering(gyro_data, filterArray):
    if not(isinstance(filterArray, list)) or len(filterArray)==0:
        return gyro_data
    for f in filterArray:
        orderOfFilter = f[0]
        criticalFrequency = f[1]
        gyro_data[:, 1:4] = np.apply_along_axis(__arrayLowPassFilter, 0, gyro_data[:,1:4], orderOfFilter, criticalFrequency)
    return gyro_data

def __parseExpRecord(fin, size):  
    expo = []
    dlen = 16
    for i in range(0, size, dlen):
        buf = fin.read(dlen)
        timecode, exp = struct.unpack("<Qd", buf)
        tm = float(timecode/1000)
        expo.append((tm))
    return expo

def __parseAccRecord(fin, size):  
    dlen = 56
    d_times = []
    d_gyros = []
    d_acc = []
    for i in range(0, size, dlen):
        buf = fin.read(dlen)
        # normally it's  roll > pitch > yaw, insta360 is pitch > yaw > roll
        # timecode, _, _, _, gyroPitch, gyroYaw, gyroRoll = struct.unpack("<Q6d", buf)
        timecode, accPitch, accYaw, accRoll, gyroPitch, gyroYaw, gyroRoll = struct.unpack("<Q6d", buf)
        tm = float(timecode/1000)
        d_times.append(tm)
        d_gyros.append((tm, -gyroRoll, gyroPitch, -gyroYaw))
        d_acc.append((tm, -accRoll, accPitch, -accYaw))
    return np.array(d_times), np.array(d_gyros), np.array(d_acc)

def __arrayLowPassFilter(array, orderOfFilter, criticalFrequency):
    b, a = signal.butter(orderOfFilter, criticalFrequency)
    zi = signal.lfilter_zi(b, a)
    # Filter once
    z, _ = signal.lfilter(b, a, array, zi=zi*array[0])
    # Filter again
    z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
    return signal.filtfilt(b, a, array)

def isInsta360Video(video_path):
    read_len=32
    magicString=b'8db42d694ccc418790edff439fe026bf'
    path = video_path
    fin = open(str(path), "rb")
    fin.seek(-read_len, 2)
    buf = fin.read(read_len)
    if (buf==magicString):
        return True
    return False

# #testing
# if __name__ == "__main__":
#     bbe = BlackboxExtractor("test_clips/GX015563.MP4_emuf_004.bbl") # btfl_all.bbl
#     gyro_data = bbe.get_gyro_data()
#     print(gyro_data)
#     print(bbe.n_of_logs)
