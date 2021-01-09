import json
from gyro_integrator import *
import GPMF_gyro

from blackbox_extract import BlackboxExtractor


extrac = GPMF_gyro.Extractor("test_clips/GX016017.MP4")
realGyroData = extrac.get_gyro(True)


#bb = BlackboxExtractor("test_clips/GX015563.MP4_emuf_004.bbl")
#realGyroData = bb.get_gyro_data(cam_angle_degrees=2)


print(realGyroData.shape)

from matplotlib import pyplot as plt

FPS = 29.97 # framerate of video to stabilize
SAMPLE_RATE = 400 # sample per second

dat = list(realGyroData[:,2])
times = list(FPS * realGyroData[:,0])

from scipy import fftpack

# sampling rate
f_s = 400

X = fftpack.fft(dat)
freqs = fftpack.fftfreq(len(dat)) * f_s

#plt.plot(freqs, X)
#plt.plot(times,dat)
#plt.show()

integrator = GyroIntegrator(realGyroData)

plt.plot(integrator.get_raw_data("t") * FPS,integrator.get_raw_data("y"))
plt.show()

time_list, orientation_list = integrator.integrate_all()

time_list, orientation_list = integrator.get_orientations()

output_data = np.column_stack((time_list, orientation_list))

CSV_header = "\t".join(["Time","q0","q1","q2","q3"])

orientation_list = orientation_list[::10,:]

#print(output_data)

# save orientation data as CSV file
#np.savetxt('hero5_orientation.csv',output_data ,delimiter='\t',header=CSV_header,comments='')

#print(realGyroData)


# Visualize motion during testing
# Adapted from https://github.com/jerabaul29/IntegrateGyroData

import sys
import pygame
from operator import itemgetter
import Quaternions_temp as qt


class Point3D:
    """A class used for describing a point in 3D."""

    def __init__(self, x=0, y=0, z=0):
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.v = qt.Vector(self.x, self.y, self.z)

    def project(self, win_width, win_height, fov, viewer_distance):
        """ Transforms this 3D point to 2D using a perspective projection. """
        factor = fov / (viewer_distance + self.z)
        x = self.x * factor + win_width / 2
        y = -self.y * factor + win_height / 2
        return Point3D(x, y, self.z)

    def rotateQ(self, q):
        """Apply rotation described by quaternion q to this 3D point"""
        v_rotated = qt.apply_rotation_on_vector(q, self.v)
        return Point3D(v_rotated.vx, v_rotated.vy, v_rotated.vz)


class RenderGyroIntegration:
    """A class for rendering gyro integration as a 3D cube display."""

    def __init__(self, win_width=640, win_height=480):

        pygame.init()

        self.screen = pygame.display.set_mode((win_width, win_height))
        pygame.display.set_caption("Rendering of 3D cube")

        self.clock = pygame.time.Clock()

        self.vertices = [
            Point3D(-1, 1, -1),
            Point3D(1, 1, -1),
            Point3D(1, -1, -1),
            Point3D(-1, -1, -1),
            Point3D(-1, 1, 1),
            Point3D(1, 1, 1),
            Point3D(1, -1, 1),
            Point3D(-1, -1, 1)
        ]

        # Define the vertices that compose each of the 6 faces.
        self.faces = [(0, 1, 2, 3), (1, 5, 6, 2), (5, 4, 7, 6),
                      (4, 0, 3, 7), (0, 4, 5, 1), (3, 2, 6, 7)]

        # Define colors for each face
        self.colors = [(255, 0, 255), (255, 0, 0), (0, 255, 0),
                       (0, 0, 255), (0, 255, 255), (255, 255, 0)]

        self.angle = 0

    def run(self):
        """ Main Loop: run until window gets closed."""

        iteration = 0
        while 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.clock.tick(54)
            self.screen.fill((0, 32, 0))

            # It will hold transformed vertices.
            t = []

            # perform one gyro integration: read, update quaternion
            itemas = orientation_list[iteration]
            q = qt.Quaternion(itemas[0],itemas[1],itemas[2],itemas[3])
            iteration += 1

            for v in self.vertices:
                # rotate point according to integrated gyro
                r = v.rotateQ(q)
                # Transform the point from 3D to 2D
                p = r.project(self.screen.get_width(), self.screen.get_height(), 256, 4)
                # Put the point in the list of transformed vertices
                t.append(p)

            # Calculate the average Z values of each face.
            avg_z = []
            i = 0
            for f in self.faces:
                z = (t[f[0]].z + t[f[1]].z + t[f[2]].z + t[f[3]].z) / 4.0
                avg_z.append([i, z])
                i = i + 1

            # Draw the faces using the Painter's algorithm:
            # Distant faces are drawn before the closer ones.
            for tmp in sorted(avg_z, key=itemgetter(1), reverse=True):
                face_index = tmp[0]
                f = self.faces[face_index]
                pointlist = [(t[f[0]].x, t[f[0]].y), (t[f[1]].x, t[f[1]].y),
                             (t[f[1]].x, t[f[1]].y), (t[f[2]].x, t[f[2]].y),
                             (t[f[2]].x, t[f[2]].y), (t[f[3]].x, t[f[3]].y),
                             (t[f[3]].x, t[f[3]].y), (t[f[0]].x, t[f[0]].y)]
                pygame.draw.polygon(self.screen, self.colors[face_index], pointlist)

            self.angle += 1

            pygame.display.flip()


test = RenderGyroIntegration()
test.run()