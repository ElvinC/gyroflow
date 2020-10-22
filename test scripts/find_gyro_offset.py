from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

def cost_func(R,measured,expected):
	cam_rotation = Rotation.from_euler("xyz", R, degrees=True)
	
	diff = cam_rotation.apply(measured) - expected
	# squared error
	cost = np.sum(diff**2)
	return cost


N = 100

# a bunch of random vectors representing perfectly aligned gyro data
gyro_perfect = np.cumsum(np.random.random((N, 3)),0)

cam_rotation = Rotation.from_euler("xyz", [10,-123,0], degrees=True)

# "Rotated" vector + some noise
gyro_rotated = cam_rotation.apply(gyro_perfect) + np.random.normal(0,0.04,(N,3))

shift_samples = 3

#gyro_rotated[20,2] = -30

guess_R = [0, 0, 0] # euler angle initial guess

bounds = [(-180,180)]*3

res = minimize(cost_func, guess_R, args=(gyro_rotated, gyro_perfect),
							  method='nelder-mead',options={'maxiter':5000,'xatol': 1e-8, 'disp': True}) # , bounds=bounds


print(res)
print(cost_func([0, 0, 0], gyro_rotated, gyro_perfect))