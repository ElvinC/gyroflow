import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 100, 100 * 200)

v1 = 10
v2 = 90
d1 = 0.2
d2 = 6

err_slope = (d2-d1)/(v2-v1)
correction_slope = err_slope + 1
gyro_start = (d1 - err_slope*v1)

#interval = 1/(correction_slope * self.fps)

g1 = v1 - d1
g2 = v2 - d2
slope = (v2 - v1) / (g2 - g1)
corrected_times = slope * (t - g1) + v1


fit = np.polyfit([v1, v2], [d1, d2], 1, full=True)
coefs = fit[0]



corrected_times_2 = (t + coefs[1])/(1- coefs[0])

plt.plot(corrected_times)
plt.plot(corrected_times_2)

print(np.allclose(corrected_times_2, corrected_times))

plt.show()




exit()

vidlength = 120 # 2 minutes

end_delay = 5 # seconds

inter_delay = 20

min_slices = 2

max_slices = 6

bad_sync_chance = 0.3

del_offset = 1
del_drift = 0.01

max_error = 0.05 # 1/20 of a second

def get_delay(t):
    nbad_sync = np.random.random() > bad_sync_chance
    print(nbad_sync)
    error = 0 if nbad_sync else (np.random.random() - 0.5) * 3
    error += 0.005 * (np.random.random() - 0.5)
    return del_drift * t + del_offset + error

times = []
if vidlength < (inter_delay + 2 * end_delay):
    times = np.array([end_delay, vidlength - end_delay])

else:
    num_syncs = 2 # round((vidlength - 2 * end_delay) / inter_delay)

    times = np.linspace(end_delay, vidlength - end_delay, num_syncs)

delays = np.array([get_delay(t) for t in times])

plt.scatter(times, delays)


chosen_indices = {}
num_chosen = 0
rsquared_best = 1000
chosen_coefs = None

for i in range(num_syncs):
    for j in range(i, num_syncs):
        if i != j:
            
            del_i = delays[i]
            del_j = delays[j]

            t_i = times[i]
            t_j = times[j]

            slope = (del_j - del_i) / (t_j - t_i)
            intersect = del_i - t_i * slope

            within_error = []
            est_curve = times * slope + intersect
            within_error = np.where(np.abs(est_curve - delays) < max_error)[0]

            if within_error.shape[0] >= num_chosen and set(within_error) != chosen_indices:
                #print(times[within_error])
                fit = np.polyfit(times[within_error], delays[within_error], 1, full=True)
                coefs = fit[0]

                if within_error.shape[0] > 2:
                    rsquared = fit[1]

                    if rsquared < rsquared_best:
                        rsquared_best = rsquared
                        chosen_coefs = coefs
                        num_chosen = within_error.shape[0]
                        chosen_indices = set(within_error)
                else:
                    chosen_coefs = coefs
                    num_chosen = within_error.shape[0]
                    chosen_indices = set(within_error)

                
                #print(rsquared)



                # Linear fit and rms error

                

            
            #for i in range(num_syncs):





            #plt.plot([t_i, t_j], [del_i, del_j])
            #plt.plot(times, est_curve)



est_curve = times * chosen_coefs[0] + chosen_coefs[1]
plt.plot(times, est_curve)
print(chosen_coefs)
print(chosen_indices)
print(rsquared_best)
plt.show()
