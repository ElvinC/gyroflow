import numpy as np


a = np.random.random(100)

average = np.mean(a)

def alt_avg(lst):
	final_value = lst[0]
	num = len(lst)

	for val in lst:
		final_value += val/num

	return final_value 


print(average)
print(alt_avg(a))