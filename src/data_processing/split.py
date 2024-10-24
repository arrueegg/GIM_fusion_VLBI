import random
import numpy as np

stations = np.loadtxt("./src/data_processing/sit_train.list", dtype=str)

random.shuffle(stations)
train = sorted(stations[:int(len(stations) * 0.8)])
val = sorted(stations[int(len(stations) * 0.8):])

np.savetxt("./src/data_processing/sit_train_.list", train, fmt="%s")
np.savetxt("./src/data_processing/sit_val.list", val, fmt="%s")
