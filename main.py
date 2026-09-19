import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import time
import os


from fit import FitNet
import plot
import execution


with h5py.File("dataset/train_catvsnoncat.h5", 'r') as f:

    x_train = np.transpose(f['train_set_x'][:], (0, 3, 1, 2))
    y_train = f['train_set_y'][:]


with h5py.File("dataset/test_catvsnoncat.h5", 'r') as f:
    
    x_temp = np.transpose(f['test_set_x'][:], (0, 3, 1, 2))
    y_temp = f['test_set_y'][:]
    
    x_test, x_val, y_test, y_val = train_test_split(
        x_temp, y_temp,
        test_size = 0.5,
        stratify = y_temp,
        random_state = 0
    )

print("\n-------------------")

print("train:", 
    int(sum(y_train[y_train == 1])), "/", 
    len(y_train)
)
print("validate:", 
    int(sum(y_val[y_val == 1])), "/", 
    len(y_val)
)

print("test:", 
    int(sum(y_test[y_test == 1])), "/", 
    len(y_test)
)

print("-------------------")

# ---------------------------------------------------------------------------

norm_dim = (0, 2, 3)

mean = x_train.mean(axis = norm_dim, keepdims = True)
std =  x_train.std(axis = norm_dim, keepdims = True)

x_train_norm = (x_train - mean) / std
x_val_norm   = (x_val   - mean) / std
x_test_norm  = (x_test  - mean) / std


# ---------------------------------------------------------------------------

'''
execution.csv(
    "schedule/net_0",
    x_train_norm, y_train,
    x_val_norm, y_val
)

plot.net_variation(
    "result/net_0",
    "result/plot"
)
'''
# ---------------------------------------------
'''
execution.csv(
    "schedule/net_1",
    x_train_norm, y_train,
    x_val_norm, y_val
)

plot.net_variation(
    "result/net_1",
    "result/plot"
)
'''

# ---------------------------------------------
'''
execution.csv(
    "schedule/net_2",
    x_train_norm, y_train,
    x_val_norm, y_val
)
'''
plot.net_variation(
    "result/net_2",
    "result/plot"
)
