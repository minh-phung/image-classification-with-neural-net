import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import time

from fit import FitNet


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

# -------------------------------------------------------------------

norm_dim = (0, 2, 3)

mean = x_train.mean(axis = norm_dim, keepdims = True)
std =  x_train.std(axis = norm_dim, keepdims = True)

x_train_norm = (x_train - mean) / std
x_val_norm   = (x_val   - mean) / std
x_test_norm  = (x_test  - mean) / std


# -------------------------------------------------------------------

net_class = FitNet(
    x_train_norm, y_train,
    x_val_norm, y_val
)

# -------------------------------------------------------------------

net_0_dict = {
    "lay_1_sampler_weight":     "xavier_uniform",
    "lay_1_sampler_bias":       "zeros"
}
'''
for each_seed in range(10):

    net_0 = net_class.model("net_0", each_seed, net_0_dict)

    net_class.train(
        net_0, result_dir_name = "result/net_0/seed_" + str(each_seed)
    )
'''

# -------------------------------------------------------------------

net_1_1_dict = {
    "lay_1_sampler_weight":     "xavier_uniform",
    "lay_1_sampler_bias":       "zeros",
    "lay_1_activation":         "identity",
    "lay_2_sampler_weight":     "xavier_uniform",
    "lay_2_sampler_bias":       "zeros"
}
'''
for each_seed in range(10):
    
    print("seed", each_seed)
    
    net_1_1 = net_class.model("net_1", each_seed, net_1_1_dict)
    
    net_class.train(
        net_1_1, 
        result_dir_name = "result/net_1/var_1/seed_" + str(each_seed)
    )
'''    
#-----------------------------------------------
#-----------------------------------------------

net_1_2_1_dict = {
    "lay_1_sampler_weight":     "xavier_uniform",
    "lay_1_sampler_bias":       "zeros",
    "lay_1_activation":         "sigmoid",
    "lay_2_sampler_weight":     "xavier_uniform",
    "lay_2_sampler_bias":       "zeros"
}
'''
for each_seed in range(8,10):
    
    print("seed", each_seed)
    
    net_1_2_1 = net_class.model("net_1", each_seed, net_1_2_1_dict)
    
    net_class.train(
        net_1_2_1,
        result_dir_name = "result/net_1/var_2_1/seed_" + str(each_seed)
    )
'''    
#-----------------------------------------------

net_1_2_2_dict = {
    "lay_1_sampler_weight":     "xavier_normal",
    "lay_1_sampler_bias":       "zeros",
    "lay_1_activation":         "sigmoid",
    "lay_2_sampler_weight":     "xavier_normal",
    "lay_2_sampler_bias":       "zeros"
}
'''
for each_seed in range(10):
    
    print("seed", each_seed)
    
    net_1_2_2 = net_class.model("net_1", each_seed, net_1_2_2_dict)
    
    net_class.train(
        net_1_2_2,
        result_dir_name = "result/net_1/var_2_2/seed_" + str(each_seed)
    ) 
'''
#-----------------------------------------------
#-----------------------------------------------

net_1_3_1_dict = {
    "lay_1_sampler_weight":     "kaiming_uniform",
    "lay_1_sampler_bias":       "constant",
    "lay_1_activation":         "relu",
    "lay_2_sampler_weight":     "kaiming_uniform",
    "lay_2_sampler_bias":       "constant"
}
'''
for each_seed in range(10):
    
    print("seed", each_seed)
    
    net_1_3_1 = net_class.model("net_1", each_seed, net_1_3_1_dict)
    
    net_class.train(
        net_1_3_1,
        result_dir_name = "result/net_1/var_3_1/seed_" + str(each_seed)
    )
'''
#-----------------------------------------------

net_1_3_2_dict = {
    "lay_1_sampler_weight":     "kaiming_normal",
    "lay_1_sampler_bias":       "constant",
    "lay_1_activation":         "relu",
    "lay_2_sampler_weight":     "kaiming_normal",
    "lay_2_sampler_bias":       "constant"
}

for each_seed in range(5,10):
    
    print("seed", each_seed)
    
    net_1_3_2 = net_class.model("net_1", each_seed, net_1_3_2_dict)
    
    net_class.train(
        net_1_3_2,
        result_dir_name = "result/net_1/var_3_2/seed_" + str(each_seed)
    )
