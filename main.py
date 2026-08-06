import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import time

from fit import FitNet
import plot


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

net_class = FitNet(
    x_train_norm, y_train,
    x_val_norm, y_val
)

# ---------------------------------------------------------------------------
'''
for each_seed in range(10):
    
    net_0 = net_class.model(
        "net_0", 
        each_seed,
        {}
    )
    
    net_class.train(
        net_0,
        result_dir_name = "result/net_0/seed_" + str(each_seed)
    )


plot.model_variation(
    ["result/net_0"],
    dir_out = "result/plot/net_0"
)
'''
# ---------------------------------------------------------------------------
hidden_layer = [1, 2, 3]

'''
for each in hidden_layer:

    net_1_0_dict = {
        "lay_hidden_number":        each,
        "lay_sampler_weight":       "xavier_uniform",
        "lay_sampler_bias":         "zeros",
        "lay_activation":           "sigmoid"
    }

    for each_seed in range(10):
        
        print("\nseed", each_seed)
        
        net_1_0 = net_class.model(
            "net_1",
            each_seed,
            net_1_0_dict
        )
        
        dir_name = "result/net_1/var_0_hid_" + str(each)
        
        net_class.train(
            net_1_0,
            result_dir_name = dir_name + "/seed_" + str(each_seed)
        )


for each in hidden_layer:
    
    net_1_1_dict = {
        "lay_hidden_number":        each,
        "lay_sampler_weight":       "kaiming_uniform",
        "lay_sampler_bias":         "constant",
        "lay_activation":           "relu"
    }
    
    for each_seed in range(10):
        
        print("\nseed", each_seed)
        
        net_1_1 = net_class.model(
            "net_1",
            each_seed,
            net_1_1_dict
        )
        
        dir_name = "result/net_1/var_1_hid_" + str(each)
        
        net_class.train(
            net_1_1,
            result_dir_name = dir_name + "/seed_" + str(each_seed)
        )


plot.model_variation(
    [
        "result/net_1/var_0_hid_1",
        "result/net_1/var_0_hid_2",
        "result/net_1/var_0_hid_3",
        "result/net_1/var_1_hid_1",
        "result/net_1/var_1_hid_2",
        "result/net_1/var_1_hid_3",
    ],
    dir_out = "result/plot/net_1"
)
'''

# ---------------------------------------------------------------------------

num_conv_lay =      [1, 2, 3]
num_conv_kern =     [3, 5]
num_fc_lay =        [1, 2]


for each_num_conv_lay in [num_conv_lay[0]]:
    
    for each_num_conv_kern in num_conv_kern:
        
        for each_num_fc_lay in num_fc_lay:
            
            print("\n--------------------------\n")
            
            print(each_num_conv_lay)
            print(each_num_conv_kern)
            print(each_num_fc_lay)
            
            for each_seed in range(10):
                
                print("seed", each_seed)
                
                net_2_dict = {
                    "lay_conv_number":      each_num_conv_lay,
                    "lay_conv_kernel":      each_num_conv_kern,
                    "lay_fc_number":        each_num_fc_lay
                }
                
                net_2 = net_class.model(
                    "net_2",
                    each_seed,
                    net_2_dict
                )
                
                dir_out = "result/net_2/var_" + str(each_num_conv_lay)
                dir_out += "_" + str(each_num_conv_kern)
                dir_out += "_" + str(each_num_fc_lay)
                
                net_class.train(
                    net_2,
                    epoch_limit = 25,
                    result_dir_name = dir_out + "/seed_" + str(each_seed)
                )
                


# ---------------------------------------------------------------------------
'''

net_3_dict = {
    "lay_conv_number":  2,
    "lay_conv_kernel":  3
}

net_3 = net_class.model(
    "net_3",
    0,
    net_3_dict
)


net_class.train(
    net_3,
    epoch_limit = 50,
    result_dir_name = "result/net_3/seed_" + 0
)

'''