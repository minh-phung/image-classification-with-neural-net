import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import time
from pathlib import Path


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

'''
'''
plot.variation(
    dir = "result/net_0",
    dir_out = "result/plot",
    name = "net_0"
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
        
        dir_name = "result/net_1/0_" + str(each)
        
        net_class.train(
            net_1_0,
            result_dir_name = dir_name + "/seed_" + str(each_seed)
        )
    
    
    plot.variation(
        dir = "result/net_1/0_" + str(each),
        dir_out = "result/plot",
        name = "net_1_0_" + str(each)
    )
    
'''
    

'''
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
        
        dir_name = "result/net_1/1_" + str(each)
        
        net_class.train(
            net_1_1,
            result_dir_name = dir_name + "/seed_" + str(each_seed)
        )
    
    
    plot.variation(
        dir = "result/net_1/1_" + str(each),
        dir_out = "result/plot",
        name = "net_1_1_" + str(each)
    )
    
'''


# ---------------------------------------------------------------------------

num_conv_lay =      [1, 2, 3]
num_conv_kern =     [3, 5]
num_fc_lay =        [1, 2]

'''
for each_num_conv_lay in [num_conv_lay[2]]:
    
    for each_num_conv_kern in num_conv_kern:
        
        for each_num_fc_lay in num_fc_lay:
            
            print("\n--------------------------\n")
            
            print(each_num_conv_lay)
            print(each_num_conv_kern)
            print(each_num_fc_lay)
            
            for each_seed in range(5):
                
                print("\nseed", each_seed)
                
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
                
                name = str(each_num_conv_lay)
                name += "_" + str(each_num_conv_kern)
                name += "_" + str(each_num_fc_lay)
                
                dir_out = "result/net_2/" + name
                
                net_class.train(
                    net_2,
                    epoch_limit = 75,
                    result_dir_name = dir_out + "/seed_" + str(each_seed)
                )
                
            
            plot.variation(
                dir = dir_out,
                dir_out = "result/plot",
                name = "net_2_" + name
            )
            
'''


# ---------------------------------------------------------------------------

num_conv_lay =      [1, 2, 3, 4, 5]
num_conv_kern =     [3, 5]

'''
for each_num_conv_lay in [num_conv_lay[4]]:
    
    for each_num_conv_kern in [num_conv_kern[0]]:
        
        print("\n-------------------")
        
        print("num conv lay", each_num_conv_lay)
        print("num conv kern", each_num_conv_kern)
        
        
        for each_seed in range(5):
        
            print("seed", each_seed)
        
            net_3_dict = {
                "lay_conv_number":  each_num_conv_lay,
                "lay_conv_kernel":  each_num_conv_kern
            }
            
            net_3 = net_class.model(
                "net_3",
                each_seed,
                net_3_dict
            )
            
            name = str(each_num_conv_lay)
            name += "_" + str(each_num_conv_kern)
            
            dir_out = "result/net_3/" + name 
            
            net_class.train(
                net_3,
                epoch_limit = 500,
                result_dir_name = dir_out + "/seed_" + str(each_seed)
            )
            
        
        
        plot.variation(
            dir = dir_out,
            dir_out = "result/plot",
            name = "net_3_" + name
        )

'''

# ---------------------------------------------------------------------------

num_conv_lay =      [1, 2, 3]
num_conv_kern =     [3, 5]
num_pool_lay =      [1, 2, 3]
num_fc_lay =        [1, 2]


'''
for each_num_conv_lay in [num_conv_lay[2]]:
    
    for each_num_conv_kern in [num_conv_kern[1]]:
        
        for each_num_pool_lay in [num_pool_lay[2]]:
            
            for each_num_fc_lay in [num_fc_lay[1]]:
                
                print("\n-------------------\n")
                print("num_conv_lay", each_num_conv_lay)
                print("num_conv_kern", each_num_conv_kern)
                print("num_pool_lay", each_num_pool_lay)
                print("num_fc_lay", each_num_fc_lay)
                print("\n-------------------\n")
                
                for each_seed in range(5):
                    
                    print("\neach seed", each_seed)
                    
                    net_4_dict = {
                        "lay_conv_number":      each_num_conv_lay,
                        "lay_conv_kernel":      each_num_conv_kern,
                        "lay_pool_number":      each_num_pool_lay,
                        "lay_fc_number":        each_num_fc_lay
                    }
                    
                    net_4 = net_class.model(
                        "net_4",
                        each_seed,
                        net_4_dict
                    )
                    
                    name = str(each_num_conv_lay)
                    name += "_" + str(each_num_conv_kern)
                    name += "_" + str(each_num_pool_lay)
                    name += "_" + str(each_num_fc_lay)
                    
                    dir_out = "result/net_4/" + name
                    
                    net_class.train(
                        net_4,
                        epoch_limit = 100,
                        result_dir_name = dir_out + "/seed_" + str(each_seed) 
                    )
                    
                    time.sleep(10)
                    
                
                
                plot.variation(
                    dir = dir_out,
                    dir_out = "result/plot",
                    name = "net_4_" + name
                )
'''

# ---------------------------------------------------------------------------


num_conv_lay =      [1, 2, 3, 4]
num_conv_kern =     [3, 5]
num_pool_lay =      [1, 2, 3]



for each_num_conv_lay in num_conv_lay[2:]:
    
    for each_num_conv_kern in num_conv_kern:
        
        for each_num_pool_lay in num_pool_lay:
            
            print("\n-------------------\n")
            
            print(each_num_conv_lay)
            print(each_num_conv_kern)
            print(each_num_pool_lay)
            
            print("\n-------------------\n")
            
            for each_seed in range(5):
                
                print("seed", each_seed)
                
                net_5_dict = {
                    "lay_conv_number"   :   each_num_conv_lay,
                    "lay_conv_kernel"   :   each_num_conv_kern,
                    "lay_pool_number"   :   each_num_pool_lay
                }
                
                net_5 = net_class.model(
                    "net_5",
                    each_seed,
                    net_5_dict
                )
                
                name = str(each_num_conv_lay)
                name += "_" + str(each_num_conv_kern)
                name += "_" + str(each_num_pool_lay)
                
                dir_out = "result/net_5/" + name
                
                Path(dir_out).mkdir(parents = True, exist_ok = True)
                
		full_dir = dir_out + "/seed_" + str(each_seed)
		full_dir += "_d1" 

                net_class.train(
                    net_5,
                    epoch_limit = 100,
                    result_dir_name = full_dir   
                )



            
            
