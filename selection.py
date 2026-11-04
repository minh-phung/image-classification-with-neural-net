import pandas as pd
import numpy as np
import torch
import os
from thop import profile

from fit import FitNet


def collect_variable(dir_result, dir_csv):
    
    print("\n\ncollect variable\n\n")
    
    net_class = FitNet(
        np.array([]), np.array([]),
        np.array([]), np.array([])
    )
    
    df_name = "selection/variable.csv"
    
    if not os.path.exists(df_name):
        
        df = pd.DataFrame(
            columns = [
                "model",
                "learn_rate",
                "para_val",
                "para_trainable",
                "mac_val",
                "expected_loss",
                "min_loss",
                "hyperpara"
            ]
        )
        
        df.to_csv(df_name, index = False)
        
    
    for dir_sub in os.listdir(dir_result):
        
        print("\n\n-------")
        model_var = dir_sub[4:]
        
        print("net_" + str(model_var))
        print("-------")
        
        
        dir_sub_full = os.path.join(dir_result, dir_sub)
        
        
        for i, dir_sub_each in enumerate(os.listdir(dir_sub_full)):
        
            var = dir_sub_each.split("__")
            
            if len(var[0].split("_")) > 1:
                
                hyper_var = []
                
                for each in var[0].split("_"):
                    
                    try:
                        hyper_var.append(int(each))
                        
                    except:
                        hyper_var.append(each)
            
            else:
                hyper_var = []
            
            
            learn_rate_var = var[1]
            
            #--------------------------------------------
            
            net = net_class.model(
                "net_" + str(model_var),
                0,
                hyper_var,
                learn_rate = 0.1
            )
            
            para_val_var = sum(
                p.numel() for p in net.parameters()
            )
            
            para_trainable = sum(
                p.numel() for p in net.parameters() if p.requires_grad
            )
            
            para_trainable_var = para_val_var == para_trainable
            
            mac_var = model_mac_val(net)
            
            #--------------------------------------------
            
            dir_sub_each_full = os.path.join(dir_sub_full, dir_sub_each)
            
            print(dir_sub_each_full)
            
            #--------------------------------------------
            
            
            
            #--------------------------------------------
            
            print("\n")
            
            print("model", model_var)
            print("learn_rate", learn_rate_var)
            print("para_val", para_val_var)
            print("para_trainable", para_trainable_var)
            print("mac_val", mac_var)
            print("expected_loss")
            print("min_loss")
            print("hyperpara", hyper_var)
            
            print("--\n")
            
            #quit()
            
            
    
def model_mac_val(
    torch_class, input_shape = (1, 3, 64, 64)
):
    
    input_zeros = torch.zeros(*input_shape)
    
    macs, params = profile(
        torch_class,
        inputs = (input_zeros,),
        verbose = False
    )
    
    return macs