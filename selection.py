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
                "expected_min_loss",
                "std_min_loss,"
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
            
            expected_min_loss_var, std_min_loss_var = expected_std_min_loss(
                dir_sub_each_full
            )
            
            min_loss_var = min_loss(dir_sub_each_full)
            
            #--------------------------------------------
            
            print("\n")
            
            print(dir_sub_each_full)
            
            row = {
                "model":                model_var,
                "learn_rate":           learn_rate_var,
                "para_val":             para_val_var,
                "para_trainable":       para_trainable_var,
                "mac_val":              mac_var,
                "expected_min_loss":    expected_min_loss_var,
                "std_min_loss":         std_min_loss_var,
                "min_loss":             min_loss_var,
                "hyperpara":            hyper_var
            }
            
            pd.DataFrame([row]).to_csv(
                df_name,
                mode = "a",
                header = False,
                index = False
            )
            
            
    
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
    
    
def expected_std_min_loss(dir):
    
    min_loss = []
    
    for dir_each_seed in os.listdir(dir):
        
        dir_full = os.path.join(dir, dir_each_seed)
        
        df = pd.read_csv(dir_full)
        
        min_loss.append(df["val_loss"].min())
        
    return np.mean(min_loss), np.std(min_loss)
    
    
def min_loss(dir):
    
    min_loss = 1
    
    for dir_each_seed in os.listdir(dir):
        
        dir_full = os.path.join(dir, dir_each_seed)
        
        df = pd.read_csv(dir_full)
        
        if min_loss > df["val_loss"].min():
            
            min_loss = df["val_loss"].min()
    
    return min_loss