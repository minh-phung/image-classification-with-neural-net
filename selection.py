import pandas as pd
import numpy as np
import torch
import os

from fit import FitNet


def collect_variable(dir_result, dir_csv):
    
    print("\n\ncollect variable\n\n")
    
    net_class = FitNet(
        np.array([]), np.array([]),
        np.array([]), np.array([])
    )
    
    count = 0
    
    for dir_sub in os.listdir(dir_result):
        
        print("\n\n-------")
        
        model_label = dir_sub[4:]
        print("net_" + str(model_label))
        
        print("-------")
        
        dir_sub_full = os.path.join(dir_result, dir_sub)
        
        for i, dir_sub_each in enumerate(os.listdir(dir_sub_full)):
        
            var = dir_sub_each.split("__")
            
            if len(var[0].split("_")) > 1:
                
                para = []
                
                for each in var[0].split("_"):
                    
                    try:
                        para.append(int(each))
                        
                    except:
                        para.append(each)
            
            else:
                para = []
            
            
            learn_rate = var[1]
            
            print(para)
            print(learn_rate)
            
            #--------------------------------------------
            
            '''
            net = net_class.model(
                "net_" + str(model_label),
                0,
                para,
                learn_rate = 0.1
            )
            '''
            
            #--------------------------------------------
            
            dir_sub_each_full = os.path.join(dir_sub_full, dir_sub_each)
            
            expected_val_loss(dir_sub_each_full)
            
            
            
            #--------------------------------------------
            
            
            print("--\n")
            
            count+= 1
            
            
            
    print(count)
    

def expected_val_loss(dir):
    
    print(dir)
    
    for dir_each_seed in os.listdir(dir):
        
        print(dir_each_seed)
    
    return
    
def min_val_loss(dir):
    
    return