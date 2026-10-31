import pandas as pd
import numpy as np
import torch
import os


def collect_variable(dir_result, dir_csv):
    
    print("\n\ncollect variable\n\n")
    
    
    for dir_sub in os.listdir(dir_result):
        
        print("\n\n-------")
        
        model_label = dir_sub[4:]
        print(model_label)
        
        print("-------")
        
        dir_sub_full = os.path.join(dir_result, dir_sub)
        
        for i, dir_sub_each in enumerate(os.listdir(dir_sub_full)):
        
            print(dir_sub_each)
            
            var = dir_sub_each.split("__")
            
            para = var[0].split("_")
            
            print(para)
            
            
            
            