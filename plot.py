import matplotlib.pyplot as plt
import os
import pandas as pd
from functools import reduce


color_plot = [
    'b', 'g', 'r', 'c', 'm', 'y', 'k'
]

def model_variation(dir_list, dir_out):
    
    for each_variable in ["mean", "std"]:
        
        
        for i, each_dir in enumerate(dir_list):
            
            for root, dirs, files in os.walk(each_dir):
                
                all_dfs = [None] * len(files)
                
                for j, file in enumerate(files):
                    
                    full_path = os.path.join(root, file)
                    all_dfs[j] = pd.read_csv(full_path)[["epoch","val_loss"]]
                    
                variable = mean_std(all_dfs, each_variable)
                
                plt.plot(
                    variable["epoch"], 
                    variable[each_variable],
                    label = os.path.basename(root),
                    color = color_plot[i]
                )
            
        plt.xlabel("epoch")
        plt.ylabel("validation loss - " + each_variable)
        plt.legend()
        
        #plt.show()
        plt.savefig(dir_out + "_" + each_variable +".png")
        plt.close()
    
    return


def mean_std(array_df, variable):
    
    variable_dict = {
        "mean": pd.DataFrame.mean,
        "std" : pd.DataFrame.std
    }
    
    dfs = [
        array_df.rename(columns = {"val_loss": f"val_loss_{i}"})
        for i, array_df in enumerate(array_df)
    ]
    
    result = reduce(
        lambda left, right: pd.merge(
            left, right,
            on = "epoch",
            how = "outer"
        ),
        dfs
    )
    
    out = variable_dict[variable](
        result.filter(like = "val_loss"),
        axis = 1
    )
    
    result[variable] = out
    
    return result[["epoch", variable]]