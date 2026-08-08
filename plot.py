import matplotlib.pyplot as plt
import os
import pandas as pd
from functools import reduce
import numpy as np

plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.figsize'] = (10, 7)

colors = plt.get_cmap('tab10').colors



def variation(dir, dir_out, name):
    
    for root, dirs, files in os.walk(dir):
        
        for j, file in enumerate(files):
    
            full_path = os.path.join(root, file)
            
            df = pd.read_csv(full_path)[["epoch","val_loss"]]
            
            plt.plot(
                np.log(df["epoch"].values),
                df["val_loss"].values,
                color = colors[j],
                label = file[0:6]
            )
    
    plt.ylim(0, 1.5)
    
    plt.xlabel("log(epoch)")
    plt.ylabel("val loss")
    
    plt.legend()
    
    plt.savefig(dir_out + "/" + name + ".png")
    
    plt.close()
    
    
    return


def compute(array_df, variable):
    
    
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
    

