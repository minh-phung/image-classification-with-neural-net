import matplotlib.pyplot as plt
import os
import pandas as pd
from functools import reduce
import numpy as np

plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.figsize'] = (10, 7)

colors = plt.get_cmap('tab10').colors


def net_variation(dir_net, dir_out):
    
    print("\nplot - net variation")
    
    for sub_dir in os.listdir(dir_net):
        
        print("--------")
        
        variation(
            os.path.join(dir_net, sub_dir),
            os.path.basename(dir_net) + "__" + str(sub_dir),
            dir_out
        )
    
    
def variation(dir, name, dir_out):
    
    print("dir_name", name)
    
    for i, sub_dir in enumerate(os.listdir(dir)):
        
        print(sub_dir)
        
        full_path = os.path.join(dir, sub_dir)
    
        df = pd.read_csv(full_path)[["epoch","val_loss"]]
            
        plt.plot(
            np.log(df["epoch"].values),
            df["val_loss"].values,
            color = colors[i],
            label = str(sub_dir)[:-4]
        )

    plt.ylim(0, 1.5)
    
    plt.xlabel("log(epoch)")
    plt.ylabel("val loss")
    
    plt.legend()
    
    plt.savefig(dir_out + "/" + name + ".png")
    
    plt.close()
    


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
    

