import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import ast

from fit import FitNet
import plot
import execution
import selection


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
'''
execution.csv(
    "schedule/net_0",
    x_train_norm, y_train,
    x_val_norm, y_val
)

plot.net_variation(
    "result/net_0",
    "plot"
)
'''
# ---------------------------------------------
'''
execution.csv(
    "schedule/net_1",
    x_train_norm, y_train,
    x_val_norm, y_val
)

plot.net_variation(
    "result/net_1",
    "plot"
)
'''
# ---------------------------------------------
'''
execution.csv(
    "schedule/net_2",
    x_train_norm, y_train,
    x_val_norm, y_val
)

plot.net_variation(
    "result/net_2",
    "plot"
)
'''
# ---------------------------------------------
'''
execution.csv(
    "schedule/net_3",
    x_train_norm, y_train,
    x_val_norm, y_val
)

plot.net_variation(
    "result/net_3",
    "plot"
)
'''
# ---------------------------------------------
'''
execution.csv(
    "schedule/net_4",
    x_train_norm, y_train,
    x_val_norm, y_val
)

plot.net_variation(
    "result/net_4",
    "plot"
)
'''
# ---------------------------------------------
'''
execution.csv(
    "schedule/net_5",
    x_train_norm, y_train,
    x_val_norm, y_val
)

plot.net_variation(
    "result/net_5",
    "plot"
)
'''
# ---------------------------------------------
'''
execution.csv(
    "schedule/net_6",
    x_train_norm, y_train,
    x_val_norm, y_val
)

plot.net_variation(
    "result/net_6",
    "plot"
)
'''
# ---------------------------------------------
'''
execution.csv(
    "schedule/net_7",
    x_train_norm, y_train,
    x_val_norm, y_val
)

plot.net_variation(
    "result/net_7",
    "plot"
)
'''
# ---------------------------------------------
'''
execution.csv(
    "schedule/net_8",
    x_train_norm, y_train,
    x_val_norm, y_val
)

plot.net_variation(
    "result/net_8",
    "plot"
)
'''
# ---------------------------------------------
'''
execution.csv(
    "schedule/net_9",
    x_train_norm, y_train,
    x_val_norm, y_val
)

plot.net_variation(
    "result/net_9",
    "plot"
)
'''
# ---------------------------------------------
'''
execution.csv(
    "schedule/net_10",
    x_train_norm, y_train,
    x_val_norm, y_val
)

plot.net_variation(
    "result/net_10",
    "plot"
)
'''

# ---------------------------------------------------------------------------
'''
selection.collect_variable(
    "result",
    "selection"
)
'''
# ---------------------------------------------------------------------------

df_variable = pd.read_csv("selection/variable.csv")

#print(df_variable.columns)
#print(df_variable)

#--------------------------------

plt.rcParams["figure.figsize"] = (10, 6)

x = ["para_val", "mac_val"]
y = ["expected_min_loss", "min_loss"]


for each_x in x:
    
    for each_y in y:
        
        plt.scatter(
            df_variable[each_x], df_variable[each_y],
            s = 20,
        )
        plt.xscale("log", base = 10)
        
        plt.xlabel(each_x)
        plt.ylabel(each_y)
        
        plt.savefig("selection/log_" + each_x + "_vs_" + each_y + ".png")
        plt.close()

    

#--------------------------------
'''
expected_min_loss_sd = np.std(df_variable["expected_min_loss"])
expected_min_loss_upper_bound = df_variable["expected_min_loss"].min() + expected_min_loss_sd

df_expected_min_loss = df_variable[
    df_variable["expected_min_loss"] < expected_min_loss_upper_bound
]


min_loss_sd = np.std(df_variable["min_loss"])
min_loss_upper_bound = df_variable["min_loss"].min() + min_loss_sd

df_min_loss = df_variable[
    df_variable["min_loss"] < min_loss_upper_bound
]


overlaps_df = pd.merge(
    df_expected_min_loss, df_min_loss,
    how = "inner"
)
'''
#--------------------------------
'''
para_val_df = overlaps_df.sort_values(by = "para_val").iloc[0:10]
mac_val_df = overlaps_df.sort_values(by = "mac_val").iloc[0:10]

overlaps_least_df = pd.merge(
    para_val_df, mac_val_df,
    how = "inner"
)

overlaps_least_df.to_csv("selection/chosen.csv")
'''
# ---------------------------------------------------------------------------

chosen_df = pd.read_csv("selection/chosen.csv")

chosen_model = chosen_df.iloc[4]
chosen_dir = "result/net_5/1_3_2_1__1e-03"

seed_count = []

chosen_min_loss = []
chosen_min_loss_epoch = []


for each_seed in os.listdir(chosen_dir):
    
    seed_chosen_dir = os.path.join(chosen_dir, each_seed)
    df_seed = pd.read_csv(seed_chosen_dir)
    
    seed_count.append(each_seed.split("_")[1].split(".")[0])
    
    idx_min_loss = df_seed["val_loss"].idxmin()
    chosen_min_loss.append(df_seed["val_loss"].iloc[idx_min_loss])
    chosen_min_loss_epoch.append(df_seed["epoch"].iloc[idx_min_loss])
    
    
weight = selection.softmax_by_fraction(chosen_min_loss, 0.3)

df_weight = pd.DataFrame({
    "seed": seed_count,
    "min_loss": chosen_min_loss,
    "epoch": chosen_min_loss_epoch,
    "weight": weight
})

# ---------------------------------------------------------------------------

print(df_weight)

print("")
print(chosen_model)

'''
print("-------------------\nprediction \n-------------------")

net_class_predict = FitNet(
    x_train_norm, y_train,
    x_val_norm, y_val,
    x_pred = x_test_norm
)


for row in df_weight.itertuples():
    
    print("\nseed", row.seed)
    
    net = net_class_predict.model(
        model = "net_" + str(chosen_model["model"]),
        seed = float(row.seed),
        args = ast.literal_eval(chosen_model["hyperpara"]),
        learn_rate = float(chosen_model["learn_rate"])
    )
    
    prediction = net_class_predict.train(
        net,
        epoch_limit = row.epoch,
        result_dir_name = "selection/prediction/" + str(row.seed) + ".csv"
    )
    
    np.save(
        "selection/prediction/" + str(row.seed) + "_pred.npy",
        prediction.detach().cpu().numpy()
    )
    
'''
# ---------------------------------------------------------------------------

full_prediction = 0
weight_sum = 0

for row in df_weight.itertuples():
    
    prediction = np.load(
        "selection/prediction/" + str(row.seed) + "_pred.npy"
    )
    
    full_prediction += prediction * row.weight

label_prediction = np.where(full_prediction >= 0.5, 1, 0)

# ---------------------------------------------------------------------------

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

for i, each_y in enumerate(y_test):
    
    label = label_prediction[i]
    
    if each_y == 1 and label == 1:
        true_positive += 1
        
    elif each_y == 0 and label == 0:
        true_negative += 1
        
    elif each_y == 0 and label == 1:
        false_positive += 1
        
    else: 
        false_negative += 1
        
print("")

print("true_positive", true_positive)
print("true_negative", true_negative)
print("false_positive", false_positive)
print("false_negative", false_negative)

accuracy = (true_positive + true_negative)
accuracy /= np.sum([true_positive, true_negative, false_positive, false_negative])

true_positive_rate = true_positive / (true_positive + false_positive)
true_negative_rate = true_negative / (true_negative + false_negative)

print("")
print("accuracy", accuracy)
print("true_positve_rate", true_positive_rate)
print("true_negative_rate", true_negative_rate)
