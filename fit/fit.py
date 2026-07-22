import torch
from collections import deque
import pandas as pd
from pathlib import Path
import numpy as np

from fit.model import Net0

from fit.feature import LOSS, SAMPLER, ACTIVATION
from fit.feature import first_der as first_der_stop


model_dict = {
    "net_0": Net0
}

def main(
    x_train, y_train,
    x_val, y_val,
    model,
    sampler,
    loss = "BCE_with_logit",
    seed = 0,
    n_stop = 10,
    epoch_limit = 50
):
    
    torch.manual_seed(0)
    
    #---------------------------------------------------

    train_data = torch.utils.data.TensorDataset(
        torch.from_numpy(y_train).float(), 
        torch.from_numpy(x_train).float()
    )
    
    train_size = len(train_data)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size = 4,
        shuffle = False
    )
    
    #---------------------------------------------------
    
    val_data = torch.utils.data.TensorDataset(
        torch.from_numpy(y_val).float(),
        torch.from_numpy(x_val).float()
    )
    
    val_size = len(val_data)
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size = 4,
        shuffle = False
    )   
    
    #---------------------------------------------------
    
    net = model_dict[model]()
    
    print("loss function", loss)
    loss_func = LOSS[loss]
    
    print("sampler", sampler)
    net.apply(SAMPLER[sampler])
    
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr = 10e-5
    )
    
    result_label = ["epoch", "train_loss", "val_loss"]
    result = []
    
    
    epoch_count = 0
    val_loss_queue = deque(maxlen = n_stop)

    while(
        epoch_count < epoch_limit
        and not first_der_stop(val_loss_queue, n_stop)
    ):
        train_loss = 0.0
        val_loss = 0.0
        
        # --------------------------------------------
        
        for i, (y_train, x_train) in enumerate(train_loader):
            
            optimizer.zero_grad()
            
            f_x = net(x_train)
            
            loss = loss_func(f_x, y_train)
            
            loss.backward()
            
            train_loss += loss.item()
            
            optimizer.step()
        
        print("total train_loss", train_loss)
        
        # --------------------------------------------
        
        for i, (y_val, x_val) in enumerate(val_loader):
            
            f_x = net(x_val)
            
            loss = loss_func(f_x, y_val)
            
            val_loss += loss.item()
            
        #print("total val_loss", val_loss)
        
        # --------------------------------------------
        val_loss_queue.append(val_loss / val_size)
        
        result.append([
            epoch_count, 
            train_loss / train_size, 
            val_loss / val_size
        ])
        
        epoch_count += 1
        
    result_out = pd.DataFrame(result, columns = result_label)
    print(result_out)
    
    
    
    return