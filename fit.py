import torch
from collections import deque
import pandas as pd
from pathlib import Path
import numpy as np



from model import Net0, Net1, Net2, Net3, Net4, Net5

from feature import LOSS
from feature import first_der as first_der_stop




model_dict = {
    "net_0": Net0,
    "net_1": Net1,
    "net_2": Net2,
    "net_3": Net3,
    "net_4": Net4,
    "net_5": Net5
}

class FitNet():
    
    def __init__(
        self,
        x_train, y_train,
        x_val, y_val,
        x_pred = None,
        loss = "BCE_with_logit"
    ):
        
        print("Fit class Net")
        
        #---------------------------------------------------
        
        train_data = torch.utils.data.TensorDataset(
            torch.from_numpy(y_train).float(), 
            torch.from_numpy(x_train).float()
        )
        
        self.train_size = len(train_data)
        self.train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size = 4,
            shuffle = False
        )
        
        #---------------------------------------------------
        
        val_data = torch.utils.data.TensorDataset(
            torch.from_numpy(y_val).float(),
            torch.from_numpy(x_val).float()
        )
        
        self.val_size = len(val_data)
        self.val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size = 4,
            shuffle = False
        ) 
        
        #---------------------------------------------------
        if x_pred is None:
            self.x_pred = None
        else:
            print("predict:", True)
            self.x_pred = torch.from_numpy(x_pred).float().contiguous()
            print(self.x_pred.shape)
            
        #---------------------------------------------------
        
        print("Loss function", loss)
        self.loss_func = LOSS[loss]
        
        
    
    
    def model(
        self, 
        model, 
        seed,
        kwarg_dict,
        learn_rate = 10e-5):
        
        torch.manual_seed(seed)
        
        net = model_dict[model](**kwarg_dict)
        
        print("\n----------------------\n")
        
        print("parameter list")
        for name, param in net.named_parameters():
            print(name, param.shape)
        
        self.optimizer = torch.optim.SGD(
            net.parameters(),
            lr = learn_rate
        )
        
        return net
        
    
    def train(
        self, 
        net,
        n_stop = 10,
        epoch_limit = 25,
        result_dir_name = "result/result",
        predict = False
    ):
        
        result_label = ["epoch", "train_loss", "val_loss"]
        result = []
        
        
        epoch_count = 0
        val_loss_queue = deque(maxlen = n_stop)
        
        print("\n\ntrain -----------")
        
        while(
            epoch_count < epoch_limit
            and True #not first_der_stop(val_loss_queue, n_stop)
        ):
            
            train_loss = 0.0
            val_loss = 0.0
            
            for i, (y_train, x_train) in enumerate(self.train_loader):
                
                self.optimizer.zero_grad()
                
                f_x = net(x_train)
                
                loss = self.loss_func(f_x, y_train)
                
                loss.backward()
                
                train_loss += loss.item()
                
                self.optimizer.step()
            
            print("total train_loss", train_loss)
            
            for i, (y_val, x_val) in enumerate(self.val_loader):
            
                f_x = net(x_val)
                
                loss = self.loss_func(f_x, y_val)
                
                val_loss += loss.item()
                
            #print("total val_loss", val_loss)
            
            # --------------------------------------------
            val_loss_queue.append(val_loss / self.val_size)
            
            result.append([
                epoch_count, 
                train_loss / self.train_size, 
                val_loss / self.val_size
            ])
            
            epoch_count += 1
            
        result_out = pd.DataFrame(result, columns = result_label)
        print(result_out)
        
        
        
        result_out.to_csv(
            result_dir_name + ".csv",
            index = False
        )
        
        if self.x_pred is not None:
            
            return torch.sigmoid(net(self.x_pred))
        
        
    
    
