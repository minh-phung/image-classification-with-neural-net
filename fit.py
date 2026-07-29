import torch
from collections import deque
import pandas as pd
from pathlib import Path
import numpy as np



from model import Net0, Net1, Net2

from feature import LOSS
from feature import first_der as first_der_stop




model_dict = {
    "net_0": Net0,
    "net_1": Net1,
    "net_2": Net2
}

class FitNet():
    
    def __init__(
        self,
        x_train, y_train,
        x_val, y_val,
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
        print("Loss function", loss)
        self.loss_func = LOSS[loss]
        
        
    
    
    def model(self, model, seed, kwarg_dict):
        
        torch.manual_seed(seed)
        
        net = model_dict[model](**kwarg_dict)
        
        self.optimizer = torch.optim.SGD(
            net.parameters(),
            lr = 10e-5
        )
        
        return net
        
    
    def train(
        self, 
        net,
        n_stop = 10,
        epoch_limit = 50,
        result_dir_name = "result/result"
    ):
        
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
        
    
    
'''    
    def predict(
        self,
        net,
        x_predict,
        epoch_count
    ):
        
        predict_loader = torch.utils.data.DataLoader(
            x_predict,
            batch_size = 4,
            shuffle = False
        )
        
        prob_x_test = []
        
        
        for i in range(epoch_count):
            
            for i, (y_train, x_train) in enumerate(self.train_loader):
            
                self.optimizer.zero_grad()
                
                f_x = net(x_train)
                
                loss = self.loss_func(f_x, y_train)
                
                loss.backward()
                
                self.optimizer.step()
            
        prob_x_test = []
        
        with torch.no_grad():
            for y_val, x_val in predict_loader:
                
                prob_x = torch.sigmoid(net(x_val))
                prob_x_test.append(prob_x)
                
        return torch.cat(prob_x_test, dim = 0)
        
'''