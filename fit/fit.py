from fit.model import Net0

model_dict = {
    "net_0": Net0
}

def main(
    x_train, y_train,
    x_val, y_val,
    model,
    seed = 0,
    n_stop = 10,
    epoch_limit = 50
):
    
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
    
    
    
    
    return