import pandas as pd
import os

from fit import FitNet


def csv(
    dir,
    x_train, y_train,
    x_val, y_val
):
    
    print("\n\nExecute - schedule.csv")
    
    sche = pd.read_csv(dir + "/schedule.csv")
    
    sche_inte = sche.copy()
    
    print(sche)
    
    except_write = False
    
    #----------------------
    
    net_class = FitNet(
        x_train, y_train,
        x_val, y_val
    )
    
    net_number = str(dir[-5:])
    
    
    for row in sche.itertuples():
        
        print("\n\n----")
        print("Execute - row iterating")
        
        try:
            print(row.Index)
            
            print("model", row.model)
            print("learn_rate", row.learn_rate)
            print("epoch", row.epoch)
            
            para = row[4:]
            
            id = "_".join(map(str, para))
            
            id = id + "__" + str(row.learn_rate)
            
            print(id)
            
            #------------------
            
            dir_result = "result/" + net_number + "/" + id
            
            os.makedirs(dir_result, exist_ok = True)
            
            
            for each_seed in range(1):
                
                print("\nseed", each_seed)
                
                net = net_class.model(
                    net_number,
                    each_seed,
                    para,
                    learn_rate = float(row.learn_rate)
                )
                
                net_class.train(
                    net,
                    epoch_limit = int(row.epoch),
                    result_dir_name = dir_result + "/seed_" + str(each_seed)
                )
                
            
            #------------------
            
            sche_inte = sche_inte.drop(row.Index)
        
        except KeyboardInterrupt:
            
            quit()
        
        except: 
            except_write = True
            
            pass
    
    
    if except_write:
        
        print("\n\n-----------")
        
        print("sche_inte", except_write)
        
        sche_inte.to_csv(dir + "/schedule_interupted.csv", index = False)
        
        print("-----------")
    