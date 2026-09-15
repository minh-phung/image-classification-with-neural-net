import pandas as pd

from fit import FitNet


def csv(
    dir,
    x_train, y_train,
    x_val, y_val
):
    
    print("\n\nExecute - schedule.csv")
    
    sche = pd.read_csv(dir + "/schedule.csv")
    
    sche_inte = sche.copy()
    
    
    print(df)
    
    
    except_write = False
    
    for row in df.itertuples():
        
        print("\n\n----")
        
        try:
            print("model", row.model)
            print("learn_rate", row.learn_rate)
            print("epoch", row.epoch)
            
            para = row[3:]
            
            
            
            sche_inte = df_copy.drop(row.Index)
            
        except: 
            except_write = True
            
            pass
    
    
    if except_write:
        sche_inte.to_csv("schedule_interupted.csv", index = False)
            
    