import pandas as pd

from fit import FitNet


def csv(
    dir
):
    
    print("\n\nExecute - schedule.csv")
    
    df = pd.read_csv(dir)
    
    df_copy = df.copy()
    
    
    print(df)
    
    
    except_write = False
    
    for row in df.itertuples():
        
        print("\n\n----")
        
        try:
            print("model", row.model)
            print("learn_rate", row.learn_rate)
            print("epoch", row.epoch)
            
            print(row[3:])
            
            print(type(row[3:]))
            
            
            df_copy = df_copy.drop(row.Index)
            
        except: 
            except_write = True
            
            pass
    
    if except_write:
        df_copy.to_csv("schedule_interupted.csv", index = False)
            
    