import pandas as pd

def csv(
    dir
):
    
    print("scheduler - schedule.csv")
    
    df = pd.read_csv(dir)
    
    print(df)
    
    