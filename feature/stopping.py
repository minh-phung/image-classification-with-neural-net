import numpy as np

def first_der(y, n = 5):
    
    if len(y) < n:
        return False
    
    dydx = np.gradient(y, range(n))
    
    if np.mean(dydx) >= 0:
        return True
    
    return False