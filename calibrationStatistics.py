import numpy as np

def calibrationStatistics(data,res):
    F = res['F']
    
    Stat = {
        'rmseGain': rms(F[0,0:-1]-data.F[0,0:-1]),
        'nrmseGain': rms(F[0,0:-1]-data.F[0,0:-1])/rms(data.F[0,0:-1]),
        'rmseOffset': rms(F[1,0:-1]-data.F[1,0:-1]),
        'nrmseOffset': rms(F[1,0:-1]-data.F[1,0:-1])/rms(data.F[1,0:-1])
    }  
    return Stat

def rms(x):
    return np.sqrt(np.mean(x ** 2))