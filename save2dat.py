import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def save2dat(RMSE,T,calibrationMethods,numRuns):
    k = 0
    for method in calibrationMethods:
        min_gain    = np.min(RMSE[method][::2],axis=0)
        min_offset  = np.min(RMSE[method][1::2],axis=0)
        max_gain    = np.max(RMSE[method][::2],axis=0)
        max_offset  = np.max(RMSE[method][1::2],axis=0)
        med_gain    = np.median(RMSE[method][::2],axis=0)
        med_offset  = np.median(RMSE[method][1::2],axis=0)
        if numRuns > 1:
            k = k+1
            plt.subplot(len(calibrationMethods),2,k)
            plt.semilogy(T[method][0],min_gain)
            plt.semilogy(T[method][0],max_gain)
            plt.semilogy(T[method][0],med_gain)
            k = k+1
            plt.subplot(len(calibrationMethods),2,k)
            plt.semilogy(T[method][0],min_offset)
            plt.semilogy(T[method][0],max_offset)
            plt.semilogy(T[method][0],med_offset)
        else:
            k = k+1
            plt.subplot(len(calibrationMethods),2,k)
            plt.semilogy(T[method][:],min_gain)
            plt.semilogy(T[method][:],max_gain)
            plt.semilogy(T[method][:],med_gain)
            k = k+1
            plt.subplot(len(calibrationMethods),2,k)
            plt.semilogy(T[method][:],min_offset)
            plt.semilogy(T[method][:],max_offset)
            plt.semilogy(T[method][:],med_offset)
    plt.show()