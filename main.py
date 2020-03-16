#!/usr/bin/env python3

__author__ = "Olivier Vu thanh"
__email__ = "olivier.vu-thanh@grenoble-inp.org"

'''
Run "python main.py --config_file=config.json"

Available calibration methods are   : emwnenmf (EM-W-NeNMF from [quote paper])
                                    : coming soon...
'''

import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from dataCreator import dataCreator
from calibrationStatistics import calibrationStatistics
# from calibrationMethods.emwnenmf_updateinsideNNLS import emwnenmf
from calibrationMethods.emwnenmf_seprestart import emwnenmf
from calibrationMethods.incal import incal
from save2dat import save2dat


print('Work in progress')

'''
Get the config (json) file, see "config.json" for default one
'''
parser = argparse.ArgumentParser(description='Parse location of config file (json).')
parser.add_argument('--config_file', type=str, default='config.json',
                    help='path to json config file, see config.json for default')

args = parser.parse_args()
with open(args.config_file) as json_data_file:
    config = json.load(json_data_file)

'''
Main loop
'''

data = dataCreator(config['sceneWidth'],
        config['sceneLength'],
        config['sensorR'],
        config['refR'],
        config['rdvR'],
        config['mvR'],
        config['phenLowerBound'],
        config['phenUpperBound'],
        config['Mu_beta'],
        config['Mu_alpha'],
        config['Bound_beta'],
        config['Bound_alpha'])
m = data.numArea
n = data.numSensor+1
RMSE = {}
T = {}
for method in config['calibrationMethods']:
    RMSE.update({method : []}) 
    T.update({method : []}) 

for run in range(config['numRuns']):
    data.create_scene(run)
    # ONLY EMWNENMF AND INCAL HAVE BEEN CODED FOR NOW
    # data.show_scene()
    print('run : '+str(run))
    for method in config['calibrationMethods']:
        print('method : '+method)
        calMethod = locals()[method]
        res = calMethod(data,data.Ginit,data.Finit,config['r'],config['Tmax'])
        if run == 0:
            RMSE[method] = res['RMSE']
            T[method]    = res['T']
        else:
            RMSE[method] = np.vstack((RMSE[method],res['RMSE']))
            T[method]    = np.vstack((T[method],res['T']))
        print('RMSE : '+str(res['RMSE'][0][-1])+'   '+str(res['RMSE'][1][-1]))
               
if config['save2dat']:
        save2dat(RMSE,T,config['calibrationMethods'],config['numRuns'])
