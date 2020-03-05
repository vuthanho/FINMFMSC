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
from calibrationMethods.emwnenmf import emwnenmf


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
Res = {}

for run in range(config['numRuns']):
    data.create_scene(run)
    # ONLY EMWNENMF HAS BEEN CODED FOR NOW
    for method in config['calibrationMethods']:
        calMethod = locals()[method]
        res = calMethod(data,np.random.rand(m,2),np.random.rand(2,n),config['r'],config['Tmax'])
        calStats = calibrationStatistics(data,res)
        Res.update({ method + '_run_'+str(run): calStats})
        print(calStats)
        print()
        print(res['F'][:,0:5])
        print()
        print(data.F[:,0:5])
        print()
        plt.semilogy(res['T'],res['RRE'])
        plt.show()
    # data.show_scene()           

# plot the results NOT DONE FOR NOW
# if config['statsToPlot']:
#     ResultPlotter(Results,config['statsToPlot'],config['calibrationMethods'],config['numRuns'])