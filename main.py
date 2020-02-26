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

for run in range(config['numRuns']):
    # Data creation
    data = dataCreator(config['sceneWidth'],
        config['sceneLength'],
        config['missingR'],
        config['refR'],
        config['rdvR'],
        config['phenLowerBound'],
        config['phenUpperBound'],
        run) # iteration is used as the random seed
    data.create_scene()
    plt.imshow(data.S)
    plt.show()