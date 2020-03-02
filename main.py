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

data = dataCreator(config['sceneWidth'],
        config['sceneLength'],
        config['sensorR'],
        config['refR'],
        config['rdvR'],
        config['phenLowerBound'],
        config['phenUpperBound'],
        config['Mu_beta'],
        config['Mu_alpha'],
        config['Bound_beta'],
        config['Bound_alpha'])

for run in range(config['numRuns']):
    data.create_scene(run)      # Ok
    data.show_scene()           # Ok
    data.show_measured_scene()  # Ok