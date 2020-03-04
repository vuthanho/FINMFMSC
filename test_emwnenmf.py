import numpy as np
import matplotlib.pyplot as plt
from dataCreator import dataCreator
from calibrationMethods.emwnenmf import emwnenmf
import argparse
import json

parser = argparse.ArgumentParser(description='Parse location of config file (json).')
parser.add_argument('--config_file', type=str, default='config.json',
                    help='path to json config file, see config.json for default')

args = parser.parse_args()
with open(args.config_file) as json_data_file:
    config = json.load(json_data_file)

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

data.create_scene(987)
m,n = data.X.shape
# Ginit = 1+np.random.rand(m,2)
# Finit = 1+np.random.randn(2,n)
# Finit[1,:] = Finit[1,:]+1
r = 15
Tmax = 10
res = emwnenmf(data,np.random.rand(m,2),np.random.rand(2,n),r,Tmax)
plt.plot(res['RRE'])
plt.show()
data.show_measured_scene()
# plt.subplot(1,2,1)
# plt.plot(res['RRE'])
# plt.subplot(1,2,2)
# plt.plot(res['RREb'])
# plt.show()
# print(data.F.T)
# print()
# print(res['F'].T)

