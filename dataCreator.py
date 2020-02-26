#!/usr/bin/env python3
import numpy as np
from scipy.stats import multivariate_normal

class dataCreator():
    '''
    Generates a pollution scenario
    '''
    def __init__(self, sceneWidth, sceneLength, missingR, refR, rdvR, phenLowerBound, phenUpperBound, randSeed):
        self.sceneWidth = sceneWidth
        self.sceneLength = sceneLength
        self.missingR = missingR
        self.refR = refR
        self.rdvR = rdvR
        self.phenLowerBound = phenLowerBound # lower bound on the phenomena (air pollution concentrations) standard deviation (log-normal distribution with mean 0)
        self.phenUpperBound = phenUpperBound # upper bound "
        np.random.seed(randSeed)

    def create_scene(self):
        self.S = np.zeros(shape=[self.sceneWidth,self.sceneLength])
        x = np.linspace(-1,1, num=self.sceneLength)
        y = np.linspace(-1,1, num=self.sceneWidth)
        # Create meshgrid
        xx, yy = np.meshgrid(x,y)
        pos = np.dstack((xx,yy))
        # pos = np.vstack([xx.ravel(), yy.ravel()]).T
        # number of pulltion pic fixed to 3
        for _ in range(5):
            mean = np.squeeze(np.array([2*(np.random.rand(1)-0.5),2*(np.random.rand(1)-0.5)]))
            cxy = 0
            cov= np.squeeze(np.array([[self.phenLowerBound + (self.phenUpperBound-self.phenLowerBound)*np.absolute(np.random.randn(1)+0.5),cxy],
                    [cxy,self.phenLowerBound + (self.phenUpperBound-self.phenLowerBound)*np.absolute(np.random.randn(1)+0.5)]]))
            z = multivariate_normal.pdf(pos,mean=mean,cov=cov)
            self.S = self.S + z


