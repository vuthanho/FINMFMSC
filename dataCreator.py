#!/usr/bin/env python3
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

class dataCreator():
    '''
    Generates a pollution scenario
    '''
    def __init__(self, sceneWidth, sceneLength, sensorR, refR, rdvR, phenLowerBound, phenUpperBound):
        self.sceneWidth = sceneWidth # Width of the scene
        self.sceneLength = sceneLength # Length of the scene
        self.numArea = self.sceneWidth*self.sceneLength # Number of sampled area in the scene
        self.sensorR = sensorR # Sensor rate : number of sensors/number of areas
        self.refR = refR # Reference rate : number of references/number of areas
        self.rdvR = rdvR # Rendez-vous rate : number of rendez-vous/number of sensors
        self.phenLowerBound = phenLowerBound # lower bound on the phenomena (air pollution concentrations) standard deviation (log-normal distribution with mean 0)
        self.phenUpperBound = phenUpperBound # upper bound "
        self.numSensor = round(self.numArea*self.sensorR) # Number of sensors in the scene
        self.numRef = round(self.numArea*self.refR) # Number of references in the scene
        self.numRdv = round(self.numSensor*self.rdvR) # Number of sensors having a rendez-vous in the scene
        # self.numAreaVisited = self.numSensor+self.numRef-self.numRdv # Number of not empty areas 

    def create_scene(self,randSeed):
        np.random.seed(randSeed) # Random seed
        # Creation of the map
        self.S = np.zeros(shape=(self.numArea))
        x = np.linspace(-1,1, num=self.sceneLength)
        y = np.linspace(-1,1, num=self.sceneWidth)
        # Create meshgrid
        xx, yy = np.meshgrid(x,y)
        coord = np.vstack((xx.ravel(),yy.ravel())).T
        # number of pulltion pic fixed to 5
        for _ in range(5):
            mean = np.squeeze(np.array([2*(np.random.rand(1)-0.5),2*(np.random.rand(1)-0.5)]))
            cxy = 0
            cov= np.squeeze(np.array([[self.phenLowerBound + (self.phenUpperBound-self.phenLowerBound)*np.absolute(np.random.randn(1)+0.5),cxy],
                    [cxy,self.phenLowerBound + (self.phenUpperBound-self.phenLowerBound)*np.absolute(np.random.randn(1)+0.5)]]))
            z = multivariate_normal.pdf(coord,mean=mean,cov=cov)
            self.S = self.S + z
        
        # Random locations for the mobile sensors and the references
        posRef = np.random.permutation(self.numArea)[0:self.numRef] # Selection of the references
        idxSenRdv = np.random.permutation(self.numSensor)[0:self.numRdv] # Selection of the sensors having a rendez-vous
        idxRefRdv = np.random.randint(self.numRef,size=(self.numRdv)) # Selection of the references corresponding to idxSenRdv

        idxSen = np.arange(self.numSensor)
        idxSen = np.delete(idxSen,idxSenRdv) # Still available sensors
        freePos = np.arange(self.numArea) 
        freePos = np.delete(freePos,posRef) # Still available positions
        posSen = np.random.choice(freePos,size=(self.numSensor-self.numRdv)) # Selection of the positions
        self.posAll = np.unique(np.concatenate((posRef,posSen))) # All unique positions of sensors and references
        self.posEmpty = np.arange(self.numArea)
        self.posEmpty = np.delete(self.posEmpty,self.posAll)

        # Computation of W,G,F and X
        self.W = np.zeros(shape=[self.numArea,self.numSensor+1])
        self.W[posRef,-1] = 1
        # np.put(self.W,(self.numSensor+1)*posRef[idxRefRdv]+idxSenRdv,1)
        self.W[posRef[idxRefRdv],idxSenRdv] = 1
        self.G = np.ones([self.numArea,2]) # The last column of G is only composed of ones
        self.G[:,0] = self.S.flat # The first column of G is composed of the true concentration for each area

        # self.F = np.zeros([2,self.numSensor+1])
        # self.X = np.zeros([self.numAreaVisited,self.numSensor+1])

    def show_scene(self):
        plt.imshow(self.S.reshape((self.sceneWidth,self.sceneLength)))
        plt.show()



