#!/usr/bin/env python3
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

class dataCreator():
    '''
    Generates a pollution scenario
    '''
    
    def __init__(self, sceneWidth, sceneLength, sensorR, refR, rdvR, mvR, phenLowerBound, phenUpperBound,Mu_beta,Mu_alpha,Bound_beta,Bound_alpha):
        self.sceneWidth = sceneWidth # Width of the scene
        self.sceneLength = sceneLength # Length of the scene
        self.numArea = self.sceneWidth*self.sceneLength # Number of sampled area in the scene
        self.sensorR = sensorR # Sensor rate : number of sensors/number of areas
        self.refR = refR # Reference rate : number of references/number of areas
        self.rdvR = rdvR # Rendez-vous rate : number of rendez-vous/number of sensors
        self.mvR = mvR # missing value rate
        self.phenLowerBound = phenLowerBound # lower bound on the phenomena (air pollution concentrations) standard deviation (normal distribution)
        self.phenUpperBound = phenUpperBound # upper bound "
        self.Mu_beta    = Mu_beta # Mean sensors offset
        self.Mu_alpha   = Mu_alpha # Mean sensors gain
        self.Bound_beta = Bound_beta # Offset boundaries
        self.Bound_alpha= Bound_alpha # Gain boundaries
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
        # number of pulltion pic fixed to 10
        for _ in range(10):
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

        # ##################################################################
        # Pourquoi les rendez-vous devraient être limités à un par capteur ?
        # ##################################################################

        # idxSen = np.arange(self.numSensor)
        # idxSen = np.delete(idxSen,idxSenRdv) # Still available sensors
        # freePos = np.arange(self.numArea) 
        # freePos = np.delete(freePos,posRef) # Still available positions
        # posSen = np.random.choice(freePos,size=(self.numSensor-self.numRdv)) # Selection of the positions
        # self.posAll = np.unique(np.concatenate((posRef,posSen))) # All unique positions of sensors and references
        # self.posEmpty = np.arange(self.numArea)
        # self.posEmpty = np.delete(self.posEmpty,self.posAll)

        # Computation of W,G,F and X
        self.W = np.zeros(shape=[self.numArea,self.numSensor+1])

        # The references
        self.W[posRef,-1] = 1 

        # The sensors having a rendez-vous
        self.W[posRef[idxRefRdv],idxSenRdv] = 1 # np.put(self.W,(self.numSensor+1)*posRef[idxRefRdv]+idxSenRdv,1)
        
        # The other sensors
        Ndata = round((1-self.mvR)*self.numSensor*(self.numArea-self.numRef))
        posSen = np.delete(np.arange(self.numArea),posRef)
        xx, yy = np.meshgrid(posSen,np.arange(self.numSensor))
        idx_mesh_sensor = np.random.permutation((self.numArea-self.numRef)*self.numSensor)[0:Ndata]
        self.W[xx.flat[idx_mesh_sensor],yy.flat[idx_mesh_sensor]] = 1

        # The areas that are not measured
        self.nW = 1-self.W
        
        self.G = np.ones([self.numArea,2]) # The last column of G is only composed of ones
        self.G[:,0] = self.S.flat # The first column of G is composed of the true concentration for each area

        self.F = np.squeeze([np.maximum(self.Bound_alpha[0], np.minimum(self.Bound_alpha[1], self.Mu_alpha+0.5*np.random.randn(1,self.numSensor+1))),
                np.maximum(self.Bound_beta[0], np.minimum(self.Bound_beta[1], self.Mu_beta+0.5*np.random.randn(1,self.numSensor+1)))])
        self.F[:,-1] = [1,0]

        self.Xtheo = np.dot(self.G,self.F)
        self.X = np.multiply(self.Xtheo,self.W)

        # Computation of omega and phi 
        self.Omega_G = np.vstack((self.W[:,-1],np.ones(shape=(self.numArea)))).T 
        self.Omega_F = np.hstack((np.zeros(shape=(2,self.numSensor)), np.ones(shape=(2,1)))) 
        self.Phi_G = np.vstack((self.X[:,-1],np.ones(shape=(self.numArea)))).T 
        self.Phi_F = np.zeros(shape=(2,self.numSensor+1))
        self.Phi_F[0,-1] = 1 

        # Faster access to the fixed values, done to avoid time consuming Hadamard product
        self.idxOG = np.argwhere(self.Omega_G.flat)
        self.sparsePhi_G = self.Phi_G.flat[self.idxOG]
        self.idxOF = np.argwhere(self.Omega_F.flat)
        self.sparsePhi_F = self.Phi_F.flat[self.idxOF]

    def show_scene(self):
        plt.imshow(self.S.reshape((self.sceneWidth,self.sceneLength)))
        plt.show()

    # def show_measured_scene(self):
    #     obs = np.zeros(shape=(self.sceneWidth,self.sceneLength))
    #     np.put(obs,self.posAll,1)
    #     plt.imshow(np.multiply(obs,self.S.reshape((self.sceneWidth,self.sceneLength))))
    #     plt.show()



