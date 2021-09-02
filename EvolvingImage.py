import numpy as np
from numpy import random
from PIL import Image

class EvolvingImage:
    def __init__(self,referenceImgPath,imgDim=(30,30),populationSize=10,crossoverRate=1,mutationRate=0.03):
        self.imgDim = imgDim
        self.population = [random.randint(0,2,imgDim[0]*imgDim[1]) for i in range(populationSize)]
        self.referenceImgArray = self.downSampleImg(referenceImgPath)
        self.referenceFlattened = self.referenceImgArray.flatten()
        self.crossoverRate = crossoverRate
        self.mutatationRate = mutationRate
    
    def downSampleImg(self,imgPath):
        #convert to greyscale, downsample to targetDimension,quantize to 1,0
        img = Image.open(imgPath).convert("L")
        downSampledImage = img.resize(self.imgDim)
        quantizedImageArray = np.array(downSampledImage) > np.median(downSampledImage)
        return quantizedImageArray.astype(int)

    def fitness(self,X):
        return 1/np.sum(np.logical_xor(X,self.referenceFlattened))

    def crossover(self,X,Y):
        crossoverPoint = int(np.random.random()*(len(X)-1))
        X_new = np.concatenate((X[:crossoverPoint],Y[crossoverPoint:]))
        Y_new = np.concatenate((Y[:crossoverPoint],X[crossoverPoint:]))
        return [self.mutatate(X_new),self.mutatate(Y_new)]

    def mutatate(self,X):
        mutationMask = np.random.random(len(X))<=self.mutatationRate
        X[mutationMask] = np.logical_not(X[mutationMask])
        return X

    def mate(self,X,Y):
        if np.random.random()<=self.crossoverRate:
            return self.crossover(X,Y)
        return (X,Y)

    def getMatingPool(self):
        fitnessScores = [self.fitness(X) for X in self.population]
        normalizedFitnessScores = [fitness/np.sum(fitnessScores) for fitness in fitnessScores]
        
        
test = EvolvingImage("mickeyMouse.jpg")
