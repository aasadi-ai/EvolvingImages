import numpy as np
from numpy import random
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

class EvolvingImage:
    def __init__(self,referenceImgPath,imgDim=(15,15),populationSize=100,crossoverRate=0.5,mutationRate=0.03):
        self.imgDim = imgDim
        self.populationSize = populationSize
        self.population = [random.randint(0,2,imgDim[0]*imgDim[1]) for i in range(populationSize)]
        self.referenceImgArray = self.downSampleImg(referenceImgPath)
        self.referenceFlattened = self.referenceImgArray.flatten()
        self.crossoverRate = crossoverRate
        self.mutatationRate = mutationRate
        self.generationNumber = 0
        self.topPerformer = None
        self.fitnessRecord = []
    
    def downSampleImg(self,imgPath):
        #convert to greyscale, downsample to targetDimension,quantize to 1,0
        img = Image.open(imgPath).convert("L")
        downSampledImage = img.resize(self.imgDim)
        quantizedImageArray = np.array(downSampledImage) > np.median(downSampledImage)
        return quantizedImageArray.astype(int)

    def fitness(self,X):
        return 1/(np.sum(np.logical_xor(X,self.referenceFlattened))+0.00001)

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
        return [X,Y]

    def getMatingPool(self):
        fitnessScores = [self.fitness(X) for X in self.population]
        self.topPerformer = self.population[np.argmax(fitnessScores)]
        self.fitnessRecord.append(np.mean(fitnessScores))
        normalizedFitnessScores = [fitness/np.sum(fitnessScores) for fitness in fitnessScores]
        return np.random.default_rng().choice(a=self.population,size=self.populationSize,p=normalizedFitnessScores)

    def breed(self):
        matingPool = self.getMatingPool()
        matingPool[0]=self.topPerformer
        poolSize = len(matingPool)
        nextGeneration = []
        for i in range(self.populationSize//2):
            offSpring = self.mate(matingPool[np.random.randint(0,poolSize)],matingPool[np.random.randint(0,poolSize)])
            nextGeneration.append(offSpring[0])
            nextGeneration.append(offSpring[1])

        self.population = nextGeneration
        self.generationNumber+=1
    
    def n_Steps(self,n):
        for i in tqdm(range(n)):
            self.breed()
    
    def displayRecord(self):
        plt.plot(self.fitnessRecord)
        plt.title("Training Results")
        plt.xlabel("Generation Number")
        plt.ylabel("Fitness")
        plt.show()

        

