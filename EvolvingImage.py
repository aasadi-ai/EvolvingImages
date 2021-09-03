#%%
import numpy as np
from numpy import random
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
#%%
class EvolvingImage:
    def __init__(self,referenceImgPath,imgDim=(30,30),populationSize=200,crossoverRate=0.5,mutationRate=0.005,crossoverPercentLength=0.05,selectionMethod="tournament"):
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
        self.crossoverLength = int(imgDim[0]*imgDim[1]*crossoverPercentLength)
        self.selectionMethod = {"tournament":self.tournamentSelection,"roulette":self.rouletteSelection}[selectionMethod]
    
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
        X_new = np.concatenate((X[:crossoverPoint],Y[crossoverPoint:crossoverPoint+self.crossoverLength],X[crossoverPoint+self.crossoverLength:]))
        Y_new = np.concatenate((Y[:crossoverPoint],X[crossoverPoint:crossoverPoint+self.crossoverLength],Y[crossoverPoint+self.crossoverLength:]))
        return [self.mutatate(X_new),self.mutatate(Y_new)]

    def mutatate(self,X):
        mutationMask = np.random.random(len(X))<=self.mutatationRate
        X[mutationMask] = np.logical_not(X[mutationMask])
        return X

    def mate(self,X,Y):
        if np.random.random()<=self.crossoverRate:
            return self.crossover(X,Y)
        return [X,Y]

    def rouletteSelection(self):   
        fitnessScores = [self.fitness(X) for X in self.population]
        self.topPerformer = self.population[np.argmax(fitnessScores)]
        self.fitnessRecord.append(np.mean(fitnessScores))
        normalizedFitnessScores = [fitness/np.sum(fitnessScores) for fitness in fitnessScores]
        return np.random.default_rng().choice(a=self.population,size=self.populationSize,p=normalizedFitnessScores)

    def tournamentSelection(self,tournamentSize=2):
        fitnessScores = [self.fitness(X) for X in self.population]
        self.topPerformer = self.population[np.argmax(fitnessScores)]
        self.fitnessRecord.append(np.mean(fitnessScores))
        matingPool = []
        for i in range(self.populationSize):
            loc0 = np.random.randint(0,self.populationSize)
            loc1 = np.random.randint(0,self.populationSize)
            if fitnessScores[loc0]>=fitnessScores[loc1]:
                matingPool.append(self.population[loc0])
            else:
                matingPool.append(self.population[loc1])
        return matingPool

    def breed(self):
        matingPool = self.selectionMethod()
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

    def tuneParams(self):
        pass
    
    def displayRecord(self):
        plt.plot(self.fitnessRecord,label="Fitness")
        #plt.plot(self.fitness(self.referenceFlattened),label="Global Optimum")
        plt.title("Training Results")
        plt.xlabel("Generation Number")
        plt.ylabel("Fitness")
        #plt.legend()
        plt.show()
#%%

test = EvolvingImage("mickeyMouse.jpg")
test.n_Steps(3000)
test.displayRecord()
Image.fromarray(np.reshape(test.topPerformer*255,test.imgDim)).resize((1000,1000)).show()
Image.fromarray(test.referenceImgArray*255).resize((1000,1000)).show()
