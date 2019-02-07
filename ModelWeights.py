import torch 
import numpy as np
from jsonLoader import getAllVariablesAndShapes as get

class ModelWeights():
    def __init__(self, weightData):
        self.weightData = weightData
    
    def weights(self, layerName):
        fullName = 'MobilenetV1/{}/weights'.format(layerName)
        w = torch.tensor(self.weightData[fullName]['variable'])
        w.reshape(self.weightData[fullName]['shape'])
        return w

    def convBias(self, layerName):
        return self.depthwiseBias(layerName)
    
    def depthwiseBias(self, layerName):
        fullName = 'MobilenetV1/{}/biases'.format(layerName)
        bias = torch.tensor(self.weightData[fullName]['variable'])
        bias.reshape(self.weightData[fullName]['shape'])
        return bias

    