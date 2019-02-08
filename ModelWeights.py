import tensorflow as tf
import numpy as np
from jsonLoader import getAllVariablesAndShapes as get

class ModelWeights():
    def __init__(self, weightData):
        self.weightData = weightData
    
    def weights(self, layerName):
        fullName = 'MobilenetV1/{}/weights'.format(layerName)
        w = tf.constant(self.weightData[fullName]['variable'])
        w = tf.reshape(w, self.weightData[fullName]['shape'])
        return w

    def depthwiseBias(self, layerName):
        fullName = 'MobilenetV1/{}/biases'.format(layerName)
        bias = tf.constant(self.weightData[fullName]['variable'])
        bias = tf.reshape(bias, self.weightData[fullName]['shape'])
        return bias    

    def convBias(self, layerName):
        return self.depthwiseBias(layerName)
    
    def depthwiseWeights(self, layerName):
        fullName = 'MobilenetV1/{}/depthwise_weights'.format(layerName)
        depth = tf.constant(self.weightData[fullName]['variable'])
        depth = tf.reshape(depth, self.weightData[fullName]['shape'])
        return depth
    
    

    