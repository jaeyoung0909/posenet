import tensorflow as tf 
import numpy as np
from functools import reduce 
from PIL import Image

from jsonLoader import getAllVariablesAndShapes
from ModelWeights import ModelWeights 

VALID_OUTPUT_STRIDES = [8, 16, 32]
ConvolutionDefinition = [
    ['conv2d', 2],
    ['separableConv', 1],
    ['separableConv', 2],
    ['separableConv', 1],
    ['separableConv', 2],
    ['separableConv', 1],
    ['separableConv', 2],
    ['separableConv', 1],
    ['separableConv', 1],
    ['separableConv', 1],
    ['separableConv', 1],
    ['separableConv', 1],
    ['separableConv', 2],
    ['separableConv', 1]
]

rate = 1
currentStride = 1
def toOutputStridedLayers(convolutionDefinition, outputStride):
    global currentStride
    currentStride = 1
    global rate 
    rate = 1
    def f(convMeta, blockId):
        if len(convMeta) is not 2:
            raise RuntimeError('convMeta is invalid')
        convType = convMeta[0]
        stride = convMeta[1]
        layerStride = None
        layerRate = None
        global rate 
        global currentStride 

        if currentStride == outputStride:
            layerStride = 1
            layerRate = rate
            rate *= stride 
        else:
            layerStride = stride 
            layerRate = 1 
            currentStride *= stride 
        
        return {'blockId' : blockId, 'convType' : convType, 'stride':layerStride,  'rate' : layerRate, 'outputStride':outputStride}
    return list(map(f, convolutionDefinition))


class Mobile():
    def __init__(self):
        super(Mobile, self).__init__()
        self.PREPROCESS_DIVISOR = tf.constant(255.0/2)
        self.modelWeights = ModelWeights(getAllVariablesAndShapes())
        self.convolutionDefinitions = ConvolutionDefinition
        self.ONE = tf.constant(1.0)

    def predict(self, input, outputStride):
        normalized = tf.div(tf.cast(input, tf.float32), self.PREPROCESS_DIVISOR)
        preprocessedInput = tf.subtract(normalized, self.ONE)
        layers = toOutputStridedLayers(self.convolutionDefinitions, outputStride)
        
        def reducer(previousLayer, layerDict):
            blockId = layerDict['blockId']
            convType = layerDict['convType']
            stride = layerDict['stride']
            rate = layerDict['layer']

            if convType == 'conv2d':
                return self.conv(previousLayer, stride, blockId)
            elif convType == 'separableConv':
                return self.separableConv(previousLayer, stride, blockId, rate)
            else:
                raise RuntimeError('Unknown conv type of {}'.format(convType))
        layers = layers.insert(0, preprocessedInput)
        return reduce(reducer, layers)


    def convToOutput(self, mobileNetOutput, outputLayerName):
        return mobileNetOutput.conv2d(self.weights(outputLayerName), 1, 'same').add(self.convBias(outputLayerName))
    
    def conv(self, inputs, stride, blockId):
        weights = self.weights('conv2d_{}'.format(blockId))
        a = inputs.conv2d(weights, stride, 'same')
        b = tf.add(a, self.convBias('conv2d_{}'.format(blockId)))
        return tf.clip_by_value(b, 0, 6)

    def separableConv(self, inputs, stride, blockId, dilations = 1):
        dwLayer = 'Conv2d_{}_depthwise'.format(blockId)
        pwLayer = 'Conv2d_{}_pointwise'.format(blockId)
        x1 = inputs.depthwiseConv2D(
                self.depthwiseWeights(dwLayer), stride, 'same', 'NHWC',
                dilations
            ).add(self.depthwiseBias(dwLayer)
            ).clip_by_value(0, 6)
        x2 = x1.conv2d(self.weights(pwLayer), [1,1], 'same'
            ).add(self.convBias(pwLayer)
            ).clip_by_value(0, 6)
        return x2

    def weights(self, layerName):
        return self.modelWeights.weights(layerName)
    
    def convBias(self, layerName):
        return self.modelWeights.convBias(layerName)
    
    def depthwiseBias(self, layerName):
        return self.modelWeights.depthwiseBias(layerName)
    
    def depthwiseWeights(self, layerName):
        return self.modelWeights.depthwiseWeights(layerName)



inputImg = Image.open('jeus.jpg')
inputImg = tf.constant(np.array(inputImg))

model = Mobile()
model.predict(inputImg, 16)