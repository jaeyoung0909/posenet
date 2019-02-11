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
blockId = 0
def toOutputStridedLayers(convolutionDefinition, outputStride):
    global currentStride
    currentStride = 1
    global rate 
    rate = 1
    global blockId 
    blockId = 0 
    def f(convMeta):
        if len(convMeta) is not 2:
            raise RuntimeError('convMeta is invalid')
        convType = convMeta[0]
        stride = convMeta[1]
        layerStride = None
        layerRate = None
        global rate 
        global currentStride 
        global blockId
        tempId = blockId 
        blockId += 1

        if currentStride == outputStride:
            layerStride = 1
            layerRate = rate
            rate *= stride 
        else:
            layerStride = stride 
            layerRate = 1 
            currentStride *= stride 
        
        return {'blockId' : tempId, 'convType' : convType, 'stride':layerStride,  'rate' : layerRate, 'outputStride':outputStride}
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
            rate = layerDict['rate']

            if convType == 'conv2d':
                return self.conv(previousLayer, stride, blockId)
            elif convType == 'separableConv':
                return self.separableConv(previousLayer, stride, blockId, rate)
            else:
                raise RuntimeError('Unknown conv type of {}'.format(convType))
        layers.insert(0, preprocessedInput)
        return reduce(reducer, layers)


    def convToOutput(self, mobileNetOutput, outputLayerName):
        tensor = tf.nn.conv2d(mobileNetOutput, self.weights(outputLayerName), [1,1,1,1], 'SAME')
        tensor = tf.nn.bias_add(tensor, self.convBias(outputLayerName))
        return tensor
    
    def conv(self, inputs, stride, blockId):
        weights = self.weights('Conv2d_{}'.format(blockId))
        a = tf.nn.conv2d(inputs, weights, [1, stride, stride, 1], 'SAME')
        b = tf.nn.bias_add(a, self.convBias('Conv2d_{}'.format(blockId)))
        return tf.clip_by_value(b, 0, 6)

    def separableConv(self, inputs, stride, blockId, dilations = 1):
        dwLayer = 'Conv2d_{}_depthwise'.format(blockId)
        pwLayer = 'Conv2d_{}_pointwise'.format(blockId)
        x1 = tf.nn.depthwise_conv2d(inputs, self.depthwiseWeights(dwLayer), [1, stride, stride, 1],'SAME')
        x1 = tf.nn.bias_add(x1, self.depthwiseBias(dwLayer))
        x1 = tf.clip_by_value(x1, 0, 6)
        x2 = tf.nn.conv2d(x1, self.weights(pwLayer), [1,1,1,1], 'SAME')
        x2 = tf.nn.bias_add(x2, self.convBias(pwLayer))
        x2 = tf.clip_by_value(x2, 0, 6)
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
inputImg = tf.constant(np.reshape(np.array(inputImg), (1, 578,466,3)))
net = Mobile()


print(tf.Session().run(net.predict(inputImg, 16)))

