import torch 
import torch.nn as nn 
import torch.nn.functional as F
from functools import reduce 

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
def toOutputStridedLayers(convolutionDefinition, outputStride):
    currentStride = 1
    rate = 1
    def f(convMeta, blockId):
        if len(convMeta) is not 2:
            raise RuntimeError('convMeta is invalid')
        convType = convMeta[0]
        stride = convMeta[1]
        layerStride = None
        layerRate = None
        if currentStride is outputstride:
            layerStride = 1
            layerRate = rate 
        else:
            layerStride = stride 
            layerRate = rate 
            rate *= stride 
        
        return {'blockId' : blockId, 'convType' : convType, 'stride':layerStride,  'rate' : layerRate, 'outputStride':outputStride}
    return list(f, convolutionDefinition)


class Mobile(nn.Module):
    def __init__(self):
        super(Mobile, self).__init__()
        this.PREPROCESS_DIVISOR = torch.scalar(255.0/2)
        this.modelWieghts = ModelWieghts 
        this.convolutionDefinitions = ConvolutionDefinition
        this.ONE = torch.scalar(1.0)

    def predict(self, input, outputStride):
        normalized = torch.div(input.float(), this.PREPROCESS_DIVISOR)
        preprocessedInput = torch.sub(normalized, this.ONE)
        layers = toOutputStridedLayers(this.convolutionDefinition, outputStride)
        
        def reducer(previousLayer, layerDict):
            blockId = layerDict['blockId']
            convType = layerDict['convType']
            stride = layerDict['stride']
            rate = layerDict['layer']

            if convType == 'conv2d':
                return this.conv(previousLayer, stride, blockId)
            elif convType == 'separableConv':
                return this.separableConv(previousLayer, stride, blockId, rate)
            else:
                raise RuntimeError('Unknown conv type of {}'.format(convType))
        
        return reduce(reducer, layers)


    def convToOutput(mobileNetOutput, outputLayerName):
        conv2d(this.weights(outputLayerName), 1, 'same').add(this.convBias(outputLayerName))
        ret = nn.Conv2d(1,1,)
        ret.add(this.convBias(outputLayerName))
        return ret 
    
    def conv(intputs, stride, blockId):
        weights = this.weights('conv2d_{}'.format(blockId))
        a = torch.Conv2d()
        b = a.add(this.convBias('conv2d_{}'.format(blockId)))
        
    def separableConv(inputs, stride, blockId, dilations = 1):
    
    def weights(layerName):
        return this.modelWeights.weights(layerName)
    
    def convBias(layerName):
        return this.modelWeights.convBias(layerName)
    
    def depthwiseBias(layerName):
        return this.modelWeights.depthwiseBias(layerName)
    
    def depthwiseWeights(layerName):
        return this.modelWeights.depthwiseWeights(layerName)
    


