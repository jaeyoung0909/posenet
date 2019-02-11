import tensorflow as tf
import numpy as np
from PIL import Image

from MobileNet import Mobile

class PoseNet():
    def __init__(self, mobileNet):
        self.mobileNet = mobileNet
    
    def predictForMultiPose(self, input, outputStride = 16):
        mobileNetOutput = self.mobileNet.predict(input, outputStride)
        heatmaps = self.mobileNet.convToOutput(mobileNetOutput, 'heatmap_2')
        offsets = self.mobileNet.convToOutput(mobileNetOutput, 'offset_2')
        displacementFwd = self.mobileNet.convToOutput(mobileNetOutput, 'displacement_fwd_2')
        displacementBwd = self.mobileNet.convToOutput(mobileNetOutput, 'displacement_bwd_2')
        return {'heatmapScores': tf.math.sigmoid(heatmaps), 
                'offsets':offsets, 
                'displacementFwd' : displacementFwd, 
                'displacementBwd' : displacementBwd}

    # def estimateMultiplePoses(
    #     self,
    #     input, 
    #     imageScaleFactor = 0.5, 
    #     flipHorizontal = False, 
    #     outputStrid = 16, 
    #     maxDetections = 5, 
    #     scoreThreshold = 0.5, 
    #     nmsRadius = 20):
    #     [height, weight] = getInputTensorDimensions(input)
    #     resizedHeight = getValidReso

net = Mobile()
posenet = PoseNet(net)

inputImg = Image.open('jeus.jpg')
inputImg = tf.constant(np.reshape(np.array(inputImg), (1, 578,466,3)))
heatmap = posenet.predictForMultiPose(inputImg)['heatmapScores']

print(tf.Session().run(heatmap))