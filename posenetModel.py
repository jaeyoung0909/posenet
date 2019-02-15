import tensorflow as tf
import numpy as np
from sklearn.preprocessing import normalize 
from skimage.transform import resize
from PIL import Image

from MobileNet import Mobile
from util import getInputTensorDimensions, getValidResolution, toResizedInputTensor, scalePoses
from multi_pose.decode_multiple_poses import decodeMultiplePoses
from drawer import drawPose, drawHeatmap, drawSegment


class PoseNet():
    def __init__(self, mobileNet):
        self.mobileNet = mobileNet
    
    def predictForMultiPose(self, input, outputStride = 16):
        mobileNetOutput = self.mobileNet.predict(input, outputStride)
        heatmaps = self.mobileNet.convToOutput(mobileNetOutput, 'heatmap_2')
        heatmapScores = tf.nn.sigmoid(heatmaps)
        heatmapValues = tf.argmax(heatmapScores, 2)
        offsets = self.mobileNet.convToOutput(mobileNetOutput, 'offset_2')
        displacementFwd = self.mobileNet.convToOutput(mobileNetOutput, 'displacement_fwd_2')
        displacementBwd = self.mobileNet.convToOutput(mobileNetOutput, 'displacement_bwd_2')
        segment = self.mobileNet.convToOutput(mobileNetOutput, 'segment_2')
        return {'heatmapScores': tf.reshape(heatmapScores,(heatmaps.shape[1], heatmaps.shape[2],heatmaps.shape[3])), 
                'heatmapValues' : heatmapValues,
                'offsets':tf.reshape(offsets, (offsets.shape[1], offsets.shape[2],offsets.shape[3])), 
                'displacementFwd' : tf.reshape(displacementFwd, (displacementFwd.shape[1], displacementFwd.shape[2],displacementFwd.shape[3])), 
                'displacementBwd' : tf.reshape(displacementBwd, (displacementBwd.shape[1], displacementBwd.shape[2],displacementBwd.shape[3])),
                'segment' : tf.reshape(segment, (segment.shape[1], segment.shape[2]))}

    def estimateMultiplePoses(
        self,
        input, 
        imageScaleFactor = 0.5, 
        flipHorizontal = False, 
        outputStride = 16, 
        maxDetections = 5, 
        scoreThreshold = 0.5, 
        nmsRadius = 20):

        [height, width] = getInputTensorDimensions(input)
        resizedHeight = getValidResolution(imageScaleFactor, width, outputStride)
        resizedWidth = getValidResolution(imageScaleFactor, width, outputStride)
        
        inputTensor = toResizedInputTensor(input, resizedHeight, resizedWidth, flipHorizontal)
        [multipleRet, img] = tf.Session().run([self.predictForMultiPose(inputTensor, outputStride), inputTensor])
        
        heatmapScores = multipleRet['heatmapScores']
        offsets = multipleRet['offsets']
        displacementFwd = multipleRet['displacementFwd']
        displacementBwd = multipleRet['displacementBwd']

        poses = decodeMultiplePoses(heatmapScores, offsets, displacementFwd, displacementBwd, outputStride, maxDetections, scoreThreshold, nmsRadius)
        
        scaleY = height / resizedHeight
        scaleX = width / resizedWidth
        return [scalePoses(poses, scaleY, scaleX), img.astype(int), heatmapScores]
    
def argmax(npArray):
    height = npArray.shape[0]
    weight = npArray.shape[1]
    channel = npArray.shape[2]
    maxs = []
    for k in range(channel):
        temp = (0, 0)
        for i in range(height):
            for j in range(weight):
                if npArray[temp[0]][temp[1]][k] < npArray[i][j][k]:
                    temp = (i, j)
        maxs.append(temp)

    ret = np.zeros((height, weight))
    for element in maxs:
        ret[element[0]][element[1]] = 1
    
    return ret

net = Mobile()
posenet = PoseNet(net)

inputImg = Image.open('trump.jpeg')
Img = np.array(inputImg)
inputImg = tf.constant(np.reshape(Img, (1,) + Img.shape))

output = tf.Session().run(posenet.predictForMultiPose(inputImg))
heatmap = output['heatmapScores']
offset = output['offsets']
segment = output['segment']

img = Image.fromarray(drawSegment(Img, segment))
# img.show()



# [output, img, heatmap] = posenet.estimateMultiplePoses(inputImg)
# print(img.shape)
# print(output)


# heatmap = [
#     {'x': 59.5228271484375, 'y': 46.82022273540497},
#     {'x': 67.39776873588562, 'y': 56.60334300994873},
#     {'x': 35.487648010253906, 'y': 97.75773429870605},
#     {'x': 65.34354972839355, 'y': 41.31705141067505},
#     {'x': 101.55524158477783, 'y': 87.48803615570068},
#     {'x': 78.0368971824646, 'y': 40.01420736312866},
#     {'x': 119.14924335479736, 'y': 129.39313542842865},
#     {'x': 44.64396142959595, 'y': 53.29807949066162}
#     ]

# img = Image.fromarray(drawPose(Img, output))
# img.show()