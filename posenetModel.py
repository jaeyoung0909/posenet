import tensorflow as tf
import numpy as np
from sklearn.preprocessing import normalize 
from skimage.transform import resize
from PIL import Image

from MobileNet import Mobile
from util import getInputTensorDimensions, getValidResolution, toResizedInputTensor, scalePoses
from multi_pose.decode_multiple_poses import decodeMultiplePoses


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
                'segment' : tf.reshape(segment, (segment.shape[0], segment.shape[1], segment.shape[2]))}

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
        print(scaleY)
        print(scaleX)
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
                
def drawPose(npImg, output):

    def line(A, B):
        f = lambda x: (B['y'] - A['y']) / (B['x'] - A['x']) * (x - A['x']) + A['y']
        ret = []
        step = -1
        if A['x'] <= B['x']:
            step = 1
        for i in range(int(A['x']), int(B['x']), step):
            ret.append([i, int(round(f(i)))])
        return ret
            

    color = {'heart' : [0, 255, 255], 'mid' : [255, 255, 0] ,'leftEye': [255, 0, 0],'rightEye': [255, 0, 0],'nose': [0, 255, 0],'leftEar': [0, 0, 255],'rightEar': [0, 0, 255],'leftShoulder': [255, 0, 0],'rightShoulder': [255, 0, 0],'leftElbow': [0, 255, 0],'rightElbow': [0, 255, 0],'leftWrist': [0, 0, 255],'rightWrist': [0, 0, 255],'leftHip': [255, 0, 0],'rightHip': [255, 0, 0],'leftKnee': [0, 255, 0],'rightKnee': [0, 255, 0], 'leftAnkle':[0,0,255], 'rightAnkle' : [0,0,255]}
    path = {'mid': 'heart', 'heart' : 'mid', 'leftEye': 'nose' ,'rightEye': 'nose','nose': 'heart' ,'leftEar': 'leftEye' ,'rightEar': 'rightEye' ,'leftShoulder': 'heart' ,'rightShoulder': 'heart' ,'leftElbow': 'leftShoulder' ,'rightElbow': 'rightShoulder' ,'leftWrist': 'leftElbow' ,'rightWrist': 'rightElbow' ,'leftHip': 'mid' ,'rightHip': 'mid' ,'leftKnee': 'leftHip' ,'rightKnee': 'rightHip' , 'leftAnkle':'leftKnee', 'rightAnkle' : 'rightKnee'}
    
    for person in output:
        position = {}
        shoulder = []
        hip = []
        for keypoint in person['keypoints']:
            position[keypoint['part']] = {'x' : keypoint['position']['x'], 'y':keypoint['position']['y']}

            if keypoint['part'] is 'leftShoulder' or keypoint['part'] is 'rightShoulder':
                shoulder.append([keypoint['position']['y'], keypoint['position']['x']])
                continue 
            if keypoint['part'] is 'leftHip' or keypoint['part'] is 'rightHip':
                hip.append([keypoint['position']['y'], keypoint['position']['x']])
                continue

        position['heart'] = {'x':(shoulder[0][1] + shoulder[1][1]) / 2, 'y':(shoulder[0][0] + shoulder[1][0]) / 2}
        position['mid'] = {'x':(hip[0][1] + hip[1][1]) / 2, 'y':(hip[0][0] + hip[1][0]) / 2}

        for keypoint in position.keys():
            startPoint = position[keypoint]
            endPoint = position[path[keypoint]]
            for point in line(startPoint, endPoint):
                if npImg.shape[0] <= point[1] or npImg.shape[1] <= point[0]:
                    continue
                npImg[point[1]][point[0]] = [255, 255, 255]

            y = int(round(startPoint['y']))
            x = int(round(startPoint['x']))
            if npImg.shape[0] < y or npImg.shape[1] < x:
                continue

            for i in range(-1, 2, 1):
                for j in range(-1, 2, 1):
                    if npImg.shape[0] <= y + j or npImg.shape[1] <= x + i:
                        continue 
                    npImg[y + j][x + i] = color[keypoint]
    return npImg  


net = Mobile()
posenet = PoseNet(net)

inputImg = Image.open('trump.jpeg')
Img = np.array(inputImg)
inputImg = tf.constant(np.reshape(Img, (1,) + Img.shape))



[output, img, heatmap] = posenet.estimateMultiplePoses(inputImg)
# print(img.shape)
# print(output)
heatmap = [
    {'x': 59.5228271484375, 'y': 46.82022273540497},
    {'x': 67.39776873588562, 'y': 56.60334300994873},
    {'x': 35.487648010253906, 'y': 97.75773429870605},
    {'x': 65.34354972839355, 'y': 41.31705141067505},
    {'x': 101.55524158477783, 'y': 87.48803615570068},
    {'x': 78.0368971824646, 'y': 40.01420736312866},
    {'x': 119.14924335479736, 'y': 129.39313542842865},
    {'x': 44.64396142959595, 'y': 53.29807949066162}
    ]


for i in heatmap:
    y = int(i['y'] * 1.4186046511627908)
    x = int(i['x'] * 2.131782945736434)
    if Img.shape[0] <= y or Img.shape[1] <= x:
        continue
    Img[y][x] = [0, 255, 255]
img = Image.fromarray(Img)
# img = Image.fromarray(drawPose(Img, output))
# img.show()



# output = posenet.predictForMultiPose(inputImg)

# heatmap = output['heatmapScores']
# npArray = tf.Session().run(heatmap).reshape(heatmap.shape[1], heatmap.shape[2],heatmap.shape[3])
# npArray = argmax(npArray)

# segment = output['segment']
# npArray = tf.Session().run(segment).reshape(segment.shape[1], segment.shape[2])


# npArray = 255*(npArray - np.min(npArray))/np.ptp(npArray).astype(int)
# img = Image.fromarray(npArray)
# img.show()