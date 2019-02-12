import tensorflow as tf 
import numpy as np 

from build_part_with_score_queue import buildPartWithScoreQueue 
from decode_pose import decodePose 
from util import getImageCoords, squaredDistance

def withinNmsRadiusOfCorrespondingPoint(poses, squaredNmsRadius, coord, keypointId):
    x = coord['x']
    y = coord['y']
    for keypoints in poses:
        correspondingKeypoint = keypoints[keypointId]['position']
        if squaredDistance(y, x, correspondingKeypoint['y'], correspondingKeypoint['x']) <= squaredNmsRadius:
            return True 
    return False

def getInstanceScore(existingPoses, squaredNmsRadius, instanceKeypoints):
    def instanceReducer(keypointScores):
        ret = 0.0
        for keypointId in range(len(keypointScores)):
            if not withinNmsRadiusOfCorrespondingPoint(existingPoses, keypointScores[keypointId]['position'], position, keypointId):
                ret += keypointScores[keypointId]['score']
        return ret 
    notOverlappedKeypointScores = instanceReducer(instanceKeypoints)
    notOverlappedKeypointScores /= len(instanceKeypoints)
    return notOverlappedKeypointScores

def decodeMultiplePoses(heatmapScores, offsets, displacementsFwd, displacementsBwd, outputStride, maxPoseDetections, scoreThreshold = 0.5, nmsRadius = 20):
    kLocalMaximumRadius = 1
    poses = []
    # [scoresBuffer, offsetsBuffer, displacementsFwdBuffer, displacementsBwdBuffer] = toTensorBuffers3D
