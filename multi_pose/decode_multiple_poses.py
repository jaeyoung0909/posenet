import tensorflow as tf 

from multi_pose.build_part_with_score_queue import buildPartWithScoreQueue 
from multi_pose.decode_pose import decodePose 
from multi_pose.util import getImageCoords, squaredDistance

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
            if not withinNmsRadiusOfCorrespondingPoint(existingPoses, squaredNmsRadius, keypointScores[keypointId]['position'], keypointId):
                ret += keypointScores[keypointId]['score']
        return ret 
    notOverlappedKeypointScores = instanceReducer(instanceKeypoints)
    notOverlappedKeypointScores /= len(instanceKeypoints)
    return notOverlappedKeypointScores

def decodeMultiplePoses(heatmapScores, offsets, displacementsFwd, displacementsBwd, outputStride, maxPoseDetections, scoreThreshold = 0.5, nmsRadius = 20):
    kLocalMaximumRadius = 1
    poses = []

    queue = buildPartWithScoreQueue(scoreThreshold, kLocalMaximumRadius, heatmapScores)
    squaredNmsRadius = nmsRadius * nmsRadius
    while (len(poses) < maxPoseDetections and not queue.empty()):
        root = queue.dequeue()

        rootImageCoords = getImageCoords(root[1]['part'], outputStride, offsets)
        if withinNmsRadiusOfCorrespondingPoint(poses, squaredNmsRadius, rootImageCoords, root[1]['part'][2]['id']):
            continue 
        
        keypoints = decodePose(
            root, heatmapScores, offsets, outputStride, displacementsFwd, displacementsBwd
        )
        score = getInstanceScore(poses, squaredNmsRadius, keypoints)
        poses.append({'keypoints' : keypoints, 'score' : score})

    return poses
