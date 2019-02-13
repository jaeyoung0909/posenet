import tensorflow as tf 

from multi_pose.build_part_with_score_queue import buildPartWithScoreQueue 
from multi_pose.decode_pose import decodePose 
from multi_pose.util import getImageCoords, squaredDistance

def withinNmsRadiusOfCorrespondingPoint(poses, squaredNmsRadius, coord, keypointId):
    x = coord['x']
    y = coord['y']
    for keypoints in poses:
        correspondingKeypoint = keypoints['keypoints'][keypointId]['position']
        if squaredDistance(y, x, correspondingKeypoint['y'], correspondingKeypoint['x']) <= squaredNmsRadius:
            return True 
    return False
[{'keypoints': [{'score': 0.9796014, 'position': {'x': 64.55803614854813, 'y': 50.58435344696045}, 'part': 'nose'}, {'score': 0.94894516, 'position': {'x': 74.615234375, 'y': 42.24597358703613}, 'part': 'leftEye'}, {'score': 0.99661225, 'part': 'rightEye', 'position': {'x': 59.5228271484375, 'y': 46.82022273540497}}, {'score': 0.9377625, 'position': {'x': 78.0368971824646, 'y': 40.01420736312866}, 'part': 'leftEar'}, {'score': 0.40034768, 'position': {'x': 45.988951206207275, 'y': 51.6635217666626}, 'part': 'rightEar'}, {'score': 0.8416014, 'position': {'x': 100.70621967315674, 'y': 86.14618587493896}, 'part': 'leftShoulder'}, {'score': 0.013474599, 'position': {'x': 43.49920463562012, 'y': 103.0239429473877}, 'part': 'rightShoulder'}, {'score': 0.63185906, 'position': {'x': 115.32553768157959, 'y': 128.18260589241982}, 'part': 'leftElbow'}, {'score': 0.4471693, 'position': {'x': 22.22228240966797, 'y': 133.50413465499878}, 'part': 'rightElbow'}, {'score': 0.31756526, 'position': {'x': 118.12984943389893, 'y': 137.63752269744873}, 'part': 'leftWrist'}, {'score': 0.021781273, 'position': {'x': 51.73404860496521, 'y': 141.17013359069824}, 'part': 'rightWrist'}, {'score': 0.12980562, 'position': {'x': 101.26399421691895, 'y': 136.9250612258911}, 'part': 'leftHip'}, {'score': 0.024537366, 'position': {'x': 50.693115234375, 'y': 138.140061378479}, 'part': 'rightHip'}, {'score': 0.0076348614, 'position': {'x': 108.5172803401947, 'y': 147.859956741333}, 'part': 'leftKnee'}, {'score': 0.0043650963, 'position': {'x': 47.92179296910763, 'y': 143.55535316467285}, 'part': 'rightKnee'}, {'score': 0.001882823, 'position': {'x': 113.838263630867, 'y': 130.66105198860168}, 'part': 'leftAnkle'}, {'score': 0.001205178, 'position': {'x': 48.95908850431442, 'y': 132.3537302017212}, 'part': 'rightAnkle'}], 'score': 0.3944794588914031}]
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

        rootImageCoords = getImageCoords(root['part'], outputStride, offsets)
        if withinNmsRadiusOfCorrespondingPoint(poses, squaredNmsRadius, rootImageCoords, root['part']['id']):
            continue 
        
        keypoints = decodePose(
            root, heatmapScores, offsets, outputStride, displacementsFwd, displacementsBwd
        )
        score = getInstanceScore(poses, squaredNmsRadius, keypoints)
        poses.append({'keypoints' : keypoints, 'score' : score})

    return poses
