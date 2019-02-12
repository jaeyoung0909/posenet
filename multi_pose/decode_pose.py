from util import clamp, getOffsetPoint, addVectors, getImageCoords 
from keypoints import partIds, partNames, poseChain 



parentChildrenTuples = list(map(lambda x: [partIds[x[0], partIds[x[1]]]], poseChain))
parentToChildEdges = list(map(lambda x: x[1], parentChildrenTuples))
childToParentEdges = list(map(lambda x: x[0], parentChildrenTuples))

def getDisplacement(edgeId, point, displacements):
    numEdges = displacements.shape[2] // 2
    return {'y': displacements[point['y']][point['x']][edgeId], 'x': displacements[point['y']][point['x']][numEdges + edgeId]}

def getStridedIndexNearPoint(point, outputStride, height, width):
    return {'y': clamp(round(point['y']/outputStride), 0, height -1), 'x': clamp(round(point['x']/outputStride), 0, width - 1)}

def traverseToTargetKeypoint(edgeId, sourceKeypoint, targetKeypointId, scoresBuffer, offsets, outputStride, displacements):
    [height, width] = scoresBuffer.shape 
    sourceKeypointIndices = getStridedIndexNearPoint(
        sourceKeypoint['position'], outputStride, height, width
    )
    displacement = getDisplacement(edgeId, sourceKeypointIndices, displacements)
    displacedPoint = addVectors(sourceKeypoint.position, displacement)
    displacedPointIndices = getStridedIndexNearPoint(displacedPoint, outputStride, height, width)
    offsetPoint = getOffsetPoint(displacedPointIndices['y'], displacedPointIndices['x'], targetKeypointId, offsets)
    score = scoresBuffer[displacedPointIndices['y']][displacedPointIndices['x']][targetKeypointId]
    targetKeypoint = addVectors({'x':displacedPointIndices['x'] * outputStride, 'y': displacedPointIndices['y'] * outputStride}, {'x': offsetPoint['x'], 'y': offsetPoint['y']})
    return [{'position' : targetKeypoint, 'part': partNames[targetKeypointId]}, score]

def decodePose(root, scores, offsets, outputStride, displacementsFwd, displacementsBwd):
    global parentToChildEdges
    global childToParentEdges
    
    numParts = scores.shape[2]
    numEdges = len(parentToChildEdges)

    instanceKeypoints = [0 for i in range(numParts)]
    rootPart = root['part']
    rootScore = root['score']
    rootPoint = getImageCoords(rootPart, outputStride, offsets)

    instanceKeypoints[rootPart['id']] = {'score': rootScore, 'part' : partNames[rootPart.id], 'position': rootPoint}
    
    for edge in range(numEdges - 1, -1, -1):
        sourceKeypointId = parentToChildEdges[edge]
        targetKeypointId = childToParentEdges[edge]
        if instanceKeypoints[sourceKeypointId] and not instanceKeypoints[targetKeypointId]:
            instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores, offsets, outputStride, displacementsBwd)
    for edge in range(numEdges):
        sourceKeypointId = childToParentEdges[edge]
        targetKeypointId = parentToChildEdges[edge]
        if instanceKeypoints[sourceKeypointId] and not instanceKeypoints[targetKeypointId]: 
            instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(
                edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores, 
                offsets, outputStride, displacementsFwd)
        
    return instanceKeypoints
