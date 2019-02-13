import math
import tensorflow as tf 

from multi_pose.max_heap import MaxHeap

def scoreIsMaximumInLocalWindow(
    keypointId, score, heatmapY, heatmapX, localMaximumRadius, scores
    ):
    height = scores.shape[0]
    width = scores.shape[1]

    localMaximum = True 
    yStart = max(heatmapY - localMaximumRadius, 0)
    yEnd = min(heatmapY + localMaximumRadius + 1, height)
    for yCurrent in range(yStart, yEnd):
        xStart = max(heatmapX - localMaximumRadius, 0)
        xEnd = min(heatmapX + localMaximumRadius + 1, width)
        for xCurrent in range(xStart, xEnd):
            if scores[yCurrent][xCurrent][keypointId] > score:
                localMaximum = False 
                break 
        if not localMaximum:
            break 
    return localMaximum 

def buildPartWithScoreQueue(
    scoreThreshold, localMaximumRadius ,scores 
    ):
    [height, width, numKeypoints] = scores.shape 
    def identity(x):
        return x['score']
    queue = MaxHeap(height * width * numKeypoints, identity)
    
    for heatmapY in range(height):
        for heatmapX in range(width):
            for keypointId in range(numKeypoints):
                score = scores[heatmapY][heatmapX][keypointId]
                if score < scoreThreshold:
                    continue 
                if scoreIsMaximumInLocalWindow(
                    keypointId, score, heatmapY, heatmapX, localMaximumRadius, scores
                    ):
                    queue.enqueue({'score' : score, 'part': {'heatmapY':heatmapY, 'heatmapX' : heatmapX, 'id': keypointId}})
    return queue 

