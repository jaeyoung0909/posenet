import math

from max_heap import MaxHeap

def scoreIsMaximumInLocalWindow(
    keypointId, score, heatmapY, heatmapX, localMaximumRadius, scores
    ):
    [height, width] = scores.shape
    localMaximum = True 
    yStart = max(heatmapY - localMaximum Radius, 0)
    