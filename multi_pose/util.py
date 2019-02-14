import tensorflow as tf

from multi_pose.keypoints import NUM_KEYPOINTS

def getOffsetPoint(y, x, keypoint, offsets):
    x = int(x)
    y = int(y)
    return {'y': offsets[y][x][keypoint], 'x': offsets[y][x][keypoint + NUM_KEYPOINTS]}

def getImageCoords(part, outputStride, offsets): 
    heatmapY = part['heatmapY']
    heatmapX = part['heatmapX']
    keypoint = part['id']
    offsetPoint = getOffsetPoint(heatmapY, heatmapX, keypoint, offsets)
    x = offsetPoint['x']
    y = offsetPoint['y']

    return {'x': heatmapX * outputStride + x, 'y': heatmapY * outputStride + y}

def fillArray(element, size):
    return [element] * size

def clamp(a, min, max):
    if a < min:
        return min 
    if a > max:
        return max 
    return a

def squaredDistance(y1, x1, y2, x2):
    dy = y2 - y1 
    dx = x2 - x1 
    return dy * dy + dx * dx 

def addVectors(a, b):
    return {'x': a['x'] + b['x'], 'y': a['y'] + b['y']}

def clampVector(a, min, max):
    return {'y': clamp(a.y, min, max), 'x': clamp(a.x, min, max)}

# def toTensorBuffer(tensor, type):
    

# def toTensorBuffers3D(tensors):
#     return list(map(lambda tensor : toTensorBuffer(tensor, 'float32'), tensors))