from keypoints import NUM_KEYPOINTS

def getOffsetPoint(y, x, keypoint, offsets):
    return {'y': offsets.get(y, x, keypoint), 'x':offsets.get(y, x, keypoint + NUM_KEYPOINTS)}

def getImageCoords(part, outputStride, offsets):
    [heatmapY, heatmapX, id] = part 
    keypoint = id['id']
    [y, x] = getOffsetPoint(heatmapY, heatmapX, keypoint, offsets)
    return {'x': part.heatmapX * outputStride + x, 'y': part.heatmapY * outputStride + y}

def fillArray(element, size):
    result = []
    for i in range(size):
        result.append(element)
    return result 

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
    return {'x': a.x + b.x, 'y': a.y + b.y}

def clampVector(a, min, max):
    return {'y': clamp(a.y, min, max), 'x': clamp(a.x, min, max)}
