import numpy as np 
from skimage.transform import resize

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

def drawHeatmap(npImg, heatmap, offset, threshold = 0.5):

    for i in range(heatmap.shape[2]):
        for j in range(heatmap.shape[0]):
            for k in range(heatmap.shape[1]):
                if  heatmap[j][k][i] > 0.5:
                    for dj in range(-1, 2, 1):
                        for dk in range(-1, 2, 1):
                            npImg[j * 16 + int(offset[j][k][i]) + dj][k * 16 + int(offset[j][k][i+17]) + dk] = [255,255,255]
    return npImg

def drawSegment(npImg, seg, threshold=-0.1):
    seg = 2 * (seg - np.min(seg))/np.ptp(seg) - 1
    seg = resize(seg, (npImg.shape[0], npImg.shape[1]))
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            if seg[i][j] > -0.1:
                npImg[i][j] = [255,255,255]
    return npImg
