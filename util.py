import tensorflow as tf 

def scalePose(pose, scaleY, scaleX):
    def keypointsMapper(keypoints):
        for keypoint in keypoints:
            keypoint['position'] = {'x': keypoint['position']['x'] * scaleX, 'y': keypoint['position']['y'] * scaleY}
        return keypoints 
    return {
        'score' : pose['score'],
        'keypoints' : keypointsMapper(pose['keypoints'])
    }

def scalePoses(poses, scaleY, scaleX):
    if scaleX == 1 and scaleY == 1:
        return poses 
    return list(map(lambda pose: scalePose(pose, scaleY, scaleX), poses))

def getValidResolution(imageScaleFactor, inputDimension, outputStride):
    evenResolution = inputDimension * imageScaleFactor - 1
    return evenResolution - (evenResolution % outputStride) + 1

def getInputTensorDimensions(input):
    shape = input.get_shape().as_list()
    return [shape[1], shape[2]]

def toResizedInputTensor(input, resizeHeight, resizeWidth, flipHorizontal):
    imageTensor = input 
    if flipHorizontal:
        return tf.image.resize_bilinear(tf.reverse(imageTensor, 1), (int(resizeHeight), int(resizeWidth)))
    else:
        return tf.image.resize_bilinear(imageTensor, (int(resizeHeight), int(resizeWidth)))
