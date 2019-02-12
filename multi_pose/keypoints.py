from functools import reduce 

partNames = ['nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder',
	'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist',
	'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle']

NUM_KEYPOINTS = len(partNames)

def pardIdsReducer(jointName):
	result = {}
	for i in range(len(jointName)):
		result[jointName[i]] = i 
	return result 
		
partIds = pardIdsReducer(partNames)

poseChain = [
	['nose', 'leftEye'], ['leftEye', 'leftEar'], ['nose', 'rightEye'],
	['rightEye', 'rightEar'], ['nose', 'leftShoulder'],
	['leftShoulder', 'leftElbow'], ['leftElbow', 'leftWrist'],
	['leftShoulder', 'leftHip'], ['leftHip', 'leftKnee'],
	['leftKnee', 'leftAnkle'], ['nose', 'rightShoulder'],
	['rightShoulder', 'rightElbow'], ['rightElbow', 'rightWrist'],
	['rightShoulder', 'rightHip'], ['rightHip', 'rightKnee'],
	['rightKnee', 'rightAnkle']
]

