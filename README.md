posenet
=======
implementation of posenet

Requirements
------------
python 3.6.5  
numpy 1.14.3  
tensorflow 1.8.0  
cv2 3.4.1  
PIL 5.1.0  
skimage 0.13.1  
sklearn 0.19.1  

Demo
----
    python3 posenetModel.py 

<object data="http://yoursite.com/the.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="http://yoursite.com/the.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="http://yoursite.com/the.pdf">Download PDF</a>.</p>
    </embed>
</object>
File description
----------------
각 파일의 목적과 내부에 있는 함수들의 설명을 작성함.  

![alt text](structure.png)

## jsonLoader.py
tf js 팀에서 학습시켜놓은 모델 weight 의 크롤링.  

### jsonLoader
저장소에 저장된 manifest 를 json 형태로 출력  
input : manefest 저장소의 url  
output : python dictionary (manifest 의 json)  

### manifestLoader
local 파일에 저장된 manifest 를 json 형태로 출력  
input : local 에 저장된 weight file 의 주소  
output : python dictionary (manefest 의 json)  

### fileLoader
manifest 저장소의 url 로 부터 weight 를 가져오고 binary 형태로 model 폴더에 저장  
input : manefest 저장소의 url  

### binaryReader
local 에 저장된 weight 를 binary 에서 float32 로 변환하고 파이썬 리스트에 저장  
input : local 에 저장된 weight의 파일 경로  

### getAllVariablesAndShapes
manifest 의 저장소와 weight 가 저장된 local fila path 를 가지고 파이썬 dictionary 에 변수명과 행렬 모양을 기준으로 정리  
input : manefest 저장소의 url, local 에 저장된 weight의 파일 경로  
output : variable과 shape 을 key 로 가지는 dictionary  

## ModelWeights.py
Model 의 weight 를 python class 형태로 정리하는 파일.  

### weights
input : layer 이름  
output : tf constant (해당 layer 의 weight) 
### depthwiseBias
input : layer 이름  
output : tf constant (해당 layer 의 bias)
### convBias
input : layer 이름  
output : tf constant (해당 layer 의 bias)
### depthwiseWeights
intput : layer 이름  
output : tf constant (해당 layer 의 depthwise weights)

## MobileNet.py
mobile net class 를 구현한 파일

### toOutputStridedLayers
mobile net 의 convolutional layers 의 정보를 list 형태로 출력. list 의 element 는 각 layer 의 block id, convolution type, stride, layer rate, output stride 을 포함.  
input : 전역 변수로 설정한 contolution definition, output stride  
output : convolution layers information list  
### MobileNet
Mobile net 을 구현한 class. ModelWeights 함수로 부터 구한 weights 를 이용하여 Mobile net 구조에 쓰인 convolution layers 를 완성시킴.   
### predict
image 를 받아서 모바일 넷을 실행한 결과를 출력.  
input : tf data type 의 image, output stride  
output : tf array  
### convToOutput
array 에 하나의 layer를 적용시킨 결과를 출력.  
input : tf array, 적용시킬 layer name  
output : tf array
### separableConv
input : tf array input, stride, block id, dilations  
output : tf array (input 정보를 가진 separable conv 를 tf array input 에 적용시킨 결과)
### weights
input : layer name  
output : tf array (layer name 를 가진 weights array)
### convBias  
input : layer name  
output : tf array (layer name 를 가진 convBias array)
### depthwiseBias 
input : layer name  
output : tf array (layer name 를 가진 depthwiseBias array)  
### depthwiseWeights
input : layer name  
output : tf array (layer name 를 가진 depthwiseWeights array)  

## multi_pose
pose net 에서 heatmap 과 offset 을 이용하여 multi pose 를 복호화하기 위한 작업을 함. 결과적으로 쓰이는 함수는 decode_multiple_poses.py 에 있는 decodeMultiplePoses 이고 나머지 파일은 주로 decoder를 구현하기 위한 help function 과 class 가 있다.

## posenetModel.py

### drawer.py
numpy array 형태로 저장된 이미지를 보기 좋게 도와줌.    
### drawPose
input : numpy array (image), estimateMultiplePoses 의 출력값   
output :  key points 를 선으로 연결하고 입력 이미지를 투영한 numpy array   
### drawHeatmap 
input : numpy array (image), predictForMultiPose 의 출력값에서 heatmapScores   값 , predictForMultiPose 의 출력값에서 offsets 값   
output :  keypoints 와 입력 이미지를 투영한 numpy array  
### drawSegment
input : numpy array (image), predictForMultiPose 의 출력값에서 segment 값  
output : segmentation 과 입력 이미지를 투영한 numpy array  





