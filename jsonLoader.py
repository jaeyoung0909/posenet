import urllib.request as urllib
import json 
from struct import unpack

url = 'https://storage.googleapis.com/tfjs-models/weights/posenet/mobilenet_v1_101/'

def jsonLoader(url = url):
    manifestURL = url+'manifest.json'
    jsonData = urllib.urlopen(manifestURL).read()
    return json.loads(jsonData)

def fileLoader(url = url): 
    manifest = jsonLoader(url)

    for value in manifest.values():
        name = value['filename'] 
        datumUrl = url + name 
        urllib.urlretrieve(datumUrl, "./model/{}".format(name))

def binaryReader(filePath = 'model/'):
    f = open(filePath, 'rb').read()
    binarySize = len(f)
    if binarySize % 4 is not 0:
        print("invalid binarySize at {}".format(filePath))
    bufferSize = binarySize//4
    data = []
    for i in range(bufferSize):
        data.append(unpack('f', f[i:i+4])[0])
    return data

def getAllVariables(url = url, filePath = 'model/'):
    jsonData = jsonLoader(url)
    variables = {}
    for key, value in jsonData.items():
        variables[key] = binaryReader(filePath + value['filename'])
    return variables 

