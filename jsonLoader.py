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
        data.append(unpack('f', f[i*4:i*4+4])[0])
    return data

def getAllVariablesAndShapes(url = url, filePath = 'model/'):
    jsonData = jsonLoader(url)
    ret = {}
    for key, value in jsonData.items():
        temp = {}
        temp['variable'] = binaryReader(filePath + value['filename'])
        temp['shape'] = tuple(value['shape'])
        ret[key] = temp
    return ret

def getAllShapes(url=url):
    jsonData = jsonLoader(url)
    shapes = {}
    for key, value in jsonData.items():
        shapes[key] = value['shape']
    return shapes


# a = getAllVariablesAndShapes()["MobilenetV1/segment_2/biases"]
# print(a['variable'])
