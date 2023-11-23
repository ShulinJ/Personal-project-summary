'''
生成lmdb数据集
'''
import os
import lmdb
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def parse_arguments():
    args = argparse.ArgumentParser()
    args.add_argument('-i',
                      '--image_dir',
                      default='',
                      type=str,
                      help='The directory of the datasets , which contains the images')
    args.add_argument('-o',
                      '--output_dir',
                      default=r'D:\JSL\shouxie\data1\xiaoshuotxt\cn_lmdb2',
                      type=str
                      , help='The generated mdb file save dir')
    args.add_argument('-m',
                      '--map_size',
                      help='map size of lmdb',
                      type=int,
                      default=5e9)
    return args.parse_args()
def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
    except:
        return False
    else:
        if imgH * imgW == 0:
            return False
    return True
def writeCache(env, cache):
    with env.begin(write=True) as txn: # 建立事务
        for k, v in cache.items():
            if type(k) == str:
                k = k.encode()
            if type(v) == str:
                v = v.encode()
            txn.put(k,v) #写入数据
def createDataset(outputPath, imagePathList, labelList, map_size, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset .
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=map_size) #创建lmdb环境
    cache = {}
    for i in tqdm(range(nSamples)):
        imagePath = imagePathList[i]#.replace('\n', '').replace('\r\n', '')
        label = labelList[i]
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue
        imageKey = 'image-%09d' % (i+1)
        labelKey = 'label-%09d' % (i+1)
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % (i+1)
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if i % 1000 == 0:
            writeCache(env, cache)
            cache = {}
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    env.close() # 关闭事务
    print('Created dataset with %d samples' % nSamples)
if True:
    args = parse_arguments()
    # path1 = args.image_dir
    pathA = r"D:\JSL\DataGenerator\output\cn_long"
    pathB = r"D:\JSL\DataGenerator\output\cn_long1"
    pathC= r"D:\JSL\DataGenerator\output\cn_long2"
    pathD= r"D:\JSL\DataGenerator\output\cn_long3"
    pathE= r"D:\JSL\DataGenerator\output\cn_long5"
    # pathF = r"D:\JSL\DataGenerator\output\cn_long5"
    list1=[pathA,pathB,pathC,pathD,pathE]
    outputPath = args.output_dir
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    list_path=[]
    labelList=[]
    imagePathList=[]
    for path1 in list1:
        directiory=os.walk(path1)
        for root, dirs, files in directiory:
            for file in files:

                if ".jpg" in  file.split("_")[-1]:
                    # if "_" == file[0]:
                    #     imagePathList.append(os.path.join(path1, file))
                    #     file = file[1:]
                    # else:
                    imagePathList.append(os.path.join(path1, file))
                    label1=file.split("_")[:-1]#.replace("_"," ")
                    label="".join(i for i in label1)
                    # print(label)
                    labelList.append(label)

    createDataset(outputPath,imagePathList,labelList,map_size=int(5e10))
    env = lmdb.Environment(outputPath)
    txn = env.begin(write=True)
    print(int(txn.get("num-samples".encode()).decode()))