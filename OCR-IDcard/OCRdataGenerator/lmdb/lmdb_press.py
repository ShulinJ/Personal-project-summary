'''
生成lmdb数据集
'''
import os
import lmdb
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def init_args():
    args = argparse.ArgumentParser()
    args.add_argument('-i',
                      '--image_dir',
                      default='',
                      type=str,
                      help='The directory of the datasets , which contains the images')
    args.add_argument('-l',
                      '--label_file',
                      default='/datassd/hzl/text_render_data/mingpian/new_gray.txt',
                      type=str,
                      help='The file which contains the paths and the labels of the data set')
    args.add_argument('-s',
                      '--save_dir',
                      default='/datassd/hzl/text_render_data/mingpian/lmdb_gray/',
                      type=str
                      , help='The generated mdb file save dir')
    args.add_argument('-m',
                      '--map_size',
                      help='map size of lmdb',
                      type=int,
                      default=40000000000000)

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
    Create LMDB dataset for CRNN training.
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
    a=0
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
        # print(cache[labelKey])
        a+=1
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % (i+1)
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if i % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print("SOLVE",i)
    print(a)
    cache['num-samples'] = str(a)
    writeCache(env, cache)
    env.close() # 关闭事务
    print('Created dataset with %d samples' % nSamples)
if True:
    path1=r"C:\PROJECT\data\001-shuibiaoshujuji\shengc\b_balck_d_white\hdcz\hdcz_shuibiao"
    outputPath = r"C:\PROJECT\data\001-shuibiaoshujuji\shengc\b_balck_d_white\hdcz_lmdb"
    list_path=[]
    labelList=[]
    imagePathList=[]
    directiory=os.walk(path1)
    for root, dirs, files in directiory:
            for file in files:
                label=file.split(".")[0]
                print(label)
                labelList.append(label)
                imagePathList.append(os.path.join(path1, file))
    print("开始，{},{}".format(len(labelList),len(imagePathList)))
    createDataset(outputPath,imagePathList,labelList,map_size=int(5e9))
    env = lmdb.Environment(outputPath)
    txn = env.begin(write=True)
    print(int(txn.get("num-samples".encode()).decode()))
