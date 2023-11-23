# -*- coding: UTF-8 -*-
import argparse
import cv2
from thop import profile
import torch
import copy
from lprnet_model import build_lprnet
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_plate
from torch.autograd import Variable
from crnn.model import get_model
import crnn.utils as utils
import crnn.json_io as json_io
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    coords[:, :8] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4

    return coords

def show_results(img, xywh, conf, landmarks, class_num):
    h,w,c = img.shape
    #tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=3, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(4):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), 3, clors[i], -1)

    label = str(conf)[:5]
    return img
def paixu(list_points,index_list):
    max=0
    index=0
    for i,j in enumerate(list_points):
        if j[0] > max:
            max=j[0]
            index = i
    index_list.append(list_points.pop(index))
    if list_points == []:
        return index_list
    else:
        return paixu(list_points,index_list)
def dian_fenlei(list_points):

    list_points=paixu(list_points,index_list=[])
    if list_points[1][1]> list_points[0][1]:
        list_points[0],list_points[1]=list_points[1],list_points[0]
    if list_points[3][1] > list_points[2][1]:
        list_points[2], list_points[3] = list_points[3], list_points[2]
    return [list_points[0],list_points[2],list_points[3],list_points[1]]


def detect_one(model, image_path, device):
    # Load model
    img_size = 800
    conf_thres = 0.3
    iou_thres = 0.5
    orgimg = cv2.imread(image_path)  # BGR
    img_1 = cv2.imread(image_path)
    sp = orgimg.shape
    h = sp[0]
    w = sp[1]
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found ' + image_path
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    # t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    # t1 = time_synchronized()
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression_plate(pred, conf_thres, iou_thres)

    # print('img.shape: ', img.shape)
    # print('orgimg.shape: ', orgimg.shape)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        # gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
        gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
        if len(det):
            # Rescale boxes from img_size to im0 size
            # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            # for c in det[:, -1].unique():
            #     n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], orgimg.shape).round()

            for j in range(det.size()[0]):
                # xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                # conf = det[j, 4].cpu().numpy()
                landmarks = (det[j, 5:13].view(1, 8) / gn_lks).view(-1).tolist()

                pointSrc = [[landmarks[0]*w, landmarks[1]*h], [landmarks[2]*w, landmarks[3]*h], [landmarks[4]*w, landmarks[5]*h],
                            [landmarks[6]*w, landmarks[7]*h]]

                pointSrc=dian_fenlei(pointSrc)
                # print(pointSrc)
                save_h, save_w = 160, 32
                srcPoints = np.float32(pointSrc)
                pointDest = [  [save_h,save_w],[ 0,save_w],[0, 0],[save_h,0] ]
                destPoints = np.float32(pointDest)
                m = cv2.getPerspectiveTransform(srcPoints, destPoints)
                resultImg = cv2.warpPerspective(img_1, m, (160, 32), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_WRAP)
                # cv2.imwrite("result_img.jpg", resultImg)
                return resultImg
                # class_num = det[j, 13].cpu().numpy()
                # orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)
    # return resultImg
    # cv2.imshow("img",orgimg)
    # cv2.waitKey(0)

def show_files(path, all_files):
    # 首先遍历当前目录所有文件及文件夹
    file_list = os.listdir(path)
    # 准备循环判断每个元素是否是文件夹还是文件，是文件的话，把名称传入list，是文件夹的话，递归
    for file in file_list:
        # 利用os.path.join()方法取得路径全名，并存入cur_path变量，否则每次只能遍历一层目录
        cur_path = os.path.join(path, file)
        # 判断是否是文件夹
        if os.path.isdir(cur_path):
            show_files(cur_path, all_files)
        else:
            if not cur_path.endswith(('jpg')):
                continue
            else:
                all_files.append(cur_path )

    return all_files

# f = show_files("/home/zeusee/plate/", [])
CHARS  = [ '使','领','澳','港',"皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学",
         '0' , '1' , '2' , '3' , '4' , '5' , '6' , '7' , '8' , '9' ,
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', '-'
         ]
def transform(img):
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (0, 1))
    img = img[np.newaxis, :]
    return img
def Greedy_Decode_Eval(Net, img):
        # assert os.path.exists(img_path), 'file is not exists'
        # img =cv2.imdecode(np.fromfile(img_path, dtype=np.uint8),flags=cv2.IMREAD_GRAYSCALE)
        # img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        img_size=[94, 24]
        height, width = img.shape
        if height != img_size[1] or width != img_size[0]:
            img = cv2.resize(img, img_size)
        img = transform(img)
        img = torch.from_numpy(img)
        img = Variable(img.cuda())
        # forward
        img = img.view(1, *img.size())
        prebs = Net(img)
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
        lb = ""
        for i in no_repeat_blank_label:
            lb += CHARS[i]
            lb += "_"
        return lb.replace("_","")

def Classification(img):
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img=Image.fromarray(np.uint8(img))
    img=transform_test(img).unsqueeze(0).to("cuda")
    output = model_class(img)
    _, pred = torch.max(output.data, 1)
    return pred
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.weights, device)
    direction=os.walk(r".\demo\test")
    i = 0
    model_class=torch.load("model_class.pth").to("cuda")
    cfg = json_io.load_json_data(r".\crnn\config.json")
    DICT, ALPHABETS, cfg["model_parameter"]["nclass"] = utils.DICT_create(cfg)
    converter = utils.strLabelConverter(DICT, ALPHABETS)

    model_class_shuang=get_model(cfg,converter).to("cuda")
    checkpoint = torch.load(cfg['test_option']['pre_model_path'])
    model_class_shuang.load_state_dict(checkpoint['state_dict'])

    cfg = json_io.load_json_data(r".\crnn\config1.json")
    DICT, ALPHABETS, cfg["model_parameter"]["nclass"] = utils.DICT_create(cfg)
    converter = utils.strLabelConverter(DICT, ALPHABETS)
    model_class_dan=get_model(cfg,converter).to("cuda")
    checkpoint = torch.load(cfg['test_option']['pre_model_path'])
    model_class_dan.load_state_dict(checkpoint['state_dict'])

    for roots,_,files in direction:
        for file in files:
            img_path = os.path.join(roots, file)
            resultImg=detect_one(model, img_path, device)
            if resultImg is not None:
                class_flags=Classification(resultImg)
                resultImg1 = cv2.cvtColor(resultImg, cv2.COLOR_BGR2GRAY)
                if class_flags==0:
                    new_name = utils.recognition(cfg, resultImg1, model_class_dan, converter, "cuda",32,220)
                    file_name = new_name + "_{}.jpg".format(str(i))
                    cv2.imencode('.jpg', resultImg)[1].tofile(
                        os.path.join(r".\demo\dan", file_name))
                    i += 1
                    # file_name=Greedy_Decode_Eval(lprnet,resultImg1)+"_{}.jpg".format(str(i))
                    # cv2.imencode('.jpg', resultImg)[1].tofile(os.path.join(r"F:\jishulin_data\chepaishuju\CCPD\Joint_Test\dan", file_name))
                    # i += 1
                else:
                    new_name = utils.recognition(cfg, resultImg1, model_class_shuang, converter, "cuda",64,220)
                    file_name=new_name+"_{}.jpg".format(str(i))
                    cv2.imencode('.jpg', resultImg)[1].tofile(os.path.join(r".\demo\shuang", file_name))
                    i += 1