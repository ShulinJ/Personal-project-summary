import cv2
import numpy as np
import random
import os
dirc_license_plate=os.listdir(r"F:\jishulin_data\chepaishuju\CCPD\shuangchepai")
dirc_image=os.listdir(r"F:\jishulin_data\chepaishuju\CCPD\CCPD")
a=0
def add_noise(image, mean=0, val=0.1):
    size = image.shape
    image = image / 255

    gauss = np.random.normal(mean, val ** 0.5, size)
    noise = image + gauss
    return  (noise*255)
def augment_hsv(img, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    aug_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return aug_img
def Gray_World(image):
    b, g, r = cv2.split(image)
    b_avg = cv2.mean(b)[0]
    g_avg = cv2.mean(g)[0]
    r_avg = cv2.mean(r)[0]

    avg = (b_avg + g_avg + r_avg) / 3
    b_k = avg / b_avg
    g_k = avg / g_avg
    r_k = avg / r_avg

    b = ((b * b_k)+random.randint(-30,30)).clip(0, 255)
    g = ((g * g_k)+random.randint(-30,30)).clip(0, 255)
    r = ((r * r_k)+random.randint(-30,30)).clip(0, 255)
    image = cv2.merge([b, g, r]).astype(np.uint8)
    return image
def zengguang(img):
        if random.random() < 0.5:
            l = random.randint(3, 12)
            img=cv2.blur(img, (l, l))
            kernel = np.ones((9, 9), np.float32) / 81
            img = cv2.filter2D(img, -1, kernel)

    #####################亮度##############################
        if random.random() < 0.5:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            img[:, :, 1] = (random.random()+0.5)*0.5 * img[:, :, 1]
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    #####################颜色空间转换##############################
        if random.random() < 0.5:
            img=augment_hsv(img)
            # cv2.imencode('.jpg', img)[1].tofile(os.path.join(out_path, i))
    #####################白平衡##############################
        if random.random() < 0.5:
            img=Gray_World(img)


    #####################直方图均衡##############################

        if random.random() < 0.2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
    #####################加噪声##############################
        if random.random() < 0.3:
            l = (random.random()+0.001)*0.08
            img=add_noise(img,val=l)
        return img
for i in dirc_license_plate:
    original_license_plate=cv2.imdecode(np.fromfile(os.path.join(r"F:\jishulin_data\chepaishuju\CCPD\shuangchepai",i), dtype=np.uint8), -1)
    l=random.randint(0,len(dirc_image)-1)
    original_image = cv2.imread(os.path.join(r"F:\jishulin_data\chepaishuju\CCPD\CCPD",dirc_image[l]))
    try:
        lp_h,lp_w,_=original_license_plate.shape
    except:
        continue
    img_h,img_w,_=original_image.shape
    img_blank = np.zeros((lp_h, lp_w, 3), np.uint8)+255
    pointSrc = [[0, 0], [lp_w, 0],[lp_w, lp_h] , [0, lp_h]]
    srcPoints = np.float32(pointSrc)
    r_w,r_h=random.randint(200,300),random.randint(200,300)
    coordinate_w,coordinate_h=random.randint(0,img_w-r_w),random.randint(0,img_h-r_h)
    a1=random.randint(coordinate_w+int(r_w/4),coordinate_w+r_w-int(r_w/4))
    b1=coordinate_h
    a2=coordinate_w+r_w
    b2=random.randint(coordinate_h+int(r_h/4),coordinate_h+r_h-int(r_h/4))
    a3=random.randint(coordinate_w+int(r_w/4),coordinate_w+r_w-int(r_w/4))
    b3=coordinate_h+r_h
    a4=coordinate_w
    b4=random.randint(coordinate_h+int(r_h/4),coordinate_h+r_h-int(r_h/4))
    pointDest = [[a1,b1],
                 [a2,b2],
                 [a3, b3],
                 [a4,b4]]
    destPoints = np.float32(pointDest)
    m = cv2.getPerspectiveTransform(srcPoints, destPoints)
    resultImg = cv2.warpPerspective(original_license_plate, m, (img_w, img_h))
    resultImg2 = cv2.warpPerspective(img_blank, m, (img_w, img_h))
    out_img=cv2.subtract(original_image,resultImg2)+resultImg

    pointDest = [[a1+random.randint(-15,15),b1+random.randint(-15,0)],
                 [a2+random.randint(0,15) ,b2+random.randint(-15,15)],
                 [a3+random.randint(-15,15), b3+random.randint(0,15)],
                 [a4+random.randint(-15,0),b4+random.randint(-15,15)]]
    destPoints = np.float32(pointDest)
    m = cv2.getPerspectiveTransform(destPoints, srcPoints)
    resultImg = cv2.warpPerspective(out_img, m, (lp_w, lp_h))

    resultImg = cv2.resize(resultImg,( random.randint(150,300), random.randint(30,72)))
    # resultImg = zengguang(resultImg)
    a+=1
    cv2.imencode('.jpg', resultImg)[1].tofile(os.path.join(r"F:\jishulin_data\chepaishuju\CCPD\shuangchepai",i))