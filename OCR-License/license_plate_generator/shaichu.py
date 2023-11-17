import cv2
import numpy as np
import random
import os
import tqdm
def add_noise(image, mean=0, val=0.1):
    size = image.shape
    dtype = image.dtype
    image = image / 255
    gauss = np.random.normal(mean, val ** 0.5, size).astype(np.uint8)
    noise = image + gauss
    return  np.clip(noise*255, 0, 255).astype(dtype)
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
def aug_img(img):
        if random.random() < 0.3:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        else:
            if random.random() < 0.3:
                l = random.randint(3, 9)
                img = cv2.blur(img, (l, l))
                # kernel = np.ones((3, 3), np.float32) / 81
                # img = cv2.filter2D(img, -1, kernel)

            #####################亮度##############################
            if random.random() < 0.3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                img[:, :, 1] = (random.random() + 0.5) * img[:, :, 1]
                img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

            #####################颜色空间转换##############################
            if random.random() < 0.3:
                img = augment_hsv(img)

                # cv2.imencode('.jpg', img)[1].tofile(os.path.join(out_path, i))
            #####################白平衡##############################
            if random.random() < 0.3:
                img = Gray_World(img)

            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #####################直方图均衡##############################
            # if random.random() < 0.2:
            #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            #     img = clahe.apply(img)
            if random.random() < 0.3:
                l = (random.random() + 0.001) * 0.08
                img = add_noise(img, val=l)

            if random.random() < 0.3:
                h, w,_ = img.shape
                M = np.float32([[1, 0, random.random() * 0.05 * w], [0, 1, random.random() * 0.1 * h]])
                img = cv2.warpAffine(img, M, (w, h))
            if random.random() < 0.3:
                h, w,_ = img.shape
                M = np.float32([[1, random.random() * 0.1, 0], [random.random() * 0.1, 1, 0]])
                img = cv2.warpAffine(img, M, (w, h))
            if random.random() < 0.3:
                h, w,_ = img.shape
                if random.random() > 0.5:
                    M = np.float32([[1 + random.random() * 0.1, 0, 0], [0, 1 + random.random() * 0.1, 0]])
                else:
                    M = np.float32([[1 - random.random() * 0.1, 0, 0], [0, 1 - random.random() * 0.1, 0]])
                img = cv2.warpAffine(img, M, (w, h))
            return img
out_path=r"F:\jishulin_data\chepaishuju\CCPD\shuangchepai"
dirc_license_plate=os.listdir(out_path)
a=0
flags = ["cv2."+i for i in dir(cv2) if i.startswith('COLOR_') and ("BGR2" in i)]
for i in tqdm.tqdm(dirc_license_plate):
        img=cv2.imdecode(np.fromfile(os.path.join(out_path,i), dtype=np.uint8), -1)
        img=aug_img(img)
        cv2.imencode('.jpg', img)[1].tofile(os.path.join(out_path, i))
