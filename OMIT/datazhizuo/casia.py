import os
import random
import cv2
import numpy as np
pathRGB=r"D:\HFR-NIR-VIS\VIS"
pathIR=r"D:\HFR-NIR-VIS\NIR"
out_path=r"D:\HFR-NIR-VIS\NIR_VIS"
file_path_RGB=os.listdir(pathRGB)
file_path_IR=os.listdir(pathIR)
for file_name_RGB in file_path_RGB:
    flag_name=file_name_RGB.split("_")[2]
    random.shuffle(file_path_IR)
    a=0
    b=0
    for file_name_IR in file_path_IR:
        if flag_name in file_name_IR:
            RGB=cv2.imread(os.path.join(pathRGB,file_name_RGB))
            IR=cv2.imread(os.path.join(pathIR,file_name_IR))
            out_img=np.concatenate((IR,RGB),axis=1)
            cv2.imwrite(os.path.join(out_path,file_name_IR),out_img)
            a+=1
            b=1
            if a == 3:
                break
    if b==0:
        print(flag_name)
