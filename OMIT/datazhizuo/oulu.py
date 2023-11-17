import os
import random
import cv2
import numpy as np
import tqdm
pathRGB=r"E:\RGB"
pathIR=r"E:\IR"
out_path=r"E:\IR_RGB"
file_path1=r"C:\Users\deng\Desktop\input\rgb"
file_path2=r"C:\Users\deng\Desktop\RGB"

A=os.walk(pathRGB)
file_path_RGB=[]
for roots,_,files in A:
    for file in files:
        if random.random()>0.9:
            file_path_RGB.append(os.path.join(roots,file))
            break
A=os.walk(pathIR)
file_path_IR = []
for roots, _, files in A:
    for file in files:
        file_path_IR.append(os.path.join(roots, file))

a=182
for i in tqdm.tqdm(file_path_RGB):
    RGB = cv2.imread(i)
    random.shuffle(file_path_IR)
    flag_name=os.path.join("E:\IR",i.split("\\")[2])
    for j in file_path_IR:
        if (flag_name+"\\") in j:
            print(flag_name,j)
            IR = cv2.imread(j)
            # a+=1
            # out_img = np.concatenate((IR, RGB), axis=1)
            # file_name_IR=file_name="NNIR_VIS__{}.jpg".format(b)
            # cv2.imwrite(os.path.join(r"E:\IR_RGB", file_name_IR), out_img)
            cv2.imwrite(os.path.join(file_path2, "IR_oulu_00000" + str(a) + ".png"), IR)
            cv2.imwrite(os.path.join(file_path1, "RGB_oulu_00000" + str(a) + ".png"), RGB)

        # if a == 3:
            a += 1
            break
    # break

