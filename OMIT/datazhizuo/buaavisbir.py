import os
import random
import cv2
import numpy as np
path=r"D:\HFR-BUAAVISNIR"
out_path=r"D:\buaaNIR_VIS"
list_rgb=[]
list_ir=[]
for i in range(1,29,2):
    list_rgb.append("{}.bmp".format(i))
for i in range(37,41):
    list_rgb.append("{}.bmp".format(i))
for i in range(2,30,2):
    list_ir.append("{}.bmp".format(i))
print(list_ir)
a=0
roots=os.listdir(path)
file_path1=r"C:\Users\deng\Desktop\input\rgb"
file_path2=r"C:\Users\deng\Desktop\RGB"
a=44
for root in roots:
    root=os.path.join(path,root)
    for i in list_rgb:
            RGB = cv2.imread(os.path.join(root, i))
            random.shuffle(list_ir)
            for j in range(1):
                IR = cv2.imread(os.path.join(root, list_ir[j]))
                try:
                    if a>=100:
                        cv2.imwrite(os.path.join(file_path2, "IR_oulu_00000"+str(a)+".png"), IR)
                        cv2.imwrite(os.path.join(file_path1, "RGB_oulu_00000"+str(a)+".png"), RGB)
                    else:
                        cv2.imwrite(os.path.join(file_path2, "IR_oulu_000000"+str(a)+".png"), IR)
                        cv2.imwrite(os.path.join(file_path1, "RGB_oulu_000000"+str(a)+".png"), RGB)
                    a += 1
                except:
                    continue
            break

#
# for file_name_RGB in file_path_RGB:
#     flag_name=file_name_RGB.split("_")[2]
#     random.shuffle(file_path_IR)
#     a=0
#     b=0
#     for file_name_IR in file_path_IR:
#         if flag_name in file_name_IR:
#             RGB=cv2.imread(os.path.join(pathRGB,file_name_RGB))
#             IR=cv2.imread(os.path.join(pathIR,file_name_IR))
#             out_img=np.concatenate((IR,RGB),axis=1)
#             cv2.imwrite(os.path.join(out_path,file_name_IR),out_img)
#             a+=1
#             b=1
#             if a == 3:
#                 break
#     if b==0:
#         print(flag_name)