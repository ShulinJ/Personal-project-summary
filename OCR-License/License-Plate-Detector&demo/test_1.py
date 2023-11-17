import os
import cv2
import numpy as np
import random
if __name__ == '__main__':
        for i in range(10):
            img_path=r"F:\jishulin_data\CCPD\0298-31_25-233&367_379&538-379&454_243&538_233&451_369&367-11_0_17_25_12_31_32-108-67.jpg"
            file="0298-31_25-233&367_379&538-379&454_243&538_233&451_369&367-11_0_17_25_12_31_32-108-67.jpg"
            img_1=cv2.imread(img_path)
            file_name = "zATIN78"+"_"+str(i)+".jpg"
            list_point=file.split("-")[3].replace("&","_").split("_")
            list_point=[np.float32(i) for i in list_point]
            pointSrc = [[list_point[0]+random.randint(-8,8) ,list_point[1]+random.randint(-8,0) ],
                        [list_point[2]+random.randint(0,8) , list_point[3]+random.randint(-8,8) ],
                        [list_point[4]+random.randint(-8,8) , list_point[5]+random.randint(0,8) ],
                        [list_point[6]+random.randint(-8,0) , list_point[7]+random.randint(-8,8) ]]


            srcPoints = np.float32(pointSrc)
            save_h,save_w=280,64
            pointDest = [[save_h, save_w], [0, save_w], [0, 0], [save_h, 0]]
            destPoints = np.float32(pointDest)
            m = cv2.getPerspectiveTransform(srcPoints, destPoints)
            resultImg = cv2.warpPerspective(img_1, m, (save_h, save_w), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_WRAP)
            cv2.imwrite(os.path.join(r"F:\jishulin_data\CCPD\test",file_name), resultImg)

            # cv2.waitKey()