import os
import random
direction= os.walk(r"D:\JSL\shouxie\data1\xiaoshuotxt\cn")
file_save=r"D:\JSL\shouxie\data1\xiaoshuotxt\cn.txt"
f_save=open(file_save, 'w',encoding="utf-8")
with open("main/dicts/char2.txt", 'r',encoding="utf-8") as file:
        DICT = {char.replace("\n",""): num + 1 for num, char in enumerate(file.readlines())}
for roots,dirs,files in direction:
    for file in files:
        name_txt=os.path.join(roots,file)
        f = open(name_txt, 'r',encoding="ANSI")
        s=f.read().replace("\n","").replace("　　","").replace("    ","").replace("   ","").replace(" ","")
        flag=True
        a=0
        while flag:
                len_nums=random.randint(20,40)
                if a+len_nums>len(s):
                    flag=False
                line=s[a:a+len_nums]
                a+=len_nums
                flag1=True
                for i in line:
                    if i not in DICT:
                        flag1=False
                if flag1==True:
                 f_save.write(line+"\n")



        # line = True
        # while line:
        #     line = f.readline()
        #     f_save.write(line)
