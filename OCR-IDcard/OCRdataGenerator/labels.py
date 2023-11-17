import os
# -*- coding: utf-8 -*-
import os
file_dir=r"C:\PROJECT\data\TextRecognitionDataGenerator\trdg\out"
f = open(r"C:\PROJECT\data\TextRecognitionDataGenerator\labels.txt","w")
for root, dirs, files in os.walk(file_dir):
    for file in files:
        list=file.split("_")
        f.write(file+" "+"".join(i for i in list[:-1])+"\n")
