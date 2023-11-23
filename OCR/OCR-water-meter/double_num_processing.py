import os
import cv2
import random
import argparse
from tqdm import tqdm
def parse_arguments():
    """
        Parse the command line arguments of the program.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic text data for text recognition."
    )
    parser.add_argument(
        "-f","--file_path", type=str, nargs="?", help="The output directory",required=True
    )
    return parser.parse_args()
args = parse_arguments()
file_path=args.file_path
dir=os.walk(file_path)
image_list=[]
for roots,dirs,files in dir:
    for file in files:
        image_list.append(os.path.join(file_path,file))
for i in tqdm(image_list):
    img=cv2.imread(i)
    high,width,_=img.shape
    a,b,c,d=int(high*0.20),int(high*0.35),int(high-high*0.25),int(high-high*0.15)
    rand_up,rand_down=random.randint(a,b),random.randint(c,d)
    img=img[rand_up:rand_down,0:width]
    cv2.imwrite(i,img)