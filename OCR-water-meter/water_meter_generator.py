import os
import cv2
import argparse
import random
import copy
from tqdm import tqdm
def coordinate_(coordinates):
    list_ = ["[", "]", "(", ")"]
    out=[]
    for coordinate in coordinates.split("],["):
        for i in list_:
            coordinate = coordinate.replace(i, "")
        coordinate=[int(m) for m in coordinate.split(",")]
        out.append(coordinate)
    return out
def parse_arguments():
    """
        Parse the command line arguments of the program.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic text data for text recognition."
    )
    parser.add_argument(
        "-wmi","--water_meter_img", type=str, nargs="?", help="The water meter img directory",required=True
    )
    parser.add_argument(
        "-fpd", "--file_path_double", type=str, nargs="?", help="The double number directory", required=True
    )
    parser.add_argument(
        "-fps", "--file_path_single", type=str, nargs="?", help="The single number directory", required=True
    )
    parser.add_argument(
        "-o", "--out_path", type=str, nargs="?", help="The output directory", required=True,
    )
    parser.add_argument(
        "-n", "--numpy_out", type=int, help="The Calibration coordinates", required=True, )
    parser.add_argument(
        "-c",
        "--coordinate",
        type=coordinate_,
        nargs="?",
        help="Define the margins around the text when rendered. In pixels",
        default=(7, 7, 7, 7),
    )
    return parser.parse_args()
args = parse_arguments()
out_path = args.out_path
list_path = [args.file_path_double, args.file_path_single]
if not os.path.exists(out_path):
    os.makedirs(out_path)
img_list = []
for i in list_path:
    directiory1 = os.walk(i)
    for root, dirs, files in directiory1:
        for file in files:
            img_list.append(os.path.join(root, file))
for i in tqdm(range(args.numpy_out)):
    file = ""
    image = cv2.imread(args.water_meter_img)
    coordinate = copy.deepcopy(args.coordinate)
    for i in coordinate:
        i[0] += int(random.randint(-3, 3))
        i[1] += int(random.randint(-3, 3))
    coordinate[0][2] += int(random.randint(-5, 5))
    coordinate[0][3] += int(random.randint(-5, 5))
    coordinate[1][2] = coordinate[0][3]
    coordinate[1][3] += int(random.randint(-5, 5))
    coordinate[2][2] = coordinate[1][3]
    coordinate[2][3] += int(random.randint(-5, 5))
    coordinate[3][2] = coordinate[2][3]
    coordinate[4][2] += int(random.randint(-5, 5))
    coordinate[4][3] += int(random.randint(-5, 5))
    for j in coordinate:
        a = random.randint(0, len(img_list) - 1)
        _, temp = os.path.split(img_list[a])
        file += temp.split("_")[0]
        file += "_"
        img = cv2.imread(img_list[a])
        img = cv2.resize(img, (j[3] - j[2], j[1] - j[0]))
        image[j[0]:j[1], j[2]:j[3]] = img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(out_path, file + "{}.png".format(i)), image)