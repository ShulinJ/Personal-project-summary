# 中国车牌模拟生成器

#### 使用说明
1、随机生成车牌
```
python generate_multi_plate.py --number 10 --save-adr multi_val
```
随机生成车牌图片并保存

daiti.py  此文件可以将生成的车牌投射到任意图片上，然后再透射变换回来。模拟正常的检测网络的检测误差。效果如result.jpg所示。

shaichu.py   此文件可以对生成的车牌进行一些数据增广操作，使其更接近真实车牌的样式

2、生成指定车牌
```
python generate_special_plate.py --plate-number 湘999999 --double True --bg-color yellow
```
|  参数   | 说明  |
|  ----  | ----  |
| plate-number  | 车牌号码 |
| double        | 是否双层车牌 |
| bg-color      | 底板颜色|
目前支持的底板颜色有：
|  参数   | 说明  |
|  ----  | ----  |
| black | 粤港澳 |
| black_shi | 使领馆 |
| blue | 普通轿车|
| green_car | 新能源轿车 |
| green_truck | 新能源卡车 |
| white | 白色警车 |
| white_army | 白色军车（仅支持单层） |
| yellow | 中型车 |


