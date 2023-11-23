# OCR-水表数据集制作+识别

# 所使用的算法：

识别算法：LPRNET -轻量，纯CNN网络，运行速度块，训练容易。

# 整体流程：

**1.生成单个数字0-9，以及双字01-90。**

代码：python main/run.py 所有参数的介绍都已在源代码run.py中备注。

重点使用的几个参数：

--output_dir  输出的图片保存地址，默认为根目录下的output文件夹

-dt  设置所使用的词典地址，设置有二，1.双数字（double.txt）2.单数字（single.txt） 位置在main/dicts/

-c 设置生成的图片的数量

-f  设置生成图片的高度

-t  设置生成图片使用的线程数，越高生成速度越快

-k 设置生成图片的倾向角度

-al 设置生成的图片中文字的位置

-or 设置生成图片的方向

-tc  设置生成文字的颜色，16进制，白色：#000000，黑色：#FFFFFF

-sw 生成的文字之间空格的距离。

-m 设置生成图片中文字和边界的距离

-fd 如果想使用新的字体，请自行建立文件夹，并将此项指定到对应文件夹中，文件夹中应为ttf字体文件。

-id 生成的图像所使用的背景图片，默认为main/image

-stw 字体线条宽度

-rm 使用设定好的随机化参数进行生成

其中图像背景尽量选择和水表的字体所在背景相似，相似之外也要有一定的噪声存在，本次使用的背景如下，存放地址为 main\images：

![background](readme_src\background.jpg)

建议使用方式：

为了生成数据集的泛化性，生成过程中需要控制颜色、字体位置、字体线条宽度、字体种类、倾斜角度、以及图像模糊、图像背景。因此集成了一个-rm开关，正常使用仅需要给定生成数量和输出地址，打开-rm即可。

运行 python main/run.py -dt double.txt -c 2000 -rm --output_dir output/double  10-20次（主要为了随机文字在图片中的位置）。

运行 python main/run.py -dt single.txt -c 2000 -rm --output_dir output/single  10-20次（主要为了随机文字在图片中的位置）。

也可以根据需要自行更改参数进行生成。

 生成结果：

![I9Zr6AUCXx](readme_src\I9Zr6AUCXx.jpg)

double：

![img-result0](readme_src\img-result0.jpg)

single：0-9的生成效果如下，词典设置为0，1，2，3，4，5，6，7，8，9

![img-result1](readme_src\img-result1.jpg)

对double number进行二次处理（使其更接近水表中的双字）：

运行 double_num_processing.py 文件。（-f 后面为刚刚生成的double图像所在文件夹）

python double_num_processing.py -f output\double

二次处理后双字图片应为：

![img-result2](C:\Users\jishulin\Desktop\新建文件夹 (8)\Personal-project-summary\OCR-water-meter\readme_src\img-result2.jpg)

### 2.对所需要的水表图像进行标定。

随机拍摄一张水表图像：

![img_0](readme_src\img_0.jpg)

剪切出数字所在位置的表盘，转正，并且对图像进行灰度化：

![img_1](readme_src\img_1.jpg)

标定出水表中的每个数字的具体像素位置。如下图所示：

第一个数字0：13:110,9:86           左上角坐标（x，y），右下角坐标（x1，y1），标定：[y,y1,x,x1]

第二个数字0：13:110, 86:177

第三个数字1：13:110, 177:271

第四个数字78：13:110, 271:350

第五个数字90：3:113, 376:435

### 3.随机采样上述40000张数字图像，代替上面标定的5个位置的数字（灰度化之后代替，最终生成的也是灰度化的图像）。

使用water_meter_generator.py文件

运行python water_meter_generator.py -wmi demo.jpg -fpd output/double -fps output/single -o water_meter -n 20000 -c [13,110,9,86],[13,110,86,177],[13,110,177,271],[13,110,271,350],[3,113,376,435]

参数:

-wmi 水表表盘图片

-fpd  双半字生成图片

-fps  单字生成图片

-o    输出图片文件夹

-n    输出图片数量

-c    标定的数字框的坐标，左上角坐标（x，y），右下角坐标（x1，y1），输入：[y,y1,x,x1],[y,y1,x,x1],[y,y1,x,x1]

最终生成效果：

![img3](readme_src\img3.jpg)

![img_3](readme_src\img_3.jpg)

使用lmdb_propress.py文件将数据集转为lmdb格式：

python lmdb_generator.py -i water_meter -o water_meter_lmdb

-i 生成的水表照片所在文件夹

-o 输出的文件夹

运行结果如下所示：

![img_4](readme_src\img_4.jpg)

### 5.结果：

![img_1](readme_src\img_1.jpg)

识别效果：

![img_5](readme_src\img_5.jpg)
