# Two-stage multi-domain generative adversarial network for landscape image translation

## 介绍
TMGAN 是一种创新方法，旨在解决中国山水画、真实景观照片和油画这三种不同艺术领域之间图像到图像的转换任务。
#### TMGAN效果
![TMGAN网络结构](TMGAN_out.jpg)

## 整体研究思路：
### 1. 收集数据集：
  调研发现并没有现成的多域多模态类型的风景迁移数据集。首先确定了多个域为：现实风景域；山水画（中国）；油画（外国）。然后分别收集各个风景域的公开数据集，例如：[山水画数据集](https://github.com/alicex2020/Chinese-Landscape-Painting-Dataset),油画数据集以及现实风景数据集几乎没有高质量的公开数据集，因此个人从[flick](https://www.flickr.com/)上收集了大量的图片，并且经过自己筛选后，每个域留下了3000张左右的图片（按照之前做的各种图像转换实验的经验，3000是一个合理的数字）。
### 2. 经典图像转换模型在风景迁移任务上的表现：
#### 转换效果图
![效果图](Comparative_Results.png)
可以


#### TMGAN网络结构
![TMGAN网络结构](TMGAN.jpg)



```shell
docker pull huiiji/ubuntu_torch1.13_python3.8:latest  #docker images约20G，请耐心下载
docker run -it --gpus all -v /yourpath:/mnt huiiji/ubuntu_torch1.13_python3.8:latest /bin/bash   #/yourpath为你的根路径
git clone https://github.com/HuiiJi/AISP_NR.git
chmod -R 777 AISP_NR/
cd AISP_NR
```
> *Tips: 该项目的依赖项包括pytorch、torchvision、numpy、opencv-python、pyyaml、tensorboard、torchsummary、torchsummaryX、torch2trt、onnx、onnxruntime等，你可以通过docker pull来下载镜像images并启动容器container来完成环境配置，docker的安装请参考[官方文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)。*
