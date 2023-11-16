# Two-stage multi-domain generative adversarial network for landscape image translation

## 介绍
TMGAN 是一种创新方法，旨在解决中国山水画、真实景观照片和油画这三种不同艺术领域之间图像到图像的转换任务。
![TMGAN网络结构](TMGAN_out.jpg)
### 项目出发点：
一个多域风景迁移的网络框架研究，主要解决一些经典的图像翻译网络（当时）在风景风格迁移任务上无法很好传递风格，而最新的应用最广的图像翻译网络无法很好保留内容的问题。



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
