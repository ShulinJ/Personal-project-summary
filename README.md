# AISP_NR:  2D AI-Noise Reduction for RAW Images
![pipe](assets/pipe.png)
## 介绍
这是一个关于AI-ISP模块：Noise Reduction 的工程实现文档，针对目标camera如（sensor：IMX766）梳理AI降噪的实现流程，该项目包含：数据准备、模型设计、模型训练、模型压缩、模型推理等。请先确保安装该项目的依赖项，通过git clone下载该项目，然后在该项目的根目录下执行以下命令安装依赖项。

```shell
docker pull huiiji/ubuntu_torch1.13_python3.8:latest  #docker images约20G，请耐心下载
docker run -it --gpus all -v /yourpath:/mnt huiiji/ubuntu_torch1.13_python3.8:latest /bin/bash   #/yourpath为你的根路径
git clone https://github.com/HuiiJi/AISP_NR.git
chmod -R 777 AISP_NR/
cd AISP_NR
```
> *Tips: 该项目的依赖项包括pytorch、torchvision、numpy、opencv-python、pyyaml、tensorboard、torchsummary、torchsummaryX、torch2trt、onnx、onnxruntime等，你可以通过docker pull来下载镜像images并启动容器container来完成环境配置，docker的安装请参考[官方文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)。*
