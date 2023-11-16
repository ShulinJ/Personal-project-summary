# SUNIT: Multimodal Unsupervised Image-to-Image Translation with Shared Encoder
介绍了一种名为Sunit的多模态无监督图像到图像翻译方法，旨在解决图像从源域到目标域的转换任务。
#### AFHQ数据集上的结果图
[results_1](jieguo.jpg)

#### Celeb_HQ数据集上的结果图
[results_1](jieguo2.jpg)

## 创新点
Sunit是一种带有共享编码器的多模态无监督图像到图像翻译方法。
Sunit在鉴别器（discriminator）和风格编码器（style encoder）之间共享一个编码器网络。这种方法减少了网络参数的数量，并利用鉴别器的信息来提取风格。
此外，还设计了一种新颖的训练策略，其中风格编码器仅使用风格重建损失进行训练，而不跟随生成器（generator）进行训练。这使得风格编码器的目标变得更加明确
