# 数据增强论文阅读

## [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/pdf/1904.10045.pdf)

文章来自google brain，做的方法非常简单粗暴有效，自己在输入的mel spectrogram中进行数据增强，增强的方式也很简单，将mel spectrogram看作2D的语谱图，在time domain 和 frequency domain进行抹除操作，具体可以看下图所示
![](./figures/data_augument1.jpg)
