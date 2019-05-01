<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>
>提示：在github中无法正常显示公式，download repo到本地用markdown打开可以正常显示，也可以使用[GitHub with MathJax](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima/related)插件来部分解决此问题，不过行内公式使用方式跟本地方式并不统一，所以更加建议用本地.md方式打开查看
# 数据增强论文阅读

## [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/pdf/1904.10045.pdf)

文章来自google brain，做的方法非常简单粗暴有效，自己在输入的mel spectrogram中进行数据增强，增强的方式也很简单，将mel spectrogram看作2D的语谱图，在time domain 和 frequency domain进行抹除操作，具体可以看下图所示
![](./figures/data_augument1.jpg)

## [Unsupervised Data Augmentation](https://arxiv.org/pdf/1904.12848.pdf)

文章依然来自google和carnegie大学的合作，主要探索了如何使用无监督数据来进行数据增强，这篇文章可以认为是在[Takeru Miyato](https://arxiv.org/pdf/1704.03976.pdf)工作上面的一个升级版本。
这边文章的related work里面对于数据增强还是有非常好的描述的，这些数据增强的方式，可以统一成如下的framework
![](./figures/data_augument2.jpg)
其中，$\hat x$表示数据增强的样本，$y^*$表示$x$对应的label。
从上面的framework来看，这类的数据增强在无论是语音，图像(crop, resize, flip等)，NLP(BERT)都是非常广为人知的操作了，但是这类方法，也仅仅是适用于有监督，无监督的进行数据增强的方法，可以参考[Takeru Miyato](https://arxiv.org/pdf/1704.03976.pdf)的文章，这篇文章做的工作总结成一个公式
![](./figures/data_augument3.jpg)
