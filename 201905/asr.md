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


>提示：在github中无法正常显示公式，download repo到本地用markdown打开可以正常显示，也可以使用[GitHub with MathJax](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima/related)插件来解决此问题

# [Very Deep Self-Attention Networks for End-to-End Speech Recognition](https://arxiv.org/pdf/1904.13377.pdf)

这篇文章算是少数用纯transformer来做ASR的了，结果还算不错，有几个结论值得借鉴：
> #transformer的层数一定要足够深，论文中的层数都达到了48层(encoder尤其需要更深一些，encoder用了36层，decoder用了12层)；同时，作者做了另外的实验，把层数减少一半，模型节点数目double，发现性能掉了挺多，说明transformer的确需要更深一些，而不是模型容量大就足够

> #为了防止transformer过拟合，中间的一些drop，stochastic layers的设计也挺重要
作者的工作目测还是很solid的，[代码](https://github.com/quanpn90/NMTGMinor/tree/audio-encoder/)可循


# [AISHELL-2: Transforming Mandarin ASR Research Into Industrial Scale](https://arxiv.org/pdf/1808.10583.pdf)
国内一家专注做语音交互的公司[希尔贝克](http://www.aishelltech.com/sy)release的1000h的中文语音数据，非常高的质量，数据量也足够大，并且作为学术研究可以免费申请

# [Hard Sample Mining for the Improved Retraining of Automatic Speech Recognition](https://arxiv.org/pdf/1904.08031.pdf) 
作者的出发点我觉得还是可以的，使用一种类似半监督的方式，来进行数据筛选。
流程如下：
> 1.使用300h SB数据训练End2End的ASR系统

> 2.使用100h的Fisher数据，送入ASR中，得到识别结果$\hat y$，同时也有golden $y$，再训练一个模型，分辨$\hat y$和$y$。再使用500h的Fisher语料，一个是随机挑的，一个是"hard" samples的数据，然后发现这种"harde"语料的性能比随机挑选的更好

文章的出发点我还是很认可，不过论文中有很多关键步骤根本没有讲清楚，所以我觉得有2种原因: 1.中间的trick作者并不愿意详细讲出来；2.另外一种嘛，emmmm