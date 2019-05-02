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
