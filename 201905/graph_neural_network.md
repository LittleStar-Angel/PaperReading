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

# GNN
## [Graph Matching Networks for Learning the Similarity of Graph Structured Objects](https://arxiv.org/pdf/1904.12787.pdf)
这篇文章作者在abstruct里面明确说了文章干了2件事情
> #如何将structure形式的sample转换为vector的representatio，并且使用此vector做一些similarity度量
> #作者提出了Graph Matching Network，这个网络结构输入的是一个pair的graph stucture sample，直接计算两者的similarity 
在看这篇文章的时候，个人觉得有几篇GNN的文章是必须要看的，这样才能把GNN的来龙去脉弄清楚，不至于对NLP中的GNN(GCN一窍不通)
## [SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS](https://openreview.net/pdf?id=SJU4ayYgl)