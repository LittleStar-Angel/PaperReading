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
在看这篇文章之前，首先要明确，图这种数据结构，一般都有有向图和无向图。图包括节点和边(nodes and edge)，假设一幅图有N个节点，则使用一个adjecency matrix可以表示N个节点之间的相互连接情况，一般情况下，adjancency matrix都是一些0,1的mask matrix，1的位置表示节点之间有连接，0表示没有连接，无向图的adjacency matrix都是对称的，而有向图却不是。传统方法中，训练图结构的Loss函数一般都是包括两项，如下面公式所示

<img src="./figures/gnn_fig1.jpg" width="400">

其中，$L_0$是大图中，带有label位置的loss，$L_{reg}$是正则化项，直白的理解是$f(.)$函数需要把图中相邻节点隐射到一个比较接近的矢量空间中， $X$是节点，$A_{ij}$是adjacency matrix的$i$$j$个item。这篇文章算是把DeepLearning引入图网络的一个很早且意义重大的工作，主要有2点比较突出的贡献
> #介绍了一种可以直接对图网络进行建模的神经网络结构，可以把图中的节点信息和与其相连节点，边的信息柔和成一个矢量表征的方法，这样便于后续objective loss的计算

> #探索了GNN建模后，针对图中部分标注数据，可以做到有对无标注图部分信息表征的能力，对图网络进行半监督训练的一种方法

首先可以看看图网络的一个信息流表示如图所示

<img src="./figures/gnn_fig2.png" width="600" >

用公式可以表示为

<img src="./figures/gnn_fig3.jpg" width="300">

这篇文章的一个贡献在于用local spectral filters的1阶近似来进行上面信息流的neural network传递。文章期望使用spectral convolution来对图网络的信息流做一个前向的传递，这里spectral convolution的模型参数是sptracl filters，表示为$g_\theta=diag(\theta)$，spectral convolution的过程可以表示为
$$g_\theta{\star}x=Ug_{\theta}U^Tx$$
其中$U$是归一化图拉普拉斯算子作用出来的特征向量，其中拉普拉斯算子定义为
$$L=I_N-D^{-\frac{1}{2}}AD^{\frac{1}{2}}=U{\Lambda}U^T$$
把$g_{\theta}$看作拉普拉斯算子的特征值，即$g_{\theta}(\Lambda)$，上面的spectral convolution变能够比较好地进行求解。但是这种求解方式带来的问题是，当图网络很大的时候，计算$U$的复杂度是$O(N^2)$，作者沿用已有的方法，使用切比雪夫多项式来近似$g_{\theta}(\Lambda)$，得到

<img src="./figures/gnn_fig4.jpg" width="200">
<img src="./figures/gnn_fig5.jpg" width="200">

切比雪夫多项式的迭代公式为
$$T_k(x)=2xT_{k-1}(x)-T_{k-2}(x)$$
这样便能够可以把整个spectral convolution的计算量给控制下来。在论文中，作者为了进一步简化计算量，把切比雪夫多相似的K阶给近似成1阶，对应的操作是在图拉普拉斯算子谱中的线性映射操作，作者论文表述，这样做的原因有如下：
> #通过堆叠多层网络，依然可以对特征有非常强的表征能力

> #对于那种非常宽的图网络，比如社交网络，这种轻量级的变换，可以防止过拟合

> #只有计算量比较合适，才能够把网络堆叠深，才符合"深度学习"的身份

通过上述的1阶线性简化后，图上的卷积操作变成如下：

<img src="./figures/gnn_fig6.jpg" width="500">

作者为了share模型参数，防止overfit把$\theta_{0}^{'}$和$\theta_{1}^{'}$给share起来，图卷积操作可以进一步简化为

<img src="./figures/gnn_fig7.jpg" width="300">

为了防止$I_N +D^{-\frac{1}{2}}AD^{\frac{1}{2}}$值域范围[0,2]产生梯度消失和爆炸的问题，作者使用renormalization trick对上面的修正为

<img src="./figures/gnn_fig8.jpg" width="350">

最终，一层GCN的变换可以表示为

<img src="./figures/gnn_fig9.jpg" width="250">