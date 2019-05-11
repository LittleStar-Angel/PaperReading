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
[TOC]
# Generative Model
[Generating High Fidelity Images With Subscale Pixel Networks And Multidimensional Upscaling](https://openreview.net/forum?id=HylzTiC5Km
)
层级自回归模型，解决pixel cnn类似无法生成大图的问题(最近open ai的sparse transformer用大力出奇迹的方式解决自回归)


[Temporal Difference Variational Auto-encoder](https://openreview.net/forum?id=S1x4ghC9tQ)
提出一种temporal的VAE方法，对temporal的历史state进行编码，模型对temporal的建模不仅仅在observation有预测能力，还得在hidden code层面有预测能力

[Enabling Factorized Piano Music Modeling And Generation With The Maestro Dataset](https://openreview.net/forum?id=r1lYRjC9F7)
文章2个点比较吸引人：1.release了170小时的音乐数据，并且有很好的标注；2.实验做得非常solida，效果很棒；实验部分有点吸引人的是其waveform上面的data augument工程上面的经验


[Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse Rl, And Gans By Constraining Information Flow](https://openreview.net/forum?id=HyxPx3R9tm)
解决GAN训练过程中D网络太强，导致梯度太小，网络训练探索，文章在D网络中使用了信息bottleneck的一种方式


[Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://openreview.net/forum?id=B1xsqj09Fm)
公号震惊部的文章BigGAN，典型大力出奇迹，效果很经验，很多工程trick值得借鉴，计算量太大，不过作者后续开源了pytorch的经济适用版，做GAN的值得follow


[Lagging Inference Networks and Posterior Collapse in Variational Autoencoders](https://openreview.net/forum?id=rylDfnCqF7)
作者使用一种简单有效的策略来缓解VAE的posterior collapse问题，直接优化inference 网络结构


[A Variational Inequality Perspective on Generative Adversarial Networks](https://openreview.net/forum?id=r1laEnA5Ym)
GAN训练的不稳定性，一般大家都是重新设计更加鲁邦稳定的目标函数，这篇文章从优化的角度出发，将GAN的优化问题隐射到数学理论中的变分不等式，并且根据数学中已有的一些方法，加以改造和近似，变成一种深度学习常用的adam的优化方法来优化GAN的训练过程


[KnockoffGAN: Generating Knockoffs for Feature Selection using Generative Adversarial Networks](https://openreview.net/forum?id=ByeZ5jC5YQ)
作者用GAN来做feature selection，并且为了降低false discovery rate，使用GAN来生成knockoffs


[Kernel Change-point Detection with Auxiliary Deep Generative Models](https://openreview.net/forum?id=r1GbfhRqF7)

[Approximability of Discriminators Implies Diversity in GANs](https://openreview.net/forum?id=rJfW5oA5KQ)
作者主要分析GAN训练的结果多样性不足或者模式坍缩的原因不是模型没有学习到输入样本的多样性分布，而从W或者KL距离来看，仅仅是没有收敛好

[Diagnosing and Enhancing VAE Models](https://openreview.net/forum?id=B1e0X3C9tQ)
作者提出了一种VAE增强手段，使得VAE生成的图像得到了较高的提升

[ClariNet: Parallel Wave Generation in End-to-End Text-to-Speech](https://openreview.net/forum?id=HklY120cYm)
百度的parallel-wavenet，比较偏工程化，写了比较多的实现trick，但是不release 代码是几个意思？？？

[GAN Dissection: Visualizing and Understanding Generative Adversarial Networks](https://openreview.net/forum?id=Hyg_X2C5FX)
论文给出了一套工具来对GAN的不同维度的信息进行分析

[InstaGAN: Instance-aware Image-to-Image Translation](https://openreview.net/forum?id=ryxwJhC9YX)
作者提出了一些方法，从实例中获取属性，来完成image2image的训练

[Generative Code Modeling with Graphs](https://openreview.net/forum?id=Bke4KsA5FX)
源码生成任务：给定自然语言描述、程序的输入输出对、程序框架等上下文信息，根据这些信息生成代码；文章改进点主要利用context信息进行解码的方法，在AST（抽象语法树）基础上将问题变成了一个序列分类（每次生成边或者点可以看成是一个action）问题

[FFJORD: Free-Form Continuous Dynamics for Scalable Reversible Generative Models](https://openreview.net/forum?id=rJxgknCcK7)
文章改进了可逆生成模型，使用一些trick来优化Trace的计算，使得模型可以避开Jacobian的计算。

[On Computation And Generalization Of Generative Adversarial Networks Under Spectrum Control](https://openreview.net/forum?id=rJNH6sAqY7)
GAN训练使用谱正则来提高训练稳定性是主流方法，文章在此基础上进行了改进，细节没看

[GANSynth: Adversarial Neural Audio Synthesis](https://openreview.net/forum?id=H1xQVn09FX)
基于GAN的音频合成，可以beatwavenet，谱建模，并且对相位也有考量，挺有意思

[DISTRIBUTIONAL CONCAVITY REGULARIZATION FOR GANS](https://openreview.net/forum?id=SklEEnC5tQ)
对GAN引入分布上的正则化

[A Universal Music Translation Network](https://openreview.net/forum?id=HJGkisCcKm)
使用auto-encoder把音乐音色器材做转换

[Layoutgan: Generating Graphic Layouts With Wireframe Discriminator](https://openreview.net/forum?id=HJxB5sRcFQ)
提出了一种layourGAN用来图形学中的布局，主要使用了GAN,self-attention,并提出了wireframe discriminator.

[Improving Generalization And Stability Of Generative Adversarial Networks](https://openreview.net/forum?id=ByxPYjC5KQ)
本文提出了一种zero-centered gradient penalty来提高GAN的泛化和收敛性能。

[Whitening and Coloring Batch Transform for GANs](https://openreview.net/forum?id=S1x2Fj0qKQ)
提出基于Whitening and Coloring的BN，提高GAN的稳定性

[Generating Multiple Objects at Spatially Distinct Locations](https://openreview.net/forum?id=rkxciiC9tm)
在GAN上增加“object pathway”，能够控制生成图片中目标位置

[Residual Non-local Attention Networks for Image Restoration](https://openreview.net/forum?id=HkeGhoA5FX)
本文提出了用残差非局部注意网络来用作高质量图像恢复。

[RelGAN: Relational Generative Adversarial Networks for Text Generation](https://openreview.net/forum?id=rJedV3R5tm)
文章提出 RelGAN,一种新的用于文本生成的GAN，

[Policy Transfer with Strategy Optimization](https://openreview.net/forum?id=H1g6osRcFQ)
本文提出了一种简单的思想，将原始的学习策略（Policy）在不同domain进行转换

[Don't let your Discriminator be fooled](https://openreview.net/forum?id=HJE6X305Fm)
提出多种与Wasserstein objective有类似平滑、稳定特性的GAN训练目标。

[Improving MMD-GAN Training with Repulsive Loss Function](https://arxiv.org/abs/1812.09916)
MMD-GAN（maximum mean discrepancy）的loss function可能抑制学习数据中试图contrat判别器的细节特征。本文提出一种repulsive loss function来解决此问题，同时，提出了一种有界的高斯核来稳定GAN的训练。模型应用于CIFAR-10，STL-10，CelebA，LSUN，能比MMD loss显著提高。在CIFAR-10上达到FID分16.21

[PROBGAN: Towards Probabilistic GAN with Theoretical Guarantees](https://openreview.net/forum?id=H1l7bnR5Ym)
文章提出一种概率模型的GAN, ProbGAN，带着carefully crafted prior迭代地学习生成器的概率分布。训练时使用Hamiltonian Monte Carlo的随机梯度方法来四号线贝叶斯推断。在CIFAR-10，STL-10，ImageNet等数据集上比其他 SOTA multi-generator GAN和probabilistic treatment for GAN都要好

[Diversity-Sensitive Conditional Generative Adversarial Networks](https://arxiv.org/abs/1901.09024)
在condition GAN方法中，模式坍缩是常见问题。模式坍缩的主要原因是模型用latent code中的大部分去编码一些非常相似的输出。文章通过加正则化的方法鼓励生成器用latent code生成不同输出来减轻模式坍缩的问题。该方法可以很容易地应用到不同的cGAN结构上。

[Sample Efficient Adaptive Text-To-Speech](https://openreview.net/pdf?id=rkzjUoAcFX)
文章提出一个meta-learning方法用少量数据训练自适应TTS。作者把模型分为task-dependent parameters和task-independent parameters，每个speaker相当于一个task。在训练过程中，同时训练所有参数，但是舍弃task-dependent parameters。训练的目标是对新task用少量数据快速学习task-dependent parameters

[PATE-GAN Generating Synthetic Data with Differential Privacy Guarantees](https://openreview.net/forum?id=S1zk9iRqF7)
在一些任务上，真实数据不是十分必要，同时还有隐私性问题。可以用GAN从真实数据生成数据，同时不包含任何隐私信息。训练D网络时使用Private Aggregation of Teacher Ensembles（PATE），不知道什么内容。

[CoT Cooperative Training for Generative Modeling of Discrete Data](https://openreview.net/forum?id=SkxxIs0qY7)
文章提出Cooperative Trainning（CoT）方法，协同训练一个生成器G和一个辅助的predictive mediator M。M的训练目标是G的分布的mixture density，G的训练目标是最小化和M估计的分布之间的JS散度。CoT在生成质量、多样性、predictive generalization ability和计算代价都超过SeqGAN、RankGAN、MaliGAN、LeakGAN等方法

[Learning From Incomplete Data With Generative Adversarial Networks](https://openreview.net/forum?id=S1lDV3RcKm)
让GAN从不完整数据中学习完整数据生成器和掩摸生成器

[Monge-amp ere Flow For Generative Modeling](https://openreview.net/forum?id=rkeUrjCcYQ)
没看懂，基于最优传输理论的Monge-amp`ere等式设计Flow生成模型

[Discriminator Rejection Sampling](https://openreview.net/forum?id=S1GkToR5tm)
使用rejection sampling来提升GAN的性能，效果很不错

[Camou: Learning Physical Vehicle Camouflages To Adversarially Attack Detectors In The Wild](https://openreview.net/forum?id=SJgEl3A5tm)
关于对抗样本，论文提出了一种车辆迷彩的生成方法，喷印在3d物体上后能够欺骗物体检测算法

[Timbretron: A Wavenet(cyclegan(cqt(audio))) Pipeline For Musical Timbre Transfer](https://openreview.net/forum?id=S1lvm305YQ)
将音乐由一种乐器转换到另一种乐器的风格，使用了CycleGAN

[On Self Modulation for Generative Adversarial Networks](https://openreview.net/forum?id=Hkl5aoR5tm)
对GAN中的generator进行了很简单的修改，在很多指标上都有明显提升。能够应用于任何GAN的变种。

[MAE: Mutual Posterior-Divergence Regularization for Variational AutoEncoders](https://openreview.net/forum?id=Hke4l2AcKQ)
VAE当decoder复杂是KL项会逐渐消失，作者认为不同的输入，在隐层空间表示不同的部分，通过新增加一个正则项来惩罚成对数据的隐层后验的相似性及多样性来防止KL消失


# Sequence to Sequence
[Posterior Attention Models for Sequence to Sequence Learning](https://openreview.net/forum?id=BkltNhC9FX
)
把attention看作是ED中间的hidden variable，用前后向算法的方式，做V组attention的类似工作，挺有意思


[Pay Less Attention With Lightweight And Dynamic Convolutions](https://openreview.net/forum?id=SkVhlh09tX)
对transformer中self-attention的改进，增加效率

[Adaptive Input Representations For Neural Language Modeling](https://openreview.net/forum?id=ByxZX20qFQ)
作者对adaptive softmax进行了扩充，做了adaptive input embedding,并在语音模型建模上取得非常好的效果


[Detecting Egregious Responses in Neural Sequence-to-sequence Models](https://openreview.net/forum?id=HyNA5iRcFQ)
这篇文章主要用于测试，一个已经训练好的对话系统，是否会产生一些充满恶意的回答

[Multilingual Neural Machine Translation with Knowledge Distillation](https://openreview.net/forum?id=S1gUsoR9YX)
多个单语种teacher指导multilingual student，student比teacher好时不学teacher

[Execution-Guided Neural Program Synthesis](https://openreview.net/forum?id=H1gfOiAqYm)
程序代码生成任务，通过执行结果指导语义

[Trellis Networks For Sequence Modeling](https://openreview.net/forum?id=HyeVtoRqtQ)
该文提出了一种trellis network，实际上就是一个temporal convolutional network。文中证明了trellis network在一定条件下等价于截断的RNN。在语言模型上验证了效果。

[Representation Degeneration Problem in Training Natural Language Generation Models](https://openreview.net/forum?id=SkEYojRqtm)
提出“表征退化问题”，用大规模语料训练时embedding表示容易退化，文章增加一个正则约束，在翻译和语言模型上有一定提升

[Universal Transformers](https://arxiv.org/abs/1807.03819)
作者引入了一个新的Active Learning，其中oracle提供了一个部分或弱标签，而不是特定示例的标签。

[Optimal Completion Distillation for Sequence Learning](https://openreview.net/forum?id=rkMW1hRqKX)
文章提出一种独立的算法OCD来优化基于编辑距离的seq2seq模型。该算法快速有效，比各种MLE方法效果更好，在没有语言模型rescore的情况下，在Wall Street Journal数据集上达到CER 3.1%，WER 9.3%，在Librispeech数据集达到WER 4.5%（test-clean）/13.3%（test-other）

[Multilingual Neural Machine Translation With Soft Decoupled Encoding](https://openreview.net/forum?id=Skeke3C5Fm)
提出软解耦编码，在小数据集上无需特殊预处理就能学习到拼写信息和共享的文本信息

[Minimum Divergence Vs. Maximum Margin: An Empirical Comparison On Seq2seq Models](https://openreview.net/forum?id=H1xD9sR5Fm)
比较了分别用最小化离散程度和最大化边缘距离训练的seq2seq模型，提出了新的训练准则，在机器翻译和句子概括上取得不错性能

[Preventing Posterior Collapse With Delta-vaes](https://openreview.net/forum?id=BJe0Gn0cY7)
让VAE中KL loss有一个最小值来防止后验坍缩

[Von Mises-fisher Loss For Training Sequence To Sequence Models With Continuous Outputs](https://openreview.net/forum?id=rJlDnoA5Y7)
做语言生成(language generation)，直接产生word embedding，而不是使用softmax，生成速度更快

[The Relativistic Discriminator: A Key Element Missing From Standard Gan](https://openreview.net/forum?id=S1erHoR5t7)
对GAN进行了改进，提出了relativistic discriminator，提升了训练的稳定性以及生成时的采样质量

[Augmented Cyclic Adversarial Learning for Low Resource Domain Adaptation](https://openreview.net/forum?id=B1G9doA9F7)
关于低资源的domain adaptation，在做对抗训练时，对原来CycleGAN的使用方式做了一些改进


# Network efficience
[Slimmable Neural Networks](https://openreview.net/forum?id=H1gMCsAqY7)
一种可扩展的 网络结构，节点数目越多，计算量越大，性能越高，节点数目越少，计算量越小，性能越低

[Energy-Constrained Compression for Deep Neural Networks via Weighted Sparse Projection and Layer Input Masking](https://openreview.net/forum?id=BylBr3C9K7)
模型压缩的一篇文章

[Snip: Single-shot Network Pruning Based On Connection Sensitivity](https://openreview.net/forum?id=B1VZqjAcYX)
模型剪枝，把dense变sparse来减少计算量

[Context-adaptive Entropy Model For End-to-end Optimized Image Compression](https://openreview.net/forum?id=HyxKIiAqYQ)
该文提出了一种图片压缩的算法，基于内容自适应的压缩，将图像内容分为两部分，一是要消耗额外bit进行压缩的，一种是不用消耗额外bit就可以压缩的。提升了压缩效果。

[Minimal Random Code Learning: Getting Bits Back From Compressed Model Parameters](https://openreview.net/forum?id=r1f0YiCctm)
该文利用了一些新的信息论里的东西对网络进行压缩，提高压缩性能。

[K for the Price of 1: Parameter-efficient Multi-task and Transfer Learning](https://openreview.net/pdf?id=BJxvEh0cFQ)
提出一种新的方法通过重新训练最少（少于2%）的参数，使预训练的神经网络适应新任务。

[A Data-Driven and Distributed Approach to Sparse Signal Representation and Recovery](https://openreview.net/forum?id=B1xVTjCqKQ)
用深度学习来学习稀疏信号表示和恢复

[Relaxed Quantization for Discretized Neural Networks](https://openreview.net/forum?id=HkxjYoCqKX)
这篇文章提出了一项可以用基于梯度训练（ gradient based training ）量化神经网络（ quantized neural networks）的方法

[ProxQuant Quantized Neural Networks via Proximal Operators](https://openreview.net/pdf?id=HyGxNg9SiQ)
参数量化，没看懂

[Analysis of Quantized Models](https://openreview.net/forum?id=ryM_IoAqYX)
参数量化，看不懂

[Double Viterbi: Weight Encoding For High Compression Ratio And Fast On-chip Reconstruction For Deep Neural Network](https://openreview.net/forum?id=HkfYOoCcYX)
结合剪枝和权重量化提出新的稀疏矩阵格式，让稀疏矩阵的解码过程高度并行化

[Rethinking The Value Of Network Pruning](https://openreview.net/forum?id=rJlnB3C5Ym)
审查主流的模型压缩算法，得到一些与常识相悖的结论

[Accumulation Bit-width Scaling For Ultra-low Precision Training Of Deep Networks](https://openreview.net/forum?id=BklMjsRqY7)
关于神经网络的低精度训练，论文提出如何找到合适的精度来进行训练

[Data-Dependent Coresets for Compressing Neural Networks with Applications to Generalization Bounds](https://openreview.net/forum?id=HJfwJ2A5KX)
提出了一种有效的基于coresets的神经网络压缩算法，该算法在保证其输出值与网络输出值近似的情况下，将训练后的神经网络的参数稀疏化，



# New Network Structure
[Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks](https://openreview.net/forum?id=B1l6qiR5F7) best paper
提出一种新的LSTM cell，这个cell的更新准则引入了tree类似的信息来提高LSTM对自然语言信息的处理能力


[Improving Differentiable Neural Computers Through Memory Masking, De-allocation, and Link Distribution Sharpness Control](https://openreview.net/forum?id=HyGEM3C9KQ) (DNC后续)
对deepmind提出的DNC的一个改进，并且大幅提升了bAbI数据的性能


[Quaternion Recurrent Neural Networks](https://openreview.net/forum?id=ByMHvs0cFQ)
bengio实验室的，他们为了解决RNN输入只能是1d，提出了quaternion recurrent neural networ，其输入可以有多个维度特征，并且在asr中验证比普通rnn效果更优


[Learning to Remember More with Less Memorization](https://openreview.net/forum?id=r1xlvi0qYm) MANN后续

[Learning to Screen for Fast Softmax Inference on Large Vocabulary Neural Networks](https://openreview.net/forum?id=ByeMB3Act7)
softmax加速，两层分类

[Rotdcf: Decomposition Of Convolutional Filters For Rotation-equivariant Deep Networks](https://openreview.net/forum?id=H1gTEj09FX)
在DCFNet基础上引入了旋转等边性，增加了稳定性

[Eidetic 3D LSTM: A Model for Video Prediction and Beyond](https://openreview.net/forum?id=B1lKS2AqtX)
作者提出一种3DLSTM结构，用于处理时间空间维度的context信息，对比Spatiotemporal LSTM（在lstm的基础上增加一个表示空间memory的cell，以及与之对应的遗忘门和输入门），在Spatiotemporal LSTM基础上将输入改为3D张量，内部运算用3D conv代替fc

[Learning Implicitly Recurrent CNNs Through Parameter Sharing](https://openreview.net/forum?id=rJgYxn09Fm)
文章提出了一种参数共享的方法，给定一个参数池T，而每一层网络则学习参数池中各组参数的权重，在前向时，对T加权得到w

[Differentiable Learning-to-Normalize via Switchable Normalization](https://openreview.net/forum?id=ryggIs0cYQ)
提出了switchable normalization来优化CNN的训练

[DARTS: Differentiable Architecture Search](https://openreview.net/forum?id=S1eYHoC5FX)
NAS做神经网络搜索的文章，极大地减少了搜索代价，并且免去reinforcement 更新

[Learning concise representations for regression by evolving networks of trees](https://openreview.net/forum?id=Hke-JhA9Y7)
把神经网络结构表示成语法树，进行演化学习

[Padam: Closing the Generalization Gap of Adaptive Gradient Methods in Training Deep Neural Networks](https://openreview.net/forum?id=BJll6o09tm)
引入adaptive gradient来加速网络的训练

[Learning What And Where To Attend With Humans In The Loop](https://openreview.net/forum?id=BJgLg3R9KQ)
提出利用更强的监督信号来学习attention，在图像识别上有更好的效果。

[Antisymmetricrnn: A Dynamical System View On Recurrent Neural Networks](https://openreview.net/forum?id=ryxepo0cFX)
该文将RNN和常微分方程联系到一起，提出了一种新的antisymmetricRNN,实验效果要比一般的LSTM要好。

[Learning Recurrent Binary/Ternary Weights](https://openreview.net/forum?id=HkNGYjR9FX)
文章提出一种高性能的LSTM，权重为二元/三元。能够大大减低实现复杂度

[Big-Little Net: An Efficient Multi-Scale Feature Representation for Visual and Speech Recognition](https://openreview.net/forum?id=HJMHpjC9Ym)
本文提出了一种新的卷积神经网络（cnn）结构来学习多尺度的特征表示，在速度和精度之间取得了很好的平衡。

[Predicting the Generalization Gap in Deep Networks with Margin Distributions](https://openreview.net/forum?id=HJlQfnCqKX)
深度网络对集外数据的扩展性很差，说明CE loss等不是好的generalization的indicator。本文用基于margin distribution的measure，从log空间（margin的积）而非线性空间利用margin的特性而非margin本身。

[Hyperbolic Attention Networks](https://openreview.net/forum?id=rJxHsjRqFQ)
双曲结构的attention 网络

[Spherical CNNs on Unstructured Grids](https://openreview.net/pdf?id=Bkl-43C9FQ)
文章针对不规则网格图像（如全景图像），设计了一种高效的卷积核（对全景图像是球形卷积核）。卷积核用可微分算子的线性组合来替换传统卷积核。可微分算子的参数可以用标准后向传播更新。在非规则网格图像任务，如外形分类、气象模式分割、全方向图像语义分割等，文中提出的方法都超过SOTA或与之相当，同时模型的参数量要小。

[SNAS Stochastic Neural Architecture Search](https://openreview.net/forum?id=rylqooRqK7)
NAS的一种economical实现方法。在保持NAS pipeline的完备性和可微性的同时，在一轮后向传播中同时训练网络参数和结构参数。但是把固定reward的feedbackguo过程替换成generic loss

[Dynamic Channel Pruning: Feature Boosting And Suppression](https://openreview.net/forum?id=BJxh2j0qYm)
通过特征增强和抑制前瞻性地放大显著卷积通道并跳过不重要的通道，保存完整的网络结构，但是能降低计算量

[Equi-normalization Of Neural Networks](https://openreview.net/forum?id=r1gEqiC9FX)
用Sinkhorn-Knopp正则化网络权重，加快收敛速度

[Neural Speed Reading With Structural-jump-lstm](https://openreview.net/forum?id=B1xf9jAqFQ)
关于自然语言处理，提出了一种speed reading model，通过跳过一些token的方式加速RNN处理token序列的速度，并且不影响精度

[Proxylessnas: Direct Neural Architecture Search On Target Task And Hardware](https://openreview.net/forum?id=HylVB3AqYm)
NAS相关，实验效果很棒state-of-the-art，做NAS的话值得关注

# Reinforcement Learning
[Learning Finite State Representations Of Recurrent Policy Networks](https://openreview.net/forum?id=S1gOpsCctm)
和强化学习相关，指出RNN policies比较复杂，很难量化的解释和理解，该文提出了通过有限维的特征，来更好的理解RNN policies。

[A Rotation-equivariant Convolutional Neural Network Model Of Primary Visual Cortex](https://openreview.net/forum?id=H1fU8iAqKX)
从abstract看不是特别明白，大概是提出了一种rotation-equivariant 的卷积网络，能够表现的更好。


[Near-optimal Representation Learning For Hierarchical Reinforcement Learning](https://openreview.net/forum?id=H1emus0qF7)
对强化学习中短期和长期的reward做representation的学习

[Composing Complex Skills by Learning Transition Policies](https://openreview.net/forum?id=rygrBhC5tQ)
作者出发点是人类在做一系列复杂的技能时都是condition on前面所学的技能，并且会学习如何从前面的技能直接进行跳转，称之为"transition skills"，作者定义了类似的学习方式来进行reinforece training

[Exploration by random network distillation](https://openreview.net/forum?id=H1lJJnR5Ym)
作者定义了一种exploration bonus，这个可以解决很多雅达利游戏中的问题，在 Montezuma's Revenge取得了超过人类的水平，而这种exploaration bonus来源很简单，就是一个随机初始化,fix的神经网络

[Probabilistic Recursive Reasoning for Multi-Agent Reinforcement Learning](https://openreview.net/forum?id=rkl6As0cF7)
作者在多智能体强化学习领域，提出了一种新的基于概率体系的递归推理

[Learning to Navigate the Web](https://openreview.net/forum?id=BJemQ209FQ)
作者要解决的问题是网页导航，使用的方法是基于reward增强、课程表学习、meta learning的强化学习方法

[Variance Reduction for Reinforcement Learning in Input-Driven Environments](https://openreview.net/forum?id=Hyg1G2AqtQ)
作者通过对强化学习中input的一些限制来提高强化学习中提的方差，使得强化学习更加容易训练

[ProMP: Proximal Meta-Policy Search](https://openreview.net/forum?id=SkxXCi0qFX)
一种基础并且新的meta+reinforcement learning方法，个人不看好，两个都没谱的东西和到一起不可能靠谱的

[Learning Self-Imitating Diverse Policies](https://openreview.net/forum?id=HyxzRsR9Y7)
文章的一个特色是policy的优化来自good roll out，从而保证了policy的多样性

[Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/forum?id=r1lyTjAqYX)
把recurrent neural networks 和experience replay结合起来，打雅达利游戏效果很厉害


[Large-Scale Study of Curiosity-Driven Learning](https://openreview.net/forum?id=rJNwDjAqYX)
用curiosity来训练智能体而不是reward，搞不清这两者有何本质区别，还是说叫法不同

[Diversity is All You Need: Learning Skills without a Reward Function](https://openreview.net/forum?id=SJx63jRqFm)
作者提出强化学习不需要使用reward来进行训练，而是使用diversity这种指标

[Learning to Schedule Communication in Multi-agent Reinforcement Learning](https://openreview.net/forum?id=SJxu5iR9KQ)
多智能体的强化学习训练


[Information-Directed Exploration for Deep Reinforcement Learning](https://openreview.net/forum?id=Byx83s09Km)
文章证明强化学习的偏差由于受状态和action的影响存在异方差性，传统的探索策略解决不好这个问题，文章定义了regret-information ratio，是一个比率，分子是选择action与最优action的reward差，分母没太理解

[Woulda, Coulda, Shoulda: Counterfactually-Guided Policy Search](https://openreview.net/forum?id=BJG0voC9YQ)
强化学习这块的一个比较偏理论前沿的研究探索工作

[Solving the Rubik's Cube with Approximate Policy Iteration](https://openreview.net/forum?id=Hyfn2jCcKm)
用强化学习解决魔方问题

[Hindsight policy gradients](https://openreview.net/forum?id=Bkg2viA5FQ)
在强化学习中，目标可能是随环境变化的，或者说在通往最终目的前存在许多子目标，而这些子目标与最终目标的差异会模型探索不到最优策略；这要求模型对近期reward不要那么敏感，文章提出一种方法解决该问题

[Episodic Curiosity through Reachability](https://openreview.net/forum?id=SkeK3s0qKQ)
RL中鼓励探索改进reward稀疏问题

[StrokeNet: A Neural Painting Environment](https://openreview.net/forum?id=HJxwDiActX)
strokenet，基于RL画字符

[Discriminator-Actor-Critic: Addressing Sample Inefficiency and Reward Bias in Adversarial Imitation Learning](https://openreview.net/forum?id=Hk4fpoA5Km)
RL的A-C基础上引入D

[Supervised Policy Update for Deep Reinforcement Learning](https://openreview.net/forum?id=SJxTroR9F7)
RL有监督学习策略更新，有些意思

[Deep Reinforcement Learning With Relational Inductive Biases](https://openreview.net/forum?id=HkxaFoC9KQ)
本文主要是提出了一种关系推断机制的深度强化学习，提高了RL的学习效率和效果。

[Recall Traces: Backtracking Models For Efficient Reinforcement Learning](https://openreview.net/forum?id=HygsfnR9Ym)
利用backtracking model来提高reinforcement learning的训练效率。

[Dom-q-net: Grounded Rl On Structured Language](https://openreview.net/forum?id=HJgd1nAqFX)
提出了一种DOM-Q-NET来提高强化学习在网页导航上的应用。

[Value Propagation Networks](https://openreview.net/forum?id=SJG6G2RqtX)
该文和reinforcement learning 有关，基于Value Iteration提出了Value Propagation,能够高效的训练强化学习。

[Contingency-aware Exploration In Reinforcement Learning](https://openreview.net/forum?id=HyxGB2AcY7)
该文属于强化学习，应该是解决了一个关于环境中contingency-aware的问题，能够有利于更好的探索。

[NADPEx: An on-policy temporally consistent exploration method for deep reinforcement learning](https://openreview.net/forum?id=rkxciiC9tm)
本文介绍了一种新的策略-神经自适应辍学策略探索（nadpex）-用于深层强化学习。

[The Laplacian in RL: Learning Representations with Efficient Approximations](https://openreview.net/forum?id=HJlNpoA5YQ)
文章提出了一种可扩展的方法学习强化学习中上下文的拉普拉斯特征向量。可以提升强化学习的效果

[CEM-RL: Combining evolutionary and gradient-based methods for policy search](https://openreview.net/forum?id=BkeU5j0ctQ)
本文将现有的两种优化方法CEM（ cross-entropy
metho）/CMA-ES和DDPG/TD3（ Deep Deterministic Policy Gradient）结合起来进行策略优化

[Marginal Policy Gradients: A Unified Family of Estimators for Bounded Action Spaces with Applications](https://openreview.net/forum?id=HkgqFiAcFm)
本文Policy Gradient过程中，减小梯度方差法，该方法适用于有向和限幅action，并保证梯度是低方差的。

[Unsupervised Control Through Non-Parametric Discriminative Rewards](https://openreview.net/forum?id=r1eVMnA9K7)
文章提出了一种无监督的学习算法来训练agent，使其仅使用一系列的观察和动作来实现感知指定的目标。

[Directed-Info Gail Learning Hierarchical Policies from Unsegmented Demonstrations Using Directed Information](https://openreview.net/forum?id=BJeWUs05KQ)
大多数任务都可以分解成若干简单的子任务。作者提出一个模仿学习框架学习子任务的策略，同时学习宏观策略来切换子任务。

[Emergent Coordination through Competition](https://openreview.net/forum?id=BkG8sjR5Km)
在多agent游戏中population-based training（PBT）可以产生合作行为，

[Learning to Understand Goal Specifications by Modelling Reward](https://openreview.net/forum?id=H1xsSjC9Ym)
用reward model输出reward，reward model的训练方式类似于GAN中的discriminator。文章提出的AGILE-A3比A3C更容易学会完成任务，且成功率更高，但是比A3C-RP（Reward Prediction）慢

[Off-Policy Evaluation and Learning from Logged Bandit Feedback: Error Reduction via Surrogate Policy](https://arxiv.org/abs/1808.00232)
在learning from batch of logged bandit feedback问题中，提出一种从logged action-context pairs中估计出一个最大似然surrogated policy作为proposal，而非直接选取历史policy。

[Hierarchical RL Using An Ensemble of Proprioceptive Periodic Policies](https://openreview.net/forum?id=SJz1x20cFQ)
训练agent学习high/low level分层的policy，low level policy学习修改internal, proprioceptive dimensions，high level policy修改non-proprioceptive dimensions。用这种学习方法，可以在复杂的迷宫问题、稀疏reward的航线问题和类人agent问题等上超越其他分层方法。

[Stable Opponent Shaping In Differentiable Games](https://openreview.net/forum?id=SyGjjsC5tQ)
在多玩家非凸强化学习中，通过Stable Opponent Shaping改进Opponent-Learning Awareness的收敛效果

[Reward Constrained Policy Optimization](https://openreview.net/forum?id=SkfrvsA9FX)
在策略优化中提出奖励的多时间尺度约束



# Representation Learning
[Understanding And Improving Interpolation In Autoencoders Via An Adversarial Regularizer](https://openreview.net/forum?id=S1fQSiCcYm)
文章研究的是VAE的差值问题，比如mnist里面的2和3，在Z domain里面，做差值，可以生成2和3之间的一些图，文章在Z domain加了一个GAN这样的loss来约束，使得差值生成的图像更加自然


[Meta-Learning Update Rules for Unsupervised Representation Learning](https://openreview.net/forum?id=HkNDsiC9KQ)
文章主要出发点想解决在unsupervised learning阶段提取的representation与后续downstream tasks没有关联，作者期望通过meta-learning的方式，使得unsupervised learning与target任务产生关联，取得更好的representation


[Unsupervised Learning Of The Set Of Local Maxima](https://openreview.net/forum?id=H1lqZhRcFm)
文章定义了1个classifier和value function，每个sample都认为是value函数的 local maximum，这个可以认为是正例，负例是这个sample的领域，优化过程中使得c区分是local maximum的正例还是反例，value函数又打分来保证local maximum，构造类似对抗的训练信号


[Learning Robust Representations By Projecting Superficial Statistics Out](https://openreview.net/forum?id=rJEjjoR9K7)
作者为了解决domain不匹配的问题，首先提出了superficial feature的概念，并且用一些prior知识定义了GLCM的一类superficial feature，并提出了2中解决思路：1个是用GLCM特征的反向梯度来迫使模型学习superficial feature不易学到的特征；2是将模型提取的representation映射到跟GLCM正交的表征空间去


[Automatically Composing Representation Transformations as a Means for Generalization](https://openreview.net/forum?id=B1ffQnRcKX)
这个文章思路挺宏伟的，首先把一个问题，拆解成很多小问题，每个小问题可能已经有一个model来建模了，当来了一个新的问题，就可以把这个问题拆解成已经有轮子的小问题，这个拆解过程是可学习的(学习的监督信号来自跟这个问题比较类似的之前问题)，这样小问题的model就可以重用，从而实现knowledge的传承


[Learning Deep Representations By Mutual Information Estimation And Maximization](https://openreview.net/forum?id=Bklr3j0cKX)
MINE论文，是bengio实验室期望使用互信息做representation learning的基石文章


[On the Minimal Supervision for Training Any Binary Classifier from Only Unlabeled Data](https://openreview.net/forum?id=B1xWcj0qYm)
Empirical risk minimization 相关的一篇文章，说是使用无监督的数据做2分类，还不清楚这玩意是干嘛用的


[Dimensionality Reduction for Representing the Knowledge of Probabilistic Models](https://openreview.net/forum?id=SygD-hCcF7)
作者出发点很简单，我们现在deepmodels中间特征维度很高，很容易过拟合。作者介绍了一种降维的方法来提高模型的泛化能力


[Transferring Knowledge across Learning Processes](https://openreview.net/forum?id=HygBZnRctX)
作者提出了“Leap"的方法，在训练过程中进行transfer learning

[Label super-resolution networks](https://openreview.net/forum?id=rkxwShA9Ym)
超分辨图像任务，把超分辨弄成低分辨，并且利用地分辨数据，来提升高分辨的任务效果

[Learning Latent Superstructures in Variational Autoencoders for Deep Multidimensional Clustering](https://openreview.net/forum?id=SJgNwi09Km)
作者声称训练了一个VAE，其中在latent space中的feature有一些superstructure特征

[Time-Agnostic Prediction: Predicting Predictable Video Frames](https://openreview.net/forum?id=SyzVb3CcFX)
对于timporal信号，让模型能够有选择能力，在比较自信的位置再做预测

[Deep Online Learning Via Meta-Learning: Continual Adaptation for Model-Based RL](https://openreview.net/forum?id=HyxAfnA5tm)
关于连续学习也就是在线连续自适应（输入数据是非平稳分布），文章似乎是用了多模型混合，并使用中餐厅过程来自洽的训练混合模型

[Meta-Learning Probabilistic Inference for Prediction](https://openreview.net/forum?id=HkxStoC5F7)
文章提出了一种新的meta-learning的框架，具体细节没看，感觉挺有意思，后面再细看

[Learning a Meta-Solver for Syntax-Guided Program Synthesis](https://openreview.net/forum?id=Syl8Sn0cK7)
meta-learning的一篇程序生成？

[What do you learn from context? Probing for sentence structure in contextualized word representations](https://openreview.net/forum?id=SJzSgnRcKX)
对近期NLP表征学习的方法进行了分析

[Modeling Uncertainty with Hedged Instance Embeddings](https://openreview.net/forum?id=r1xQQhAqKX)
主流的表示学习，都是将实例（图像、词...）映射到一个embedding space，作者认为在embedding space空间中关系是确定的，在实际问题中很多不确定、模糊的关系。作者针对该问题提出了HIB（引入随机变量的embed）具体方法没细看

[ARM: Augment-REINFORCE-Merge Gradient for Stochastic Binary Networks](https://openreview.net/forum?id=S1lg0jAcYm)
对离散隐变量梯度的无偏估计，并且能够有效地降低variance

[Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference](https://openreview.net/forum?id=B1gTShAct7)
连续学习在向新目标迁移和引入干扰上存在trade-off（对新样本多度学习会带来偏差overfit，反之模型更新缓慢，无法适应新环境），作者提出一种新算法MER来解决该问题，细节没看

[Reasoning About Physical Interactions with Object-Oriented Prediction and Planning](https://openreview.net/forum?id=HJx9EhC9tQ)
文章提出一种方法对物体结构进行理解或者拆解，不使用物体结构监督信号，而通过拆解生成的无监督过程来学习物体结构信息

[Unsupervised Domain Adaptation for Distance Metric Learning](https://openreview.net/forum?id=BklhAj09K7)
文章提出一种无监督迁移学习的方法，定义了FTN网络：source与source trans的CE loss，即希望原领域表征以及变换后的表征都有区分性，同时，增加对抗loss1，区分source表征与target表征和source trans表征的 ，对抗loss2，不能区分target表征和source trans表征

[Unsupervised Speech Recognition via Segmental Empirical Output Distribution Matching](https://openreview.net/forum?id=Bylmkh05KX)
文章提出一种两步交替训练的无监督方法：音素预测和边界预测；首先给定边界预测音素，使用了ODM和一个平滑的组合loss，ODM loss是使得语言模型概率越高的预测路径概率越高（loss 公式看上去计算量很大）；其次是给定预测模型来优化边界，用MAP准则优化，实际是在根据语言模型概率跳转概率来学习合适的自跳概率（自跳和语言模型跳转是trade-off关系，因此可以优化至平衡）

[SOM-VAE: Interpretable Discrete Representation Learning on Time Series](https://openreview.net/forum?id=rygjcsR9Y7)
基于VAE的表征学习

[Variational Autoencoders with Jointly Optimized Latent Dependency Structure](https://openreview.net/forum?id=SJgsCjCqt7)
VAE联合优化隐含表示结构

[Learning Factorized Multimodal Representations](https://openreview.net/forum?id=rygqqsA9KX)
主要是针对多模态学习提出了一种基于generative-discriminative 目标函数的方法，学习模态之间的区分性和具体模态的特征。

[Unsupervised Learning via Meta-Learning](https://openreview.net/forum?id=r1My6sR9tX)
这篇文章提出一种无监督的学习方法，它使用meta-learning来实现对下游图像分类任务的有效学习，优于最先进的方法。

[Active Learning with Partial Feedback](https://openreview.net/forum?id=HJfSEnRqKQ)
作者引入了一个新的Active Learning，其中oracle提供了一个部分或弱标签，而不是特定示例的标签。

[Meta-Learning For Stochastic Gradient MCMC](https://openreview.net/forum?id=HkeoOo09YX)
传统SG-MCMC计算复杂，用meta-learning自动实现SG-MCMC

[Latent Convolutional Models](https://openreview.net/forum?id=HJGciiR5Y7)
用latent model实现image restoration

[AUTOLOSS Learning Discrete Schedule for Alternate Optimization](https://openreview.net/forum?id=BJgK6iA5KX)
文章提出了一种meta-learning框架AutoLoss来自动学习和决定optimization schedule。作者把AutoLoss应用于d元二次方程回归、MLP、GAN生成图像、多任务NMT等四个任务上，AutoLoss可以学习到更好的optimization schedule，并获得更好的收敛结果。

[Differentiable Perturb-And-Parse Semi-Supervised Parsing with A Structured Variational Autoencoder](https://openreview.net/forum?id=BJlgNh0qKQ)
文章提出一种半监督的latent-variable generative句法解析模型，模型可以视为一个VAE。模型可以通过扰动候选关系的权重和可微分动态规划的structured argmax inference来获取一个近似的采样样本。

[Measuring Compositionality In Representation Learning](https://openreview.net/forum?id=HJz05o0qK7)
衡量真实产生表征的模型能被一个显式组合原始表征的模型近似到何种程度，来评价真实表征的组合性

[Regularized Learning For Domain Adaptation Under Label Shifts](https://openreview.net/forum?id=rJl0r3R9KX)
标签移位的正则学习，先用标注源域数据和非标注目标域估计重要性权重，然后用加权的源域数据训练分类器，并估计分类器在目标域的泛化边界

[Delta: Deep Learning Transfer Using Feature Map With Attention For Convolutional Networks](https://openreview.net/forum?id=rkgbwsAcYm)
关于迁移学习，效果很好。以往迁移学习更关注源网络的weight，这篇论文更关注源网络和目标网络输出的feature map的差异性。论文认为在所有层中，都会有一些feature map对迁移是有帮助的，另一些feature map是没帮助的，通过Attention的机制来利用这些信息。

[Revealing interpretable object representations from human behavior](https://openreview.net/forum?id=ryxSrhC9KX)
用人类的一些行为来表示图像的语义表征，这些表征能表示对象之间潜在的相似性结构，捕获了人类行为判断中大部分可解释的差异。

[Hierarchical Generative Modeling for Controllable Speech Synthesis](https://openreview.net/forum?id=rygkk305YQ)
作者描述了一种使用变分自编码方法学习韵律内部表示的方法。可以对说话风格、噪声条件等进行细粒度控制



# 有意思的小现象
[An Empirical Study Of Example Forgetting During Deep Neural Network Learning](https://openreview.net/forum?id=BJlxm30cKm)
文章从一个非常有意思的角度出发，研究训练集中的forgeten/unforgetten样本，发现对于分类重要的unforgetten样本非常重要，哪些fogetten样本，即使label有误也不会影响模型的鲁棒性和最终的泛化性能，很多reviewer对文章评价都非常高，值得仔细看，默默看了作者，是bengio实验室和carnegie的做的，大厂文章确实值得佩服



[Identifying And Controlling Important Neurons In Neural Machine Translation](https://openreview.net/forum?id=H1z-PsR5KX)
作者首先抛出一个问题，在机器翻译中，重要的linguistic 信息是全自由分布的还是更侧重于某些节点；作者提出一种无监督发现重要节点的方法，并且通过控制这些节点的激活来控制机器翻译的结果


[Robustness May Be at Odds with Accuracy](https://openreview.net/forum?id=SyxAb30cY7)
作者发现，用对抗样本训练出来较为鲁邦的模型，在常规测试集上面可能出现性能下降的问题，但是除了能处理对抗样本外，模型有一些比较有意思，接近人的感知的一些优势


[The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://openreview.net/forum?id=rJl-b3RcF7) best paper
作者从“模型剪枝”这个问题出发，对于通常训练模型的随机初始化，我们并不清楚哪些连接是重要的，所以把初始模型弄得很大，这样来增大“中奖”的概率，从而获得较好的模型，而模型剪枝就把那些不重要的连接干掉。但是一开始如果初始化一个较小的网络，是无法完成此工作的。作者通过初始化“重要连接”，网络虽然小，但是认为这些连接都是“高概率中奖"的，也能收获很好的性能


[Critical Learning Periods In Deep Networks](https://openreview.net/forum?id=BkeStsCcKQ)
作者从一个很有意思的角度分析了深度学习中的一个现象，模型初始几个epoch的性能对模型最终收敛效果影响非常大，初识节点建立起来的“强连接”，会影响模型所有的后续训练，所以提出了"遗忘"的重要性

[ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness](https://openreview.net/forum?id=Bygh9j09KX)
作者发现image-net训练的网络更倾向于使用纹理来判断物体，而我们人类实际上对于形状来判断物体更加信任些，作者提出了"stylized-imagenet"来解决此问题，在object detection上面取得了更加好的效果

[Small nonlinearities in activation functions create bad local minima in neural networks](https://openreview.net/forum?id=rke_YiRct7)
作者说他发现在激活函数中，即使有略微的非线性，也能够是的模型陷入坏的局部最优点，这有点扯蛋吧、、

[On the loss landscape of a class of deep neural networks with no bad local valleys](https://openreview.net/forum?id=HJgXsjA5tQ)
作者提出一类网络，这类网络没有不好的谷点，也就是不存在次优解，不会陷入局部最优

[Approximating Cnns With Bag-of-local-features Models Works Surprisingly Well On Imagenet](https://openreview.net/forum?id=SkfMWhAqYQ)
比较有意思，提出了利用bag-of-local features也能够在大规模的图像分类上实现很好地效果。

[Deep, Skinny Neural Networks are not Universal Approximators](https://openreview.net/forum?id=ryGgSsAcFQ)
本文证明了无论神经网络有多深，某些函数也不能完全近似

[Bias-Reduced Uncertainty Estimation for Deep Neural Classifiers](https://openreview.net/forum?id=SJfb5jCqKm)
通过未收敛模型的不确定性（抖动）来估计最终分类模型给出的高置信度点的不确定性

[Adaptive Estimators Show Information Compression in Deep Neural Networks](https://openreview.net/forum?id=SkeZisA5t7)
本文提出一种DNN的互信息估计方法，并用它来观察非饱和激活网络的压缩

[Detecting Memorization in ReLU Networks](https://openreview.net/forum?id=HJeB0sC9Fm)
观察发现，在ReLU网络中，推广性好的模型对相似输入的响应更接近于线性，而记住训练数据（过拟合）的模型对相似输入的响应更接近于非线性。作者对每层的响应做非负矩阵分解，来分析模型记忆训练数据的程度。用此方法可以指导何时做early stopping。

[From Hard To Soft: Understanding Deep Network Nonlinearities Via Vector Quantization And Statistical Inference](https://openreview.net/forum?id=Syxt2jC5FX)
从VQ的角度对神经网络中常见的一些非线性激活函数进行了分析

[On The Sensitivity Of Adversarial Robustness To Input Data Distributions](https://openreview.net/forum?id=S1xNEhR9KX)
作者发现一个有关对抗训练的有趣现象，对抗的鲁棒性很容易受输入数据分布的影响

[interpretability, natural language processing, computer vision](https://openreview.net/forum?id=SkEqro0ctQ)
dnn效果很好，但仍是个黑盒不好理解。作者利用分层放法ACD来解释DNN的预测输出， ACD将DNN的预测分为层级的聚类及每个聚类的对预测的贡献度


# knowledeg graph/GNN
[Supervised Community Detection With Line Graph Neural Networks](https://openreview.net/forum?id=H1g0Z3A9Fm)
作者用GNN来解决“communit detection"的问题，并且整个训练过程是一个有监督训练，并且为了解决实际应用场景的效率和规模问题，作者号称使用了增强的GNN，line GNN，并且这类简化后GNN也能够取得不错的效果

[Smoothing the Geometry of Probabilistic Box Embeddings](https://openreview.net/forum?id=H1xSNiRcF7)
一篇结构化embedding的论文，主要做entailment的，对此领域了解少，不知道干嘛用

[Diffusion Scattering Transforms on Graphs](https://openreview.net/forum?id=BygqBiRcFQ)
GNN的一篇文章，介绍一种散射变换表征方法，不知道有啥牛逼之处

[Learning Localized Generative Models for 3D Point Clouds via Graph Convolution](https://openreview.net/forum?id=SJeXSo09FQ)
一个使用GNN提取特征的GAN网络，做一些3d generation

[RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://openreview.net/forum?id=HkgEQnRqYQ)
任务为知识图谱补全，主要是将图谱的节点与关系映射到一个表示空间。作者提出将其映射到一个复数空间，这样可以点与点之间的关系很容易定义成一个旋转，并且在复数空间使用点乘即可完成旋转操作

[LanczosNet: Multi-Scale Deep Graph Convolutional Networks](https://openreview.net/forum?id=BkedznAqKQ)
GCN+multiscale

[How Powerful are Graph Neural Networks? ](https://openreview.net/forum?id=ryGs6iA5Km)
GNN中一篇比较经典的文章，挺有意思的

[Deep Graph Infomax](https://openreview.net/forum?id=rklz9iAcKQ)
在图上做无监督学习

[Graph Hypernetworks For Neural Architecture Search](https://openreview.net/forum?id=rkgW0oA9FX)
网络搜索的文章，主要是解决网络搜索效率问题，提出了Graph HyperNetwork（GHN）,大致也是在网络空间搜索然后利用GHN在validation set上测试效果找搜索方向。效率大大的提高。

[Invariant and Equivariant Graph Networks](https://openreview.net/forum?id=Syx72jC9tm)
图网络的一篇偏理论一些的研究

[DyRep: Learning Representations over Dynamic Graphs](https://openreview.net/forum?id=HyePrhR5KX)
学习动态图的语义表征

[Building Dynamic Knowledge Graphs from Text Using Machine Reading Comprehension](https://openreview.net/forum?id=S1lhbnRqF7)
用machine reading comprehension（MRC）模型做知识图谱

[Structured Neural Summarization](https://openreview.net/forum?id=H1ersoRqtm)
将sequence models和graph model结合起来用于Summarization任务，比纯sequence models或者纯graph model效果都要好


# 鲁棒性，对抗样本
[Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://openreview.net/forum?id=HJz6tiCqYm)
加入扰动来提升模型鲁棒性

[Evaluating Robustness of Neural Networks with Mixed Integer Programming](https://openreview.net/forum?id=HyGIdiRqtm)
作者设计了10000层relu来验证模型的鲁棒性，并且找到一种快速获得对抗样本的方法

[Prior Convictions: Black-box Adversarial Attacks with Bandits and Priors](https://openreview.net/forum?id=BkMiWhR5K7)
使用priors来进行黑盒对抗样本攻击

[Are adversarial examples inevitable?](https://openreview.net/forum?id=r1lWUoA9FQ)
文章从理论层面分析了对抗样本是不是不可避免的，结论是对于确定的分类问题，无法完全避免对抗样本

[A Statistical Approach to Assessing Neural Network Robustness](https://openreview.net/forum?id=S1xcx3C5FX)
使用统计学方法来增加模型鲁棒性的文章

[Phase-aware Speech Enhancement With Deep Complex U-net](https://openreview.net/forum?id=SkeRTsAcYm)
该文主要是提出了基于U-net的complex masking的方法来增强语音，并给出了一个有效的很想相位估计的损失函数。

[Janossy Pooling: Learning Deep Permutation-invariant Functions For Variable-size Inputs](https://openreview.net/forum?id=BJluy2RcFm)
本文提出一种Jannosy pooling，通过对扰动敏感函数集的pooling，来构造扰动不敏感的函数。从理论和实验证明了效果。

[Query-efficient Hard-label Black-box Attack: An Optimization-based Approach](https://openreview.net/forum?id=rJlk6iRqKX)
提出了一种攻击黑盒的对抗样本方法，并且黑盒只提供hard label。大致是将hard-label的攻击转为real-value的优化问题。

[Do Deep Generative Models Know What They Don't Know?](https://openreview.net/forum?id=H1xwNhCcYm)
这篇文章指出以前大家认为生成模型要比判别模型的鲁邦性能好，即能更有效地抵抗对抗样本之类的输入，但是这篇文章发现生成模型也有相应的问题。

[The Limitations Of Adversarial Training And The Blind-spot Attack](https://openreview.net/forum?id=HylTBhA5tQ)
该文指出了目前adversarial training的缺点，如果对抗样本存在于流形上的盲点的话，模型很容易受到攻击。并提出了一种对抗样本方法blind-spot。

[Adversarial Attacks On Graph Neural Networks Via Meta Learning](https://openreview.net/forum?id=Bylnx209YX)
提出用meta-learning对图神经网络进行对抗攻击。

[Training For Faster Adversarial Robustness Verification Via Inducing Relu Stability](https://openreview.net/forum?id=BJfIVjAcKm)
该文通过relu stability不仅提高了模型对于对抗样本的鲁棒性，还能够比较快速的确定模型具有鲁棒性。

[Neural network gradient-based learning of black-box function interfaces](https://openreview.net/forum?id=r1e13s05YX)
训练一个神经网络去学习“黑盒系统”，训练完成后，该网络可代替该黑盒系统

[Structured Adversarial Attack: Towards General Implementation and Better Interpretability](https://openreview.net/forum?id=BkgzniCqY7)
本文提出了一种针对DNN的对抗性攻击方法（StrAttack），旨在利用图像的底层结构。具体地说，在对抗性样本的生成过程中引入群稀疏正则化，并使用基于ADMM的实现来生成对抗性扰动。

[Defensive Quantization: When Efficiency Meets Robustness](https://openreview.net/forum?id=ryetZ20ctX)
文章设计了一种新的量化方法，同时优化深度学习模型的效率和鲁棒性。

[There Are Many Consistent Explanations of Unlabeled Data: Why You Should Average](https://openreview.net/forum?id=rkgKBhA5Y7)
文章指出权重加权平均有助于提高泛化性能

[Detecting Adversarial Examples via Neural Fingerprinting](https://openreview.net/forum?id=SJekyhCctQ)
从真实数据中编码fingerprint patterns，这些patterns可以表征模型对真实数据期望行为，因此可以用于检测对抗样本。该方法达到98-99%的ROC AUC。

[Minimal Images in Deep Neural Networks Fragile Object Recognition in Natural Images](https://openreview.net/pdf?id=S1xNb2A9YX)
minimal image指人能识别出物体的最小的图像区域（只有部分物体）。文章分析了DNN与人类对FRI（微小变化就会极大影响识别率的minimal image）的不同表现，并使用数据增强、正则化和大尺度pooling等方法缩小DNN与人类的gap。

[Adv-bnn: Improved Adversarial Defense Through Robust Bayesian Neural Network](https://openreview.net/forum?id=rk4Qso0cKm)
针对对抗攻击的新训练方法，将盲添加随机性的方式改成用贝叶斯神经网络对随机性建模，并在对抗攻击下最优化BNN的min-max问题来学习最佳模型分布

[Adef: An Iterative Algorithm To Construct Adversarial Deformations](https://openreview.net/forum?id=Hk4dFjR5K7)
新的对抗攻击算法，用梯度上升的方式迭代地微变形图像

[Peernets: Exploiting Peer Wisdom Against Adversarial Attacks](https://openreview.net/forum?id=Sk4jFoA9K7)
将传统欧几里得卷积替换成图卷积来利用伙伴图的信息，对抗健壮性提升了3倍

[L2-nonexpansive Neural Networks](https://openreview.net/forum?id=ByxGSsR9FQ)
提出了一种正则化方法，可以提升模型的鲁棒性，可以抵御对抗攻击

[Verification Of Non-linear Specifications For Neural Networks](https://openreview.net/forum?id=HyeFAsRctQ)
作者提出了一种新的验证网络鲁棒性的方法：convex-relaxable specifications可以模拟许多预测任务的specifications （如能量守恒原则、语义一致性...）

[Characterizing Audio Adversarial Examples Using Temporal Dependency](https://openreview.net/forum?id=r1g4E3C9t7)
时间依赖性可以增加音频对对抗性样本的识别能力，并且可以抵抗实验中的自适应攻击


# 理论研究论文
[Towards Robust, Locally Linear Deep Networks](https://openreview.net/forum?id=SylCrnCcFX)
这是一篇偏理论的论文，文章主要期望解决深度学习过程中，有很多求导过程并不稳定，或者说在较小值域范围才稳定,文章主要给出，如何扩充这个稳定的值域范围(包括如何优化模型参数)，并给出如何松弛此方法用于实际模型


[Deep Frank-Wolfe For Neural Network Optimization](https://openreview.net/forum?id=SyVU6s05K7)
使用Frank-Wolfe算法来优化神经网络


[Gradient descent aligns the layers of deep linear networks](https://openreview.net/forum?id=HJflg30qKX)
对梯度下降的一堆理论分析，不知道有啥用

[Learning Grid Cells as Vector Representation of Self-Position Coupled with Matrix Representation of Self-Motion](https://openreview.net/forum?id=Syx0Mh05YQ)
比较偏理论研究的一些智能体和grid cells论文，不知道干嘛用的

[Theoretical Analysis Of Auto Rate-tuning By Batch Normalization](https://openreview.net/forum?id=rkxQ-nA9FX)
文章理论分析了BN是怎么样work的，作者认为BN具有调节学习率的作用，使得较少的学习率调节就可以训练很好，证明没看懂

[Learning Mixed-Curvature Representations in Product Spaces](https://openreview.net/forum?id=HJxeWnCcF7)
文章针对非欧空间的embedding，但对于非欧空间大多数数据结构不够规整，因此，提出在内积流形上学习embedding的方法，可以得到一个适应多种结构的曲率空间，细节没看

[Learning Neural PDE Solvers with Convergence Guarantees](https://openreview.net/forum?id=rklaWn0qK7)
用神经网络做PDE，结合NIPS的ODE一起看看可能比较有趣

[An analytic theory of generalization dynamics and transfer learning in deep linear networks](https://openreview.net/forum?id=ryfMLoCqtQ)

[The Comparative Power of ReLU Networks and Polynomial Kernels in the Presence of Sparse Latent Structure](https://openreview.net/forum?id=rJgTTjA9tX)
RELU和多项式核的分析，没意思

[The role of over-parametrization in generalization of neural networks](https://openreview.net/forum?id=BygfghAcYX)
分析一坨过参数化模型，又能取得好泛化性能的方法

[Analyzing Inverse Problems With Invertible Neural Networks](https://openreview.net/forum?id=rJed6j0cKX)
该文通过invertible neural networks来解决Inverse problems，inverse problems可以看做是从度量空间（measurement-space）到参数空间的问题。

[Toward Understanding The Impact Of Staleness In Distributed Machine Learning](https://openreview.net/forum?id=BylQV305YQ)
该文偏理论，主要是讲了分布式机器学习中对延迟影响的理解

[Non-vacuous Generalization Bounds At The Imagenet Scale: A Pac-bayesian Compression Approach](https://openreview.net/forum?id=BJgqqsAct7)
这文章应该是证明了一个bound，偏理论，和网络压缩有关系。

[Understanding Straight-through Estimator In Training Activation Quantized Neural Nets](https://openreview.net/forum?id=Skh4jRcKQ)
该文偏理论，给出了13年bengio提出的更新量化网络方法straight-through estimator （STE）的证明，并指出STE不是唯一方法，并在小网络上验证了效果。

[Generalized Tensor Models For Recurrent Neural Networks](https://openreview.net/forum?id=r1gNni0qtm)
该文偏理论，主要是用理论证明了RNN的一些性质。

[A Convergence Analysis of Gradient Descent for Deep Linear Neural Networks](https://openreview.net/forum?id=rkxciiC9tm)
文章分析了基于梯度下降的深度线性神经网络收敛性问题

[How Important is a Neuron](https://openreview.net/forum?id=SylKoo0cKm)
这篇文章通过多种比较方法解释了神经元（隐层节点）的重要性。

[Bayesian Deep Convolutional Networks with Many Channels are Gaussian Processes](https://openreview.net/forum?id=B1g30j0qF7)
文章通过实验验证了，具有多通道的贝叶斯深度卷积网络是高斯过程，然而并不知道有啥实际意义

[Adaptivity of deep ReLU network for learning in Besov and mixed smooth Besov spaces: optimal rate and curse of dimensionality](https://openreview.net/forum?id=H1ebTsActm)
这篇文章展示了在Besov空间中带RELU激活的深度神经网络在估计非参数方程的收敛速度

[Doubly Reparameterized Gradient Estimators for Monte Carlo Objectives](https://openreview.net/pdf?id=HkG3e205K7)
IWAE是一种用于估计latent variable model的log likelihood下界的方法。在样本数增加时，其梯度估计表现很差，用DReG则没有这个问题。同时DReG可以降低IWAE，RWS，JVI等方法的梯度的方差

[Universal Stagewise Learning for Non-Convex Problems with Convergence on Adveraged Solutions](https://openreview.net/forum?id=Syx5V2CcFm)
在凸优化问题中，逐步的优化算法理论非常成熟，可以设计出合适的stagewise策略，但是在非凸优化问题中还没有良好的理论。文章提出一种普遍性的逐步的优化算法框架，用于解决非凸优化问题，并在文章中实验了多种变种

[Principled Deep Neural Network Training through Linear Programming](https://openreview.net/forum?id=HkMwHsCctm)
系统分析了常用的模型结构、激活函数、目标函数，可以用线性规划在输入数据、参数空间维度的指数时间和数据集的多项式时间内训练至近乎最优。特别地，给定一个模型结构，可以找出一个多项式时间的方法。

[On the Turing Completeness of Modern Neural Network Architectures](https://openreview.net/forum?id=HyGBdo0qFm)
图灵完备性和计算能力分析，看不懂

[Fluctuation-dissipation Relations For Stochastic Gradient Descent](https://openreview.net/forum?id=SkNksoRctQ)
证明了 fluctuation-dissipation relations for SGD，可用于动态调节学习率，探测loss surface，理论性比较强。



# Sparse
[ALISTA: Analytic Weights Are As Good As Learned Weights in LISTA](https://openreview.net/forum?id=B1lnzn0ctQ)
也是sparse coding相关的文章吧，得分很高，不知道能干嘛

[Dynamic Sparse Graph for Efficient Deep Learning](https://openreview.net/forum?id=H1goBoR9F7)
文章主要讲如何对稀疏图进行压缩，提取特征用于DNN训练

[Sparse Dictionary Learning by Dynamical Neural Networks](https://openreview.net/forum?id=B1gstsCqt7)
Sparse coding的论文，评论都说好有趣，我看不懂，不知道能干嘛用

[Provable Online Dictionary Learning And Sparse Coding](https://openreview.net/forum?id=HJeu43ActQ)
该文提出了Neurally plausible alternating optimization-based online Dictionary learning算法，能够恢复dictionary和组合系数。

# Variational Inference
[The Deep Weight Prior](https://openreview.net/forum?id=ByGuynAct7)
提出了一种更适用于深度网络的先验分布deep weight prior，使得在贝叶斯推断（变分推断）中得到更好的效果

[Function Space Particle Optimization for Bayesian Neural Networks](https://openreview.net/forum?id=BkgtDsCcKQ)
使用基于粒子优化（近期一篇文章提出）的变分推断来解决贝叶斯神经网络的后验推断问题，$p(w|X,Y)$

[Auxiliary Variational MCMC](https://openreview.net/forum?id=r1NJqsRctX)
变分推断的一篇理论文章

[Deterministic Variational Inference for Robust Bayesian Neural Networks](https://openreview.net/forum?id=B1l08oAct7)
贝叶斯网络的确定化推理，没太看懂

[Information Theoretic Lower Bounds On Negative Log Likelihood](https://openreview.net/forum?id=rkemqsC9Fm)
将隐变量模型中的优化先验问题等同为计算率失真函数过程中的变分优化问题，以此推出NLL的下界，从率失真角度将改变先验与优化似然函数等同起来


# Optimizer
["A2BCD: Asynchronous Acceleration with Optimal Complexity"](https://openreview.net/forum?id=rylIAsCqYm)
一篇做异步更新优化的，理论推导很多，并没有DL相关的实验，看这公式看得头大

[Quasi-hyperbolic Momentum And Adam For Deep Learning](https://openreview.net/forum?id=S1fUpoR5FQ)
文章提出了在对momentum与plain sgd加权平均的优化方法，momentum可以降低梯度的方差，但也会使得梯度变得迟钝；QHM保留了momentum的一阶矩，并在一阶矩基础上保留了当前梯度，但该方法本身同样是trade-off

[Local SGD Converges Fast and Communicates Little](https://openreview.net/forum?id=S1g2JnRcFX)
为克服大网络batchsize不能太大的问题，提出多路独立sgd，然后定时平均的方法。具体的没细看

[AdaShift: Decorrelation and Convergence of Adaptive Learning Rate Methods](https://openreview.net/forum?id=HkgTkhRcKQ)
作者就adam难以收敛至最优问题进行了分析，认为是二阶距与梯度存在相关性导致，因此提出了一个去相关的更新方法，方法有一些细节存在疑问

[Riemannian Adaptive Optimization Methods](https://openreview.net/forum?id=r1eiqi09K7)
作者提出基于黎曼自适应的优化方法，并证明其收敛性

[G-SGD: Optimizing ReLU Neural Networks in its Positively Scale-Invariant Space](https://openreview.net/forum?id=SyxfEn09Y7)
文章提出正值不变空间来是的RELU网络更好收敛，设计了一个G-space，在G-space中将传统的优化weight，改为了优化路径（G-space的定义产生作用上存在疑问，这一块没看懂）

[Deep Layers as Stochastic Solvers](https://openreview.net/forum?id=ryxxCiRqYX)
这篇文章主要想把深度学习的网络层跟随机优化算法扯上关系，以此来提升模型性能和设计网络结构

[Knowledge Flow: Improve Upon Your Teachers](https://openreview.net/forum?id=BJeOioA9Y7)
学多个teacher隐层

[Stochastic Optimization of Sorting Networks via Continuous Relaxations](https://openreview.net/forum?id=H1eSS3CcKX)
松弛，随机优化

[signSGD via Zeroth-Order Oracle](https://openreview.net/forum?id=BJe-DsC5Fm)
零阶梯度优化

[Go Gradient For Expectation-based Objectives](https://openreview.net/forum?id=ryf6Fs09YX)
该文针对基于期望的目标函数，提出了a general and one-sample 的梯度方法，有效的应用到不能够reparameterization的离散或者连续参数，并且有着较小的估计方差。。

[Learning Preconditioner On Matrix Lie Group](https://openreview.net/forum?id=Bye5SiAqKX)
本文看着比较有意思，用了一种在Matrix Lie Group上的preconditioner的SGD，指出RMSprop, Adam, batch normalization等，都是他这篇文章的一种特例。

[Complement Objective Training](https://openreview.net/forum?id=HyM7AiA5YX)
有点类似 label smoothing, 还算有意思。本文指出一般的softmax cross entropy只考虑了正确类别的概率，而没有考虑其他类别的概率损失，该文提出了一种complement objective来改善该缺点。

[On the Convergence of A Class of Adam-Type Algorithms for Non-Convex Optimization](https://openreview.net/forum?id=H1x-x309tm)
文章对Adam型算法的收敛性进行了分析，给出了保证算法收敛的一些温和的充分条件，并指出违反这些条件会使算法发散。

[A Mean Field Theory of Batch Normalization](https://openreview.net/forum?id=SyMDXnCcF7)
Mean Field Theory, 用于解释深度网络中的梯度爆炸问题，理论认为梯度爆炸问题可以通过tuning the network close to the linear regime来减轻，同时可以不使用残差连接

[Three Mechanisms of Weight Decay Regularization](https://openreview.net/forum?id=B1lz-3Rct7)
总结weight decay的三个机制：1）增大有效学习率；2）近似地正则化输入-输出的Jacobian norm；3）降低优化器二阶估计的阻尼系数。

[Residual Learning Without Normalization Via Better Initialization](https://openreview.net/forum?id=H1gsz30cKX)
在训deep residual networks时，只要做好初始化，Batch Normalization Layer Normalization这些都不需要

[optimization, generalization, theory of deep learning, SGD, hessian](https://openreview.net/forum?id=SkgEaj05t7)
论文分析了SGD中不同学习率或batch size大小与loss surface之间的变化关系

[DeepOBS: A Deep Learning Optimizer Benchmark Suite](https://openreview.net/forum?id=rJg6ssC5Y7)
深度学习优化领域缺乏标准的基准，本文为神经网络优化器提供了一个简单的基准测试套件


# NLP
[Wizard of Wikipedia: Knowledge-Powered Conversational Agents](https://openreview.net/forum?id=r1l73iRqKm)
作者提出一个数据集，用于基于知识的对话系统，并提供了一套评估方法，以及一个基线方法

[Don't Settle for Average, Go for the Max: Fuzzy Sets and Max-Pooled Word Vectors](https://openreview.net/forum?id=SkxXg2C5FX)
文章提出了一种新的基于词向量表示句子及其相似性度量的方法。具体来说，本文提出了模糊词袋法(FBoW)，与传统词袋法不同在于其包含所有词典词，并根据word vector相似度给了一个度量

[Generative Question Answering: Learning to Answer the Whole Question](https://openreview.net/forum?id=Bkx0RjA9tX)
文章对question、answer的联合概率进行建模，模型变成两部分：answer的先验概率和question的likehood，作者认为用语言模型式的链式建模likehood，对answer更具可解释性

[Efficient Training on Very Large Corpora via Gramian Estimation](https://openreview.net/forum?id=Hke20iA9Y7)
作者使用Gramin Estimian来训练embedding神经网络，并且效率很高

[No Training Required: Exploring Random Encoders For Sentence Classification](https://openreview.net/forum?id=BkgPajAcY7)
随机初始化网络参数可获得句子分类任务不错的结果，

[The Neuro-Symbolic Concept Learner: Interpreting Scenes, Words, and Sentences From Natural Supervision](https://openreview.net/forum?id=rJgMlhRctm)
用神经网络来提取symbolic的表征，做下游的NLP任务

[textTOvec: DEEP CONTEXTUALIZED NEURAL AUTOREGRESSIVE TOPIC MODELS OF LANGUAGE WITH DISTRIBUTED COMPOSITIONAL PRIOR](https://openreview.net/forum?id=rkgoyn09KQ)
基于自回归document矢量表示

[Global-to-local Memory Pointer Networks for Task-Oriented Dialogue](https://openreview.net/forum?id=ryxnHhRqFm)
对话系统的一篇文章

[FLOWQA: Grasping Flow in Histiry for Conversational Machine Comprehension](https://openreview.net/forum?id=ByftGnR9KX)
为了使单轮问答模型能处理对话式阅读理解问题，作者引入FLOW机制，FLOW可以用一个额外并行的结构编码回答历史问题时的中间表示。该方法在CoQA上提升F1分7.2%，在QuAC上提升4.0%。在SCONE的三个domain比其他最优模型的accuracy提高1.8%到4.4%。

[Glue: A Multi-task Benchmark And Analysis Platform For Natural Language Understanding](https://openreview.net/forum?id=rJ4km2R5t7)
通用自然语言理解评估基线，衡量模型特征解决多任务的能力，文中的多任务模型超过了state-of-the-art，但是绝对性能较低


# 目标检测
[Feature Intertwiner for Object Detection](https://openreview.net/forum?id=SyxZJn05YX)
ICLR中少见的检测的论文

[LeMoNADe: Learned Motif and Neuronal Assembly Detection in calcium imaging videos](https://openreview.net/forum?id=SkloDjAqYm)
端到端的一种无监督做检测的文章

[Deep Anomaly Detection With Outlier Exposure](https://openreview.net/forum?id=HyxCxhRcY7)
这篇文章也比较有意思，使用一个叫outlier exposure的方法来做异常检测。


