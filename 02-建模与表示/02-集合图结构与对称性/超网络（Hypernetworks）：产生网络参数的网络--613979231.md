# 【神经网络】超网络（Hypernetworks）：产生网络参数的网络

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2023-03-14  
来源：[https://zhuanlan.zhihu.com/p/613979231](https://zhuanlan.zhihu.com/p/613979231)

[知识库首页](../../README.md)

---

> QMIX和HPN中都用到了超网络，之前我只知道超网络是用来产生参数的，但至于为什么要用超网络，一直没有非常理解，所以今天把《HyperNetworks》阅读了一遍，但其中涉及到一些CNN和RNN的知识，我不是非常了解，所以只能把全文大致过了一遍，简单写了一些个人的理解~~欢迎大家交流讨论

原文：《Hypernetworks》

[[1609.09106] HyperNetworks (arxiv.org)](https://arxiv.org/abs/1609.09106)![](../../assets/images/55f361bfe3a55c98c1d9.jpg)

**超网络（Hypernetworks）**表示用于产生较大规模网络参数的小规模网络，在这一过程中，主网络的作用与其他任意神经网络一样，将输入样本映射到对应的目标值，而超网络的作用则是接收一系列包含主网络参数结构信息的值作为输入然后产生主网络某一层的参数。超网络的思想源于**进化计算（evolutionary computing）**：在一个包含成千上万参数的空间中搜索参数是十分困难的，一种更加高效的方式是用较小的网络产生较大网络的参数，这样搜索空间就会被限制在一个小得多的空间中，超网络的参数和主网络通过**端到端**的方式进行训练。本文以卷积神经网络（convolutional neural networks，CNNs）与循环神经网络（recurrent neural networks，RNNs）探索了利用超网络产生网络参数的原理与优势。

## **Hypernetworks**

作者首先提出，CNN和RNN是两种极端。RNN的不同层之间共享参数，十分灵活，但会遇到**梯度消失**的问题，因此难以训练；而CNN的不同层之间则没有共享参数，因此当网络很深时，会包含需要**冗余参数**。超网络可以看做是一种**非严格参数共享（relaxed weight sharing）**，能够在参数共享与非参数共享之间达到折中。

针对CNN和RNN，本文分别提出了静态超网络（static hypernetworks）和动态超网络（dynamic hypernetworks）。

### **Static Hypernetworks：针对CNN的参数分解**

在标准CNN中，大多数模型参数来源于卷积层中的卷积核，每个核包含 $N_{in}\times N_{out}$ 个滤波器，每个滤波器的维度是 $f_{size}\times f_{size}$。假设每一层的参数被储存在矩阵 $K^j\in\mathbb{R}^{N_{in}f_{size}\times N_{out}f_{size}}$ 中，$j=1,...,D$，$D$ 表示CNN的深度。对于每一层 $j$，可以使用一个超网络来产生 $K^j$，超网络的输入是层嵌入 $z^j\in\mathbb{R}^{N_z}$，即 $K^j$ 可以表示为



$$
K^j=g(z^j),\ \forall j=1,...,D \\
$$



$K^j$ 可以划分为 $N_{in}$ 个维度为 $f_{size}\times N_{out}f_{size}$ 的小矩阵切片，每个核的每个切片可以记为 $K^j_i\in\mathbb{R}^{f_{size}\times N_{out}f_{size}}$。因此，作者提出将超网络构造成一个两层的线性网络，首先第一层利用 $N_{in}$ 个不同的矩阵 $W_i\in\mathbb{R}^{d\times N_z}$ 和偏置向量 $B_i\in\mathbb{R}^d$ 将 $z^j$ 线性投射成 $N_{in}$ 个隐层特征 $a^j_i\in\mathbb{R}^d$，这里 $d$ 表示超网络隐藏层的维度，将 $d$ 设置为 $N_z$。然后第二层利用一个共享的张量 $W_{out}\times\mathbb{R}^{f_{size}\times N_{out}f_{size}\times d}$ 和偏置矩阵 $B_{out}\in\mathbb{R}^{f_{size}\times N_{out}f_{size}}$ 将 $a_i^j$ 线性投射成 $K^j_i$。最终所有的 $K^j_i$ 堆叠起来就行成了 $K^j$。因此 $g(z^j)$ 的计算表达式如下：



$$
\begin{aligned} & a_i^j=W_i z^j+B_i, \quad \forall i=1, \ldots, N_{i n}, \forall j=1, \ldots, D \\ & K_i^j=\left\langle W_{\text {out }}, a_i^j\right\rangle+B_{\text {out }}, \quad \forall i=1, \ldots, N_{\text {in }}, \forall j=1, \ldots, D \\ & K^j=\left(\begin{array}{llllll} K_1^j & K_2^j & \ldots & K_i^j & \ldots & K_{N_{\text {in }}}^j \end{array}\right), \quad \forall j=1, \ldots, D \\ \end{aligned} \\
$$



这里 $\left\langle \right\rangle$ 表示 $W\in\mathbb{R}^{m\times n \times d}$ 与 $a\in\mathbb{R}^d$ 之间的点乘，结果 $\left\langle W, a \right\rangle\in\mathbb{R}^{m\times n}$ 。

在该框架下，可学习的参数有 $W_i$，$B_i$，$W_{out}$，$B_{out}$ 和所有的 $z^j$ 。在学习过程中，模型将层嵌入向量 $z^j$ 作为输入，然后产生主CNN在第 $j$ 层的核参数。此时，超网络的参数总量为 $N_z\times D + d \times (N_z+1)\times N_i+f_{size}\times N_{out}\times f_{size}\times(d+1)$，而主CNN的网络参数总量为 $D\times N_{in}\times f_{size}\times N_{out}\times f_{size}$，可以看出**超网络的参数远远小于主CNN**。

另外，上面假设每一层核的尺寸相同，作者在论文中介绍了如何利用上述模型构造不同尺寸的核，在此不赘述。

### **Dynamic Hypernetworks：针对RNN的动态参数生成**

本节介绍利用一个基于RNN的超网络**HyperRNN**为另一个RNN动态生成参数，这样RNN在每一个时间步的参数就可以不同。在该场景下，超网络被称为动态超网络，它可以被当做是**非严格参数共享（relaxed weight-sharing），即在传统RNN的严格参数共享以及CNN的非参数共享之间折中**。这种非严格的参数共享可以在**模型参数的数量**以及**模型的表征能力**之间达到平衡。

HyperRNN将输入 $x_t$ 和主RNN的隐层特征 $h_{t-1}$ 作为输入，然后输出向量 $\hat{h}_t$，该向量之后会被用于产生同一时间步的主RNN参数，HyperRNN和主RNN通过端到端的方式联合训练。一个基本RNN的标准形式为



$$
h_t=\phi\left(W_h h_{t-1}+W_x x_t+b\right) \\
$$



这里 $h_t$ 表示隐层状态，$\phi$ 表示非线性激活函数，对于输入 $X=\{x_1,x_2,...,x_T\}$ 权重矩阵和偏置 $W_h \in \mathbb{R}^{N_h \times N_h}, W_x \in \mathbb{R}^{N_h \times N_x}, b \in \mathbb{R}^{N_h}$ 在每一个时间步是固定的。

![](../../assets/images/c9dfe8e3303f5679cdee.jpg)

在HyperRNN中，我们可以利用一个更小的超网络产生主RNN在每个时间步的参数，使得 $W_h$ 和 $W_x$ 随时间变化。具体来说，$W_h, W_x, b$ 在每个时间步不同，因此 $h_t$ 可以计算为



$$
\begin{aligned} h_t & =\phi\left(W_h\left(z_h\right) h_{t-1}+W_x\left(z_x\right)+b\left(z_b\right)\right), \text { where } \\ W_h\left(z_h\right) & =\left\langle W_{h z}, z_h\right\rangle \\ W_x\left(z_x\right) & =\left\langle W_{x z}, z_x\right\rangle \\ b\left(z_b\right) & =W_{b z} z_b+b_0 \end{aligned} \\
$$



其中 $W_{h z} \in \mathbb{R}^{N_h \times N_h \times N_z}, W_{x z} \in \mathbb{R}^{N_h \times N_x \times N_z}, W_{b z} \in \mathbb{R}^{N_h \times N_z}, b_0 \in \mathbb{R}^{N_h}$, $z_h, z_x, z_z \in\mathbb{R}^{N_z}$。这里的 $z_h, z_x, z_z$ 用超网络生成：



$$
\begin{aligned} \hat{x}_t & =\left(\begin{array}{c} h_{t-1} \\ x_t \end{array}\right) \\ \hat{h}_t & =\phi\left(W_{\hat{h}} \hat{h}_{t-1}+W_{\hat{x}} \hat{x}_t+\hat{b}\right) \\ z_h & =W_{\hat{h} h} \hat{h}_{t-1}+b_{\hat{h} h} \\ z_x & =W_{\hat{h} x} \hat{h}_{t-1}+b_{\hat{h} x} \\ z_b & =W_{\hat{h} b} \hat{h}_{t-1} \end{aligned} \\
$$



其中 $W_{\hat{h}} \in \mathbb{R}^{N_{\hat{h}} \times N_{\hat{h}}}, W_{\hat{x}} \in \mathbb{R}^{N_{\hat{h}} \times\left(N_h+N_z\right)}, b \in \mathbb{R}^{N_{\hat{h}}}, \text { and } W_{\hat{h} h}, W_{\hat{h} x}, W_{\hat{h} b} \in \mathbb{R}^{N_z \times N_{\hat{h}}}$ ，$b_{\hat{h} h}, b_{\hat{h} x} \in \mathbb{R}^{N_z}$。

HyperRNN可以修改主RNN的权重矩阵和偏置，这样就相当于对激活函数前的输入进行了线性缩放，并且不同时间步与不同输入样本的缩放系数都不同。

## **仿真实验**

### **针对深度残差网络和CIFAR-10识别的实验**

![](../../assets/images/4d3de40213692ecadf4b.jpg)

实验结果表明，在深度残差网络中利用超网络实现非严格参数共享会造成分类准确率下降1.25~1.5%，但优势是网络参数的数量大大减少。准确率下降的原因之一是**深度网络通常会利用不同层捕获不同层次的特征，因此需要不同的滤波器来联合达到最优**。超网络要求主网络的不同层之间存在一定的共性，但也为每个层提供了64个自由度使其与其他层区分开来。尽管利用了超网络之后，主网络无法为每个层学到最优的滤波器参数，但是它会在给定的约束下，得到最优的滤波器，并且需要学习的模型参数数量也大大减小，学习效率提升。

### **针对HyperLSTM与语言建模的实验**

![](../../assets/images/9f579cae1b436b37fd6f.jpg)

可以看出，HyperLSTM的性能优于LSTM，以及更宽更深的LSTM，以及Layer Norm LSTM，这说明通过动态调整权重缩放向量，HyperLSTM能够有效学习到如何缩放激活函数的输入，代价是**训练参数更多**。

## **超网络在MADRL中的应用**

我目前知道的超网络在MADRL算法中的应用有两个，一个是QMIX算法中利用超网络接收全局状态作为输入产生mixing network的网络参数，另一个是HPN算法中利用超网络为每个智能体的观测产生entity-wise的嵌入层。

[XuanAxuan：【多智能体强化学习】[learn to cooperate] QMIX 阅读笔记](../../04-协作系统/01-协作稳定性与信用分配/QMIX 阅读笔记--433670116.md)[XuanAxuan：【多智能体强化学习】HPN：利用超网络实现置换不变性（permutation invariance）与置换同变性（permutation equivariance，PE）](../../04-协作系统/03-规模变化与策略泛化/HPN：利用超网络实现置换不变性（permutation invariance）与置换同变性（permutation equiv--613723150.md)

### **QMIX：利用超网络产生mixing network的网络参数**

在QMIX中，超网络接收全局状态作为输入产生mixing network的网络参数，原因如下：

- QMIX要求mixing network的网络参数非负，利用超网络可以满足这一点；
- Mixing network需要将环境全局状态作为额外的输入，如果直接将状态作为mixing network的输入，就会限制状态与全局Q值之间的关系也单调递增，而把状态作为超网络的输入则不再有此限制；

上面两个理由是原文中提到的，除此之外，使用超网络产生mixing network的参数，这使得每个不同的环境状态都对应着一套不同的mixing network网络参数，也就是说网络参数会随时间动态变化，类似于HyperRNN，提升了网络的表征能力。

### **HPN：利用超网络为观测产生entity-wise的嵌入层参数**

在HPN中，超网络接收观测中每个entity的特征作为输入产生每个entity对应嵌入层的网络参数，从而实现置换不变性，与这种方法相对的有：

- MLP：网络参数固定，不满足置换不变性；
- GNN、Attention、Deep Sets：每个entity共享嵌入层的网络参数，且网络参数固定，满足置换不变性；
- DPN：存在一个固定的参数集，每个entity从该参数集中选择网络参数作为其嵌入层参数，满足置换不变性；

不同于以上几种网络结构，HPN利用超网络接收entity特征作为输入产生嵌入层网络参数，这意味着每个entity的特征与嵌入层参数一一对应，满足置换不变性。相对共享嵌入层参数而言，每个entity的嵌入层参数不同，网络表征能力提升。并且，同一entity的嵌入层参数在不同时刻也不同，说明网络参数是动态变化的，类似于上面所介绍的HyperRNN，进一步提升了网络的表征能力。

这里HPN中的超网络类似于本文中静态超网络和动态超网络的结合，同时实现了空间和时间两个维度上网络参数的差异：1）不同entity之间通过超网络实现非严格参数共享，即通过共同的超网络捕获嵌入层之间的共同性，通过不同的entity特征，实现每个entity独特的嵌入层参数，类似于静态超网络；2）由于不同时刻的观测特征不同，所以每个entity的嵌入层参数在不同时刻也会发生变化，这是时间维度上的非严格参数共享，类似于HyperRNN。

## **总结与个人思考**

这篇文章中超网络主要有两个作用：1. 针对非参数共享的CNN，利用一个较少参数的超网络去产生较多参数的主CNN，这样减少了需要学习的参数数量，代价仅仅是性能的轻微下降；2.针对参数共享的RNN，利用HyperRNN动态生成每个时间步的网络参数，这种方法被称之为非严格参数共享，这样做能够在仅仅增加少量网络参数的情况下，超越更大更深的标准RNN的性能。

在QMIX和HPN两个MADRL算法中，都利用了超网络动态生成网络参数，提升了网络的表征能力；另外，在HPN中，利用超网络实现了观测中不同entity之间的非严格参数共享，相对于Attention、Deep Sets等严格参数共享而言，网络的表征能力更强。

## **参考文献**

[1] D. Ha, A. Dai, and Q. V. Le, ‘HyperNetworks’. arXiv, Dec. 01, 2016. Accessed: Mar. 13, 2023. [Online]. Available: <http://arxiv.org/abs/1609.09106>
