# 【多智能体强化学习】UPDeT：基于transformer的策略分解模型

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2023-03-10  
来源：[https://zhuanlan.zhihu.com/p/613020553](https://zhuanlan.zhihu.com/p/613020553)

[知识库首页](../../README.md)

---

> 这篇论文主要考虑的是利用**置换不变性（permutation-invariant）**模型解决**动态数目**智能体任务间的**迁移**问题，拿到了ICLR2021的spotlights，因此后来常被当作baseline对比置换不变性模型/迁移算法的性能。听说这个算法效果不是很好，论文也写得非常含糊，模型部分几乎一笔带过，我看了两遍也没有完全看明白，而且与代码也对应不上。但既然是baseline算法，我还是尽量理解理解好了~~推荐指数：★★★

<http://arxiv.org/abs/2101.08001>

当前大多多智能体强化学习算法将每个智能体对于环境中不同实体的观测当做是一个整体，例如将对应不同实体的观测部分堆叠成一个向量作为策略网络或值网络的输入，这种方法没有对不同观测部分与动作之间的关系进行挖掘，而是假设神经网络能够自动探索观测不同部分与动作之间的关系从而得到观测与策略之间的最佳策略，因而通常难以捕捉到不同任务之间的共同点。另外，这种方法要求网络的输入输出维度固定，因此难以实现不同数目智能体任务之间的迁移。

基于上述问题，本文提出了一种基于transformer的多智能体强化学习框架——Universal Policy Decoupling Transformer (UPDeT)，该框架的中心思想是将单个观测当做由各个实体（entities，智能体或其他非智能体部分）组成的集合，然后对应这些实体，将动作空间划分成多个子集，将不同动作子集与不同实体对应的观测部分匹配起来，从而在动作子集层面对策略进行了优化，这种方式被称为**policy decoupling**。另外本模型还利用了**transforme**r机制对观测中不同实体部分之间的关系进行了学习。总结来说，UPDeT就是利用Policy decoupling和transformer提升了模型性能，另外，UPDeT还有以下优点：1）训练好之后能够部署在不同数目智能体的任务中；2）通过策略解耦提升了模型的表征能力；3）更加具有解释性；4）能够与任意MARL算法结合。

## **Universal Policy Decoupling Transformer (UPDeT)**

模型部分写得非常非常简略，我只能大致理解理解....

UPDeT模型只对单个智能体的局部Q网络进行了处理，主要分为两个部分，首先是利用transformer处理观测，然后进行policy decoupling。

### **Transformer**

首先将观测 $O$ 转换到一个语义embedding空间。假设智能体 $a_i$ 在时刻 $t$ 能够观测到其他 $k$ 个实体 $\left\{o_{i, 1}, \ldots, o_{i, k}\right\}$，这些实体的观测会通过一个嵌入层 $E$ 转换成如下embedding：



$$
e_i^t=\left\{E\left(o_{i, 1}^t\right), \ldots, E\left(o_{i, k}^t\right)\right\} \\
$$



这里 $i$ 表示智能体序号，$i\in\{1,...,n\}$，智能体 $i$ 的局部Q值可以表示为



$$
q_i^t=Q_i\left(h_i^{t-1}, e_i^t, \mathbf{u}_t\right) \\
$$



这里 $h^{t-1}_i$ 表示上一个时刻的隐层特征（RNN）。接下来介绍如何利用 transformer 构造 $Q_i$。

在self-attention中，首先需要利用模型输入构造三个矩阵 $\mathbf{K},\mathbf{Q},\mathbf{V}$​ 分别用于表示keys，queries和values，然后进行如下操作：



$$
\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})=\operatorname{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V} \\
$$



这里 $d_k$ 表示缩放因子，等于key的维度。

不了解attention结构的可以看

[XuanAxuan：【深度学习】Attention is all you need](../../02-建模与表示/01-序列注意力与随机变量/Attention is all you need--604339233.md)

UPDeT利用self-attention来学习观测中各个实体embedding以及全局时序信息之间的关系。具体地，为每个智能体 $a_i$ 构造矩阵 $K_i,Q_i,V_i$，这里假设这三个矩阵相等，$R_i^l=K_i=Q_i=V_i$，这里 $l\in\{1,...,L\}$ 表示transformer的层数。Transformer的计算方式如下：



$$
\begin{aligned} R_i^1 & =\left\{h_i^{t-1}, e_i^t\right\} \\ Q_i^l, K_i^l, V_i^l & =L F_{Q, K, V}\left(R_i^l\right) \\ R_i^{l+1} & =\operatorname{Attention}\left(Q_i^l, K_i^l, V_i^l\right) \end{aligned} \\
$$



这里 $LF$ 表示用于计算 $\mathbf{K},\mathbf{Q},\mathbf{V}$ 的线性方程。最终 $Q_i$ 的计算如下：



$$
Q_i\left(h_i^{t-1}, e_i^t, u_i\right)=P\left(R_i^L, u_i\right) \\
$$



这里 $P$ 表示线性方程，$u_i$ 表示动作。

Transformer的主要作用是学习观测中不同实体对应部分之间的关系，但此时仍然不能解决输入输出维度变化的问题，接下来介绍如何通过policy decoupling构造一个能够具有强大表征能力且能够处理动态输入输出维度的映射函数 $P$。

### **Policy Decoupling**

![](../../assets/images/aab726d63010a44b19a4.jpg)![](../../assets/images/df71f0d4485a6299930f.jpg)

这一部分本应该重点叙述，但原文值给了短短一段话，连公式都没有，非常不理解......

UPDeT首先根据任务特征，将观测中的各实体部分与输出的各动作匹配起来，这样就能够通过self-attention机制大大减轻模型表征学习的负担（？？？这里如何用self-attention）。另外考虑到一个实体可能对应着多个关联动作，因此可以将动作空间划分中多个动作子集，每个子集对应于一个实体，划分过程在上面图三的左边部分。如果一个实体对应的动作子集包含一个动作，则利用一个全连接层将embedding转换成维度等于动作数目的输出。如果一个实体没有对应的动作，则直接丢弃其embedding，由于之前已经通过transformer将各个实体之间的信息聚合了起来，所以不用担心会丢失任何信息。UPDeT的原理如上面图三的右边部分所示。

实名吐槽图三，我是完全没看明白这是在干嘛，各个字符也没有标注，$A,B,C$，$1,...,j$，$1,...,m$ 分别代表什么都不知道，Aggregation在文中也没有提及，self-attention在原文中提及了却没有体现在图中，我真的一个大无语。

我自己的理解就是直接按照entity-wise的方式将每个实体的embedding映射到对应有关联的动作子集的Q值，没有对应动作的实体embedding就丢弃掉，由于用了transformer，所以各个实体的信息包含在了transformer输出的每个embedding向量中，即使丢弃掉了部分embedding向量，也不会损失任何信息。所以这样看UPDeT或许就是在ASN的基础上加了一个transformer？

### **时序单元结构**

类似大多多智能体强化学习算法，UPDeT也提出用隐层状态来记录时序信息，但是按照原文意思是，UPDeT没有采用GRU/LSTM等RNN结构，而是直接将transformer和隐层状态结合起来（好奇怪！！），论文提出两种方式处理时序信息：

1）Global：将隐层状态作为全局信息直接作为transformer模块的额外输入，此时公式（4）可以补充为 $R^1=\left\{h_G^{t-1}, e_1^t\right\} $ 以及 $\left\{h_G^t, e_L^t\right\}=R^L$，这里忽略了原本的 $i$ 下标，然后用 $G$ 表示全局global。

2）Individual：将隐层状态作为各个实体的局部信息，也就是说，每个实体都有对应的隐层状态，用 $j$ 表示实体序号，最终的关系可以表示为 $R^1=\left\{h_1^{t-1} \ldots h_j^{t-1}, e_1^t\right\}$ 以及 $\left\{h_1^t \ldots h_j^t, e_L^t\right\}=R^L$。

第二种方式能够更加精确地描述时序信息，但增加了学习困难。

### **训练**

直接按照端到端地方式进行训练。

## **实验结果**

![](../../assets/images/946f67e5b378a6e97688.jpg)

对比实验：

- Vanilla transformer：直接将transformer的输出映射到所有动作的Q值
- Aggregation transformer：将transformer输出的所有实体embedding加起来映射到所有动作的Q值
- GRU：即普通算法，无transformer也无policy decoupling

结果表明：

- Transformer只有与policy decoupling结合才能发挥作用（图（a））
- 增加时序信息能够提升性能，并且global和individual差不多
- UPDeT能够与各类算法结合提升性能

![](../../assets/images/d81688fc3de8083291b6.jpg)

- UPDeT能够帮助算法在不同数目智能体的任务间进行迁移

## **总结与个人思考**

这篇论文写得真的一言难尽......算法完全没表述不清楚，而且transformer和self-attention这两个词总是换来换去，图片也看不懂，不理解！如果有复现过这篇论文的小伙伴，欢迎指出我理解中的错漏，以及讲讲这个算法效果究竟如何~~

我个人的理解是这篇论文就是用了个transformer来表征不同实体特征之间的关系，然后将动作分了个组，将实体和动作匹配起来，一定程度上优化了参数空间，并且使得模型能够处理动态输入输出。

## **参考文献**

[1] S. Hu, F. Zhu, X. Chang, and X. Liang, ‘UPDeT: Universal Multi-agent Reinforcement Learning via Policy Decoupling with Transformers’. arXiv, Feb. 07, 2021. Accessed: Jan. 19, 2023. [Online]. Available: <http://arxiv.org/abs/2101.08001>
