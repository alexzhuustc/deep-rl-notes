# 【深度学习】Attention is all you need

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2023-02-07  
来源：[https://zhuanlan.zhihu.com/p/604339233](https://zhuanlan.zhihu.com/p/604339233)

[知识库首页](../../README.md)

---

> 虽然知道attention已经很久了，看论文的时候也经常看到，但是不总结出来总感觉自己还没有完全理解，所以参照原论文简单写了写~~  
> 之后想做关于动态数目/可扩展多智能体方面的内容，首先是要结合QMIX设计一个population-invariant structure，然后结合迁移学习/课程学习等等使得模型可以更好地适应不同数目的智能体。看了许多论文，很多都是为SMAC特别设计的，感觉不太实用~~一直在找普适性强、实用性强、限制条件少，不复杂可复现的算法，友友们有什么好的文章好的方法欢迎多多和我交流交流呀~~~

注意力机制（Attention）函数可用于将一个问询（query）和一系列密钥-数值（key-value）对映射成一个输出（output），这里的query，keys，values以及output都是向量。Output是values的加权和，其中分配给每个value的权重（weights）通过query和对应key的一个紧凑函数（compatibility function）得到。

### **缩放点积注意力（Scaled Dot-Product Attention）**

![](../../assets/images/6081511d01ddc427ab28.jpg)

本文所介绍的Attention名为Scaled Dot-Product Attention，该模型的输入包含一系列维度为 $d_k$ 的queries与keys，维度为 $d_v$ 的values。首先计算每一个query与所有keys的点乘，然后除以 $\sqrt{d_k}$ ，最后利用softmax函数得到每个values的weightes。

在实际中，我们会将一系列queries、keys和values构造成矩阵 $Q$、 $K$ 和 $V$， 然后同时计算所有queries的attention函数，最终的output矩阵为



$$
\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V \\
$$



两种最常用的attention函数是additive attention和dot-product（multiplicative）attention。其中dot-product attention和上面所介绍的相同，只是没有放缩系数 $\frac{1}{\sqrt{d_k}}$。Additive attention函数采用一个只有一个隐藏层的前向网络作为计算weights的compatibility function。这两种方法在理论上复杂度相似，但在实际操作中dot-product attention可以使用优化过的矩阵乘法实现，因而更快、空间更高效。放缩系数 $\frac{1}{\sqrt{d_k}}$ 是为了防止随着 $\sqrt{d_k}$ 增大，输出越来越大，梯度越来越小。

### **多头注意力机制（Multi-Head Attention）**

相比于对 $d_{\text{model}}$ 维的keys、values和queries采用单个attention函数，一种更好的方式是将这些keys、values和queries线性投射 $h$ 次分别得到不同的$d_k$， $d_k$ 和 $d_v$ 维向量。对于每一个线性投射的结果，并行地执行attention函数，得到 $d_v$ 维的output值。然后将所有的output连接起来再次进行线性投射，得到最终的结果。

Multi-head attention使得模型能够从不同的表达空间对不同位置的信息进行关注，其表达式为



$$
\begin{aligned} \operatorname{MultiHead}(Q, K, V) & =\operatorname{Concat}\left(\operatorname{head}_1, \ldots, \operatorname{head}_{\mathrm{h}}\right) W^O \\ \text { where head } & =\operatorname{Attention}\left(Q W_i^Q, K W_i^K, V W_i^V\right) \end{aligned} \\
$$



这里的 $W_i^Q \in \mathbb{R}^{d_{\text {model }} \times d_k}, W_i^K \in \mathbb{R}^{d_{\text {model }} \times d_k}, W_i^V \in \mathbb{R}^{d_{\text {model }} \times d_v}$ 以及 $W^O \in \mathbb{R}^{hd_v \times d_{\text {model}}}$ 表示投射的参数矩阵。

### **解析**

Attention机制最主要用于NLP领域，这方面我不是很懂，所以我直接根据个人理解对上述的attention操作进行一个详细的拆解。

Attention的核心逻辑实际上就是**从关注全部到关注重点**，将有限的注意力集中在重点信息上，从而节省资源，快速获得最有效的信息。从实操角度来看，Attention的思路简单来说就是**“带权求和”**。假设有 $N$ 个entities（例如MADRL里的智能体obs），每个entities的feature可以用一个 $d_{\text{embedding}}$ 的向量 $z_n$ 表示，在self-attention中，每个entities有三个向量，分别是query，key和value，它们是通过 $d_{\text{feature}}$ 乘以三个不同的权重矩阵 $W^Q$，$W^K$，$W^V$ 得到，这三个矩阵的维度分别是 $d_{\text{feature}}\times d_k$，$d_{\text{feature}}\times d_k$ 和 $d_{\text{feature}}\times d_v$。整个attention的过程可以分为以下几个步骤：

- 首先利用全连接层网络将所有entities的features $o_n$ 转换成 $d_\text{embedding}$ 维的embedding向量，即 $e_n=f(o_n)$；
- 利用 $W^Q$，$W^K$，$W^V$ 将每个entities的embedding向量转换成对应的query，key和value三个向量，即 $q_n^T=e_n^T\times W^Q$，$k_n^T=e_n^T\times W^K$ 和 $v_n^T=e_n^T\times W^V$，因此 $q_n$、$k_n$ 与 $v_n$ 的维度分别为 $d_k$，$d_k$ 和 $d_v$；
- 对于每一个entity：

- 将其query向量 $q_n$ 与所有entities的keys向量进行点乘得到两个entities的相似度score（为了梯度的稳定，需要进行归一化，即除以 $\sqrt{d_k}$），即 $s_{n,n'}=q_n\cdot k_{n'}$；
- 对所有的 $score_{n,n'}$ 采用softmax操作，得到每个value值的weights $w_{n,n'}=\frac{\exp{s_{n,n'}}}{\sum_{n"}{\exp{s_{n,n"}}}}$；
- 最终可以得到该entity的output向量：$z_n=\sum_{n'}{w_{n,n'}v_{n'}}$。

可以看出上面的步骤实际上就是针对每一个entity，对所有entities的features进行了一个聚合，通过加权求和的方式得到了一个全新的向量，权重因子为该entity与其他entity的相似度，也可以当做对其他entity的关注程度。

在实际操作中，为了计算的方便，可以将所有queries，keys和values向量堆叠成矩阵 $Q\in \mathbb{R}^{n\times d_k}$、$K\in \mathbb{R}^{n\times d_k}$ 和 $V\in \mathbb{R}^{n\times d_v}$，此时整个self-attention的操作就可以紧凑地写成



$$
\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V \\
$$



最后所得结果的维度是 $n\times d_v$。

而multi-head attention实际上就是进行 $h$ 次self-attention操作，然后将所得结果堆叠起来变成一个 $n\times hd_v$ 维的大矩阵，最后进行一次线性操作得到最终结果。

### **总结**

Attention实际上就是将不同entities的信息进行了聚合，相比于将所有的信息简单的加起来，这里通过对不同entities的相似度进行了计算，从而得到了不同entities的权重，使得每个entity可以对其他entities的信息有所侧重的关注，更加符合人类的信息处理方式。
