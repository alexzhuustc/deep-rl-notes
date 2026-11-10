# 【多智能体强化学习】HPN：利用超网络实现置换不变性（permutation invariance）与置换同变性（permutation equivariance，PE）

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2023-03-14  
来源：[https://zhuanlan.zhihu.com/p/613723150](https://zhuanlan.zhihu.com/p/613723150)

[知识库首页](../../README.md)

---

> 这篇论文针对多智能体强化学习算法所面临的维度灾难问题，提出了一种新的网络结构——**超策略网络（Hyper Policy Network，HPN）**，该网络结构能够满足**置换不变性（permutation invariance，PI）**和**置换同变性（permutation equivariance，PE）**，可与任意多智能体强化学习算法结合，最终性能超过了当前的许多算法。这篇论文从PI与PE的角度，对网络结构的设计进行了很清晰地阐述，为什么、怎么做、效果如何都讲得非常清楚，整体看下来，算法想法很巧妙，实现起来也应该不复杂，个人认为是非常有意义的一个工作。推荐指数：★★★★★

原文：《Breaking the Curse of Dimensionality in Multiagent State Space: A Unified Agent Permutation Framework》

<http://arxiv.org/abs/2203.05285>

当前多智能体强化学习（MARL）算法面临严重的**维度灾难（curse of dimensionality）**，即联合状态-动作空间随智能体数目增加而指数上升，因而样本效率低，扩展性差。解决该问题的一种合理方案是适当减小状态-动作空间的维度，本文的中心思想就是利用**置换不变性（permutation invariance，PI）**和**置换同变性（permutation equivariance，PE）**的归纳偏置（inductive bias）减小MARL算法中的状态-动作空间。

（**归纳偏置（inductive bias）**可以理解为，利用先验知识对模型做一定的约束，从而可以起到 “模型选择” 的作用。例如，深度神经网络认为层次化处理信息有更好效果；卷积神经网络认为信息具有空间局部性 (Locality)，可用滑动卷积共享权重的方式降低参数空间；循环神经网络 则将时序信息考虑进来，强调顺序重要性；图网络则认为中心节点与邻居节点的相似性会更好引导信息流动。一个好的归纳偏置可以提升样本的利用效率。）

## **置换不变性（permutation invariance，PI）与置换同变性（permutation equivariance，PE）**

用 $G$ 表示维度为 $m\times m$ 的所有置换矩阵的集合，用 $g$ 表示 $G$ 中的一个置换矩阵。对于函数 $f:X\rightarrow Y$，$X=[x_1,...,x_m]^T$，如果置换输入元素的排列不改变输出，即 $f(g[x_1,...,x_m]^T)=f([x_1,...,x_m]^T)$，$\forall g\in G$，则可以说该函数满足PI特性。相反，对于函数 $f:X\rightarrow Y$，$X=[x_1,...,x_m]^T$，$Y=[y_1,...,y_m]^T$，如果置换输入元素的排列会以同样方式置换输出的排列，即 $f(g[x_1,...,x_m]^T)=g[y_1,...,y_m]^T$，$\forall g\in G$，则说明该函数满足PE特性。不满足PI特性和PE特性统称为置换敏感（permutation-sensitive）函数。

### **动机**

一个多智能体系统包含 $m$ 个实体（entities），这其中包括 $n$ 个学习智能体（agents）和 $m-n$ 个非学习对象（行为由内嵌规则决定）。每个智能体 $i$ 的观测 $o_i$ 通常包含 $m$ 个实体的特征，$[x_1,...,x_m]$，这里 $x_i\in\mathcal{X}$ 表示每个实体的特征，$\mathcal{X}$ 表示特征空间。如果简单地将 $o_i$ 表示成固定排列下特征的堆叠，$[x_1,...,x_m]$，观测空间的维度有 $|\mathcal{X}|^m$。然而，即便所有实体总共有 $m!$ 种排列方式，**不同排列对应向量所包含的信息都是相同的**！因此设计一种对实体排序不敏感的函数能够将观测空间的维度减小到原先的 $\frac{1}{m!}$。基于此，本论文利用PI与PE函数构造样本高效的MARL算法。

![](../../assets/images/3ed36d3c2ad0ece7730f.jpg)

以SMAC中的局部Q网络为例。输入 $o_i$ 包含两类实体：同盟 $o_i^{\text{ally}}$ 与敌人 $o_i^{\text{enemy}}$。输出包含两类动作的Q值：移动 $\mathcal{A}_i^{\text{move}}$，攻击 $\mathcal{A}_i^{\text{attack}}$。变换 $o_i$ 中元素的排列，$\mathcal{A}_i^{\text{move}}$ 的Q值不应该发生变化，因此我们可以利用PI模块使得 $Q_i(\mathcal{A}_i^{\text{move}}|o_i)$ 学得更加高效。而对于 $Q_i(\mathcal{A}_i^{\text{enemy}}|o_i)$，由于 $o_i^{\text{enemy}}$ 中每个enemy的特征与 $\mathcal{A}_i^{\text{attack}}$ 中每个攻击的动作之间存在一一对应关系， $o_i^{\text{enemy}}$ 的置换应该使得 $\mathcal{A}_i^{\text{attack}}$ 发生相同的置换，因此我们可以利用PE模块使得 $Q_i(\mathcal{A}_i^{\text{attack}}|o_i)$ 更加高效。

### **当前方法**

[XuanAxuan：【神经网络】Deep Sets：满足置换不变性（permutation-invariant）与置换同变性（permutation-equivariant）的通用网络结构](../../02-建模与表示/02-集合图结构与对称性/Deep Sets：满足置换不变性（permutation-invariant）与置换同变性（permutation-equiv--612736617.md)[XuanAxuan：【深度学习】Attention is all you need](../../02-建模与表示/01-序列注意力与随机变量/Attention is all you need--604339233.md)[XuanAxuan：【多智能体强化学习】ASN：考虑动作与智能体之间的对应关系](ASN：考虑动作与智能体之间的对应关系--606055951.md)

为了实现PI，之前的方法可以分为两种。第一种是利用数据增强（data augmentation），即通过打乱输入元素的顺序产生更多的训练样本，使相同元素组成的输入映射到同样的输出，然而，利用排列不同的样本训练一个置换敏感的函数效率非常低下，并且训练出来的网络也很不稳定。第二种方法是利用一些天然的PI结构，例如Deep Sets，GNN，Attention，这些结构首先利用一个共享的嵌入层处理每个实体的特征，然后利用一个entity-wise的池化层处理这些嵌入特征，从而实现PI。然而，**共享嵌入层限制了模型的表征能力**，可能会导致性能下降。目前在MARL领域实现PE的方法还较少，一个较为相关的工作是ASN，该方法研究了不同种类动作的不同影响，但没有考虑PE。

### **目标**

将每个智能体的动作划分为两个部分：与实体相关的（entity-corrected）动作集合 $\mathcal{A}_{\text{equiv}}$ 以及与实体不相关（entity-uncorrected）的动作集合 $\mathcal{A}_{\text{inv}}$ $\mathcal{A}_i=\{\mathcal{A}_{\text{equiv}},\mathcal{A}_{\text{inv}}\}$。$\mathcal{A}_{\text{inv}}$ 中的每个动作与每个实体之间有一一对应关系，例如SMAC中的攻击一个特定的enemy。因此，$a_{\text{equiv}}\in\mathcal{A}_{\text{equiv}}$ 的排序必须与 $o_i$ 相同，而 $a_{\text{inv}}\in\mathcal{A}_{\text{inv}}$ 则必须保持不变。

用 $g\in\mathcal{G}$ 表示任意置换矩阵，$g$ 作用于 $o_i$ 时，会置换 $m$ 个实体特征的排列，即 $go_i$，$g$ 作用于 $a_i=\{a_{\text{equiv}},a_{\text{inv}}\}$，则只置换 $a_{\text{equiv}}$ 的次序，保持 $a_{\text{inv}}$ 不变，即

$g a_i \triangleq\left(g a_{\text {equiv }}, a_{\text {inv }}\right)$。本文的目标是在智能体的策略网络中引入PI与PE的归纳偏置使得：



$$
\pi_i\left(a_i \mid g o_i\right)=g \pi_i\left(a_i \mid o_i\right) \triangleq\left(g \pi_i\left(a_{\text {equivi }} \mid o_i\right), \pi_i\left(a_{\text {inv }} \mid o_i\right)\right) \quad \forall g \in G, o_i \in \mathcal{O} \\
$$



$g$ 作用于 $\pi_i$ 表示置换 $\pi_i\left(a_{\text {equivi }} \mid o_i\right)$ 的次序，保持 $\pi_i\left(a_{\text {inv }} \mid o_i\right)$ 不变。

## **模型框架**

![](../../assets/images/9a8f369705a406a53588.jpg)

本文将智能体的策略网络结构划分为4个部分：1）输入模块 $A$；2）主要模块 $B$，该模块可以是任意结构；3）动作 $a_{\text{inv}}$ 对应的输出模块 $C$；4）动作 $a_{\text{equiv}}$ 对应的输出模块 $D$。四个模块的输入输出关系为：$z_i=A(o_i)$，$h_i=B(z_i)$，$\pi_i(a_{\text{inv}}|o_i)=C(h_i)$ 以及 $\pi_i(a_{\text{equiv}}|o_i)=D(h_i)$。

为了满足公式（1），只需修改 $\{A,B,C,D\}$ 的结构使得 $C$ 的输出是PI的，$D$ 的输出是PE的。实际上，为了满足这个条件，只需要修改 $A$ 使其满足PI条件，然后修改 $D$ 使其满足PE条件即可，$B$ 和 $C$ 的结构可以保持不变。证明很简单，在此不赘述。另外，当 $A$ 满足PI条件时，为了使 $D$ 满足PE条件，必须将 $o_i$ 作为 $D$ 额外的输入。因为 $A$ 的PI特性会使得 $D$ 的输出满足PI特性，因此必须使得 $D$ 知晓 $o_i$ 中各实体特征的排序。

有了以上框架之后，作者提出以**最小修改原则**设计 $\{A,B,C,D\}$ 使得该模型能够与当前大多MARL算法结合。

本文主要是对单个智能体的局部策略网络进行了设计，具体地，首先提出 **动态置换网络（Dynamic Permutation Network，DPN）**使得网络满足PI与PE特性，随后又提出**超策略网络（Hyper Policy Network，HPN）**用于扩大参数集。

## **Dynamic Permutation Network（DPN）**

![](../../assets/images/4e6bf86ab8bd0b4b3c53.jpg)

在大多数MARL算法中，模块 $A$，$C$ 和 $D$ 都是全连接（MLP）层，模块 $B$ 则会引入RNN处理局部观测的输入。给定包含 $m$ 个实体特征的观测 $o_i$，子模块视角下的MLP层如上图左侧所示，总共有 $m$ 个独立的权重矩阵。将 $A$ MLP层整体的权重矩阵记为 $\mathcal{W}_{\text {in }}=\left[\mathbf{W}_1^{\mathrm{in}}, \ldots \mathbf{W}_m^{\mathrm{in}}\right]$，其输出可以计算为



$$
z_i=\sum_{j=1}^m o_i[j] \mathcal{W}_{\text {in }}[j] \\
$$



这里 $[j]$ 表示第 $j$ 个元素。类似地，将输出层 $D$ 的权重矩阵记为 $\mathcal{W}_{\text {out }}=\left[\mathbf{W}_1^{\mathrm{out}}, \ldots \mathbf{W}_m^{\mathrm{out}}\right]^{\top}$，输出 $\pi_i\left(a_{\text {equivi }} \mid o_i\right)$ 的计算式为



$$
\pi_i\left(a_{\text {equiv }}| o_i\right)[j]=h_i \mathcal{W}_{\text {out }}[j] \quad \forall j \in\{1, \ldots, m\} \\
$$



接下来分别设计PI的输入层 $A$ 和PE的输出层 $D$。

### **PI输入层**

根据公式（2），置换 $o_i$，即 $o_i'=go_i$，会导致输入的第 $j$ 个元素变为 $o_i'[j]$，而第 $j$ 个权重矩阵并没有发生变化，因此网络的输出会变成 $z_i'\ne z_i$。为了让MLP层满足PI特性，需要为每个 $o_i[j]$ 设计一种PI的**权重矩阵选择策略**，**使得无论** $o_i[j]$ **如何排列**，**相同的** $o_i[j]$ **都会与相同的权重矩阵相乘**。具体地，本文设计了一种输出维度为 $m$ 的权重选择网络：



$$
p_{\text {in }}\left(\left[\mathbf{W}_1^{\mathrm{in}}, \ldots \mathbf{W}_m^{\mathrm{in}}\right]| o_i[j]\right)=\operatorname{softmax}\left(\operatorname{MLP}\left(o_i[j]\right)\right) \\
$$



这里输出的第 $k$ 个元素 $p_{\text {in }}\left(\mathbf{W}_k^{\text {in }}| o_i[j]\right)$ 表明 $o_i[j]$ 选择 $\mathcal{W}_{\text {in }}$ 中第 $k$ 权重矩阵的概率。然后，对于每个 $o_i[j]$ ，只需选择概率最大的权重矩阵即可。

然而，最大操作是不可导的，因此，为了使得权重选择过程可导，本文采用Gumbel-Softmax操作得到 $\arg\max$ 索引的one-hot编码，并将其记为 $\hat{p}_{\text {in }}\left(o_i[j]\right)$，此时最大概率的权重矩阵可以表示为 $\hat{p}_{\text {in }}\left(o_i[j]\right)\mathcal{W}_{\text {in }}$，也就是从 $\mathcal{W}_{\text {in }}$ 中选择对应的行。总结来说，无论输入 $o_i$ 的排列如何变化，模块 $A$ 的输出层总是可以计算为



$$
z_i=\sum_{j=1}^m o_i^{\prime}[j]\left(\hat{p}_{\text {in }}\left(o_i^{\prime}[j]\right) \mathcal{W}_{\text {in }}\right) \quad o_i^{\prime}=g o_i, \forall g \in G \\
$$



由于权重选择网络只接收 $o_i[j]$ 作为输入，相同的 $o_i[j]$ 总会产生相同的概率分布，因此无论 $o_i[j]$ 如何排列，总会选择相同的权重矩阵。因此无论 $o_i$ 如何排列，所得到的 $z_i$ 总是相同的，即 模块 $A$ 满足PI特性。示意图如上图所示。

### **PE输出层**

为了使得 $D$ 满足PE特性，同样为每个各实体相关的输出构造一个权重选择网络 $\hat{p}_{\text {out}}\left(o_i[j]\right)$，类似地， $D$ 的第 $j$ 个输出可以计算为



$$
\pi_i\left(a_{\text {equiv }}|o_i^{\prime}\right)[j]=h_i\left(\hat{p}_{\text {out }}\left(o_i^{\prime}[j]\right) \mathcal{W}_{\text {out }}\right) \quad o_i^{\prime}=g o_i, \forall g \in G, \forall j \in\{1, \ldots, m\} \\
$$



对于 $\forall g\in G$，$o_i'$ 的第 $j$ 个元素总是对应 $\mathcal{W}_{\text {out}}$ 中相同的权重矩阵，因此 $\pi_i\left(a_{\text {equiv }}|o_i^{\prime}\right)$ 的第 $j$ 个输出也总是保持相同的值，即输入排列的变化会造成输出排列发生相同变化，从而实现PE。

### **从矩阵计算的角度理解**

实际上，DPN就是**在普通MLP层的基础上增加了一个权重选择机制**，即将观测中每一个实体的特征与权重矩阵中与其相乘的矩阵分量一一对应起来，为了加深理解，我将通过对矩阵计算的过程进行分解来进一步分析DPN的原理。

首先考虑PI特性，输入 $o_i=[\mathbf{o}_{i,1}^T,...,\mathbf{o}^T_{i,m}]^T$ ，$\mathbf{o}_{i,j}\in\mathbb{R}^{d_{\text{in},j}\times1}$ 表示第 $j$ 个实体的特征，输出为 $z_i$， 权重矩阵为 $\mathcal{W}_{\text {in }}=\left[\mathbf{W}_1^{\mathrm{in}}, \ldots \mathbf{W}_m^{\mathrm{in}}\right]$，其中 $\mathbf{W}_j^{\mathrm{in}}\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in},j}}$ 表示第 $j$ 个实体特征所对应的权重矩阵。 常规MLP的计算公式可以表示为



$$
\begin{aligned} z_i=\mathcal{W}_{\text {in }}o_i = \left[\mathbf{W}_1^{\mathrm{in}}, \ldots \mathbf{W}_m^{\mathrm{in}}\right] \begin{bmatrix} \mathbf{o}_{i,1}\\ \vdots\\ \mathbf{o}_{i,m} \end{bmatrix} =\sum_{j=1}^m \mathbf{W}_j^{\mathrm{in}}\mathbf{o}_{i,j} \end{aligned} \\
$$



可以看到，MLP层的权重矩阵可以分解为多个分量，每个分量对应一个实体的特征分量，为了满足PI特性，需要将特征分量 $\mathbf{o}_{i,j}$ 与权重分量 $\mathbf{W}_j^{\mathrm{in}}$ 一一对应起来，这就是权重选择网络存在的意义。

类似的方法考虑PE特性，输入 $h_i\in\mathbb{R}^{d_{\text{in}}\times 1}$，输出为 $y_i=[\mathbf{y}_{i,1}^T,...,\mathbf{y}^T_{i,m}]^T$，$\mathbf{y}_{i,j}\in\mathbb{R}^{d_{\text{out},j}\times1}$ 表示对应第 $j$ 个实体的输出，权重矩阵为 $\mathcal{W}_{\text {out}}=\left[(\mathbf{W}_1^{\mathrm{out}})^T, \ldots (\mathbf{W}_m^{\mathrm{out}})^T\right]^T$ ，其中 $\mathbf{W}_j^{\mathrm{out}}\in\mathbb{R}^{d_{\text{out},j}\times d_{\text{in}}}$ 表示第 $j$ 个实体所对应输出的权重矩阵。常规MLP的计算公式可以表示为



$$
\begin{bmatrix} \mathbf{y}_{i,1}\\ \vdots\\ \mathbf{y}_{i,m} \end{bmatrix} =\mathcal{W}_{\text {out}}h_i = \begin{bmatrix} \mathbf{W}_1^{\mathrm{out}} \\ \vdots \\ \mathbf{W}_m^{\mathrm{out}} \end{bmatrix} h_i \\
$$



在输出层，MLP的权重矩阵也可以分解为多个分量，每个分量对应一个实体，为了满足PE特性，同样需要将输出分量 $\mathbf{y}_{i,j}$ 与权重分量 $\mathbf{W}_j^{\mathrm{out}}$ 一一对应起来，因此也需要一个权重选择网络。

PI输入与PE输出的区别在于对MLP权重矩阵分解方式的不同，PI输入是从输入的角度进行分解，PE输出是从输出的角度进行分解，我理解本质上都是**对矩阵计算的解构**。

## **Hyper Policy Network（HPN）**

DPN的核心思想是总是为每个 $o_i[j]$ 分配相同的权重矩阵。相比于DeepSets、Attention等结构中每个 $o_i[j]$ 共享同一权重矩阵，即 $|\mathcal{W}_{\text {in }}|=1$，DPN的表征能力大大提升，即 $|\mathcal{W}_{\text {in }}|=m$。然而，$\mathcal{W}_{\text {in}}$ 和 $\mathcal{W}_{\text {out}}$ 的表征空间还是十分受限，因为对于每个 $o_i[j]$，我们只能从**有限的参数集** $\mathcal{W}_{\text {in}}$ 或 $\mathcal{W}_{\text {out}}$ 中选择权重矩阵。

我们是否能够通过某种方式**提供数量无限大的权重矩阵集合**呢？通常来说，$o_i[j]$ 是连续的，有无数多个，我们难以为每个 $o_i[j]$ 学习一个权重矩阵。在这种情况下，作者提出了**超策略网络（hyper policy network，HPN）**，即直接利用一系列**超网络（hypernetworks）**产生每个 $o_i[j]$ 所对应的权重矩阵 $\mathbf{W}_j^{\mathrm{in}}$ / $\mathbf{W}_j^{\mathrm{out}}$，每个超网络的输入为 $o_i[j]$，此时 $\mathcal{W}_{\text {in}}$ 和 $\mathcal{W}_{\text {out}}$ 的大小不再限制于 $m$，可以近似于无限大，因而表征能力更强。

![](../../assets/images/320080cd85095393c74f.jpg)

### **PI输入层**

对于输入层 $A$，构建一个超网络 $\operatorname{hp}_{\mathrm{in}}\left(o_i[j]\right)=\operatorname{MLP}\left(o_i[j]\right)$ 来产生每个 $o_i[j]$ 所对应的 $\mathbf{W}_j^{\mathrm{in}}$：



$$
z_i=\sum_{j=1}^m o_i^{\prime}[j] \mathrm{hp}_{\text {in }}\left(o_i^{\prime}[j]\right) \quad o_i^{\prime}=g o_i, \forall g \in G \\
$$



不同 $o_i[j]$ 所产生的权重矩阵不同，无论 $o_i$ 如何排列，同一 $o_i[j]$ 总是对应相同的权重矩阵。由于每个 $o_i[j]$ 分别与其对应的 $\mathcal{W}_{\text {in}}[j]$ 相乘，然后通过一个PI的求和方程合并在一起，因此PI特性得到了满足。

### **PE输出层**

类似地，对于输出层 $D$，构建一个超网络 $\operatorname{hp}_{\mathrm{out}}\left(o_i[j]\right)=\operatorname{MLP}\left(o_i[j]\right)$ 用于产生权重矩阵 $\mathcal{W}_{\text {out}}$。最终 $D$ 的第 $j$ 个输出为



$$
\pi_i\left(a_{\text {equiv }} \mid o_i^{\prime}\right)[j]=h_i \mathrm{hp}_{\text {out }}\left(o_i^{\prime}[j]\right) \quad o_i^{\prime}=g o_i, \forall g \in G, \forall j \in\{1, \ldots, m\} \\
$$



通过使用超网络，$o_i[j]$ 与 $\mathcal{W}_{\text {out}}[j]$ 一一对应起来。输入排序的改变会造成输出产生相同方式的改变，从而实现了PE。

### **理解**

在DPN中， $\mathcal{W}_{\text {in}}$ 和 $\mathcal{W}_{\text {out}}$ 的参数集是固定的，每个 $o_i[j]$ 所对应的权重矩阵只能从固定的参数集中选择，而在HPN中，每个 $o_i[j]$ 都能够产生一个独一无二的权重矩阵，参数空间大大增加，因而表征能力更强。

## **小实验：不同PI模型之间的表征能力比较**

为了对比以下PI模型的表征能力大小：1）Deep Sets；2）DA-MLP（data augmentation ）；3）DPN；4）HPN；5）Attention。本文在附录中描述了一个小实验：总共有2个智能体，每个智能体 $i$ 的特征 $x_i$ 是一个整数，且 $x_i\in\{1,...,30\}$，即每个智能体有30种不同的特征，因此联合状态空间 $([x_1,x_2])$ 的尺寸为 $30\times 30=900$。每个状态 $[x_1,x_2]$ 所对应的目标值大小为 $x_1\times x_2$。

上一节说到之所以要引入HPN，是因为DPN中的候选权重矩阵数量有限。在这个简单的任务中，我们可以为每个特征 $x_i$ 构造一个独立的权重矩阵，由于 $x_i$ 是离散的，我们可以用一个表格为不同的特征 $x_i$ 记录一个不同的权重矩阵，此时我们可以认为参数空间是无限大的（为每个 $x_i$ 分配一个与众不同的权重矩阵），因此将该方法称为DPN（$\infty$）。而HPN通过使用超网络为每个 $x_i$ 产生一个不同的权重矩阵，可近似实现无限大的参数空间。

### **参数空间扩大**

利用均方误差（mean square error，MSE）作为损失函数训练这些模型，最终的结果如下：

![](../../assets/images/ae32197c71bfa98b0890.jpg)

可以看出，DPN（$\infty$）>HPN>Attention>DPN>DeepSet>DA-MLP，可以看出增加模型的表征能力可以显著降低损失。

### **状态空间减小**

增加智能体数目，通过堆叠特征所产生的状态空间以及使用PI特性所产生的状态空间的空间维度比较如下：

![](../../assets/images/11f95898b8ce1491b001.png)

可以看出，利用PI特性，可以显著减小状态空间。

## **仿真实验**

### **对比实验**

![](../../assets/images/d57399ef9ccd3ebe8b78.jpg)[XuanAxuan：【多智能体强化学习】DyMA-CL：从少到多地训练多智能体系统](DyMA-CL：从少到多地训练多智能体系统--605456953.md)[XuanAxuan：【多智能体强化学习】ASN：考虑动作与智能体之间的对应关系](ASN：考虑动作与智能体之间的对应关系--606055951.md)[XuanAxuan：【多智能体强化学习】UPDeT：基于transformer的策略分解模型](UPDeT：基于transformer的策略分解模型--613020553.md)

Baseline：1）DA-QMIX（data augmentation）；2）SET-QMIX（Deep Sets）；3）GNN-QMIX（GNN）；4）ASN-QMIX；5）UPDeT-QMIX。

结论：HPN＞DPN≥UPDeT≥ASN>QMIX>GNN≈SET≈DA。

- HPN的性能优于DPN，这是因为HPN通过超网络扩大了参数空间，因而表征能力更强；
- UPDeT、ASN能够达到与DPN相近的性能，说明PI特性与PE特性能够提升性能；
- GNN与SET的性能比原始QMIX更差（存疑），这俩结构实现了PI，但是没有实现PE，（我的理解是PI能够缩小状态空间，而PE则能够缩小动作空间），作者给出的解释是共享嵌入层的使用限制了模型表征能力；
- DA的性能不如原始QMIX，这说明用置换敏感的函数（MLP）来实现PI是十分困难的。

### **消融实验**

![](../../assets/images/8bcce3201d4935127564.jpg)

- 增加QMIX的网络参数（BIG-QMIX），性能变差，说明单纯增加参数数量无法提升性能；
- 在SET-QMIX的基础上增加基于HPN的PE输出层（HPN-SET-QMIX），性能大幅提升，但仍然没有达到HPN-QMIX，说明了PE输出层的必要性，同时也表明了基于HPN的PI输入层相对Deep Sets的优势；

### **不同数量智能体任务间的迁移**

![](../../assets/images/12d50397a307f95b3887.jpg)

由于实现了PI特性与PE特性，HPN能够处理不同维度的输入与输出。因此，可以将在较简单任务下训练出来的HPN策略迁移到更加复杂的任务，提升训练效率与最终性能。

## **个人思考与总结**

这篇论文太棒了，想法很巧妙，论文也写得很好，逻辑清晰，每个细节都讲得很清楚，没有多余的废话，很多描述让我有豁然开朗的感觉，真的很赞。

首先，PI特性与PE特性可以缩小状态空间与动作空间， 提升算法性能；

相较于GNN、Deep Sets、Attention等PI结构而言，DPN的优势是参数空间更大（各个实体特征不共享参数），从1扩大到有限个，但是结构也比较简单；

相较于DPN，HPN利用了超网络，参数空间进一步增大，从有限个扩大到趋近于无限。

另外，GNN、Attention等实现了PI，而ASN、UPDeT、DPN和HPN既实现了PI，也实现了PE，进一步缩小了状态-动作空间的搜索范围。

HPN与基于Attention、GNN的模型相比到底效果如何，欢迎大家讨论讨论呀~~

## **参考文献**

[1] X. Hao *et al.*, ‘Breaking the Curse of Dimensionality in Multiagent State Space: A Unified Agent Permutation Framework’. arXiv, Oct. 19, 2022. Accessed: Feb. 24, 2023. [Online]. Available: <http://arxiv.org/abs/2203.05285>
