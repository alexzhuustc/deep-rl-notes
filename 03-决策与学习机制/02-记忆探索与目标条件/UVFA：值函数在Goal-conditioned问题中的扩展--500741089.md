# 【迁移强化学习】UVFA：值函数在Goal-conditioned问题中的扩展

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2022-04-18  
来源：[https://zhuanlan.zhihu.com/p/500741089](https://zhuanlan.zhihu.com/p/500741089)

[知识库首页](../../README.md)

---

今天介绍论文《Universal Value Function Approximators》。

> 本文将强化学习中的值函数扩展到了目标空间，另外针对同时定义在状态空间与目标空间上的值函数，设计了一种网络结构以及对应的训练方法，这篇论文的想法具有很强的创新性，仿真实验也很grounded，所提出来的goal-conditioned值函数对之后的一些工作例如HER、HAC也具有很强的借鉴意义，总体来说是一篇非常值得阅读的论文，但是本文避开了什么是目标这一问题，所考虑的场景也比较局限，所以UVFA在实际应用中究竟适合用在什么问题中是一个需要思考的问题。推荐指数 ★★★★。

本文所提的UVFAs（universal value function approximators）将强化学习中的值函数同时定义在了状态 $s$ 和目标 $g$ 上，并提出了一种监督学习的方法将值函数分解成状态和目标的嵌入向量，从而学习到 $s$ 和 $g$ 到它们各自嵌入向量之间的映射关系。

## **算法背景**

本文提出的UVFA在标准马尔科夫框架下增加了目标 $g\in \mathcal{G}$。对于任意 $g\in \mathcal{G}$，定义伪-奖励函数 $R_g(s,a,s')$ 与伪-折扣函数 $\gamma_g(s)$，这里 $\gamma_g(s)$ 同时决定了每一时刻与 $s$ 相关的折扣因子，以及状态是否终止。对应新加入的 $g$，状态价值函数与动作价值函数分别扩展为：

$V_{g, \pi}(s):=\mathbb{E}\left[\sum_{t=0}^{\infty} R_{g}\left(s_{t+1}, a_{t}, s_{t}\right) \prod_{k=0}^{t} \gamma_{g}\left(s_{k}\right) \mid s_{0}=s\right] \\$$Q_{g, \pi}(s, a):=\mathbb{E}_{s^{\prime}}\left[R_{g}\left(s, a, s^{\prime}\right)+\gamma_{g}\left(s^{\prime}\right) \cdot V_{g, \pi}\left(s^{\prime}\right)\right] \\$

每个目标 $g$ 均对应着一个最优策略 $\pi_g^*:=\arg\max_a Q_{\pi,g}(s,a)$，以及最优值函数 $V^*_g:=V_{g,\pi^*_g}$，$Q^*_g:=Q_{g,\pi^*_g}$。

## **UVFA网络结构**

![](../../assets/images/780b81d72f29bd37db90.jpg)

UVFA的主要思想是用一个函数拟合器来表示一系列值函数。具体来说，就是用 $V(s, g ; \theta) \approx V_{g}^{*}(s)$ 或 $Q(s, a, g ; \theta) \approx Q_{g}^{*}(s, a)$ 来近似状态空间 $s\in\mathcal{S}$ 与目标空间 $g\in\mathcal{G}$ 上的 其中 $\theta\in\mathbb{R}^d$ 为函数参数。

论文比较了两种不同的网络结构：

- **Concatenated** 结构（对应上图第一个结构）： 这种结构直接将状态 $s$ 与目标 $g$ 串联起来输入到非线性函数近似器例如MLP中：$F:\mathcal{S}\times\mathcal{G}\rightarrow\mathbb{R}$。
- **Two-Stream** 结构（对应上图第二个和第三个结构）：这种结构首先分别将状态 $s$ 与目标 $g$ 输入到两个不同的子模块中：$\phi: \mathcal{S} \mapsto \mathbb{R}^{n}$ 与 $\psi: \mathcal{G} \mapsto \mathbb{R}^{n}$ ，得到状态 $s$ 与目标 $g$ 对应的 $n$-维**嵌入（embedding）**向量，然后将embedding向量同时输入到输出函数 $h: \mathbb{R}^{n} \times \mathbb{R}^{n} \mapsto \mathbb{R}$ 中，得到一个标量输出。本文中 $\phi$ 与 $\psi$ 用通用函数近似器表示，例如MLP，$h$ 则用简单函数拟合，例如点乘（dot-product）。

采用two-steam结构能够充分利用状态与目标之间结构上的相似性。首先，目标通常定义在状态上，例如 $\mathcal{G} \subseteq \mathcal{S}$，这一特征可以通过在之间共享特征来利用。具体而言，当使用MLP来表示 $\phi$ 与 $\psi$ 时，可以共享两个网络的第一层参数，从而学习到状态和目标之间的通用特征。另外，UVFA是 **对称** 的，即 $V_{g}^{*}(s)=V_{s}^{*}(g)\  \forall s, g$ ，这种对称性可以通过为状态和目标使用相同的网络：$\phi=\psi$，以及对称的输出函数 $h$ （点乘）来实现。上述的两种情况分别被称为 **部分对称** 以及 **完全对称** 结构。特别来说，当我们选择了一个基于距离的 $h$ 的对称结构，并且 $\mathcal{G}=\mathcal{S}$ 时，UVFA可以学习到一种embedding 使得较小距离的 $h$ 表示两个状态是相近的。

## **基于监督学习的UVFA**

本节考虑的是**监督学习（supervised learning）**的场景，即假设给定一部分 $(s,g)$ 对应的值函数真实值 $V_g^*(s)$ 与 $Q_g^*(s,a)$ 作为训练标签。

论文比较了两种UVFA的训练方法：

- **端对端（End-to-End）** 训练：学习网络参数 $\theta$ 最直接的方式就是通过端对端训练，这可以通过在一个合适的损失函数上进行反向传播实现，例如首先利用均方误差（MSE）$\mathbb{E}\left[\left(V_{g}^{*}(s)-V(s, g ; \theta)\right)^{2}\right]$ 计算下降方向，然后通过随机梯度上升（SGD）的各种变体在该方向上更新参数 $\theta$，这种方法适用于上图中的任意一种结构（concatenated与two-stream）。
- **Two-Stage** 训练：对于 **two-stream** 结构，本文还提出了一种基于 **矩阵分解（matrix factorization）** 的两步训练方法：

1. Stage 1： 首先将所有标签值函数 $V^*_g(s)$ 排列成一个数据矩阵，一行对应着一个观测到的状态 $s$，一列则对应着一个观测到的目标 $g$。然后将矩阵分解，得到一个低秩近似用于定义状态与目标的 $n$-维embedding空间。 用 $\hat{\phi}_s$ 表示所得到的状态 $s$ 所在行的目标embedding向量， $\hat{\phi}_g$ 表示所得到的目标 $g$ 所在列的目标embedding向量。
2. Stage 2：根据得到的 $\hat{\phi}_s$ 与 $\hat{\phi}_g$ 通过多元回归的方式分别学习网络 $\phi$ 与 $\psi$ 的网络参数。
3. Stage 3（*）：通过端到端训练细调two-stream结构。

在上述步骤中，矩阵分解用于得到可以恢复值函数的理想行embedding与列embedding $\hat{\phi}$ 与 $\hat{\psi}$。Stage 2 则用于学习如何将实际状态 $s$ 映射到其embedding向量 $\phi(s)$ 以及如何将实际目标 $g$ 映射到其embedding向量 $\psi(g)$。

![](../../assets/images/f53dd40cf03ad7c5030d.jpg)

实验表明，采用two-stage的训练速度远远大于端对端训练。

## **UVFA在强化学习中的应用**

上节假设给定一部分 $(s,g)$ 对应的值函数真实值 $V_g^*(s)$ 与 $Q_g^*(s,a)$ 作为训练标签，本节考虑更为实际的场景，即不存在给定的值函数真实值，论文讨论了两种方法：1）通过 **Horde** 为UVFA提供训练标签；2）直接进行bootstrapping。

### **基于Horde的UVFA**

一种结合UVFA与强化学习的方法是利用 **Horde** 结构 [2]。Horde中的每个demon用于估计一个具体目标的值函数，这些值函数从一个共享的经验回放池中通过并行的，off-policy的方式进行学习。而通过将从Horde中学习到的值函数组成一个数据矩阵，然后根据上一届介绍的方法将其分解，就实现了UVFA在强化学习中的扩展，即利用已有的目标对应的值函数泛化到没有出现过的目标上，替代原Horde方法中直接学习所有goal值函数的方法。

算法流程如下：

![](../../assets/images/7e28400ad78f70bcd5de.jpg)

### **基于bootstrapping的UVFA**

除了使用Horde获取值函数标签之后，直接通过bootstrapping更新的方法端到端地训练UVFA也是可行的：



$$
Q\left(s_{t}, a_{t}, g\right) \leftarrow  \alpha\left(r_{g}+\gamma_{g} \max _{a^{\prime}} Q\left(s_{t+1}, a^{\prime}, g\right)\right)+(1-\alpha) Q\left(s_{t}, a_{t}, g\right) \\
$$



其中 $r_{g}=R_{g}\left(s_{t}, a_{t}, s_{t+1}\right)$。

然而，==论文指出同时进行bootstrapping拟合函数以及泛化目标会使学习过程变得十分不稳定。==

## **关于目标如何选取**

本论文并没有讨论目标从哪里来，以及如何表示目标这一问题，而是采用了最简单的一种情况，即目标为状态的子集：$\mathcal{G}\subset\mathcal{S}$，当状态达到目标时，获得奖励，对应的伪-奖励函数与伪-折扣可以表示成：

$R_{g}\left(s, a, s^{\prime}\right)= \begin{cases}1, & s^{\prime}=g \text { and } \gamma_{e x t}(s) \neq 0 \\ 0, & \text { otherwise }\end{cases} \\$$\gamma_{g}(s)= \begin{cases}0, & s=g \\ \gamma_{e x t}(s), & \text { otherwise }\end{cases} \\$

其中，$\gamma_{ext}$ 表示原本的折扣函数。

## **UVFA的应用**

- UVFA可用于 **迁移学习**，适用于环境动态规律相同但目标不同的任务，具体而言，采用UVFA学习到的 $V(s,g,\theta)$ 可以用于初始化新目标 $g$ 对应任务的价值函数 $V_g(s)$。下图表明这种方式的迁移学习可以大大提升学习速度。

![](../../assets/images/2fdd57392fdae6738167.jpg)

- UVFA可用于提取features来表征状态
- UVFA还可以提取时序抽象的options用于 **分层强化学习** 。对于任意的目标 $g$，可以依据 $V(s,g,\theta)$ 构造一个对应的option，从而为实现目标 $g$ 提供一个通用的option策略，这种方法允许分层策略选择任意 $g\in\mathcal{G}$ 作为时序抽象的动作。

## **个人思考**

这篇论文的想法很巧妙，实验也做得十分grounded，但是应用在实际情况中是否真的会很有用呢？我个人感觉按照文中介绍的先利用Horde结构获取值函数标签，然后通过矩阵分解、监督学习等一系列操作学习 $V(s,g,\theta)$，这种方法看起来有些过于复杂了，直接bootstrapping学 $V(s,g,\theta)$ 效果确实会非常差吗？可能需要通过实验来验证。

另外，这篇论文避开了目标是什么以及如何表示这一问题，而是简单地考虑了目标就是状态这一场景，比较具有局限性，我目前还没有想到在我的课题中有什么类似的场景。

## **参考文献**

[1] Schaul, Tom (Google DeepMind, 5 New Street Square, London; EC4A 3TW, United Kingdom); Horgan, Dan; Gregor, Karol; Silver, David Source: 32nd International Conference on Machine Learning, ICML 2015, v 2, p 1312-1320, 2015, 32nd International Conference on Machine Learning, ICML 2015

[2] Sutton, Richard S, Modayil, Joseph, Delp, Michael, Degris, Thomas, Pilarski, Patrick M, White, Adam, and Precup, Doina. Horde: A scalable real-time architecture for learning knowledge from unsupervised sensorimotor interaction. In The 10th International Conference on Autonomous Agents and Multiagent Systems-Volume 2, pp. 761–768, 2011.
