# 【深度强化学习】MDP Homomorphic Networks：强化学习中的群对称性（Group Symmetries）

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2023-11-08  
来源：[https://zhuanlan.zhihu.com/p/655937679](https://zhuanlan.zhihu.com/p/655937679)

[知识库首页](../../README.md)

---

> 这篇论文首先利用群论相关知识对强化学习中的对称性问题进行了分析，然后利用矩阵的方法对普通线性层网络进行改进，使其满足对称性，大大缩小了参数空间，仿真实验也验证了所提方法的有效性。整篇论文的数学推导严谨且清晰，看完之后很有收获~推荐指数：★★★★★

原文：《MDP Homomorphic Networks: Group Symmetries in Reinforcement Learning》

![](../../assets/images/dd9c5e4faab4e13620d3.jpg)

许多任务中存在对称性，例如在倒立摆问题中，让倒向左边的杆和倒向右边的杆所需要的策略是镜像对称的，除此之外，还有旋转对称性，排列对称性等等。本文的中心思想就是通过探索马尔科夫决策（Markov decision processes，MDPs）中的这种**对称性信息**来减小强化学习问题的求解空间。本文利用**MDP Homomorphic（MDP同态映射）**这一概念对这些对称性进行建模。MDP同态是MDP之间的映射关系，这种映射关系保留了原始MDP的基本结构，与此同时能够删除问题描述中的冗余信息。针对MDP同态问题，本文提出了一种名为**MDP Homomorphic Networks**的模型，该模型可以利用一个状态-动作对的经验更新所有等价（equivalent）状态-动作对的策略。具体来说，当所有等价的状态-动作对可以用一个可逆的转换联系起来时，我们将这种情况称为**group-structured（群结构化）**的，此时策略网络属于**group-equivariant neural network（群等变神经网络）**。群等变神经网络是一类自带对称性的神经网络，它们是具有平移不变性的卷积神经网络（convolutional neural network，CNN）在**transformation groups（变换群）**中的泛化。这种网络可以大大提升样本效率，因此，本文利用群等变神经网络对强化学习中的等变策略网络及不变值网络进行了设计，具体来说，本文设计了一种创新的数值方法用于设计针对任意变换群的等变层。接下来对本文所提方法进行介绍。

## **背景知识**

本文所涉及到的许多重要概念均来源于群论，因此本文首先对群论中的一些基础概念进行介绍。

### **群和变换**

**群（Group）**： 群是一个代数结构，包括一个非空集合 $G$ 和一个二元运算 $*$ ，满足以下四个性质：

1. **封闭性（Closure）**： $\forall g_1,g_2\in G$，$g_1*g_2\in G$。
2. **结合性（Associativity）**：$\forall g_1,g_2,g_3 \in G$，$(g_1*g_2)*g_3=g_1*(g_2*g_3)$。
3. **单位元素（Identity Element）**：$\exists e\in G$，使得 $\forall g\in G, g*e=e*g=g$，$e$ 即为 $G$ 的单位元素。
4. **逆元素（Inverse Element）**：$\forall g\in G$，$\exists g^{-1}\in G$，使得 $g*g^{-1}=g^{-1}*g=e$。

举例来说，整数加法群是一种最简单的群，对于整数集合$ {..., -3, -2, -1, 0, 1, 2, 3, ...}$，加法 $+$ 是二元运算。对于任何两个整数 $a$ 和 $b$，它们的和 $a + b$ 仍然是整数。零（$0$）是单位元素，对于任何整数 $a$，$a + 0 = 0 + a = a$。对于每个整数 $a$$，$-$a$ 是它的逆元素，因为 $a + (-a) = (-a) + a = 0$。

**群作用（Group action）**：群作用是给定群 $G$ 对集合 $X$ 的一种映射，它将每个群元素 $g\in G$ 映射到一个集合上的映射，这个映射可以表示为 $f_g(x):G \times X \rightarrow X$。群作用必须满足以下性质：

1. 对于群元素 $g \in G$ 和集合元素 $x \in X$，群作用 $f_g(x)$ 产生的结果仍然属于集合 $X$，即 $f_g(x) \in X$。
2. 对于群元素 $g_1, g_2 \in G$ 和集合元素 $x \in X$，有 $f_{g_1*g_2}(x) = f_{g_1}\left(f_{g_2}\left(x\right)\right)$。这表示群作用是结合的。
3. 群作用中的单位元素 $e$ 是群 $G$ 的恒等元素，即对于 $\forall x\in X$，有 $f_e(x)=x$。
4. 逆元素可以撤销对应元素的作用，即对于 $\forall x\in X$，有 $f_{g^{-1}}(f_g(x))=x$。

举例来说，整数加法群 $ {..., -3, -2, -1, 0, 1, 2, 3, ...}$ 可以在整数集合 $X$ 上进行作用，对于 $a\in G$，$x\in X$，作用 $f_a(x) = a + x$。

**轨道（orbits）**：给定一个群作用 $G$ 在集合 $X$ 上，元素 $x\in X$ 的**轨道**是由 $G$ 的所有元素对 $x$ 进行作用所产生结果的集合，可以表示为 $\mathcal{O}_x=\{f_g(x)|g\in G\}$。 $\mathcal{O}_x$ 表示了元素 $x$ 在群 $G$ 作用下的所有可能位置或状态。

**对称性群（Symmetry Group）**：对称性群是一种特殊的群，其中的元素是各种对称性操作，这些操作可以改变对象的位置、方向或形状，但不改变对象的基本性质。常见的例子是平面上一个正方形的对称性群。它包括了四种90度旋转、两种镜像操作以及一个恒等操作（不变操作）。这些操作构成了正方形的对称性群。

### **等价（Equivalence），不变（Invariance）与等变（Equivariance）**

**等价（Equivalence）**：如果函数 $f:\mathcal{X}\rightarrow \mathcal{Y}$ 将两个输入 $x,x'\in\mathcal{X}$ 映射到同一个输出，即 $f(x)=f(x')$，即说明 $x$ 与 $x'$ 是 $f$-**等价**的。所有 $f$-等价于 $x$ 的点被称为 $x$ 的**等价类（equivalent class）**

**不变（Invariance）**与**对称（Symmetries）**：等价类中的点通常存在一定的关系，这可以用一个变换操作 $L_g:\mathcal{X}\rightarrow\mathcal{X}$，其中 $g\in G$，$G$ 表示群。如果 $L_g$ 满足

$f(x)=f\left(L_q[x]\right), \quad \text { for all } g \in G, x \in \mathcal{X} \\$则可以说 $f$ 对于 $L_g$ 不变或对称，并且 $\left\{L_g\right\}_{g \in G}$ 是 $f$ 对称操作的集合。可以看出，满足该不变等式的先决条 件是，$L_g(\cdot)$ 只能将 $x$ 映射到其等价类中。 另外，如果 $f$ 对于 $L_g$ 不变，那么它对于 $G$ 上的轨道均不变，此时 $x$ 的轨道 $\mathcal{O}_x$ 即为 $x$ 通过变换操作 $L_g$ 可以得到的点的集合：



$$
\mathcal{O}_x \triangleq\left\{L_g[x] \in \mathcal{X} \mid g \in G\right\} \\
$$



**等变（Equivariance）**：与不变相关的另一个概念是等变。给定一个变换操作 $L_g:\mathcal{X}\rightarrow\mathcal{X}$，和一个映射 $f:\mathcal{X}\rightarrow\mathcal{Y}$，如果存在 $f$ 输出空间上的另一个变换操作 $K_g:\mathcal{Y}\rightarrow\mathcal{Y}$，使得



$$
K_g[f(x)]=f\left(L_g[x]\right), \quad \text { for all } g \in G, x \in \mathcal{X} \\
$$



我们就可以说 $f$ 对于 $L_g$ 等变。$L_g$ 与 $K_g$ 分别是是群 $G$ 在输入空间与输出空间的变换操作。一个等变的映射可以看做是输入元素轨道到输出元素轨道轨道之间的映射。当我们将 $K_g$ 设置为恒等函数时，**不变性就成了一种特殊的等变性**。给定 $L_g$ 和 $K_g$，我们可以求解任意满足等变性约束条件的等变函数 $f$ 的集合。对于线性变换操作和线性的 $f$，已经存在丰富的理论对 $f$ 进行求解，此时 $f$ 被称为**interwiner（交织算子）**。在等变深度学习领域，可以利用interleaving interwiner以及等变的非线性函数构造神经网络，本文提供了一种数值计算的方法求解interwiner。

### **马尔科夫决策过程**

**对称MDP（MDP with Symmetries）**：在对称MDP中 ，存在状态-动作空间上使得奖励函数和状态转移函数不变的变换集合。分别定义状态变换和与状态相关的动作变换为 $L_g: \mathcal{S} \rightarrow \mathcal{S}$ ， $K_q^s: \mathcal{A} \rightarrow \mathcal{A}$，奖励函数和转移函数的不变性可以描述为



$$
\begin{aligned} R(s, a) & =R\left(L_g[s], K_g^s[a]\right) & & \text { for all } g \in G, s \in \mathcal{S}, a \in \mathcal{A} \\ T\left(s^{\prime} \mid s, a\right) & =T\left(L_g\left[s^{\prime}\right] \mid L_g[s], K_g^s[a]\right) & & \text { for all } g \in G, s \in \mathcal{S}, a \in \mathcal{A} \end{aligned} \\
$$



可以看出，在对称MDP中，奖励函数与转移函数沿着变换 $(L_g,K_g^s)$ 所定义的轨道是**不变（invariant）**的。

**MDP同态映射（MDP Homomorphisms）**：MDP同态 $h$ 是一个定义在状态空间 $\mathcal{S} \times \mathcal{A}$ 上的 MDP $M=(\mathcal{S}, \mathcal{A}, R, T, \gamma)$ 到另一个定义在**抽象状态动作空间** $\overline{\mathcal{S}} \times \overline{\mathcal{A}}$ 上的抽象MDP $ \bar{M}=(\overline{\mathcal{S}}, \overline{\mathcal{A}}, \bar{R}, \bar{T}, \gamma)$ 之间的满射映射（surjective map）。具体来说，$h$ 由满射映射组 $\left(\sigma,\left\{\alpha_s \mid s \in \mathcal{S}\right\}\right)$ 组成，其中包含状态映射 $\sigma: \hat{\mathcal{S}} \rightarrow \overline{\mathcal{S}}$ 和状态相关的动作映射 $\alpha_s: \mathcal{A} \rightarrow \overline{\mathcal{A}}$。这些映射满足下面的条件：



$$
\begin{aligned} \bar{R}\left(\sigma(s), \alpha_s(a)\right) & \triangleq R(s, a) & \text { for all } s \in \mathcal{S}, a \in \mathcal{A}, \\ \bar{T} \left(\sigma\left(s^{\prime}\right) \mid \sigma(s), \alpha_s(a)\right) & \triangleq \sum_{s^{\prime \prime} \in \sigma^{-1}\left(s^{\prime}\right)} T\left(s^{\prime \prime} \mid s, a\right) & \text { for all } s, s^{\prime} \in \mathcal{S}, a \in \mathcal{A} . \end{aligned} \\
$$



给定一个MDP同态 $h$，如果 $\sigma(s)=\sigma\left(s^{\prime}\right)$ 且 $\alpha_s(a)=\alpha_{s^{\prime}}\left(a^{\prime}\right)$，那么两个状态-动作对 $(s,a)$ 与 $(s',a')$ 就是 $h$-**等价**的。

![](../../assets/images/1254a82103df98ceb170.jpg)

根据对称MDP可以定义一个对应的MDP同态：

**群结构化MDP同态映射（Group-structured MDP Homomorphisms）**：给定一个对称MDP，可以通过将状态-动作对 $(s,a)$ 以及它们的**等价类**映射到抽象状态-动作空间 $(\sigma(s),\alpha_s(a))$，此时，每一个状态-动作对和它们经过 $(L_g,K_g^s)$ 的变换被映射到压缩MDP相同的抽象状态-动作对：



$$
\left(\sigma(s), \alpha_s(a)\right)=\left(\sigma\left(L_g[s]\right), \alpha_{L_g[s]}\left(K_g^s[a]\right)\right) \quad \forall g \in G, s \in \mathcal{S}, a \in \mathcal{A} \\
$$



之所以将这种MDP同态映射称为群结构化的，是因为由群变换所定义轨道上的所有状态-动作对都会被映射到该MDP同态的相同抽象状态-动作对上。简单来说，**通过MDP同态映射，原有对称MDP上同一轨道上的状态-动作对在新的MDP上被映射成同一个状态-动作对**，这样做可以极大地压缩状态-动作空间，因此求解新的MDP比求解原MDP更加简单。

**最优值等价（Optimal Value Equivalence）**以及**抬升策略（Lifted Policies）**：$h$-等价的状态-动作对拥有相同的最优Q-值和最优值函数。另外，存在抽象最优Q-值 $\bar{Q}^*$ 与抽象值函数 $\bar{V}^*$，使得 $Q^*(s, a)=\bar{Q}^*\left(\sigma(s), \alpha_s(a)\right)$ 以及 $V^*(s)=\bar{V}^*(\sigma(s))$，这被称为**最优值等价**。我们可以直接在更加简单的抽象MDP中优化策略，所得到的的最优抽象策略 $\bar{\pi}(\bar{a} \mid \sigma(s))$ 可以通过**抬升（lifting）**这一过程推断得到原MDP的最优策略，抬升策略的计算公式为



$$
\pi^{\uparrow}(a \mid s) \triangleq \frac{\bar{\pi}(\bar{a} \mid \sigma(s))}{\left|\left\{a \in \alpha_s^{-1}(\bar{a})\right\}\right|}, \quad \text { for any } s \in \mathcal{S} \text { and } a \in \alpha_s^{-1}(\bar{a} \mid) \\
$$



这一节所介绍的主要是，将一个对称MDP通过MDP同态映射，转换成状态-动作空间更小的一个抽象MDP，然后直接在抽象MDP中求解最优策略，再通过抬升过程得到原MDP的最优策略。

## **方法**

### **证明：抬升策略具有不变性**

【定理】在群结构化对称的MDP中，抬升策略对于对称群 $(L_g,K_g^s)$，$\forall g\in G$， 具有不变性。

证明：给定由群 $G$ 上的变换操作 $(L_g,K_g^s)$ ，如果令 $s'=L_s[s]$，$a'=K_g^s[a]$，$g\in G$，$(s',a')$ 和 $(s,a)$ 对于对应的MDP同态映射 $\left(\sigma,\left\{\alpha_s \mid s \in \mathcal{S}\right\}\right)$ 是 $h$-等变的。因此，



$$
\pi^{\uparrow}(a \mid s)=\frac{\bar{\pi}\left(\alpha_s(a) \mid \sigma(s)\right)}{\left|\left\{a \in \alpha_s^{-1}(\bar{a})\right\}\right|}=\frac{\bar{\pi}\left(\alpha_{s^{\prime}}\left(a^{\prime}\right) \mid \sigma\left(s^{\prime}\right)\right)}{\left|\left\{a^{\prime} \in \alpha_{s^{\prime}}^{-1}(\bar{a})\right\}\right|}=\pi^{\uparrow}\left(a^{\prime} \mid s^{\prime}\right) \\
$$



这一等式具有通用性，可以应用于任何状态-动作变化 $(L_g,K_g^s)$ 上。

对于离散动作空间，考虑向量化参数，此时不变性可以重新描述为等变性。

**离散动作空间上的不变策略是等变的向量化策略**

对于离散动作空间，随机策略可以表示为各离散动作执行概率所组成的向量：



$$
\boldsymbol{\pi}(s) \triangleq\left[\pi\left(a_1 \mid s\right), \quad \pi\left(a_2 \mid s\right), \quad \ldots, \quad \pi\left(a_N \mid s\right)\right]^{\top} \\
$$



这里 $a_1,...,a_N$ 表示动作空间 $A$ 上 $N$ 个可选的离散动作。动作变换 $K_g^s$ 将动作可逆地映射成另一个动作，因此在向量策略上应用动作变换实际上就是对策略中的元素进行置换，将对应的置换矩阵记为 $\mathbf{K}_g$，有



$$
\mathbf{K}_g^{-1} \boldsymbol{\pi}(s) \triangleq\left[\pi\left(K_g^s\left[a_1\right] \mid s\right), \quad \pi\left(K_g^s\left[a_2\right] \mid s\right), \quad \ldots, \quad \pi\left(K_g^s\left[a_N\right] \mid s\right)\right]^{\top} \\
$$



抬升函数的不变性此时可以记为 $\boldsymbol{\pi}^{\uparrow}(s)=\mathbf{K}_g^{-1} \boldsymbol{\pi}^{\uparrow}\left(L_g[s]\right)$，可以进一步组成成下面的等变方程：



$$
\mathbf{K}_g \boldsymbol{\pi}^{\uparrow}(s)=\boldsymbol{\pi}^{\uparrow}\left(L_g[s]\right) \quad \text { for all } g \in G, s \in \mathcal{S}, a \in \mathcal{A} \\
$$



该等式表明抬升策略必须满足等变性约束，接下来，我们将介绍如何构造这样的网络。

### **构造MDP同态映射网络（MDP Homomorphic Networks）**

根据上面的理论，本文的目标是构造一个在一系列状态-策略变换下等变（即满足上一个公式）的神经网络。现有工作已经证明，只包含等变层以及逐点的非线性激活函数的神经网络也是等变的，因此，我们只需要构造单个等变层，然后将多个这样的等变层堆叠起来就构造了一个等变的神经网络。

**等变层（Equivariant Layers）**

本文考虑在线性层的基础上进行改进，使其满足等变性。考虑一个线性层 $\mathbf{z}^{\prime}=\mathbf{W} \mathbf{z}+\mathbf{b}$，其中 $\mathbf{W} \in \mathbb{R}^{D_{\text {out }} \times D_{\text {in }}}$ 表示权重矩阵，$\mathbf{b} \in \mathbb{R}^{D_{\text {in }}}$ 表示偏置向量。为了简化分析，将偏置融合进权重矩阵：$\mathbf{W} \mapsto[\mathbf{W}, \mathbf{b}]$，$\mathbf{z} \mapsto[\mathbf{z}, 1]^{\top}$，将该增强版权重的空间记为 $\mathcal{W}_{\text{total}}$。对于矩阵形式的线性群变换操作，$\left(\mathbf{L}_g, \mathbf{K}_g\right)$，这里 $\mathbf{L}_g$ 表示输入变换，$\mathbf{K}_g$ 表示输出变换，我们需要解一下线性方程组：



$$
\mathbf{K}_g \mathbf{W} \mathbf{z}=\mathbf{W L}_g \mathbf{z}, \quad \text { for all } g \in G, \mathbf{z} \in \mathbb{R}^{D_{\text {in }}+1} \\
$$



方程对所有 $\mathbf{z}$ 成立，因此可以丢掉 $\mathbf{z}$。我们的目标是求解所有满足该方程的权重，将这些等变权重的空间记为 $\mathcal{W}$，$\mathcal{W}$ 实际上可以写为



$$
\mathcal{W}\triangleq\{\mathbf{W}\in\mathcal{W}_{\text{total}}\mid\mathbf{K}_{g}\mathbf{W}=\mathbf{W}\mathbf{L}_{g},\text{for all }g\in G\} \\
$$



对于任意 $g\in G$，$\mathbf{K}_{g}\mathbf{W}=\mathbf{W}\mathbf{L}_{g}$ 对于 $\mathbf{W}$ 都是线性的。因此，为了找到 $\mathcal{W}$，需要求解一系列 $\mathbf{W}$ 的线性方程。

为了求解该线性方程组，首先构造一个运算器**symmetrizer** $S(\mathbf{W})$：



$$
S(\mathbf{W})\triangleq\frac{1}{|G|}\sum_{g\in G}\mathbf{K}_{g}^{-1}\mathbf{W}\mathbf{L}_{g} \\
$$



$S(\mathbf{W})$ 有三个重要性质：

**1. 对称性（Symmetric Property）**： $S(\mathbf{W})$ 是对称的，即 $S(\mathbf{W})\in\mathcal{W}$.  
 证明：只需证明 $\mathbf{W}=\mathbf{K}_{g}^{-1}\mathbf{W}\mathbf{L}_{g},\quad\mathrm{for~all~}g\in G$：  
 $\begin{aligned} \mathbf{K}_{g}^{-1}S(\mathbf{W})\mathbf{L}_{g}& =\mathbf{K}_{g}^{-1}\left(\frac{1}{|G|}\sum_{h\in G}\mathbf{K}_{h}^{-1}\mathbf{W}\mathbf{L}_{h}\right)\mathbf{L}_{g} && \operatorname{substitute}S(\mathbf{W})=\mathbf{K}_{g}^{-1}S(\mathbf{W})\mathbf{L}_{g}  \\ &=\frac{1}{|G|}\sum_{h\in G}\mathbf{K}_{g}^{-1}\mathbf{K}_{h}^{-1}\mathbf{W}\mathbf{L}_{h}\mathbf{L}_{g} \\ &=\frac{1}{|G|}\sum_{h\in G}\mathbf{K}_{hg}^{-1}\mathbf{W}\mathbf{L}_{hg}&& {\mathrm{representation~definition:~}}\mathbf{L}_{h}\mathbf{L}_{g}=\mathbf{L}_{hg}  \\ &=\frac{1}{|G|}\sum_{g'g^{-1}\in G}\mathbf{K}_{g'}^{-1}\mathbf{W}\mathbf{L}_{g'}&& \mathrm{change~of~variables~}g^{\prime}=hg,h=g^{\prime}g^{-1}  \\ &=\frac1{|G|}\sum_{g^{\prime}\in Gg}\mathbf{K}_{g^{\prime}}^{-1}\mathbf{W}\mathbf{L}_{g^{\prime}}&& g^{\prime}g^{-1}\in G\iff g^{\prime}\in Gg  \\ &=\frac{1}{|G|}\sum_{g^{\prime}\in G}\mathbf{K}_{g^{\prime}}^{-1}\mathbf{W}\mathbf{L}_{g^{\prime}}&& G=Gg  \\ &=S(\mathbf{W})&& \mathrm{definition~of~symmetrizer}.  \end{aligned} \\$ 因此，$S(\mathbf{W})\in\mathcal{W}$。

**2. 固定性（Fixing Property）**： $S(\mathbf{W})$ 可以固定任何对称权重 $\mathbf{W}$，即 $\mathbf{W}\in\mathcal{W}\implies S(\mathbf{W})=\mathbf{W}$。  
 证明：这一性质证明了 $S(\mathbf{W})$ 可以覆盖住整个等变空间 $\mathcal{W}$，即 $\mathcal{W}=\{S(\mathbf{W})\in\mathcal{W}|\mathbf{W}\in\mathcal{W}_{\mathrm{total}}\}$。  
 对于 $\mathbf{W}\in\mathcal{W}$：  
 $\begin{aligned} S(\mathbf{W})& =\frac{1}{|G|}\sum_{g\in G}\mathbf{K}_{g}^{-1}\mathbf{W}\mathbf{L}_{g} && \text{definition}  \\ &=\frac{1}{|G|}\sum_{g\in G}\mathbf{K}_{g}^{-1}\mathbf{K}_{g}\mathbf{W}&& \mathbf{W}\in\mathcal{W}\iff\mathbf{K}_{g}\mathbf{W}=\mathbf{W}\mathbf{L}_{g},\forall g\in G  \\ &=\frac{1}{|G|}\sum_{g\in G}\mathbf{w} \\ &=\mathbf{W} \end{aligned} \\$ 这一等式表明symmetrizer使得等变空间保持不变，用群论中，这被描述为 $\mathcal{W}$ 在 $G$ 下是固定的。由于 $S:\mathcal{W}_{\text{total}}\rightarrow \mathcal{W}$，总是存在矩阵 $\mathbf{W}$ 使得对于每个 $\mathbf{W} \in \mathcal{W}$，$S(\mathbf{W})=\mathbf{W}$，也就证明了 $\mathcal{W}=\{S(\mathbf{W})\in\mathcal{W}|\mathbf{W}\in\mathcal{W}_{\mathrm{total}}\}$。

**3. 幂等性（Idempotence Property）**： $S(\mathbf{W})$ 无论执行多少次结果都一样，即 $S(S(\mathbf{W}))=S(\mathbf{W})$。  
 证明：  
 $\begin{aligned} S(S(\mathbf{W}))& =S\left(\frac{1}{|G|}\sum_{h\in G}\mathbf{K}_{h}^{-1}\mathbf{W}\mathbf{L}_{h}\right)  \\ &=\frac{1}{|G|}\sum_{g\in G}\mathbf{K}_{g}^{-1}\left(\frac{1}{|G|}\sum_{h\in G}\mathbf{K}_{h}^{-1}\mathbf{W}\mathbf{L}_{h}\right)\mathbf{L}_{g} \\ &=\frac{1}{|G|}\sum_{g\in G}\left(\frac{1}{|G|}\sum_{h\in G}\mathbf{K}_{g}^{-1}\mathbf{K}_{h}^{-1}\mathbf{W}\mathbf{L}_{h}\mathbf{L}_{g}\right)&& \text{linearity of sum}  \\ &=\frac{1}{|G|}\sum_{g\in G}\left(\frac{1}{|G|}\sum_{h\in G}\mathbf{K}_{hg}^{-1}\mathbf{W}\mathbf{L}_{hg}\right)&& \text{definition of group representations}  \\ &=\frac{1}{|G|}\sum_{g\in G}\left(\frac{1}{|G|}\sum_{g^{\prime}g^{-1}\in G}\mathbf{K}_{g^{\prime}}^{-1}\mathbf{WL}_{g^{\prime}}\right)&& \mathrm{change~of~variables~}g^{\prime}=hg  \\ &=\frac{1}{|G|}\sum_{g\in G}\left(\frac{1}{|G|}\sum_{g'\in Gg}\mathbf{K}_{g'}^{-1}\mathbf{WL}_{g'}\right)&& g^{\prime}g^{-1}\in G\iff g^{\prime}\in Gg  \\ &=\frac{1}{|G|}\sum_{g\in G}\left(\frac{1}{|G|}\sum_{g'\in G}\mathbf{K}_{g'}^{-1}\mathbf{W}\mathbf{L}_{g'}\right)&& Gg=G  \\ &=\frac{1}{|G|}\sum_{g^{\prime}\in G}\mathbf{K}_{g^{\prime}}^{-1}\mathbf{W}\mathbf{L}_{g^{\prime}}&& \text{sum over constant}  \\ &=S(\mathbf{W}) \end{aligned} \\$

这些性质表明symmetrizer $S$ 可以将任意 $\mathbf{W}\in\mathcal{W}_{\text{total}}$ 投影到等变空间 $\mathcal{W}$ 中。

由于 $\mathcal{W}$ 是一系列线性方程的解集合，因此 $\mathcal{W}$ 是 $\mathcal{W}_{\text{total}}$ 的线性子空间，每个 $\mathbf{W}\in\mathcal{W}$ 都可以表示为基权重 $\{\mathbf{V}_{i}\}_{i=1}^{r}$ 的线性组合，这里 $r$ 是子空间的秩，并且 $\operatorname{span}(\{\mathbf{V}_i\}_{i=1}^r)=\mathcal{W}$。为了求得 $\mathbf{W}$ 的基，可以使用**Gram-Schmidt正交化方法**，即：

1. 首先在 $\mathcal{W}_{\text{total}}$ 采样多个权重矩阵；
2. 利用symmetrizer将这些矩阵投影到等变空间；
3. 根据这些等变矩阵，利用奇异值分解的方法求解得到等变空间的正交基。

完整步骤如下：

![](../../assets/images/ae1cf2c57b1b79aff8fc.jpg)

由此，任意等变层的权重参数可以写成基权重的的线性组合：



$$
\mathbf{W}=\sum_{i=1}^rc_i\mathbf{V}_i \\
$$



这里 $c_i$ 是可学的标量系数，$r$ 是等变空间的秩，矩阵 $\mathbf{V}_i$ 表示基矩阵。

由以上推导可以看出，为了执行该算法我们需要的只有 $\mathbf{L}_g$ 和 $\mathbf{K}_g$ ，这两个矩阵是根据问题对称特性预先定义的。

## **仿真实验**

### **实验环境：**

![](../../assets/images/78ecb97600782a36c1a6.jpg)

### **对比算法：**

- Nullspace：利用显示非对称的基权重构造最终的权重；
- Random：直接利用满秩矩阵作为基权重。
- Data.Aug：数据增强

### **结果：**

![](../../assets/images/f4e702a6eafd5282b9bc.jpg)![](../../assets/images/c02c61911695a43e3cb4.jpg)

- 本文所构造的MDP Homomorphic Network的性能优于MDP和CNN；
- 本文所用网络构造方法优于数据增强方法。

## **总结与个人思考**

这篇论文用非常严谨的数学推导对强化学习中的对称性问题进行了全面分析，所采用的方法具有通用性，全文思路清晰，重点突出，个人认为是无论是从工作本身，还是论文写作的角度，都是一篇非常优秀的文章。

## **参考文献**

[1] E. van der Pol, D. E. Worrall, H. van Hoof, F. A. Oliehoek, and M. Welling, ‘MDP Homomorphic Networks: Group Symmetries in Reinforcement Learning’. arXiv, Jan. 20, 2021. Accessed: Sep. 10, 2023. [Online]. Available: <http://arxiv.org/abs/2006.16908>
