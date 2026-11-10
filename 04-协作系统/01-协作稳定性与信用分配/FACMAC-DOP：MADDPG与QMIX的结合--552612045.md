# 【多智能体强化学习】FACMAC/DOP：MADDPG与QMIX的结合

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2022-08-11  
来源：[https://zhuanlan.zhihu.com/p/552612045](https://zhuanlan.zhihu.com/p/552612045)

[知识库首页](../../README.md)

---

> [QMIX](QMIX 阅读笔记--433670116.md)是当前公认性能最好的多智能体强化学习算法，不能直接用于连续任务，FACMAC把QMIX引入到了[MADDPG](MADDPG 阅读笔记--436281028.md)中。实验结果表明，值函数分解的应用让FACMAC相较MADDPG有了明显的性能提升，但与QMIX相比似乎没有特别明显的优势。

在集中式训练分布式执行（centralized training with decentralised execution）框架下，有两类经典的多智能体强化学习算法，第一类基于**值函数分解（value decomposition）**，代表算法有[VDN](VDN阅读笔记--432345078.md)、[QMIX](QMIX 阅读笔记--433670116.md)等，本文简称为VD类算法；第二类基于**集中式critic**，代表算法有[MADDPG](MADDPG 阅读笔记--436281028.md)等，本文简称为MADDPG算法。由于VD类算法以DQN算法为基础算法，所以一般用于离散动作空间，代表任务有**SMAC（StarCraft Multi-Agent Challenge）**；而MADDPG类算法以DDPG算法作为基础算法，所以一般用于连续动作空间，代表任务有**MPE（multi-agent particle environments）**。总结来说，VD类算法和MADDPG类算法的不同体现在两个方面：

- **基础算法不同**，VD类基于DQN，MADDPG类基于DDPG；
- **多智能体协同方式不同**，VD类基于值函数分解，MADDPG类基于集中式critic；

由于VD类算法和MADDPG类算法所适用的环境不同，直接将两者放在一起比较并不容易。有一些论文用SMAC或者离散动作空间的MPE对VD类算法与使用了[gumbel-softmax](../../02-建模与表示/01-序列注意力与随机变量/Gumbel-Softmax：离散随机变量的重参数化（reparameterization）--551255387.md)的MADDPG类算法进行比较，结果表明**QMIX比MADDPG性能好**，我个人认为这样比较，很难表明到底是因为MADDPG不适用于离散动作空间，还是因为值函数分解的协同方式比集中式critic更优。

实际上，集中式critic确实存在**扩展性**和**信用分配（credit assignment）**等方面的问题，而值函数分解恰好可以解决这两个方面的问题，因此，一种很直接的想法是**将值函数分解应用到MADDPG类算法中**。**FACMAC（FACtoredMulti-Agent Centralised policy gradients）**（oxwhirl组）与**DOP（decomposed policy gradient method）**（清华张崇洁组）就做了这样的尝试。为了充分展现值函数分解所带来的优势，这两篇论文对FACMAC/DOP与benchmark算法（VDN、QMIX、MADDPG）分别在离散任务（SMAC）和连续任务（MPE）中的性能进行了比较，最终结果表明值函数分解确实能够给MADDPG带来较大的性能提升，并且即便是在连续任务中，MADDPG的性能也不如经过了特殊处理的QMIX（使用CEM求解连续动作空间），进一步表明了**值函数分解相对集中式critic的优势**。FACMAC与DOP的思想大体上是一致的，只是FACMAC用了QMIX的值函数分解方式（非线性），而DOP用的是VDN分解方式（线性），另外两个算法策略梯度的计算方式也略有区别。本文主要介绍FACMAC。

## **MADDPG与值函数分解**

### **MADDPG（multi-agent deep deterministic policy gradient）**

[XuanAxuan：【多智能体强化学习】[learn to cooperate] MADDPG 阅读笔记](MADDPG 阅读笔记--436281028.md)

MADDPG以DDPG算法为基础，每个智能体均需要学习一个critic与一个actor，其中critic接收所有智能体的观测-动作作为输入，而actor则只能以单个智能体的局部观测为输入，输出其动作。由于每个智能体都会单独学习一个Q值，因此每个智能体都**能够设计任意属于自己的奖励**。MADDPG背后的思想是，如果我们知道所有智能体执行的动作，那么即使每个智能体的策略都在发生变化，整个系统的环境也是平稳变化的，因为对于任意 $\mathbf{\pi}_i\ne \mathbf{\pi}_{i^{\cdot}}'$​ ，均有



$$
P\left(s^{\prime} \mid s, a_{1}, \ldots, a_{N}, \pi_{1}, \ldots, \boldsymbol{\pi}_{N}\right)=P\left(s^{\prime} \mid s, a_{1}, \ldots, a_{N}\right)=P\left(s^{\prime} \mid s, a_{1}, \ldots, a_{N}, \pi_{1}^{\prime}, \ldots, \boldsymbol{\pi}_{N}^{\prime}\right) \\
$$



实际上，MADDPG各个智能体的学习过程可以看做是**相互独立**的，只是在学习的过程中需要利用其他智能体的观测-动作信息，也就是说在训练过程中进行了**观测共享**。这样做的好处是，MADDPG可以用在**异构智能体**的场景中，这里的异构指的是不同智能体的观测空间、动作空间或奖励函数可以不同。但是，由于critic需要接收所有智能体的观测-动作作为输入，其输入会随着智能体数目的增加而增加，存在**扩展性**问题。并且，为了保证critic输入节点固定，MADDPG中智能体的**数目还必须是固定**的，不允许智能体数目动态变化。

考虑异构智能体既是MADDPG的优势，同样也使MADDPG存在一些局限。当所有智能体都共用一个系统奖励时，每个智能体所学习的critic实际上是全局critic，评估的是所有智能体的整体观测-动作对的好坏，因此单个智能体无法从中判断自身观测-动作对整个系统的影响，这也就是所谓的**信用分配（credit assignment）**问题。

### **VDN与QMIX**

[XuanAxuan：【多智能体强化学习】[learn to cooperate] VDN阅读笔记](VDN阅读笔记--432345078.md)[XuanAxuan：【多智能体强化学习】[learn to cooperate] QMIX 阅读笔记](QMIX 阅读笔记--433670116.md)

值函数分解可以解决多智能体系统中的**信用分配**问题，在VDN中，假设对于所有智能体做动作所导致的系统奖励，有一个全局Q网络，单个智能体的动作同样也会对应一个局部Q网络，VDN假设局部Q网络与全局Q网络存在线性关系，即



$$
Q\left(\left(h^{1}, h^{2}, \ldots, h^{d}\right),\left(a^{1}, a^{2}, \ldots, a^{d}\right)\right) \approx \sum_{i=1}^{d} \tilde{Q}_{i}\left(h^{i}, a^{i}\right) \\
$$



然后通过 $Q\left(\left(h^{1}, h^{2}, \ldots, h^{d}\right),\left(a^{1}, a^{2}, \ldots, a^{d}\right)\right) $ 的损失函数进行反向传播即可训练出各个智能体的局部Q网络，而无需为单个智能体设计奖励。可以看出，不同于COMA，VDN是直接为每个智能体学习到了一个局部的Q网络，该局部Q网络可直接用于离散任务中的分布式决策。我们很容易想到，可以将VDN的思想应用在actor-critic框架下，实现连续任务的控制，MAAC就用到了这样的框架。

VDN将所有的局部Q网络加起来去近似整体Q网络，即认为两者之间的关系是简单的求和，但实际上局部Q网络与整体Q网络之间的关系很复杂。另外VDN在学习期间没有利用任何额外的状态信息。为此，QMIX提出用神经网络将局部Q网络与整体Q网络之间的关系拟合成一个**单调函数**，另外QMIX还在训练过程中额外使用了全局状态信息。

## **FACMAC/DOP**

![](../../assets/images/a266cc67f5cc784de532.jpg)

### **Centralized but Factored Critic**

所有智能体共用一个集中式的critic $Q_{tot}^{\boldsymbol{\mu}}$，且该critic可以分解成：



$$
Q_{t o t}^{\boldsymbol{\mu}}(\boldsymbol{\tau}, \mathbf{u}, s ; \phi, \psi)=g_{\psi}\left(s,\left\{Q_{a}^{\mu_{a}}\left(\tau_{a}, u_{a} ; \phi_{a}\right)\right\}_{a=1}^{n}\right) \\
$$



其中 $\boldsymbol{\mu}$ 与 $\mu$ 分别表示联合策略和单个智能体策略（参数 $\theta$），$\tau$ 表示历史轨迹 $\boldsymbol{\phi}$ 与 $\phi_a$ 分别是联合动作价值函数 $Q_{tot}^{\boldsymbol{\mu}}$ 与单个智能体局部动作价值函数 $Q_a^{\mu_a}$ 的网络参数。在FACMAC中，$g_\psi$ 是一个**非线性单调函数**，跟QMIX一样，用参数为 $\psi$ 的神经网络表示，该网络被称为混合网络（mixing network）。Critic根据下面的损失函数更新参数：



$$
\mathcal{L}(\phi, \psi)=\mathbb{E}_{\mathcal{D}}\left[\left(y^{t o t}-Q_{t o t}^{\boldsymbol{\mu}}(\boldsymbol{\tau}, \mathbf{u}, s ; \phi, \psi)\right)^{2}\right] \\
$$



其中 $y^{t o t}=r+\gamma Q_{\text {tot }}^{\boldsymbol{\mu}}\left(\tau^{\prime}, \boldsymbol{\mu}\left(\tau^{\prime} ; \boldsymbol{\theta}^{-}\right), s^{\prime} ; \phi^{-}, \psi^{-}\right)$。$\mathcal{D}$ 是经验回放池，$\boldsymbol{\theta^-}$、$\boldsymbol{\phi}^-$ 与 $\psi^-$ 分别为目标actor、目标critic与目标混合网络的参数。

这里FACMAC将 $Q_{tot}^{\boldsymbol{\mu}}$ 表示成单个智能体 $Q_a^{\mu_a}$ 的非线性单调函数，在仿真实验中，FACMAC的作者还探索了其他的集中值分解模式：

1. 取消单调函数的限制，即不要求混合网络的参数均为正（FACMAC-nonmonotonic）；
2. 跟VDN一样，将$Q_{tot}^{\boldsymbol{\mu}}$ 表示成单个智能体 $Q_a^{\mu_a}$ 的和（(FACMAC-vdn）（DOP就是用的这种方式）；
3. 将$Q_{tot}^{\boldsymbol{\mu}}$ 表示成单个智能体 $Q_a^{\mu_a}$ 的和外加一个与state有关的bias（FACMAC-vdn-s）

### **集中式策略梯度**

为了对各个智能体的策略进行更新，一种比较直接的方式是类似于MADDPG，利用集中式的critic或值分解之后的局部critic**计算单个智能体策略的梯度**：



$$
\nabla_{\theta_{a}} J\left(\mu_{a}\right)=\mathbb{E}_{\mathcal{D}}\left[\left.\nabla_{\theta_{a}} \mu_{a}\left(\tau_{a}\right) \nabla_{u_{a}} Q_{t o t}^{\boldsymbol{\mu}}\left(\boldsymbol{\tau}, u_{1}, \ldots, u_{n}, s\right)\right|_{u_{a}=\mu_{a}\left(\tau_{a}\right)}\right] \\
$$



或



$$
\nabla_{\theta_{a}} J\left(\mu_{a}\right)=\mathbb{E}_{\mathcal{D}}\left[\left.\nabla_{\theta_{a}} \mu_{a}\left(\tau_{a}\right) \nabla_{u_{a}} Q_{a}^{\mu_{a}}\left(\tau_{a}, u_{a}\right)\right|_{u_{a}=\mu_{a}\left(\tau_{a}\right)}\right] \\
$$



在这种更新方式中，只有智能体 $a$ 的动作 $u_a$ 是根据当前策略 $\mu_a$ 选取的，其他智能体的动作都从经验回放池中采样得到，而经验回放池中采样得到的动作可能会和根据这些智能体当前策略选择的动作有很大差距，这就容易导致智能体 $a$ 的策略陷入次优解。而单个智能体陷入次优又会影响critic的集中式更新，从而影响到其他智能体。

在FACMAC中，作者提出对整个联合动作空间进行**集中式更新**，也就是说在估计 $Q_{tot}^{\boldsymbol{\mu}}$ 时，根据所有智能体当前的策略采样动作，此时集中式的策略梯度可以计算为



$$
\nabla_{\theta} J(\boldsymbol{\mu})=\mathbb{E}_{\mathcal{D}}\left[\nabla_{\theta} \boldsymbol{\mu} \nabla_{\boldsymbol{\mu}} Q_{t o t}^{\boldsymbol{\mu}}\left(\boldsymbol{\tau}, \mu_{1}\left(\tau_{1}\right), \ldots, \mu_{n}\left(\tau_{n}\right), s\right)\right] \\
$$



这里 $\boldsymbol{\mu}=\left\{\mu_{1}\left(\tau_{1} ; \theta_{1}\right), \ldots, \mu_{n}\left(\tau_{n} ; \theta_{n}\right)\right\}$ 是所有智能体策略的集合，且所有智能体共用一个参数为 $\theta$ 的actor网络（parameter sharing不是硬性要求）。

如果值函数分解是线性的（DOP/FACMAC-vdn）这种集中式策略梯度更新实际上就等效于分别对各个智能体进行策略梯度更新。

![](../../assets/images/75ef7f793c156aa91db1.jpg)

为了表明集中式策略梯度的优势，作者利用一个简单的矩阵实验，对比了使用集中式策略梯度的MADDPG与原始MADDPG的性能，实验结果表明原始MADDPG很快就会陷入局部最优，而使用来了集中式策略梯度的MADDPG能够稳定地达到最优。（我想这是不是因为使用了parameter sharing的缘故，原始MADDPG没有使用parameter sharing，这种情况下如果使用集中式策略梯度需要更新的参数数目很大，应该会很难训练，而在FACMAC中，各个智能体共用同一套参数，此时用集中式策略梯度能够达到比较好的效果。）

## **性能评估**

### **连续任务（MAMuJoCo/MPE）**

对比算法：MADDPG、independent DDPG、VDN with [CEM](../../03-决策与学习机制/01-策略优化与连续控制/QT-Opt：用DQN处理连续任务--552089276.md)（COVDN）、QMIX with CEM（COMIX）。

![](../../assets/images/60f4c4116fadbb75ad72.jpg)

从这些实验结果中，可以肯定的一点是，MADDPG的效果确实不好，特别是当智能体数目增加或任务难度增大时。增加了值函数分解以及集中式策略梯度之后的FACMAC能够明显提升MADDPG的性能。另外，6个实验中有4个实验都是COMIX的性能最好，这充分说明了值函数分解的优势，也就是说，**值函数分解本身就是一种非常优秀的算法，甚至不需要利用actor-critic框架，仅仅使用简单的CEM来处理连续任务，也能获得非常好的性能**。

### **离散任务（SMAC）**

对比算法：FACMAC with [gumbel-softmax](../../02-建模与表示/01-序列注意力与随机变量/Gumbel-Softmax：离散随机变量的重参数化（reparameterization）--551255387.md)、MADDPG with gumbel-softmax，COMA，CentralV，DOP with gumbel-softmax，QMIX，QPLEX

![](../../assets/images/b9c627b33e3ed5f3bf03.jpg)

实验结果表明，FACMAC在离散任务中能够达到接近甚至超过QMIX的性能（有些奇怪）。

## **消融实验**

### **集中式策略梯度**

![](../../assets/images/272690947996211ab6cc.jpg)

集中式策略梯度有优势（但好像也不明显）

### **混合网络单调性**

QMIX与VDN均假设：



$$
\underset{\mathbf{u}}{\operatorname{argmax}} Q_{t o t}(\boldsymbol{\tau}, \mathbf{u})=\left(\begin{array}{c} \operatorname{argmax}_{u^{1}} Q_{1}\left(\tau^{1}, u^{1}\right) \\ \vdots \\ \operatorname{argmax}_{u^{n}} Q_{n}\left(\tau^{n}, u^{n}\right) \end{array}\right) \\
$$



因此，QMIX中要求混合网络的参数是非负的，但是在某些任务中，各个智能体的Q值之间有更加复杂的耦合关系，此时单调的非线性函数可能难以表征局部Q值与全局Q值之间的关系，此时需要取消单调性限制。FACMAC的作者就设计了一个这样的实验来表明某些时刻需要用到非单调的函数拟合混合网络。

![](../../assets/images/63d8d84c7322695a946f.jpg)

但是在大多数实验中，FACMAC-nonmonotonic更难学，效果不如普通FACMAC。

## **总结与思考**

相比于提出FACMAC这个算法本身，我个人认为这篇论文更大的意义应该是充分地对比了QMIX与MADDPG两类算法，同时表明了QMIX的优势，实验结果也表明，FACMAC的效果可能还不如直接使用QMIX。

## **参考文献**

[1] Y. Wang, B. Han, T. Wang, H. Dong, and C. Zhang, ‘Off-Policy Multi-Agent Decomposed Policy Gradients’. arXiv, Oct. 04, 2020. Accessed: Aug. 10, 2022. [Online]. Available: <http://arxiv.org/abs/2007.12322>

[2] B. Peng *et al.*, ‘FACMAC: Factored Multi-Agent Centralised Policy Gradients’. arXiv, May 07, 2021. Accessed: Jul. 07, 2022. [Online]. Available: <http://arxiv.org/abs/2003.06709>

[3] G. Papoudakis, F. Christianos, L. Schäfer, and S. V. Albrecht, ‘Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks’, *arXiv:2006.07869 [cs, stat]*, Nov. 2021, Accessed: Dec. 07, 2021. [Online]. Available: <http://arxiv.org/abs/2006.07869>
