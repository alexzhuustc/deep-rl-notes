# 【元强化学习】MAESN：一种让智能体学习如何高效探索（explanation）的meta-RL算法

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2022-05-10  
来源：[https://zhuanlan.zhihu.com/p/508845258](https://zhuanlan.zhihu.com/p/508845258)

[知识库首页](../../README.md)

---

今天介绍论文《Meta-Reinforcement Learning of Structured Exploration Strategies》。

> 这篇论文从提升智能体对任务的探索能力这一角度出发，以[MAML](经典meta-RL算法之MAML--508512375.md)为基础，设计了MAESN算法，想法比较新颖，整体算法也不难，可以看一看。推荐阅读指数：★★★★

在算法背景部分，作者首先讨论了以往meta-RL算法所存在的问题：

1. 在[RL^2](RL^2：meta-RL算法的开山之作--508387238.md)、MAML等算法中，策略的随机性都局限于动作分布所造成的固定噪声，这限制了智能体对新策略的探索能力；
2. 以RL$^2$为例的基于RNN的meta-RL算法，需要通过RNN的前向传播模型学习一整个RL算法，这一类算法通常没有办法达到与用RL算法直接学习（learning from scratch）一样的性能。

可以看出，作者的主要观点就是以往的meta-RL算法中智能体对新策略的探索能力不够，为此，本文从如何**高效探索**的角度设计了一种新的meta-RL算法——MAESN （model agnostic exploration with structured noise）。MAESN以**MAML**为基础，在策略的输入部分增加了一个隐层特征作为随机噪声，此时元学习的目标就是让智能体学会如何通过同时对策略参数以及隐层特征的少量梯度更新迅速适应新任务，因此MAESN可以被看做是**噪声加强版的MAML**，因此该算法也属于**gradient-based**类。

## **MAML**

[XuanAxuan：【元强化学习】经典meta-RL算法之MAML](经典meta-RL算法之MAML--508512375.md)

回顾一下在meta-RL中，我们所考虑的是服从一定分布 $p(\tau)$ 的一系列任务，其中每个任务均可以用一个不同的MDP $M_i=(S,A,T_i,R_i)$ 表示，其中 $S$ 为状态空间，$A$ 为动作空间，$T_i$ 为状态转移分布，$R_i$ 是奖励函数，不同的任务的奖励函数与状态转移分布不同。Meta-RL的目标是学习一个策略，该策略能够适应不同任务，从而在新任务中快速获得较高的期望收益。

在上一篇文章中，我们已经介绍过，MAML的目标是训练得到模型的最优初始参数，使得该模型能够通过标准的梯度上升迅速地适应新任务，其元学习的目标可以写为



$$
\begin{aligned} \max _{\theta} \sum_{\tau_{i}} & \mathbb{E}_{\pi_{\theta^{\prime}}}\left[\sum_{t} R_{i}\left(s_{t}\right)\right] \\ \theta^{\prime} &=\theta+\alpha \mathbb{E}_{\pi_{\theta}}\left[\sum_{t} R_{i}\left(s_{t}\right) \nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right)\right] \end{aligned} \\
$$



## **MAESN**

MAESN在MAML的基础上添加了一个隐层状态（latent state）作为随机噪声，这样策略的随机性就同时来源于动作分布以及隐层空间。

### **Policies with Latent State**

![](../../assets/images/788820f99bf8a528a7c3.jpg)

作者认为标准的随机策略 $\pi_{\theta}(a|s)$ 在不同时刻是相互独立的，缺少在trajectory上时间连贯（temporally coherent）的随机性，这限制了智能体的探索能力。为此，作者提出在每个episode产生一个随机特征向量 $z$ 作为策略的额外输入， $z$ 服从一个可学的隐层分布（latent distribution），该分布与任务有关。由于 $z$ 每个episode只采样一次，因此它们提供了时间上连贯的随机性。最终的策略可以写为 $\pi_{\theta}(a|s,z)$，其中 $z\sim \mathcal{N}(\mu,\sigma)$，$\mu$ 与 $\sigma$ 均为可学的参数。隐层特征的引入使得智能体能够在每个episode采样得到一整个与任务相关的行为特征或者目标，而非仅仅只是依赖于不同时刻相互独立的动作，这样智能体就获得了更加连贯的探索行为。

对于这一部分，我的理解是，$z$ 的引入将策略的随机探索划分成了两个层级，其中每个episode采样得到的 $z$ 提供的是在时间上跨度更大的一种随机性，该随机性有利于智能体根据当前任务的MDP调整其整体的策略探索；而策略本身的动作随机分布则提供了每个时刻智能体具体行为的随机性，该随机性主要是让智能体对不同的动作进行探索，这是随机策略原本就有的固有特征。这种不同时间跨度的探索有点类似于分层强化学习中的**时间抽象（temporal abstraction）**。

### **Meta-Learning Latent Variable Policies**

MAESN中的meta-training包含两部分：1）训练得到一个最优的**初始策略参数** $\theta$（与MAML相同）；2）为每个任务得到一个**初始隐层分布参数** $\left\{\left(\mu_{0}, \sigma_{0}\right),\left(\mu_{1}, \sigma_{1}\right) \ldots\right\}$（MAESN特有）。为了同时达到这两个目标，作者提出结合**变分推理（variational inference）**与MAML。具体而言，作者提出在MAML原有优化目标的基础上增加每个任务的初始隐层分布与先验分布之间的KL散度作为额外目标，本文中将先验分布设定为单位高斯分布。在元学习的每一次迭代中，我们首先根据 $\theta$，$(\mu_i,\sigma_i)$ 确定策略，然后对 $\theta$，$(\mu_i,\sigma_i)$ 进行inner梯度更新（$\theta$ 的更新可选）得到更新后的参数 $\theta'$，$(\mu_i',\sigma_i')$。然后根据 $\theta'$，$(\mu_i',\sigma_i')$ 在测试集上的性能对 $\theta,\left(\mu_{0}, \sigma_{0}\right),\left(\mu_{1}, \sigma_{1}\right) \ldots$ 进行元更新。这一过程的完整数学表达式如下：



$$
\begin{aligned} &\max _{\theta, \mu_{i}, \sigma_{i}} \sum_{i \in \text { tasks }} E_{a_{t} \sim \pi\left(a_{t} \mid s_{t} ; \theta_{i}^{\prime}, z_{i}^{\prime}\right)}\left[\sum_{z_{i}^{\prime} \sim \mathcal{N}\left(\mu_{i}^{\prime}, \sigma_{i}^{\prime}\right)} R_{i}\left(s_{t}\right)\right]-\\ &\sum_{i \in \text { tasks }} D_{K L}\left(\mathcal{N}\left(\mu_{i}, \sigma_{i}\right) \| \mathcal{N}(0, I)\right)\\ &\mu_{i}^{\prime}=\mu_{i}+\alpha_{\mu} \circ \nabla_{\mu_{i}} E_{a_{t} \sim \pi\left(a_{t} \mid s_{t} ; \theta, z_{i}\right)}\left[\sum_{z_{i} \sim \mathcal{N}\left(\mu_{i}, \sigma_{i}\right)} R_{t}\left(s_{t}\right)\right]\\ &\sigma_{i}^{\prime}=\sigma_{i}+\alpha_{\sigma} \circ \nabla_{\sigma_{i}} E_{a_{t} \sim \pi\left(a_{t} \mid s_{t} ; \theta, z_{i}\right)}\left[\sum_{t} R_{i} \left(s_{t}\right)\right]\\ &\theta_{i}^{\prime}=\theta+\alpha_{\theta} \circ \nabla_{\theta} E_{a_{t} \sim \pi\left(a_{t} \mid s_{t} ; \theta, z_{i}\right)} \underset{z_{i} \sim \mathcal{N}\left(\mu_{i}, \sigma_{i}\right)}{ }\left[\sum_{t} R_{i}\left(s_{t}\right)\right] . \end{aligned}
$$



上面表达式中 $\theta$ 到 $\theta'$ 的更新时可选的，实际上，作者表明忽略这一更新而仅仅依赖于隐层特征分布的更新能够获得更好的性能。另外，作者还表明上面的学习率 $\alpha$ 通过元学习自动获得对于取得良好性能也十分关键。

通过上述过程，最终每个任务的隐层分布参数 $(\mu_i,\sigma_i)$ 均能够接近所设定的先验分布，这样即便在我们没有获得 $(\mu,\sigma)$ 一个比较好的初始化，或者只能直接将先验分布作为初始分布时，我们也能够在新任务上获得较好的性能。

我对于这里先验分布的理解是，作者在元学习的目标函数中增加了每个任务的初始隐层分布与先验分布之间的KL散度作为额外目标，这样就保证了训练结束之后每个任务的隐层分布都接近于我们所设定的先验分布，注意这里是初始分布，在inner adaptation之后，该分布会根据不同任务的特征进行梯度更新而发生变化，变化之后的分布可以看做已知一定任务信息之后所得到的后验分布。先验分布暗含了历史经验的信息，而后验分布则包含了智能体对新任务的认知。之所以要设定这样一个先验分布，是因为当新任务出现时，我们并不知道该新任务的 $\mu_i$ 和 $\sigma_i$，此时就可以直接拿先验分布作为初始分布。我的疑惑是，为什么不能在训练的时候就直接把先验分布作为初始分布呢？为什么要费那么老大劲训练让每个任务的分布接近先验分布而不是直接让其与先验分布一致呢？

综上，MAESN的工作可以总结：1）引入隐层状态增加策略的探索；2）通过对初始策略参数以及初始隐层分布的优化让智能体获得迅速适应新任务的能力。与MAML一样，MAESN的inner problem用REINFORCE算法，而元训练则用的是TRPO算法，完整算法如下：

![](../../assets/images/718c6d3763e49d748b3c.jpg)

## **实验结果**

![](../../assets/images/0af473ad953a60197b80.jpg)

- 实验结果表明，MAESN的性能比learning from scratch以及其他meta-RL算法例如RL$^2$ 以及MAML的性能都要好。

![](../../assets/images/d6d345e33d7375e3ed34.jpg)

- MAESN确实提升了智能体的探索能力。

![](../../assets/images/6c978e176bcbea269573.jpg)

- 通过MAESN的训练，在新任务测试阶段，仅通过少量梯度更新，就能学习到对应不同任务的后验隐层概率分布。

## **个人总结与思考**

总结来说，MAESN实际上就是在MAML的基础上增加了一个用于随机探索的隐层状态，虽然思路比较简单，但是引入隐层特征这个想法在我看来还是比较新颖的，是一篇比较不错的工作。

## **参考文献**

[1] A. Gupta, R. Mendonca, Y. Liu, P. Abbeel, and S. Levine, ‘Meta-Reinforcement Learning of Structured Exploration Strategies’, *arXiv:1802.07245 [cs]*, Feb. 2018, Accessed: Apr. 26, 2022. [Online]. Available: <http://arxiv.org/abs/1802.07245>
