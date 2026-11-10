# 【深度强化学习】Constraint RL for safe exploration：primal-dual optimization 算法

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2022-04-25  
来源：[https://zhuanlan.zhihu.com/p/450524919](https://zhuanlan.zhihu.com/p/450524919)

[知识库首页](../../README.md)

---

在强化学习中，智能体通过在未知环境中探索与试错来学习如何决策。大多数RL算法允许智能体自由地探索环境，并采取任意能够提升奖励的动作，然而，能够获得较高奖励的动作同时也可能会带来较大风险。而在一些实际场景中，确保**智能体的安全**至关重要。不同于标准RL只需要最大化奖励函数，此时智能体所采取的行为必须能够避免危险的情况，因此设计者需要合理地设计奖励函数，通过不同的权重系数（奖励因子与惩罚因子）在最大化奖励以及减少危险代价之间取得平衡。这其中存在着两个问题：1）需要满足的约束条件与正确的权重参数之间不存在给定的映射关系，如果惩罚因子选取得过小，智能体可能会学习到危险的行为，相反，如果惩罚因子选取得过大，智能体可能无法学习到任何东西；2）对于给定的权重系数，即使能够让智能体最终学习到满足约束的最优策略，也依然无法保证智能体在整个训练过程中都能满足约束。一种确保智能体安全性的方法是在标准马尔科夫框架中增加约束条件，把问题转变成**受限马尔科夫决策过程（constraint Markov Decision Process，CMDP）**，此时智能体的目标是在满足long-term代价约束的条件下最大化long-term奖励。这种方法能够同时解决上述的两个问题。

当前求解CMDP的算法主要包含两大类：**原始对偶优化（primal-dual optimization，PDO）**算法与 **受限策略优化（constraint policy optimization，CPO）**算法。其中，PDO算法以 **拉格朗日松弛（Lagrangian relaxation）**技术为基础，轮流更新原始域参数与对偶域参数。具体来说，原始策略参数利用 **策略梯度上升** 的方法进行更新，而对偶域的参数则采用 **对偶梯度上升** 的方法进行更新。CPO与PPO的区别在于对偶域的更新方式，在CPO中，每一次迭代都会通过求解一个精心设计的优化问题来直接求解对偶参数，这样确保了训练过程中约束条件也能够得到满足，CPO是TRPO在CMDP中的扩展。open AI 在对于 safety RL 的benchmark中提到，CPO的实际效果不如PDO，并且CPO的算法框架基于TRPO算法，而PDO可以应用在各种标准RL算法中，因此这里只介绍PDO算法。

## **Constraint RL**

### **受限马尔科夫决策过程（CMDP）**

CDMP 在 MDP 的基础上增加了对于长远折扣代价（long-term discounted costs）的约束。具体来说，假设总共有 $m$ 个代价函数 $C_1,...,C_m$，其中每个代价函数 $C_i:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\rightarrow\mathbb{R}$ 表示状态-动作对到代价之间的映射关系。类似于奖励收益，策略 $\pi$ 下的长远折扣代价收益被定义为 $C_{i}(\pi)=\mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{\infty} \gamma^{t} C_{i}\left(s_{t}, a_{t}, s_{t+1}\right)\right]$，对应的约束门限值为 $d_i$。CMDP的目标是在满足长远代价收益 $C_{i}(\pi) \leq d_{i}, \forall i \in[m]$ 的情况下最大化奖励收益 $R(\pi)$ ，即



$$
\begin{aligned} \pi^{*}=& \arg \max _{\pi \in \Pi_{\theta}} R(\pi) \\ & \text { s.t. } C_{i}(\pi) \leq d_{i}, \forall i \in[m] . \end{aligned} \\
$$



除此之外，约束条件还有多种形式，例如机会约束 $P(\sum_t C_i(s_t,a_t,s_{t+1})\ge C_i)\le d_i$，风险条件价值约束（constraints on the conditional value at risk，与最坏情况结果的一小部分相比的预期代价总和），每个状态的独立约束 $\mathbb{E}_{a\sim \pi}[C_i(s,a,s')]\le d_i, \forall s$。实际上，在Sutton的书中提到过，所有的目标与目的都可以用奖励函数来表示，因此通过合理设计代价函数所能表示出来的约束条件种类是十分广泛的。

另外，Constraint RL 与 multi-objective RL 十分相近，但是两者有着一定的区别，在constraint RL中，当约束条件得到满足时，通常存在一个饱和点（saturation point），当到达该点时，继续减少代价函数的值就不再具有任何意义，这个点对应的代价函数的值也就是约束条件的门限值，这个门限值在multi-objective RL中不存在类似的定义。

### **原始对偶优化（primal-dual optimization）**

为了求解CMDP，可以采用拉格朗日松弛技术，具体来说，上述CMDP问题的拉格朗日函数为



$$
\mathcal{L}(\pi, \lambda)=R(\pi)-\sum_{i} \lambda_{i}\left(C_{i}(\pi)-d_{i}\right) \\
$$



其中 $\lambda=\left(\lambda_{1}, \cdots, \lambda_{m}\right)$ 是拉格朗日因子。原带约束的优化问题可以转换为如下不带约束的优化问题：



$$
\left(\pi^{*}, \lambda^{*}\right)=\arg \min _{\lambda \geq 0} \max _{\pi \in \Pi_{\theta}} \mathcal{L}(\pi, \lambda) \\
$$



为了求解这个不含约束的minmax问题，标准的方法是采用迭代原始-对偶方法，即在每一次迭代中轮流更新原始策略 $\pi$ 与对偶变量 $\lambda$ 。在第 $k$ 次迭代中的原始-对偶更新过程如下：

- 固定 $\lambda = \lambda^{(k)}$，执行策略梯度上升： $\theta_{k+1}=\theta_{k}+\left.\alpha_{k} \nabla_{\theta}\left(\mathcal{L}\left(\pi(\theta), \lambda^{(k)}\right)\right)\right|_{\theta=\theta_{k}}$，其中 $\alpha_k$ 表示更新步长。这里的策略梯度既可以是 on-policy 的似然比策略梯度（REINFORCE与TRPO等），也可以是 off-policy 的确定性策略梯度（例如DDPG）。
- 固定 $\pi=\pi_k$，执行对偶更新：$\lambda^{(k+1)}=f_{k}\left(\lambda^{(k)}, \pi_{k}\right)$。CMDP中不同方法的区别就在于对偶更新函数 $f_k(\cdot)$ 的选择。例如，PDO采用简单的对偶梯度上升 $\lambda_{i}^{(k+1)}=\left[\lambda_{i}^{(k)}+\beta_{k}\left(C_{i}\left(\pi_{k}\right)-d_{i}\right)\right]^{+}$，其中 $\beta_k$ 是步长， $[x]^{+}=\max \{0, x\}$ 是对于对偶空间 $\lambda \ge 0$ 的投影。相反，CPO通过在每一次迭代中构建新的优化问题来求解对偶变量 $\lambda^{(k+1)}$ ，进一步加强了约束。

## **实际算法举例**

### **Primal-Dual DDPG for CMDPs**

这里提供DDPG算法的原始-对偶优化版本用于求解CMDP，该算法中的原始策略更新与对偶变量更新均利用经验回访池中的 off-policy 数据样本。为了方便描述，假设此时CMDP中只存在1个约束，多个约束的情况也可以轻易地扩展得到。在原始-对偶DDPG算法中，存在以下几个神经网络：

- Reward critic Q-network $Q_{R}\left(s, a \mid \theta_{R}^{Q}\right)$ 以及reward target critic Q-network $Q'_{R}\left(s, a \mid \theta_{R}^{Q'}\right)$
- Cost critic critic Q-network $Q_{C}\left(s, a \mid \theta_{C}^{Q}\right)$ 以及cost target critic Q-network $Q'_{C}\left(s, a \mid \theta_{C}^{Q'}\right)$
- Actor policy network $\mu(s|\theta^\mu)$ 以及 actor target policy network $\mu'(s|\theta^{\mu'})$

具体算法如下：

![](../../assets/images/3651461bba9ac58917dd.jpg)

可以看出，这里的算法与标准DDPG的算法区别在于：

1. 多了1个用于表征长远折扣代价的神经网络；
2. 需对对偶域的拉格朗日因子即惩罚因子进行梯度上升。

类似的，PDO还可以应用在TRPO、PPO、TD3、SAC等算法上。

### 存在的问题

在实验中我发现这种方法存在一定的问题，在实际场景中我们的约束条件是有一定物理意义的，例如一段时间内机器人的能量消耗必须小于一定的阈值，保证机器人的正常运作，这里的阈值是根据无折扣的代价之和求得的，但是在上面的算法流程中，利用了一个神经网络去拟合代价对应的Q值，这里的Q值对应的是有折扣的代价收益，与奖励对应的收益类似，因此这里的代价Q值和我们实际中的代价之和是有区别的。在open AI提供的实验中，代价比较简单，通常为1，此时open AI的代码在假设每个step都有代价，且均为1的情况下，利用有折扣代价之和与无折扣代价之和之间的等式关系：



$$
\text{The sum of discounted cost}=1+\gamma+\gamma^2+....+\gamma^n=\frac{1-\gamma}{1-\gamma^n}
$$





$$
\text{The sum of undiscounted cost}=1+1+1+....+1=n
$$





$$
\text{The sum of discounted cost}=\text{The sum of undiscounted cost}\times\frac{1-\gamma}{n(1-\gamma^n)}
$$



将实际无折扣代价之和得到的门限值转换成了有折扣代价之和得到的门限值：

![](../../assets/images/8acf07875f3fa9e2feb7.jpg)

这种方法基于每个时刻均有代价，且代价值固定的假设，在实际场景中，代价值和奖励值一样，可能存在各种各样的取值，并且不是每个step都会存在，所以这种等式变换是十分不精确的，并不适合实际场景。因此，在我个人的实验中，并没有采用神经网络对代价对应的收益进行拟合，而是直接采用on-policy的数据计算一个episode的代价之和做为代价收益，然后利用这个值对拉格朗日因子进行梯度更新。

## 参考文献

[1] A. Ray, J. Achiam, and D. Amodei, ‘Benchmarking Safe Exploration in Deep Reinforcement Learning’, p. 25.  
[2] Q. Liang, F. Que, and E. Modiano, ‘Accelerated Primal-Dual Policy Optimization for Safe Reinforcement Learning’, *arXiv:1802.06480 [cs, stat]*, Feb. 2018, Accessed: Apr. 14, 2021. [Online]. Available: <http://arxiv.org/abs/1802.06480>  
[3]J. Achiam, D. Held, A. Tamar, and P. Abbeel, ‘Constrained Policy Optimization’, *arXiv:1705.10528 [cs]*, May 2017, Accessed: Apr. 17, 2021. [Online]. Available: <http://arxiv.org/abs/1705.10528>  
[4]Y. Chow, M. Ghavamzadeh, L. Janson, and M. Pavone, ‘Risk-Constrained Reinforcement Learning with Percentile Risk Criteria’, *arXiv:1512.01629 [cs, math]*, Apr. 2017, Accessed: Apr. 17, 2021. [Online]. Available: <http://arxiv.org/abs/1512.01629>
