# 【深度强化学习】最大熵 RL：从Soft Q-Learning到SAC

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2022-05-11  
来源：[https://zhuanlan.zhihu.com/p/444441890](https://zhuanlan.zhihu.com/p/444441890)

[知识库首页](../../README.md)

---

> SAC（**soft** actor-critic）是一种采用**off-policy**方法训练的**随机策略**算法，该方法基于 **最大熵（maximum entropy）**框架，即策略学习的目标要在最大化收益的基础上加上一个最大化每一时刻策略的熵。根据这短短一句话的介绍，我不禁产生了两个疑问：1. **Soft** actor-critic 中的 “soft” 有什么含义？2. 通常的随机策略算法都是on-policy的，为什么SAC可以采用off-policy方法训练？  
> 这两个问题必须要同时阅读soft Q-learning以及SAC的论文才能较好的理解，首先给出答案：1. **soft** 是最大熵框架下所衍生出来的一种 **SoftMax** 操作，对应的有soft Q与soft V；2. **SAC实际上可以看做是最大熵框架下的DDPG（TD3），是off-policy的。**  
> SAC算法用起来还是比较简单的，但是该算法集成了许多创新的理论，例如最大熵、基于能量的策略模型等，涉及到了许多理论推导，接下来的内容是我根据SAC作者的三篇论文总结出来的，其中还是有一些小小的问题没有弄明白，欢迎大家交流讨论~

关于**确定性策略**与**随机策略**，我个人的理解是，对于最优行为较为确定只有1个的场景而言，确定性策略能够更快对收敛到最佳策略，而对于存在多个最优或次优行为的场景（例如多目标场景）而言，应当更倾向于让智能体学习到一个较为随机的策略，使其能够探索到更多好行为。SAC针对的就是后一种场景，它希望策略尽可能地随机。而目前大多随机策略算法，例如TRPO、PPO、A3C等，都是**on-policy**算法，这些算法的每一次梯度更新都需要收集大量的新样本，样本效率较低；**Off-policy** 可以重复利用过去的经验（样本），然而当前较为常见的off-policy DRL算法均是以DDPG为基础的确定性策略算法（DDPG、TD3、D4PG）。这类算法样本效率很高，但只能学习到确定性的策略，并且对超参数较为敏感，训练不稳定。

（关于确定性策略与随机策略的区别可以见：

[强化学习中，确定性策略和随机策略的区别，以及各自经典的算法是什么?](https://www.zhihu.com/question/65005739/answer/2262274837)

在这样的背景下，SAC提出用**off-policy**方法训练一个**随机策略**。SAC算法采用的是**最大熵（maximum entropy）**框架，该框架在标准最大化奖励目标的基础上增加了一个最大化熵的项。最大熵能够使得策略更加随机，从而增加智能体的探索率，还能够降低算法对模型与估计误差的敏感性，使得算法更稳健。

## **Maximum Entropy 框架**

### **Maximum Entropy Policies**

标准RL框架的目标是学习到一个能够最大化期望收益（奖励之和的期望）的最优策略，即：



$$
\pi_{\mathrm{std}}^{*}=\arg \max _{\pi} \sum_{t} \mathbb{E}_{\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right) \sim \rho_{\pi}}\left[\gamma^tr\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)\right] \\
$$



最大熵RL框架在奖励的基础上增加了一个 **熵（entropy）**的项，此时最优策略变为：



$$
\pi_{\mathrm{MaxEnt}}^{*}=\arg \max _{\pi} \sum_{t} \mathbb{E}_{\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right) \sim \rho_{\pi}}\left[\gamma^t(r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)+\alpha \mathcal{H}\left(\pi\left(\cdot \mid \mathbf{s}_{t}\right)\right)\right)] \\
$$



其中 $\alpha$ 表示温度因子，用于平稳熵与奖励之间的相对重要程度， 熵 $\mathcal{H}\left(\pi\left(\cdot \mid \mathbf{s}_{t}\right)\right)$ 的计算公式为：



$$
\mathcal{H}\left(\pi\left(\cdot \mid \mathbf{s}_{t}\right)\right)=\mathbb{E}_{a\sim\pi(\cdot|s_t)}[-\log\pi(a|s_t)] \\
$$



它表示了随机策略 $\pi\left(\cdot \mid \mathbf{s}_{t}\right)$ 对应概率分布的随机程度，策略越随机，熵越大。

### **Soft Value Function**

分别定义**soft Q-function**（最优）与**soft value function**（最优）为

$$
\begin{aligned} &Q_{\mathrm{soft}}^{*}\left(\mathbf{~s}_{t}, \mathbf{a}_{t}\right)=r_{t}+ \mathbb{E}_{\left(\mathbf{s}_{t+1}, \ldots\right) \sim \rho_{\pi_{\mathrm{MaxEnt}}^{*} }}\left[\sum_{l=1}^{\infty} \gamma^{l}\left(r_{t+l}+\alpha \mathcal{H}\left(\pi_{\mathrm{MaxEnt}}^{*}\left(\cdot \mid \mathbf{s}_{t+l}\right)\right)\right)\right] \end{aligned}
$$

$$
V_{\mathrm{soft}}^{*}\left(\mathbf{s}_{t}\right)=\alpha \log \int_{\mathcal{A}} \exp \left(\frac{1}{\alpha} Q_{\mathrm{soft}}^{*}\left(\mathbf{s}_{t}, \mathbf{a}^{\prime}\right)\right) d \mathbf{a}^{\prime}
$$

这两个公式均有星号上标，表示这里的Q与V均对应最优策略 $\pi_{\mathrm{MaxEnt}}^{*} $，与Q-learning中直接求解最优策略的Q值类似。为了方便理解，我们回顾一下标准RL框架中最优Q与最优V的计算公式：

$$
\begin{aligned} &Q_{\mathrm{std}}^{*}\left(\mathbf{~s}_{t}, \mathbf{a}_{t}\right)=r_{t}+ \mathbb{E}_{\left(\mathbf{s}_{t+1}, \ldots\right) \sim \rho_{\pi^*}}\left[\sum_{l=1}^{\infty} \gamma^{l}r_{t+l}\right] \end{aligned}
$$

$$
V_{\mathrm{std}}^{*}\left(\mathbf{~s}_{t}\right) =\max_a Q_{\mathrm{std}}^{*}\left(\mathbf{~s}_{t}, \mathbf{a}\right)
$$

很明显， $Q_{\mathrm{soft}}^*$ 就是把标准框架中 $Q_{\mathrm{std}}^{*}$ 的计算公式中的奖励替换成了奖励加熵，这是标准RL框架在最大熵框架下的直接扩展；然而，不同于标准RL框架中，$Q_{\mathrm{std}}^{*}$ 取最大值就得到了 $V_{\mathrm{std}}^{*}$，在最大熵框架下，$V_{\mathrm{soft}}^{*}$ 值函数是一种 **SoftMax**（$\operatorname{softmax}_{\mathbf{a}} f(\mathbf{a}):=\log \int \exp f(\mathbf{a}) d \mathbf{a}$） 形式，这是一种smooth maximum，即平滑地取最大值，$\alpha\rightarrow0$ 时，SoftMax 就退化成了HardMax，此时 $V_{\mathrm{soft}}^{*}$ 与 $V_{\mathrm{std}}^{*}$ 的形式完全一致。

$Q_{\mathrm{soft}}^*$ 是根据最大熵目标直接定义得到的，而 $V_{\mathrm{soft}}^{*}$ 的定义应该是根据 $Q_{\mathrm{soft}}^*$ 的贝尔曼方程推导而来的（见下一节）。直观来看， $\alpha$ 是一个调节策略随机性的温度因子，当 $\alpha$ 较大时，策略很随机，此时 $V_{\mathrm{soft}}^{*}$ 的计算也应当遵从这样一种随机性，而不是确定性地取 $Q_{\mathrm{soft}}^*$ 的最大值，当 $\alpha=0$ 时，相当于此时策略已经趋近于确定性了，可以直接取 $Q_{\mathrm{soft}}^*$ 的最大值作为 $V_{\mathrm{soft}}^{*}$。大概这就是为什么此时价值被称为“soft”价值吧。

### **Soft Bellman Equation（最优）**

上面定义的soft Q满足以下**soft bellman equation**（最优）：

$Q_{\mathrm{soft}}^{*}\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)=r_{t}+\mathbb{E}\left[\gamma \operatorname{softmax}_{\mathbf{a}} Q_{\mathrm{soft}}^{*}\left(\mathbf{s}_{t+1}, \mathbf{a}\right)\right]$ $Q_{\mathrm{soft}}^{*}\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)=r_{t}+\gamma \mathbb{E}_{\mathbf{s}_{t+1} \sim p_{\mathrm{s}}}\left[V_{\mathrm{soft}}^{*}\left(\mathbf{s}_{t+1}\right)\right]$

这里的因果关系是，首先根据 $Q_{\mathrm{soft}}^*$ 的定义可以推导得到第一个贝尔曼方程，根据该贝尔曼方程，我们可以类似于标准框架 $Q_{\mathrm{std}}^{*}$ 与 $V_{\mathrm{std}}^{*}$ 之间的关系建立 $Q_{\mathrm{soft}}^{*}$ 与 $V_{\mathrm{soft}}^{*}$ 之间的联系，这样应该就好理解了！

看到这里有没有觉得这个 $V_{\mathrm{soft}}^{*}\left(\mathbf{s}_{t}\right)$ 的公式怎么这么复杂这么难理解！看完SAC我发现这里可能是作者没有意识到可以直接用下面这个贝尔曼方程



$$
Q_{\mathrm{soft}}^{*}\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)=r_{t}+\gamma \mathbb{E}_{\mathbf{s}_{t+1} \sim p_{\mathrm{s}},\mathbf{a}_{t+1} \sim p_{\mathrm{a}}}\left[Q_{\mathrm{soft}}^{*}\left(\mathbf{s}_{t+1}, \mathbf{a}_{t+1}\right)-\alpha\log\pi^*_{\mathrm{MaxEnt}}(\mathbf a_{t+1}|\mathbf s_{t+1})\right] \\
$$



以及直接根据Q与V和收益之间的关系得到：



$$
V_{\mathrm{soft}}^{*}\left(\mathbf{s}_{t}\right)=\mathbb{E}_{\mathbf a_t\sim \pi^*_{\mathrm{MaxEnt}}}[Q_{\mathrm{soft}}^{*}\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)-\alpha\log \pi^*_{\mathrm{MaxEnt}}(\mathbf{a}_{t}|\mathbf{s}_{t})] \\
$$



上面这两个公式在soft Q-learning中没有用到，但是在SAC中用到了，相反SAC中根本没有提到softmax这个概念，难道这就是一个作者的理解随着研究逐渐深入的过程？

### **Energy-based Policies**

![](../../assets/images/e2a858d5c206f645e17f.jpg)

考虑一个简单的任务：智能体找目标，此时智能体有两条路线到达目标处，上面的路线距离更短，因此更好，这体现在 $Q(s,a)$ 上就是如下图中的灰色区域所示，两个尖峰分别对应两条路线，左边更高的尖峰对应上面那条更短的路线。传统RL方法通常会采用以最大的Q值为中心的 **unimodal** 策略分布，分布会延伸到最大Q值的邻居区域来提供一些探索噪声，如下面左图所示的高斯分布。此时，智能体的探索会完全偏向于上面的路线，从而完全忽视了下面的路线。尽管这种策略分布下智能体能够学习到较好的行为，但是一旦环境发生了改变，例如从左图变成了右图，上面的路线被一个障碍物遮挡住了，由于智能体只学习到了往上走的路线，为了再次到达目标，必须从头学起。

![](../../assets/images/5123d4c4cb7b3b07e5cd.jpg)

为了避免上述 **unimodal** 所带来的问题，作者提出应当确保智能体能够在优先选择更好行为的情况下探索到各种各样可能的行为。一种可能的方式就是将策略分布定义为Q值的指数分布形式：



$$
\pi(\mathbf{a} \mid \mathbf{s}) \propto \exp Q(\mathbf{s}, \mathbf{a}) \\
$$



这种分布是玻尔兹曼分布（基于能量的模型）的一种形式，这里Q作为negative energy存在。该分布给所有动作都分配了一个非0的概率，因此智能体会更加倾向于学习到所有能够处理该任务的行为，从而适应环境变化所导致的最优行为不可用。

### **Energy-based Policies 与 Soft Q Function之间的关系**

在上面的描述中，作者对于soft Q 与 energy-based policies 的定义是相互独立，然而通过公式推导我们可以惊喜地发现最大熵框架下，依据soft Q推导得到的最优策略具有如下形式：



$$
\pi_{\text {MaxEnt }}^{*}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)=\exp \left(\frac{1}{\alpha}\left(Q_{\text {soft }}^{*}\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)-V_{\text {soft }}^{*}\left(\mathbf{s}_{t}\right)\right)\right) \\
$$



这不正与上面所提到的玻尔兹曼分布一致吗？这里 $\frac{1}{\alpha} Q_{\text {soft }}\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)$ 是negative energy，而 $\frac{1}{\alpha} V_{\mathrm{soft}}\left(\mathrm{s}_{t}\right)$ 则是 log-partition function（用于归一化）。这一理论上的结论建立了 energy-based policies 与最大熵、soft Q之间的联系，进一步证实了最大熵框架可以促进智能体学习到一个 **multimodal** 的探索型策略。

再次回顾标准RL框架下的Q-learning算法，在Q-learning中，最优策略为



$$
\mathbf a_t^*=\arg \max_\mathbf a Q_{\mathrm{std}}^*(\mathbf s_t,\mathbf a) \\
$$



这是一种确定性策略，直接取Q值最大的作为最优动作，而最大熵框架下的soft Q则对应随机策略，每个动作的概率正比于soft Q的指数。实际上Q-learning和soft Q-learning是十分类似的，只不过一个是 **HardMax** 而另一个是 **SoftMax**，这里hard与soft的对比体现在了算法的方方面面，例如Q的定义以及最优策略的推导等等。

## **Soft Q-Learning**

### **Soft Q-Iteration**

【**Soft Q-Iteration**】根据上面的公式，我们可以类似于 Q-lteration 对 $Q_{\text {soft }}^{*}$ 与 $V_{\text {soft }}^{*}$ 进行迭代更新，更新公式如下：



$$
\begin{aligned} Q^{t+1}_{\mathrm{soft}}\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right) & \leftarrow r_{t}+\gamma \mathbb{E}_{\mathbf{s}_{t+1} \sim p_{\mathrm{s}}}\left[V^t_{\mathrm{soft}}\left(\mathbf{s}_{t+1}\right)\right], \forall \mathbf{s}_{t}, \mathbf{a}_{t} \\ V^{t+1}_{\mathrm{soft}}\left(\mathbf{s}_{t}\right) & \leftarrow \alpha \log \int_{\mathcal{A}} \exp \left(\frac{1}{\alpha} Q^t_{\mathrm{soft}}\left(\mathbf{s}_{t}, \mathbf{a}^{\prime}\right)\right) d \mathbf{a}^{\prime}, \forall \mathbf{s}_{t} \end{aligned} \\
$$



最终会收敛到 $Q_{\mathrm{soft}}^{*}$ 与 $V_{\mathrm{soft}}^{*}$。最大熵策略可以迭代地根据公式（11），直至收敛。

尽管理论上我们可以根据上面的公式对soft Q、soft V以及MaxEnt策略进行迭代更新，但实际操作中存在以下问题：

1. 贝尔曼方程需要遍历所有状态与动作，难以直接应用于连续状态、连续动作空间；
2. 从Q到V的SoftMax过程需要在整个动作空间积分，难以实现；
3. 最优策略服从基于能量的概率分布，难以采样。

### **Soft Q-Learning**

为了解决上述soft Q-iteration所存在的问题，作者借鉴（deep）Q-learning的方法提出了Soft Q-learning。

### Soft Q 网络的迭代更新

由于soft Q-iteration中存在无限状态、动作空间上的积分，我们可以把它表示成一 个随机优化问题（stochastic optimization），然后用随机梯度上升方法求解。为了处理连续状态空间，首先将 $Q_{\mathrm{soft}}^{*}(\mathbf{s}_{t}, \mathbf{a}_{t})$ 表示成参数为 $\theta$ 的神经网络 $Q^\theta_{\mathrm{soft}}\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)$。注意到 $V_{\mathrm{soft}}^{*}\left(\mathbf{s}_{t}\right)$ 的计算公式中存在需要在整个动作空间中积分的SoftMax操作，难以处理，因此作者提出通过 **importance sampling** 的方法将 $V_{\mathrm{soft}}^{*}\left(\mathbf{s}_{t}\right)$ 表示成期望的形式：



$$
V_{\mathrm{soft}}^{\theta}\left(\mathbf{s}_{t}\right)=\alpha \log \mathbb{E}_{q_{\mathrm{a}^{\prime}}}\left[\frac{\exp \left(\frac{1}{\alpha} Q_{\mathrm{soft}}^{\theta}\left(\mathbf{s}_{t}, \mathbf{a}^{\prime}\right)\right)}{q_{\mathrm{a}^{\prime}}\left(\mathbf{a}^{\prime}\right)}\right] \\
$$



其中 $q_{a'}$ 可以是动作空间上的任意概率分布，这里的importance sampling与off-policy中的importance sampling作用一样，同样都是根据一个概率分布的期望来估计另一个概率分布的期望（这里是整个动作空间的积分）。

最终，$Q^\theta_{\mathrm{soft}}\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)$ 通过随机梯度上升的方法进行更新，目标函数为



$$
J_{Q}(\theta)=\mathbb{E}_{\mathrm{s}_{t} \sim q_{\mathrm{s}_{t}}, \mathrm{a}_{t} \sim q_{\mathrm{a}_{t}}}\left[\frac{1}{2}\left(\hat{Q}_{\mathrm{soft}}^{\bar{\theta}}\left(\mathrm{s}_{t}, \mathrm{a}_{t}\right)-Q_{\mathrm{soft}}^{\theta}\left(\mathrm{s}_{t}, \mathrm{a}_{t}\right)\right)^{2}\right] \\
$$



其中 $\hat{Q}_{\mathrm{soft}}^{\bar{\theta}}\left(\mathrm{s}_{t}, \mathrm{a}_{t}\right)=r_{t}+\gamma \mathbb{E}_{\mathrm{s}_{t+1} \sim p_{\mathrm{s}}}\left[V_{\mathrm{soft}}^{\bar{\theta}}\left(\mathrm{s}_{t+1}\right)\right]$ 表示目标soft Q函数，$V_{\mathrm{soft}}^{\bar{\theta}}\left(\mathrm{s}_{t+1}\right)$ 根据公式（14）计算得到，$\bar\theta$ 定期从 $\theta$ 复制而来。 $q_{s_t}$ 与 $q_{a_t}$ 可以是状态、动作空间上的任意分布，这与Q-learning中off-policy的思想是一致的，因为这里我们估计的是最优soft Q $Q_{\mathrm{soft}}^{*}(\mathbf{s}_{t}, \mathbf{a}_{t})$，它所对应的最优贝尔曼方程 $Q_{\mathrm{soft}}^{*}\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)=r_{t}+\gamma \mathbb{E}_{\mathbf{s}_{t+1} \sim p_{\mathrm{s}}}\left[V_{\mathrm{soft}}^{*}\left(\mathbf{s}_{t+1}\right)\right]$ 对于任意状态、动作对都是成立的，因此我们无需使用当前策略所对应的分布采样，而可以直接使用任意分布，且不需要importance sampling。

随机优化问题可以通过对采样得到的状态、动作样本进行随机梯度上升更新求解得到。尽管上面我们提到求期望所用的分布 $q_{s_t}$ 与 $q_{a_t}$ 可以是任意的，我们通常还是会直接使用当前策略 $\pi(\mathbf{a}_t \mid \mathbf{s}_t) \propto \exp \left(\frac{1}{\alpha} Q^\theta_{\mathrm{soft}}(\mathbf{s}_t, \mathbf{a}_t)\right)$。对于 $q_{a'}$ ，我们有更多的选择，例如均匀分布，但是均匀分布很难扩展到高维空间，因此更好的选择依然是当前策略。

到目前为止，我们已经知道了在soft Q-learning中如何对 $Q^\theta_{\mathrm{soft}}\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)$ 与 $V^\theta_{\mathrm{soft}}\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)$ 进行迭代更新，这背后的思想以及具体的实现过程与Q-learning都十分相似。但是在soft Q-learning中还存在一个问题，刚刚我们说到，为了求得$Q^\theta_{\mathrm{soft}}\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)$ 的更新目标函数以及 $V^\theta_{\mathrm{soft}}\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)$，都需要从当前策略中进行采样，然而当前策略分布 $\pi(\mathbf{a}_t \mid \mathbf{s}_t) \propto \exp \left(\frac{1}{\alpha} Q^\theta_{\mathrm{soft}}(\mathbf{s}_t, \mathbf{a}_t)\right)$ 十分复杂，从中进行采样是一个十分困难的过程（这一点不是特别明白，谁能告诉我为什么难以从这样的概率分布中采样）。为此，作者提出了一种近似采样的过程。

### 策略采样网络的更新

为了根据 $\pi(\mathbf{a}_t \mid \mathbf{s}_t) \propto \exp \left(\frac{1}{\alpha} Q^\theta_{\mathrm{soft}}(\mathbf{s}_t, \mathbf{a}_t)\right)$ 进行采样，作者基于 amortized Stein variational gradient descent (SVGD) 方法构建了一个采样网络（这个SVGD具体是什么不清楚，应该大概类似于一种求梯度的方法），**这个网络实际上就是actor-critic算法中的actor**。

首先构建一个参数为 $\phi$ 的随机神经网络 $\mathbf{a}_{t}=f^{\phi}\left(\xi ; \mathbf{s}_{t}\right)$，该神经网络可以在状态为 $s_t$ 的条件下，将一个服从任意分布如高斯分布的噪声样本 $\xi$ 映射成服从 $\pi(\mathbf{a}_t \mid \mathbf{s}_t) \propto \exp \left(\frac{1}{\alpha} Q^\theta_{\mathrm{soft}}(\mathbf{s}_t, \mathbf{a}_t)\right)$ 分布的无偏动作样本。将 $\mathbf{a}_{t}=f^{\phi}\left(\xi ; \mathbf{s}_{t}\right)$ 对应的动作分布记为 $\pi^\phi(\mathbf{a}_t \mid \mathbf{s}_t)$，$\phi$ 的更新目标是让 $\pi^\phi(\mathbf{a}_t \mid \mathbf{s}_t)$ 尽可能接近于 $\pi(\mathbf{a}_t \mid \mathbf{s}_t) \propto \exp \left(\frac{1}{\alpha} Q^\theta_{\mathrm{soft}}(\mathbf{s}_t, \mathbf{a}_t)\right)$。很容易想到度量两个概率密度分布相似程度的指标可以是KL散度，因此 $\phi$ 更新的目标函数（损失函数）可以记为



$$
\begin{aligned} &J_{\pi}\left(\phi ; \mathbf{s}_{t}\right)= \mathrm{D}_{\mathrm{KL}}\left(\pi^{\phi}\left(\cdot \mid \mathbf{s}_{t}\right) \| \exp \left(\frac{1}{\alpha}\left(Q_{\mathrm{soft}}^{\theta}\left(\mathbf{s}_{t}, \cdot\right)-V_{\mathrm{soft}}^{\theta}\right)\right)\right) \end{aligned} \\
$$



根据SGVD，$\phi$ 的更新方向可以计算为

$$
\begin{aligned} \Delta f^{\phi}\left(\cdot ; \mathrm{s}_{t}\right)=& \mathbb{E}_{\mathrm{a}_{t} \sim \pi^{\phi}}\left[\left.\kappa\left(\mathbf{a}_{t}, f^{\phi}\left(\cdot ; \mathrm{s}_{t}\right)\right) \nabla_{\mathbf{a}^{\prime}} Q_{\mathrm{soft}}^{\theta}\left(\mathrm{s}_{t}, \mathrm{a}^{\prime}\right)\right|_{\mathbf{a}^{\prime}=\mathbf{a}_{t}}\right.\\ &\left.+\left.\alpha \nabla_{\mathbf{a}^{\prime}} \kappa\left(\mathbf{a}^{\prime}, f^{\phi}\left(\cdot ; \mathbf{s}_{t}\right)\right)\right|_{\mathbf{a}^{\prime}=\mathbf{a}_{t}}\right] \end{aligned}
$$

$$
\frac{\partial J_{\pi}\left(\phi ; \mathbf{s}_{t}\right)}{\partial \phi} \propto \mathbb{E}_{\xi}\left[\Delta f^{\phi}\left(\xi ; \mathbf{s}_{t}\right) \frac{\partial f^{\phi}\left(\xi ; \mathbf{s}_{t}\right)}{\partial \phi}\right]
$$

根据这里的梯度计算公式，我们就可以采用任意基于梯度的优化方法求得最优采样网络的参数。

### **算法总结**

总结而言，**soft Q-learning算法实际上就是最大熵RL框架下的deep Q-learning又或者DDPG算法**，之所以说是DQN，是因为整体的框架类似于DQN，但是由于soft Q-learning里需要额外学习一个采样网络，因此也可以说它和DDPG是十分类似的。算法包含一个学习soft Q的critic以及一个学习策略采样网络的actor，算法在从环境中收集新的经验（样本）与更新soft Q function以及采样网络的参数之间交替进行。类似于deep Q-learning，收集到的经验会被储存在**经验回访池** $\mathcal{D}$ 中，每一次更新都会从 $\mathcal{D}$ 中取出一个随机的minibatch用于计算随机梯度上升。对于每个Q网络与策略采样网络，soft Q-learning也会类似于deep Q-learning一样分别学习两个网络，一个当前网络，一个目标网络，来增加算法的稳定性。另外，SAC的Q网络用到了与TD3一样的技巧。

![](../../assets/images/cab61578b33e520b9e26.jpg)

至此，我们已经能够回答**为什么soft Q-learning可以用off-policy的方法训练出一个随机策略**，原因有二：

1. 对于soft Q的学习，思路与Q-learning基本一致，都是直接学习最优策略的soft Q，其对应的最优贝尔曼方程对于任意状态、动作对都成立，因此采样的样本不一定非得是当前策略的样本，而可以利用off-policy样本。
2. 对于策略采样网络的学习，从上面算法的流程图中可以看出，此时实际上只利用到了经验样本的状态 $\mathbf s_t$ ，而动作都是通过对当前策略进行采样得到的，$\mathbf{a}_{t}^{(i, j)}=f^{\phi}\left(\xi^{(i, j)}, \mathbf{s}_{t}^{(i)}\right)$。在RL中，我们通常假设初始状态的分布与策略无关，在意的只是动作的分布，这里动作是从当前策略采样得到的，但对应的状态用到了经验回访池中的样本，因此还是可以称为off-policy方法。回忆一下DDPG，DDPG为什么是off-policy，因为它的策略是确定性的，在求actor损失函数时无需对动作进行采样，这里使用到的采样方法实际上与常规的随机策略是不太一样的，常规的随机策略采用on-policy，样本的状态和动作都是当前策略产生的，而这里状态是经验回放池采样得到的，动作是从当前策略采样得到的，这不就跟DDPG从actor网络输出一个确定性策略很类似吗。

## **Soft Actor-Critic（SAC）**

SAC与Soft Q-learning的思想基本上是一致的，但是SAC相比soft Q-learning更简洁！在soft Q-learning中，最优 soft Q与soft V的关系写成了



$$
V_{\mathrm{soft}}^{*}\left(\mathbf{s}_{t}\right)=\alpha \log \int_{\mathcal{A}} \exp \left(\frac{1}{\alpha} Q_{\mathrm{soft}}^{*}\left(\mathbf{s}_{t}, \mathbf{a}^{\prime}\right)\right) d \mathbf{a}^{\prime} \\
$$



而在SAC中，两者的关系简化成了



$$
V_{\mathrm{soft}}^{*}\left(\mathbf{s}_{t}\right)=\mathbb{E}_{\mathbf{a}_{t} \sim \pi}\left[Q_{\mathrm{soft}}^{*}\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)-\alpha \log \pi^*_{\mathrm{MaxEnt}}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)\right] \\
$$



下面这个公式实际上是更容易理解的，能够直接从定义中看出来，至于为什么soft Q-learning中会出现这么复杂的softmax形式，我不清楚，我也不知道这两者是否相等，那就斗胆推测它们相等吧！

另一个区别在于策略采样网络的损失函数，在soft Q-learning中，策略采样网络更新的损失函数为



$$
\begin{aligned} &J_{\pi}\left(\phi ; \mathbf{s}_{t}\right)= \mathrm{D}_{\mathrm{KL}}\left(\pi^{\phi}\left(\cdot \mid \mathbf{s}_{t}\right) \| \exp \left(\frac{1}{\alpha}\left(Q_{\mathrm{soft}}^{\theta}\left(\mathbf{s}_{t}, \cdot\right)-V_{\mathrm{soft}}^{\theta}\right)\right)\right) \end{aligned} \\
$$



SAC进一步将该损失函数计算成了



$$
\begin{align*} J_{\pi}(\phi ; \mathbf{s}_{t})&=\mathbb{E}_{\mathrm{s}_{t} \sim \mathcal{D}}\left[\mathbb{E}_{\mathrm{a}_{t} \sim \pi_{\phi}}\left[\alpha \log \left(\pi_{\phi}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)\right)-Q_{\theta}\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)\right]\right]\\ &=\mathbb{E}_{\mathbf{s}_{t} \sim \mathcal{D}, \epsilon_{t} \sim \mathcal{N}}\left[\alpha \log \pi_{\phi}\left(f_{\phi}\left(\epsilon_{t} ; \mathbf{s}_{t}\right) \mid \mathbf{s}_{t}\right)-Q_{\theta}\left(\mathbf{s}_{t}, f_{\phi}\left(\epsilon_{t} ; \mathbf{s}_{t}\right)\right)\right] \end{align*} \\
$$



相比soft Q-learning的损失函数来说，简化过后损失函数的梯度更为简单：



$$
\hat{\nabla}_{\phi} J_{\pi}(\phi)=\nabla_{\phi} \alpha \log \left(\pi_{\phi}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)\right)+\left(\nabla_{\mathbf{a}_{t}} \alpha \log \left(\pi_{\phi}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)\right)-\nabla_{\mathbf{a}_{t}} Q\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)\right) \nabla_{\phi} f_{\phi}\left(\epsilon_{t} ; \mathbf{s}_{t}\right) \\
$$



总体来说，SAC与soft Q-learning本质上是一样的，都是最大熵RL框架下推导出来的，不同之处只是SAC对其中迭代更新的过程进行了简化（实际上也不是简化，应该说进一步推导）。

有一点我比较疑惑的是，在SAC中提到将 $\pi^\phi(\mathbf{a}_t \mid \mathbf{s}_t)$ 建模成均值与方差由神经网络决定的高斯分布，这不就违背了之前作者所说的策略分布应当服从玻尔兹曼分布这一说法吗？尽管在更新过程中损失函数让 $\pi^\phi(\mathbf{a}_t \mid \mathbf{s}_t)$ 尽可能地接近 $\pi(\mathbf{a}_t \mid \mathbf{s}_t) \propto \exp \left(\frac{1}{\alpha} Q^\theta_{\mathrm{soft}}(\mathbf{s}_t, \mathbf{a}_t)\right)$，但这并不能改变 $\pi^\phi(\mathbf{a}_t \mid \mathbf{s}_t)$ 是高斯分布只有一个峰值的事实。后面作者又提到将 $\pi^\phi(\mathbf{a}_t \mid \mathbf{s}_t)$ 建模成 $\mathbf{a}_{t}=f^{\phi}\left(\xi ; \mathbf{s}_{t}\right)$，在这种情况下，高斯噪声 $\xi$ 经过神经网络的一系列非线性变换确实是可能服从玻尔兹曼分布。但是我查看了spinning up提供的SAC复现代码，里边并没有体现 $\mathbf{a}_{t}=f^{\phi}\left(\xi ; \mathbf{s}_{t}\right)$ 这一步骤，而是与普通随机策略方法一样将 $\pi^\phi(\mathbf{a}_t \mid \mathbf{s}_t)$ 建模成了高斯分布，不太懂这是为什么~

## **自动熵调节**



$$
\pi_{\mathrm{MaxEnt}}^{*}=\arg \max _{\pi} \sum_{t} \mathbb{E}_{\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right) \sim \rho_{\pi}}\left[\gamma^t(r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)+\alpha \mathcal{H}\left(\pi\left(\cdot \mid \mathbf{s}_{t}\right)\right)\right)] \\
$$



在最大熵框架下，存在一个温度因子 $\alpha$ 用于调节目标奖励与熵之间的相对重要程度，在soft Q-learning与最原始的SAC中，$\alpha$ 的值是固定的，那么如何选择合适的 $\alpha$ 呢？作者在改进版的SAC中通过将策略的期望熵约束为恒定值，设计了一种自动调节 $\alpha$ 的方案。注意这里约束的是所有状态对应下的平均熵，而不是每个状态对应的熵都固定，因为在某些状态下，好的动作与坏的动作区别较大，此时熵应该较小，而在另一些状态下，不同动作之间的区别不大，此时我们希望策略更为随机，即熵更大。因此，我们不希望所有状态的熵都是固定的，而只对期望熵进行了约束。

此时我们的目标变成了最小期望熵约束下最大化期望收益，即下面的带约束优化问题：



$$
\max _{\pi_{0: T}} \mathbb{E}_{\rho_{\pi}}\left[\sum_{t=0}^{T} r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)\right] \text { s.t. } \mathbb{E}_{\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right) \sim \rho_{\pi}}\left[-\log \left(\pi_{t}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)\right)\right] \geq \mathcal{H} \ \ \ \forall t \\
$$



可以看出当忽略约束条件时，问题就退化成了标准的RL问题。这里将策略 $\pi$ 拆分成了每一时刻的策略 $\pi_{o:T}=\{\pi_1,...,\pi_T\}$，这是为了方便每一时刻 $t$ 最优温度因子 $\alpha^*_t$ 的求解。

由于每一时刻 $t$ 的策略只会对未来奖励产生影响，因此可以采用动态规划方法，后向地对策略进行求解。将优化目标分解为



$$
\max _{\pi_{0}}\left(\mathbb{E}\left[r\left(s_{0}, a_{0}\right)\right]+\max _{\pi_{1}}\left(\mathbb{E}[\ldots]+\max _{\pi_{T}} \mathbb{E}\left[r\left(s_{T}, a_{T}\right)\right]\right)\right) \\
$$



从最后一个时刻开始，此时的优化问题为



$$
\max _{\pi_{T}} \mathbb{E}_{(\mathbf{s}_{T}, \mathbf{a}_{T})\sim\rho_{\pi_T}}\left[ r\left(\mathbf{s}_{T}, \mathbf{a}_{T}\right)\right] \text { s.t. } \mathbb{E}_{\left(\mathbf{s}_{T}, \mathbf{a}_{T}\right) \sim \rho_{\pi_T}}\left[-\log \left(\pi_{t}\left(\mathbf{a}_{T} \mid \mathbf{s}_{T}\right)\right)\right] \geq \mathcal{H} \ \ \ \forall t \\
$$



将该带约束的优化问题转换成其对偶问题，



$$
\max _{\pi_{T}} \mathbb{E}_{\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right) \sim \rho_{\pi}}\left[r\left(\mathbf{s}_{T}, \mathbf{a}_{T}\right)\right]=\min _{\alpha_{T} \geq 0} \max _{\pi_{T}} \mathbb{E}\left[r\left(\mathbf{s}_{T}, \mathbf{a}_{T}\right)-\alpha_{T} \log \pi\left(\mathbf{a}_{T} \mid \mathbf{s}_{T}\right)\right]-\alpha_{T} \mathcal{H} \\
$$



其中 $\alpha_T$ 为拉格朗日系数，实际上就是温度因子。由于目标函数是线性的，而约束条件是 $\pi_T$ 的凸函数，因此此时强对偶性存在。此时，最优策略与 $\alpha_T$ 有关，可以写为：$\alpha_{T}: \pi_{T}^{*}\left(\mathbf{a}_{T} \mid \mathbf{s}_{T} ; \alpha_{T}\right)$，最优的对偶变量 $\alpha_T$ 可以通过下面优化问题求解得到：



$$
\arg \min _{\alpha_{T}} \mathbb{E}_{\mathbf{s}_{t}, \mathbf{a}_{t} \sim \pi_{i}}\left[-\alpha_{T} \log \pi_{T}^{*}\left(\mathbf{a}_{T} \mid \mathbf{s}_{T} ; \alpha_{T}\right)-\alpha_{T} \mathcal{H}\right] \\
$$



采用同样的方法可以推导得到 $T-1$ 时刻的对偶优化问题



$$
\min _{\alpha_{T-1} \geq 0} \max _{\pi_{T-1}}\left(\mathbb{E}\left[Q_{T-1}^{*}\left(\mathrm{~s}_{T-1}, \mathrm{a}_{T-1}\right)\right]-\mathbb{E}\left[\alpha_{T-1} \log \pi\left(\mathrm{a}_{T-1} \mid \mathrm{s}_{T-1}\right)\right]-\alpha_{T-1} \mathcal{H}\right)+\alpha_{T}^{*} \mathcal{H} \\
$$



递归地求解 $\pi^*_{o:T}=\{\pi^*_1,...,\pi^*_T\}$。每一时刻 $t$ 的最优策略都是 $\alpha_t$ 的函数，因此我们可以在求解完 $Q^*_t$ 与 $\pi^*_t$ 之后求解最优的 $\alpha^*_t$：



$$
\alpha_{t}^{*}=\arg \min _{\alpha_{t}} \mathbb{E}_{\mathbf{a}_{t} \sim \pi_{t}^{*}}\left[-\alpha_{t} \log \pi_{t}^{*}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t} ; \alpha_{t}\right)-\alpha_{t} \overline{\mathcal{H}}\right] \\
$$



最终，带温度调节的SAC算法流程如下：

![](../../assets/images/7eecdde8d9a88da33a72.jpg)

### 阅读参考文献

[1] T. Haarnoja, H. Tang, P. Abbeel, and S. Levine, ‘Reinforcement Learning with Deep Energy-Based Policies’, *arXiv:1702.08165 [cs]*, Jul. 2017, Accessed: Dec. 10, 2021. [Online]. Available: <http://arxiv.org/abs/1702.08165>

[2] T. Haarnoja *et al.*, ‘Soft Actor-Critic Algorithms and Applications’, *arXiv:1812.05905 [cs, stat]*, Jan. 2019, Accessed: Dec. 08, 2021. [Online]. Available: <http://arxiv.org/abs/1812.05905>
