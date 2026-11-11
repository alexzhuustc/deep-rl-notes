# 【分层强化学习】The Option-Critic Architecture 阅读笔记

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2022-04-16  
来源：[https://zhuanlan.zhihu.com/p/430471198](https://zhuanlan.zhihu.com/p/430471198)

[知识库首页](../../README.md)

---

分层强化学习是强化学习领域中的一个分支。其将复杂问题分解成若干子问题（sub-problem），逐个解决子问题从而最终解决一个复杂问题。常见的分层强化学习方法可以分成以下3类：①基于选项(options)的学习，②基于分层局部策略(hierarchical partial policy)的学习，③基于子任务(sub-task)的学习。其中，基于options的方法是目前较为流行的一种分层强化学习方法，本文将对经典文献‘’The Option-Critic Architecture‘’进行解读。

> option-critic将[1]中所介绍的option框架推广到了一般算法中，论文中进行了大量理论的推导，具有很强的指导意义，本文推荐指数★★★★★。  
> 本文适合人群：熟练掌握强化学习且对分层强化学习有基本了解。

## **option框架**

option框架对时间扩展的动作进行了定义。一个马尔科夫**选项（option）** $\omega\in\Omega$ 可以用三元组 $\left(\mathcal{I}_{\omega}, \pi_{\omega}, \beta_{\omega}\right)$ 表示，其中 $\mathcal{I}_{\omega} \subseteq \mathcal{S}$ 表示能够初始化选择的状态集合， $\pi_\omega$ 表示选择内策略，$\beta_{\omega}: \mathcal{S} \rightarrow[0,1]$ 表示终止函数。假设对于 $\forall s \in \mathcal{S}, \forall \omega \in \Omega$ 均有 $s\in\mathcal{I}_\omega$​​，即所有选择均在任何状态下都可以存在。[2] 证明了option的存在使得 MDP 变成了 **半马尔科夫过程（Semi-Markov Decision Process）**，该过程有对应option 的最优价值函数 $V_\Omega(s)$ 与最优选择价值函数 $Q_\Omega(s,\omega)$。 在该设定下，需要解决的问题有：**学习option上的价值函数与策略（value function and policy over options）以及学习option内的策略与终止函数（intra-option policies and termination functions）**。本文将借鉴 **策略梯度** 的思想为学习 options 提供新的思路。

## **学习options**

考虑 **call-and-return** 的选项执行模型，假设一个智能体首先根据 policy over options $\pi_\Omega$ 来选择策略 $\omega$，然后根据 intra-option policy $\pi_\omega$ 选择动作直到该option终止，终止条件由 $\beta_\omega$ 决定，该过程会被多次重复。用 $\pi_{\omega,\theta}$ 与 $\beta_{\omega,\vartheta}$ 分别表示选项 $\omega$ 的intra-policy policy 与 termination function，其中 $\theta$ 与 $\vartheta$ 表示需要学习的模型参数。

根据策略梯度的思想，策略的优化目标是最大化指定状态 $s_0$ 与 option $\omega_0$ 下的期望收益： $\rho\left(\Omega, \theta, \vartheta, s_{0}, \omega_{0}\right)=\mathbb{E}_{\Omega, \theta, \omega}\left[\sum_{t=0}^{\infty} \gamma^{t} r_{t+1} \mid s_{0}, \omega_{0}\right]$。该收益同时取决于 policy over options以及intra-option policies与termination function的参数。（与原始策略梯度中优化目标唯一的不同是这里使用 $(s_0,\omega_0)$ 替代了 $s_0$ ）

### **定义价值函数**

在计算该目标函数对于 $\theta$ 与 $\vartheta$ 的梯度之前，我们需要定义一系列价值函数：

- **option-状态 价值函数**（对应 $V(s)$ ）：在状态 $s$ 下选择option $\omega$ 所能获得的期望收益  
   $Q_{\Omega}(s, \omega)=\sum_{a} \pi_{\omega, \theta}(a \mid s) Q_{U}(s, \omega, a)$ 这里的 $(s,\omega)$ 可以表示成一个 **增强状态空间（augmented state space）**， $Q_{U}: \mathcal{S} \times \Omega \times \mathcal{A} \rightarrow \mathbb{R}$ 表示选项 $\omega$ 内的动作价值函数。
- **intra-policy 动作价值函数** （对应 $Q(s,a)$ ）：在状态 $s$ 与 option $\omega$ 下选择动作 $a$​ 所能获得的期望收益  
   $Q_{U}(s, \omega, a)=r(s, a)+\gamma \sum_{s^{\prime}} \mathrm{P}\left(s^{\prime} \mid s, a\right) U\left(\omega, s^{\prime}\right)$ 函数 $U: \Omega \times \mathcal{S} \rightarrow \mathbb{R}$ 表示 抵达时（upon arrival）的option 价值函数（类似于普通强化学习中的 $V(s')$ ）。
- **抵达时的 option 价值函数** （对应 $V(s')$ ）：表示执行 option $\omega$ 后进入状态 $s'$​ 的期望收益  
   $U\left(\omega, s^{\prime}\right)=\left(1-\beta_{\omega, \vartheta}\left(s^{\prime}\right)\right) Q_{\Omega}\left(s^{\prime}, \omega\right)+\beta_{\omega, \vartheta}\left(s^{\prime}\right) V_{\Omega}\left(s^{\prime}\right)$ 这里的 $V_\Omega:\mathcal{S}\rightarrow\mathbb{R}$ 表示状态 $s$ 的价值函数，且 $V_{\Omega}\left(s^{\prime}\right)=\sum_{\omega^{\prime}} \pi_{\Omega}\left(\omega^{\prime} \mid s^{\prime}\right)Q_{\Omega}(s', \omega')$ 。

**上面的第二个表达式中，之所以要用 $U(\omega,s')$，大概是因为我们并不知道进入状态 $s'$ 后是什么option，有 $\beta_{\omega,\vartheta}$ 的概率转移到其他 $\omega'$，也有 $1- \beta_{\omega,\vartheta}$ 继续停留在 $\omega$。$U(\omega,s')$ 实际上就是概率加权的 $Q_\Omega(s',\omega')$。此时第二个表达式实际上就是对应于增强状态 $(s,\omega)$ 的动作价值函数的贝尔曼方程。**

可以看出 $Q_\Omega$ 与 $Q_U$ 均取决于 $\theta$ 与 $\vartheta$ ，他们分别对应于 policy over options 与 intra-options policies。

### **推导马尔科夫链**

为了推导策略梯度，还需要推导性能指标所对应的 **马尔科夫链（Markov chain）**，很明显这里可以使用 **增强状态空间**。

假设状态 $s_t$ 下 option $\omega_t$ 被初始化（执行），那么下一时刻增强状态转移到 $\left(s_{t+1}, \omega_{t+1}\right)$​ 的1步折扣概率为



$$
\begin{aligned} &\mathrm{P}^{(1)}_{\gamma}\left(s_{t+1}, \omega_{t+1} \mid s_{t}, \omega_{t}\right)=\sum_{a} \pi_{\omega_{t}, \theta}\left(a \mid s_{t}\right) \mathrm{P}\left(s_{t+1} \mid s_{t}, a\right)(\left.\left(1-\beta_{\omega_{t}, \vartheta}\left(s_{t+1}\right)\right) 1_{\omega_{t}=\omega_{t+1}}+\beta_{\omega_{t}, \vartheta}\left(s_{t+1}\right) \pi_{\Omega}\left(\omega_{t+1} \mid s_{t+1}\right)\right) \end{aligned} \\
$$



（这里的 $\gamma$ 表示未来时刻对现在时刻影响的衰减程度？不太明白这里为什么要用在概率上）

论文提出该过程是 **homogeneous** 的，在 **option available everywhere** 的条件下，该过程是 **各态历经（ergodic）**的，增强状态（状态-选项对）服从 **平稳（stationary）** 分布。我理解这里应该表示的是状态转移概率只与上一时刻的增强状态 $(s_{t-1},\omega_{t-1})$ 与本时刻所采取的动作 $a$ 有关，与之前的历史无关，在时间上是同性质的，实际上就是 **平稳（stationary）** 的。（不确定对不对）

类似的，从 $(s_t,\omega_{t-1})$ 转移到 $(s_{t+1},\omega_{t})$​ 的概率是



$$
\begin{aligned} &\mathrm{P}_{\gamma}^{(1)}\left(s_{t+1}, \omega_{t} \mid s_{t}, \omega_{t-1}\right)=\left(\left(1-\beta_{\omega_{t-1}}\left(s_{t}\right)\right) \mathbf{1}_{\omega_{t}=\omega_{t-1}}+\right.\left.\beta_{\omega_{t-1}}\left(s_{t}\right) \pi_{\Omega}\left(\omega_{t} \mid s_{t}\right)\right) \sum_{a} \pi_{\omega_{t}}\left(a \mid s_{t}\right) \gamma \mathrm{P}\left(s_{t+1} \mid s_{t}, a\right) \end{aligned} \\
$$



注意上面两个转移概率表示的含义是完全不一样的，其中 $\mathrm{P}^{(1)}\left(s_{t+1}, \omega_{t+1} \mid s_{t}, \omega_{t}\right)$ 表示从在 $s_t$ 下执行 $w_t$ 后转移到 $s_{t+1}$ 且执行 $w_{t+1}$ 的概率，而 $\mathrm{P}_{\gamma}^{(1)}\left(s_{t+1}, \omega_{t} \mid s_{t}, \omega_{t-1}\right)$ 则表示执行 $w_{t-1}$ 到达 $s_t$ 后 执行 $w_{t}$ 且到达 $s_{t+1}$ 的概率，两者的因果关系是不一样的，因此所对应的表达式也有所区别。

进一步的，我们还可以得到 $k$​ 步折扣概率的递归表达式：

$$
\begin{aligned} &\mathrm{P}_{\gamma}^{(k)}\left(s_{t+k}, \omega_{t+k} \mid s_{t}, \omega_{t}\right)=\sum_{s_{t+1}} \sum_{\omega_{t+1}}( \left.\mathrm{P}_{\gamma}^{(1)}\left(s_{t+1}, \omega_{t+1} \mid s_{t}, \omega_{t}\right) \mathrm{P}_{\gamma}^{(k-1)}\left(s_{t+k}, \omega_{t+k} \mid s_{t+1}, \omega_{t+1}\right)\right) \\ \end{aligned}
$$

$$
\mathrm{P}_{\gamma}^{(k)}\left(s_{t+k}, \omega_{t+k-1} \mid s_{t}, \omega_{t-1}\right)=\sum_{s_{t+1}} \sum_{\omega_{t}}( \left.\mathrm{P}_{\gamma}^{(1)}\left(s_{t+1}, \omega_{t} \mid s_{t}, \omega_{t-1}\right) \mathrm{P}_{\gamma}^{(k-1)}\left(s_{t+k}, \omega_{t+k-1} \mid s_{t+1}, \omega_{t}\right)\right)
$$

### **推导策略梯度**

接下来要推导期望折扣收益相对于 intra-option policies的参数 $\theta$ 以及 termination function 的参数 $\vartheta$ 的梯度。

### **Intra-Option Policy Gradient**

由于这里学习的是 intra-option即option内的策略，因此折扣收益应当也是给定 option条件下的，此时 $\rho\left(\Omega, \theta, \vartheta, s, \omega\right)=\mathbb{E}_{\Omega, \theta, \omega}\left[\sum_{t=0}^{\infty} \gamma^{t} r_{t+1} \mid s, \omega\right]$ 实际上就对应着 $Q_{\Omega}(s, \omega)$（相比于普通策略梯度里用 $Q(s)$，这里用增强状态替代了状态 ）（这里的推导过程应该主要参考了sutton的推导）

- （**Intra-Option Policy Gradient Theorem**）：对于一系列给定的马尔科夫option集合，这些集合的 intra-option policies的模型参数为 $\theta$，那么期望收益在初始条件为 $(s_0,\omega_0)$ 时对于 $\theta$ 的梯度为  

   $$
   \begin{aligned} &\frac{\partial Q_{\Omega}\left(s_{0}, \omega_{0}\right)}{\partial \theta}=\sum_{s, \omega} \mu_{\Omega}\left(s, \omega \mid s_{0}, \omega_{0}\right) \sum_{a} \frac{\partial \pi_{\omega, \theta}(a \mid s)}{\partial \theta} Q_{U}(s, \omega, a) \end{aligned}
   $$

   其中 $\mu_{\Omega}\left(s, \omega \mid s_{0}, \omega_{0}\right)$ 为从 $(s_0,\omega_0)$ 开始的状态-option对的**折扣因子**： $\mu_{\Omega}\left(s, \omega \mid s_{0}, \omega_{0}\right)=\sum_{t=0}^{\infty} \gamma^{t} \mathrm{P}\left(s_{t}=s, \omega_{t}=\omega \mid s_{0}, \omega_{0}\right)$，即在计算导数时对于每个增强状态的权重，即 $\theta$ 的改变对不同的 $\left(s,\omega\right)$ 有不同的影响，因此在算梯度时需要根据这些增强状态出现得概率进行加权，使得出现概率大的增强概率的权重更大，这里对应于原始策略梯度中的 $\mu(s)$。
   这里的梯度描述了策略上微小的改变对整个折扣收益的影响。

以下是推导过程：

目标是表达式中只包含 $\pi_{\omega,\theta}$ 对 $\theta$ 的求导项，因此我们需要将价值函数的求导项逐步分解。

首先利用链式求导公式，然后将价值函数的贝尔曼方程代入：



$$
\begin{aligned} &\frac{\partial Q_{\Omega}(s, \omega)}{\partial \theta}=\frac{\partial}{\partial \theta} \sum_{a} \pi_{\omega, \theta}(a \mid s) Q_{U}(s, \omega, a) \\ &=\sum_{a}\left(\frac{\partial \pi_{\omega, \theta}(a \mid s)}{\partial \theta} Q_{U}(s, \omega, a)+\right. \left.\pi_{\omega, \theta}(a \mid s) \frac{\partial Q_{U}(s, \omega, a)}{\partial \theta}\right)\\ &=\sum_{a}\left(\frac{\partial \pi_{\omega, \theta}(a \mid s)}{\partial \theta} Q_{U}(s, \omega, a)+\right. \left.\pi_{\omega, \theta}(a \mid s) \sum_{s^{\prime}} \gamma \mathrm{P}\left(s^{\prime} \mid s, a\right) \frac{\partial U\left(\omega, s^{\prime}\right)}{\partial \theta}\right) \end{aligned} \\
$$



此时只需计算 $\frac{\partial U\left(\omega, s^{\prime}\right)}{\partial \theta}$ 即可：



$$
\begin{aligned} &\frac{\partial U\left(\omega, s^{\prime}\right)}{\partial \theta}= \left(1-\beta_{\omega, \vartheta}\left(s^{\prime}\right)\right) \frac{\partial Q_{\Omega}\left(s^{\prime}, \omega\right)}{\partial \theta}+\beta_{\omega, \vartheta}\left(s^{\prime}\right) \frac{\partial V_{\Omega}\left(s^{\prime}\right)}{\partial \theta} \\ &=\left(1-\beta_{\omega, \vartheta}\left(s^{\prime}\right)\right) \frac{\partial Q_{\Omega}\left(s^{\prime}, \omega\right)}{\partial \theta}+\beta_{\omega, \vartheta}\left(s^{\prime}\right) \sum_{\omega^{\prime}} \pi_{\Omega}\left(\omega^{\prime} \mid s^{\prime}\right) \frac{\partial Q_{\Omega}\left(s^{\prime}, \omega^{\prime}\right)}{\partial \theta} \\ &=\sum_{\omega^{\prime}}\left(\left(1-\beta_{\omega, \vartheta}\left(s^{\prime}\right)\right) \mathbf{1}_{\omega^{\prime}=\omega}+\beta_{\omega,\vartheta}\left(s^{\prime}\right) \pi_{\Omega}\left(\omega^{\prime} \mid s^{\prime}\right)\right) \frac{\partial Q_{\Omega}\left(s^{\prime}, \omega^{\prime}\right)}{\partial \theta} . \end{aligned} \\
$$



上面的等式基于假设 $\theta$ 值存在于 intra-option policies中，即 policy over options $\pi_\Omega$ 与 $\theta$ 无关。将 $\frac{\partial U\left(\omega, s^{\prime}\right)}{\partial \theta}$ 代入到 $\frac{\partial Q_{\Omega}(s, \omega)}{\partial \theta}$​ 中可以得到：



$$
\begin{aligned} &\frac{\partial Q_{\Omega}(s, \omega)}{\partial \theta}=\sum_{a} \frac{\partial \pi_{\omega, \theta}(a \mid s)}{\partial \theta} Q_{U}(s, \omega, a)+ \sum_{a} \pi_{\omega, \theta}(a \mid s) \sum_{s^{\prime}} \gamma \mathrm{P}\left(s^{\prime} \mid s, a\right) \sum_{\omega^{\prime}}\left(\beta_{\omega, \vartheta}\left(s^{\prime}\right) \pi_{\Omega}\left(\omega^{\prime} \mid s^{\prime}\right)\right. \\ &\left.\quad+\left(1-\beta_{\omega, \vartheta}\left(s^{\prime}\right)\right) 1_{\omega^{\prime}=\omega}\right) \frac{\partial Q_{\Omega}\left(s^{\prime}, \omega^{\prime}\right)}{\partial \theta} \\ &=\sum_{a} \frac{\partial \pi_{\omega, \theta}(a \mid s)}{\partial \theta} Q_{U}(s, \omega, a)+ \sum_{s^{\prime}} \sum_{\omega^{\prime}} \mathrm{P}_{\gamma}^{(1)}\left(s^{\prime}, \omega^{\prime} \mid s, \omega\right) \frac{\partial Q_{\Omega}\left(s^{\prime}, \omega^{\prime}\right)}{\partial \theta} \\ &=\sum_{a} \frac{\partial \pi_{\omega, \theta}(a \mid s)}{\partial \theta} Q_{U}(s, \omega, a)  + \sum_{s^{\prime}} \sum_{\omega^{\prime}} \mathrm{P}_{\gamma}^{(1)}\left(s^{\prime}, \omega^{\prime} \mid s, \omega\right)(\sum_{a} \frac{\partial \pi_{\omega', \theta}(a \mid s)}{\partial \theta} Q_{U}(s', \omega', a)+ \\ &\sum_{s^{\prime\prime}} \sum_{\omega^{\prime\prime}} \mathrm{P}_{\gamma}^{(1)}\left(s^{\prime\prime}, \omega^{\prime\prime} \mid s', \omega'\right) \frac{\partial Q_{\Omega}\left(s^{\prime\prime}, \omega^{\prime\prime}\right)}{\partial \theta}) \\ &=\sum_{a} \frac{\partial \pi_{\omega, \theta}(a \mid s)}{\partial \theta} Q_{U}(s, \omega, a) + \sum_{s^{\prime}} \sum_{\omega^{\prime}} \mathrm{P}_{\gamma}^{(1)}\left(s^{\prime}, \omega^{\prime} \mid s, \omega\right)\sum_{a} \frac{\partial \pi_{\omega', \theta}(a \mid s)}{\partial \theta} Q_{U}(s', \omega', a)\\ &+ \sum_{s''}\sum_{\omega''}\mathrm{P}_{\gamma}^{(2)}\left(s^{\prime\prime}, \omega^{\prime\prime} \mid s, \omega\right) \frac{\partial Q_{\Omega}\left(s^{\prime\prime}, \omega^{\prime\prime}\right)}{\partial \theta}) \\ &=\sum_{k=0}^{\infty} \sum_{s^{\prime}, \omega^{\prime}} \mathrm{P}_{\gamma}^{(k)}\left(s^{\prime}, \omega^{\prime} \mid s, \omega\right) \sum_{a} \frac{\partial \pi_{\omega^{\prime}, \theta}\left(a \mid s^{\prime}\right)}{\partial \theta} Q_{U}\left(s^{\prime}, \omega^{\prime}, a\right) \end{aligned} \\
$$



从第一个等式到第二个等式利用到了增强状态的1步转移概率，而第二个等式到最后一个等式则利用了递归的思想以及 $k$ 步转移概率。

因此，最终我们可以得到



$$
\begin{aligned} &\frac{\partial Q_{\Omega}\left(s_{0}, \omega_{0}\right)}{\partial \theta}= \sum_{s, \omega} \sum_{k=0}^{\infty} \mathrm{P}_{\gamma}^{(k)}\left(s, \omega \mid s_{0}, \omega_{0}\right) \sum_{a} \frac{\partial \pi_{\omega, \theta}(a \mid s)}{\partial \theta} Q_{U}(s, \omega, a) \\ &=\sum_{s, \omega} \mu_{\Omega}\left(s, \omega \mid s_{0}, \omega_{0}\right) \sum_{a} \frac{\partial \pi_{\omega, \theta}(a \mid s)}{\partial \theta} Q_{U}(s, \omega, a) \end{aligned} \\
$$



### **Termination Gradient**

由于这里要推导的是 termination gradient，termination function决定了在执行 $w$ 时下一个状态 $s'$ 是否会继续停留在此option，因此这里的开始状态是 $(\omega_0,s_1)$ ，而对应的价值函数也应当是抵达时的option价值函数 $U(\omega,s')$，与计算 intra-policy gradient 时是有所区别的。

- （**Termination Gradient Theorem**）：对于一系列给定的马尔科夫option集合，这些集合的 termination function 的模型参数为 $\vartheta$，那么期望收益在初始条件为 $(s_1,\omega_0)$ 时对于 $\vartheta$​ 的梯度为  
   $\frac{\partial U\left(\omega_{0}, s_{1}\right)}{\partial \vartheta}=-\sum_{s^{\prime}, \omega} \mu_{\Omega}\left(s^{\prime}, \omega \mid s_{1}, \omega_{0}\right) \frac{\partial \beta_{\omega, \vartheta}\left(s^{\prime}\right)}{\partial \vartheta} A_{\Omega}\left(s^{\prime}, \omega\right)$ 其中 $\mu_{\Omega}\left(s^{\prime}, \omega \mid s_{1}, \omega_{0}\right)$ 表示从 $\left(s_1,\omega_0\right)$ 开始的状态-option对的折扣因子：$\mu_{\Omega}\left(s, \omega \mid s_{1}, \omega_{0}\right)=\sum_{t=0}^{\infty} \gamma^{t} \mathrm{P}\left(s_{t+1}=s, \omega_{t}=\omega \mid s_{1}, \omega_{0}\right)$，$A_{\Omega}\left(s^{\prime}, \omega\right)=Q_{\Omega}\left(s^{\prime}, \omega\right)-V_{\Omega}\left(s^{\prime}\right)$ 表示优势函数。\
   在普通马尔科夫过程的策略梯度算法中，优势函数通常作为 **基准（baseline）**用来减小梯度估计的方差，此时它的存在通常是与算法设计相关的。然而在本文所推导的option场景中，优势函数是直接作为策略梯度推导的结果而存在的，从公式中我们可以直观地看出：当选择的option的价值相对于所有option的平均价值而言是次优的时，优势函数是负的，此时它会指引着termination function朝着正方向增加，也就是增加该选项的终止概率，这是符合直觉的（该option不好，跳到下一个option）。

以下是推导过程：



$$
\begin{aligned} &U\left(\omega, s^{\prime}\right)=\left(1-\beta_{\omega, \vartheta}\left(s^{\prime}\right)\right) Q_{\Omega}\left(s^{\prime}, \omega\right)+\beta_{\omega, \vartheta}\left(s^{\prime}\right) V_{\Omega}\left(s^{\prime}\right) \\ &=\left(1-\beta_{\omega, \vartheta}\left(s^{\prime}\right)\right) \sum_{a} \pi_{\omega, \theta}\left(a \mid s^{\prime}\right) \left(r\left(s^{\prime}, a\right)+\sum_{s^{\prime \prime}} \gamma \mathrm{P}\left(s^{\prime \prime} \mid s^{\prime}, a\right) U\left(\omega, s^{\prime \prime}\right)\right) \\ &+\beta_{\omega, \vartheta}\left(s^{\prime}\right) \sum_{\omega^{\prime}} \pi_{\Omega}\left(\omega^{\prime} \mid s^{\prime}\right) \sum_{a} \pi_{\omega^{\prime}, \theta}\left(a \mid s^{\prime}\right)\left(r\left(s^{\prime}, a\right)+\sum_{s^{\prime \prime}} \gamma \mathrm{P}\left(s^{\prime \prime} \mid s^{\prime}, a\right) U\left(\omega^{\prime}, s^{\prime \prime}\right)\right) \end{aligned} \\
$$



利用链式求导公式并将增强状态的1步转移概率矩阵代入可以得到



$$
\begin{aligned} &\frac{\partial U\left(\omega, s^{\prime}\right)}{\partial \vartheta}=-\frac{\partial \beta_{\omega, \vartheta}\left(s^{\prime}\right)}{\partial \vartheta} A_{\Omega}\left(s^{\prime}, \omega\right)+ \quad \sum_{\omega^{\prime}} \sum_{s^{\prime \prime}} \mathrm{P}_{\gamma}^{(1)}\left(s^{\prime \prime}, \omega^{\prime} \mid s^{\prime}, \omega\right) \frac{\partial U\left(\omega^{\prime}, s^{\prime \prime}\right)}{\partial \vartheta} \\ &=-\sum_{\omega^{\prime}, s^{\prime \prime}} \sum_{k=0}^{\infty} \mathrm{P}_{\gamma}^{(k)}\left(s^{\prime \prime}, \omega^{\prime} \mid s^{\prime}, \omega\right) \frac{\partial \beta_{\omega^{\prime}, \vartheta}\left(s^{\prime \prime}\right)}{\partial \vartheta} A_{\Omega}\left(s^{\prime \prime}, \omega^{\prime}\right) \end{aligned} \\
$$



最终我们可以得到



$$
\begin{aligned} &\frac{\partial U\left(\omega_{0}, s_{1}\right)}{\partial \vartheta}= -\sum_{\omega, s^{\prime}} \sum_{k=0}^{\infty} \mathrm{P}_{\gamma}^{(k)}\left(s^{\prime}, \omega \mid s_{1}, \omega_{0}\right) \frac{\partial \beta_{\omega, \vartheta}\left(s^{\prime}\right)}{\partial \vartheta} A_{\Omega}\left(s^{\prime}, \omega\right) \\ &=-\sum_{\omega, s^{\prime}} \mu_{\Omega}\left(s^{\prime}, \omega \mid s_{1}, \omega_{0}\right) \frac{\partial \beta_{\omega, \vartheta}\left(s^{\prime}\right)}{\partial \vartheta} A_{\Omega}\left(s^{\prime}, \omega\right) \end{aligned} \\
$$



## **算法框架**

![](../../assets/images/adc8c38a4f9f80991029.jpg)

根据上面所介绍的两个策略梯度定理，本文设计了学习option的梯度上升算法。本文采用的是 **two-timescale** 框架：在较快的时间尺度上学习价值函数，而在较慢的时间尺度上学习 intra-option policies 和 termination functions。

本文借鉴了 **actor-critic** 的框架，其中 intra-option policies，termination function 以及 policy over options 均属于 actor 部分，而critic部分则包含 $Q_U$ 与 $A_\Omega$。option-critic框架没有对 $\pi_\Omega$ 的获取做任何规定，它对现有的策略梯度、时间差分更新等方法都是适用的。

### **最优贝尔曼方程推导**

论文中采用的是Q-learning算法，另外，论文中的伪代码并没有提到如何更新 $Q_\Omega(s,\omega)$ 以及对应的 $\pi_\Omega$ 策略，因此本部分本人将自行推导：

与论文一样，这里采用off-policy的方式，对最优策略的贝尔曼方程进行推导，记最优 policy over option 下的option-状态价值函数为 $Q_\Omega^*\left(s,\omega\right)$，显然我们有



$$
V_\Omega^*\left(s\right)=\max_\omega Q_\Omega^*\left(s,\omega\right) \\
$$



记最优 intra-option policies 下的 intra-option 动作价值函数为 $Q_{U}^*(s, \omega, a)$，同样可以得到



$$
Q_{\Omega}^*\left(s, \omega\right)=\max_a Q_{U}^*\left(s, \omega, a\right) \\
$$



在最优策略下，$Q_{U}^*\left(s, \omega, a\right)$ 的贝尔曼方程可以写为



$$
Q_{U}^*\left(s, \omega, a\right)=r(s, a)+\gamma \sum_{s^{\prime}} \mathrm{P}\left(s^{\prime} \mid s, a\right) U^*\left(\omega, s^{\prime}\right) \\
$$



根据 $U\left(\omega,s'\right)$ 的定义，可以得到



$$
\begin{aligned} U^{*} &\left(\omega, s^{\prime}\right)=\left(1-\beta_{\omega, \vartheta}\left(s^{\prime}\right)\right) Q_{\Omega}^{*}\left(s^{\prime}, \omega\right)+\beta_{\omega, \vartheta}\left(s^{\prime}\right) V_{\Omega}^{*}\left(s^{\prime}\right) \\ &=\left(1-\beta_{\omega, \vartheta}\left(s^{\prime}\right)\right) Q_{\Omega}^{*}\left(s^{\prime}, \omega\right)+\beta_{\omega, \vartheta}\left(s^{\prime}\right) \max _{\bar{\omega}} Q_{\Omega}^{*}\left(s^{\prime}, \bar{\omega}\right) \end{aligned}
$$



将 $U\left(\omega,s'\right)$ 代入到 $Q_{U}^*\left(s, \omega, a\right)$ 中可以得到



$$
\begin{aligned} &Q_{U}^{*}(s, \omega, a)=r(s, a)+\gamma \sum_{s^{\prime}} \mathrm{P}\left(s^{\prime} \mid s, a\right)\left(\left(1-\beta_{\omega, \vartheta}\left(s^{\prime}\right)\right) Q_{\Omega}^{*}\left(s^{\prime}, \omega\right)+\beta_{\omega, \vartheta}\left(s^{\prime}\right) \max _{\bar{\omega}} Q_{\Omega}^{*}\left(s^{\prime}, \bar{\omega}\right)\right) \\ &=r(s, a)+\gamma \sum_{s^{\prime}} \mathrm{P}\left(s^{\prime} \mid s, a\right)\left(\left(1-\beta_{\omega, \vartheta}\left(s^{\prime}\right)\right) \max _{\bar{a}} Q_{U}^{*}\left(s^{\prime}, \omega, \bar{a}\right)+\beta_{\omega, \vartheta}\left(s^{\prime}\right) \max _{\bar{\omega}, \bar{a}} Q_{U}^{*}\left(s^{\prime}, \bar{\omega}, \bar{a}\right)\right) \end{aligned}
$$



由于 $Q_\Omega^*\left(s,\omega\right)$ 与 $Q_{U}^*\left(s, \omega, a\right)$ 存在确定的关系，因此只需要对其中一个进行学习即可，论文中说的是学习 $Q_\Omega$ 然后推导得到 $Q_U$，但是以我个人的理解，应该是学习 $Q_{U}^*\left(s, \omega, a\right)$ 然后直接根据 $Q_{\Omega}^*\left(s, \omega\right)=\max_a Q_{U}^*\left(s, \omega, a\right)$ 更为简单？

### **算法流程**

论文给出了以Q-learning算法为基础的算法流程图，这个算法应该是结合参考了Q-learning以及actor-critic的算法。

![](../../assets/images/112b8bfdb55431d18471.jpg)

实际上，option-critic只是提出了一种框架，它应该能够与各种算法结合，例如DQN、DDPG等。

## 个人思考

这篇论文对于option的学习是完全自动的，因此只需规定option的数目即可。作者提到本框架最大的局限是 all options apply everywhere，我对于这句话的理解是在任何一个状态下都对应着一个option，即 $Q_U(s,\omega,a)$ 一定是与 option 相联系的。对于option的自动学习固然可以帮助智能体有效地处理某些复杂问题，但是在一些场景中，我们具有对option的先验知识，例如有哪些option，什么状态下需要切换option（即initial state set），我猜想先验知识的增加应当可以进一步提升分层学习的效果，但是如何处理这些先验知识呢？这篇文章并没有进行分析。

## **参考文献**

[1] P.-L. Bacon, J. Harb, and D. Precup, ‘The Option-Critic Architecture’, *arXiv:1609.05140 [cs]*, Dec. 2016, Accessed: Nov. 02, 2021. [Online]. Available: <http://arxiv.org/abs/1609.05140>

[2] R. S. Sutton, D. Precup, and S. Singh, ‘Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning’, *Artificial Intelligence*, vol. 112, no. 1–2, pp. 181–211, Aug. 1999, doi: [10.1016/S0004-3702(99)00052-1](https://doi.org/10.1016/S0004-3702(99)00052-1).
