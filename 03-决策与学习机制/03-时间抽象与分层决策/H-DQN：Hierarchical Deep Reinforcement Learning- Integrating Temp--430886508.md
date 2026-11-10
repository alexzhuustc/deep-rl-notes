# 【分层强化学习】H-DQN：Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction阅读笔记

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2022-04-16  
来源：[https://zhuanlan.zhihu.com/p/430886508](https://zhuanlan.zhihu.com/p/430886508)

[知识库首页](../../README.md)

---

**稀疏反馈（sparse feedback）**是目前强化学习领域一个十分棘手的问题，智能体低效的探索会导致价值函数学习的困难。在这种背景下，作者提出内在的驱动力（该驱动力区别于直接解决系统问题）可以提升智能体的探索能力。因此，本文提出了一种分层框架——**hierarchical-DQN（h-DQN）**，其中上层的价值函数学习 **内在（intrinsic）的任务策略**，而下层的价值函数则负责直接学习完成原始任务所需要的**动作策略**。

不同于[option-critic](The Option-Critic Architecture 阅读笔记--430471198.md)，h-DQN的分层是通过**任务（goal）**来划分的，因此该结构属于 **基于子任务（subgoal）**的学习。

关于option-critic的介绍，请看另一篇笔记

[XuanAxuan：【分层强化学习】The Option-Critic Architecture 阅读笔记](The Option-Critic Architecture 阅读笔记--430471198.md)
> 相对于option-critic而言，h-DQN基本上没有在理论上提出很大的创新，它更多的在于提出了一种分层的方法，并将该方法运用到了Montezuma's Revenge游戏中，本文对于该游戏的分解，或许对工程实践的分层有一定意义，推荐指数 ★★★★。  
> 本文适合人群：熟练掌握强化学习且对分层强化学习有基本了解。

## **模型框架**

### **智能体**

智能体：在MDPs中，有效的探索对于控制策略的学习十分重要，$\epsilon$-贪心算法对于局部探索是有效的，但它无法推动智能体去探索状态空间中的其他区域。为了解决该问题，本文提出了一个新的概念——**目标（goal）**$g\in\mathcal{G}$，用于提供智能体的**内在驱动力**。有了一个个 goal 之后，智能体可以更加关注与如何实现这些目标，从而最大化**外在收益**。

本文用 **option** 的时间扩展来定义每个目标 $g$ 的策略 $\pi_g$，在学习如何选择最优goal序列的同时，智能体还将学习这些goal所对应的option策略。为了学习到每个 $\pi_g$，智能体有一个critic，它能够提供 **内在奖励（intrinsic rewards）**，根据这些奖励智能体能够实现目标。

> （这里的option与option-critic中的option看似概念上有所区别，实际上拥有相同的本质。这里的option指代的是选择了某个goal之后，该goal所对应的策略，而在option-critic中，每个option也是对应着一个策略，这两者本质上是一样的，区别只是在于前者是目标（任务）驱动，先选择一个目标然后才有策略，而后者则直接是动作驱动，我直接选择一个option，这个option本身就对应着一个策略）

### **时间扩展**

![](../../assets/images/0fe82b22ca541365fa4e.jpg)

如上图所示，智能体是两层的分层架构，包含 **控制器（controller）**与 **元控制器（meta-controller）**。其中 meta-controller 根据状态 $s_t$ 选择一个目标 $g_t\in\mathcal{G}$，其中 $\mathcal{G}$ 表示所有目标的集合。而 controller 则根据 $s_t$ 与 $g_t$ 选择动作 $a_t$。目标 $g_t$ 会一直保持到该目标被实现或者到达 **终止状态（terminal state）**。**内在critic（internal critic）** 负责评估目标 $g_t$ 是否达到并传递给 controller 一个合适的奖励。controller 的目标函数是最大化内在奖励所对应的收益： $R_{t}(g)=\sum_{t^{\prime}=t}^{\infty} \gamma^{t^{\prime}-t} r_{t^{\prime}}(g)$。类似的，meta-controller的目标是最大化外在奖励（即环境奖励）所对应的收益：$F_{t}=\sum_{t^{\prime}=t}^{\infty} \gamma^{t^{\prime}-t} f_{t^{\prime}}$，这里 $f_t$ 表示从环境接收到的奖励信号。

> 这里的critic与我们所熟知的actor-critic中critic的含义应当是不太一样的，一般情况下critic表示的是评估Q值的模型，而这里critic表示的只是判断goal是否完成并以此提供controller相应奖励的一种装置。

### **时间扩展的深度强化学习**

本文采用DQN框架来学习 controller 和 meta-controller 的策略。具体而言，controller对以下Q-价值函数进行估计：



$$
\begin{aligned} Q_{1}^{*}(s, a ; g) &=\max _{\pi_{a g}} \mathrm{E}\left[\sum_{t^{\prime}=t}^{\infty} \gamma^{t^{\prime}-t} r_{t^{\prime}} \mid s_{t}=s, a_{t}=a, g_{t}=g, \pi_{a g}\right] \\ &=\max _{\pi_{a g}} \mathrm{E}\left[r_{t}+\gamma \max _{a_{t+1}} Q_{1}^{*}\left(s_{t+1}, a_{t+1} ; g\right) \mid s_{t}=s, a_{t}=a, g_{t}=g, \pi_{a g}\right] \end{aligned} \\
$$



其中 $g$ 是智能体在状态 $s$ 的目标，并且 $\pi_{ag}=P(a|s,g)$ 表示动作策略。

类似的，对于 meta-controller，有



$$
Q_{2}^{*}(s, g)=\max _{\pi_{g}} \mathrm{E}\left[\sum_{t^{\prime}=t}^{t+N} f_{t^{\prime}}+\gamma \max _{g^{\prime}} Q_{2}^{*}\left(s_{t+N}, g^{\prime}\right) \mid s_{t}=s, g_{t}=g, \pi_{g}\right] \\
$$



其中 $N$ 表示给定目标 $g_t$ 在完成之前所经历的时间步数，$g'$ 表示状态 $s_{t+N}$ 下智能体的目标，且 $\pi_g=P(g|s)$ 表示 **policy over goals**。 $Q_2$ 所产生的的 $\left(s_{t}, g_{t}, f_{t}, s_{t+N}\right)$ 与 $Q_1$ 所产生的 $\left(s_{t}, a_{t}, g_{t}, r_{t}, s_{t+1}\right)$ 分别存放在不同的经验回访池 $\mathcal{D}_2$ 与 $\mathcal{D}_1$ 中，controller的经验在每一个时间步进行收集，而meta-controller的经验则只在controller终止时（即goal重新产生）时进行收集。

> 在option-critic中，只包含环境奖励，在那种情况下，上层的Q值与下层的Q值有非常紧密的统计关系（上层Q值是下层Q值的统计平均），此时学习其中1个Q值就可以得到另一个Q值。但是本文分别赋予了上层Q值与下层Q值不同的奖励（环境奖励与任务驱动奖励），此时两者可以说是完全独立的，可以分别进行学习。我无法判断这两种方法哪一种更好，只是单纯地觉得option-critic的推导更为精妙，但或许h-DQN的方法更加适合工程实践。（本论点未经推敲，纯属本人直观感觉）

## **算法流程**

实际上，该算法应用起来应该还是比较简单直观的，只是在普通DQN的基础上额外增加了一个上层学习的模块。

![](../../assets/images/9fe1b0e1919a264ec27f.jpg)

## **ATARI实验**

![](../../assets/images/371ef13fa1551ec2d9ee.jpg)

本文以 `Montezuma's Revenge' 游戏为例，展示了 h-DQN 在稀疏延时奖励下相对于DQN 的优势。

### **游戏描述**

如上图所示，游戏玩家需要通过多个房间来收集珠宝。为了通过左上角和右上角的门（300奖励），玩家需要首先获得钥匙（100奖励），在此之前，玩家首先需要爬上右侧的梯子然后向左移动才能到达钥匙处，这些动作都是没有奖励的，这就是该游戏的困难之处。DQN算法在该游戏中只能获得0分。

### **目标设定**

为了让智能体在自行学习到获得钥匙所能带来的奖励之前能够探索到该游戏场景中有意义的部分，智能体需要有一个内在的驱动力。本文将场景中的 **实体（entities）**或 **物品（objects）**作为环境中的目标，该游戏中总共有6个goals：左上角的门、右上角的门、中间的梯子、左下角的梯子、右下角的梯子、钥匙。如果智能体到达了一个实体，则说明它完成了门这个目标。

### **模型训练**

训练分为两个阶段：

- 第一阶段，固定meta-controller的探索率为1，然后训练controller，此时可以让controller学习到如何实现一系列目标。
- 第二阶段，同时训练controller和metacontroller。

### **实验结果**

可以看出，H-DQN能获得奖励明显优于DQN。第三幅图表示的是随着episode的增加，不同goal的选择概率，不太懂这里为什么选门的那么少？

![](../../assets/images/36773c62e087db6397df.jpg)

下面这幅图展示了智能体的学习过程。在左上图中，智能体选择key作为goal，发现无论怎么执行动作都到达不了。于是下一次智能体选择右下角的梯子作为goal，并且最终实现了它。之后智能体分别选择钥匙和右上角的门作为目标并逐步实现了这些目标。

> 有个问题是如果一个目标怎么都实现不了，此时要怎么终止它呢？

![](../../assets/images/be1ad6212921649a13c2.jpg)

## **个人思考**

本文所提出的H-DQN无论在理论上还是应用上都是比较简单的，这是一种任务驱动的方法，该方法适用于奖励稀疏的问题，如果想要利用该方法，在拿到一个具体问题时，我们首先需要思考的是如何将该问题分解成多个子任务。相对于option-critic，我认为本方法也许更加适合工程实践，因为我们可以在算法中注入相关问题的先验知识，而不是让智能体自行去决定什么是option。然而，该方法是否能够真正解决稀疏奖励的问题还需要通过实验来检验，另外，对于奖励并不稀疏的问题呢？

## **参考文献**

[1] T. D. Kulkarni, K. R. Narasimhan, A. Saeedi, and J. B. Tenenbaum, ‘Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation’, *arXiv:1604.06057 [cs, stat]*, May 2016, Accessed: Nov. 02, 2021. [Online]. Available: <http://arxiv.org/abs/1604.06057>
