# 【分层强化学习】HIRO：一种off-policy的分层强化学习算法

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2022-04-19  
来源：[https://zhuanlan.zhihu.com/p/497424021](https://zhuanlan.zhihu.com/p/497424021)

[知识库首页](../../README.md)

---

今天介绍论文《Data-Efficient Hierarchical Reinforcement Learning》

> 本论文提出了一种适用于off-policy算法的分层模型，该模型与[FeUdal](FeUdal：一种分层的神经网络结构与训练模型--492227096.md)十分类似，但相较而言模型更加简单，理论推导也不难，感兴趣的话可以看一看。推荐指数 ★★★★。  
> 本文适合人群：熟练掌握强化学习且对分层强化学习有基本了解。

本论文的提出背景是前期分层强化学习算法中存在的两个问题：

- 许多算法需要针对每个任务进行一些人为的设计，例如H-DQN算法中，需要对事先设定goals；
- 许多算法都需要用on-policy的RL算法进行训练，这是因为每个high-level策略对应的low-level策略通常处于不断变化的过程之中，因此过去某一时刻的high-level策略所能产生的low-level动作与未来不一定相同，这导致了**high-level策略的非平稳问题**，使得过去的样本无法有效地应用在训练之中。

针对上述两个问题，本文所提出的**HIRO**算法具有通用性（无需人为设定特征）以及样本高效（采用off-policy算法）两个特点。

回顾一下我们之前所介绍的，分层强化学习中主要有两种分层方法，一种是基于**option**的（例如[option-critic算法](The Option-Critic Architecture 阅读笔记--430471198.md)），另一种是基于**subgoal**的（例如[H-DQN算法](H-DQN：Hierarchical Deep Reinforcement Learning- Integrating Temp--430886508.md)，[FeUdal](FeUdal：一种分层的神经网络结构与训练模型--492227096.md)）。这里的Option指的是一种时间上扩展的行为，可以说是更高层次的action，例如为了找到一个目标，智能体先选择去哪一个房间（option），再选择具体的移动方向（action），而subgoal指的则是一种目标，比如为了打开门，智能体先得拿到钥匙（subgoal），为了这个目标，智能体需要执行一系列action。在我的理解里，option是一种策略层面上的定义（sub-policy），即具体该如何做，而subgoal则是目标层面上的定义，即需要达到什么目的。与H-DQN以及FeUdal一样，HIRO是一种基于**subgoal**的算法，算法通过将状态作为goals，避免了人为设定中间目标的问题；而通过引入off-policy的修正训练，则避免了high-level策略的非平稳问题。另外，HIRO与FeUdal算法十分类似，区别主要在于goals语义的设计，以及RL算法的选择。

关于其他分层强化学习算法的介绍，请看：

[XuanAxuan：【分层强化学习】The Option-Critic Architecture 阅读笔记33 赞同 · 0 评论文章34 赞同 · 0 评论文章](The Option-Critic Architecture 阅读笔记--430471198.md)[XuanAxuan：【分层强化学习】H-DQN：Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction阅读笔记29 赞同 · 0 评论文章](H-DQN：Hierarchical Deep Reinforcement Learning- Integrating Temp--430886508.md)[XuanAxuan：【分层强化学习】FeUdal：一种分层的神经网络结构与训练模型](FeUdal：一种分层的神经网络结构与训练模型--492227096.md)

## **算法介绍**

![](../../assets/images/bb784edb973f82adc799.jpg)

HIRO将标准的RL框架扩展成了一个分层级的两层结构，其中low-level的策略记为 $\mu^{lo}$，high-level的策略记为 $\mu^{hi}$。High-level的策略在一个较为粗糙的抽象层级上工作，为low-level的策略设定目标（goals），这里的目标直接对应着low-level策略应当达到的状态。在每一个时间步，环境提供一个观测状态 $s_t$，high-level的策略观测该状态，产生一个high-level的动作即目标 $g_t\in\mathbb{R}^{d_s}$，当 $t\equiv0(\text{mod}\ c)$ 时，$g_t$ 通过在策略 $\mu^{hi}$ 中采样产生，否则的话就根据一个固定的goal转移函数 $g_t=h(s_{t-1},g_{t-1},s_t)$ 产生。由于high-level策略每 $c$ 个时间步才通过 $\mu^{hi}$ 产生一次决策，这样就提供了**时间抽象（temporal abstraction）**。紧接着low-level策略 $\mu^{lo}$ 根据状态 $s_t$ 和目标 $g_t$ 产生一个low-level的动作 $a_t\sim\mu^{lo}(s_t,g_t)$，并在环境中执行动作。由此环境转移到新的状态 $s_{t+1}$，并产生一个奖励 $R_t$。另外，High-level的控制器会通过一个固定的参数化奖励函数提供给low-level控制器一个内部奖励（intrinsic reward）$r_t=r(s_t,g_t,a_t,s_{t+1})$。

一个时间步结束之后，low-level的策略会将经验 $(s_t,g_t,a_t,r_t,s_{t+1},h(s_t,g_t,s_{t+1}))$ 储存起来用于 off-policy训练，而high-level策略会收集奖励 $R_t$ 并且每 $c$ 个时间步收集一次 high-level 经验 $(s_{t:t+c-1},g_{t:t+c-1},a_{t:t+c-1},R_{t:t+c-1},s_{t+c})$ 用于off-policy训练。

### **内部奖励**

刚刚提到HIRO的high-level策略用于产生目标 $g_t$ 来表示状态观测中理想的相对变化。也就是说，每个时间步 $t$，high-level策略输出一个目标 $g_t$，此时low-level需要通过采取动作使得 $c$ 个时间步后的动作尽可能地接近 $s_t+g_t$。为了在状态不断变化的情况下保持目标所表示的绝对位置不变，目标的转移模型 $h$ 定义为



$$
h\left(s_{t}, g_{t}, s_{t+1}\right)=s_{t}+g_{t}-s_{t+1} \\
$$



内部奖励则定义为当前观测与目标观测之间的距离



$$
r\left(s_{t}, g_{t}, a_{t}, s_{t+1}\right)=-\left\|s_{t}+g_{t}-s_{t+1}\right\|_{2} \\
$$



这一奖励激励这low-level策略产生接近理想状态 $s_t+g_t$ 的状态。

Low-level策略的训练可以直接通过标准方法完成，只需要在价值和策略模型中增加 $g_t$ 作为额外的输入即可。例如，在DDPG中，low-level的Q价值函数可通过在经验回放池中采样，然后最小化



$$
\left(Q_{\theta}^{l o}\left(s_{t}, g_{t}, a_{t}\right)-r\left(s_{t}, g_{t}, a_{t}, s_{t+1}\right)-\gamma Q_{\theta}^{l o}\left(s_{t+1}, g_{t+1}, \mu_{\phi}^{l o}\left(s_{t+1}, g_{t+1}\right)\right)\right)^{2} \\
$$



完成，策略 $\mu^{lo}_\phi$ 的目标则是最大化 $Q_{\theta}^{l o}\left(s_{t}, g_{t}, \mu_{\phi}^{l o}\left(s_{t}, g_{t}\right)\right)$.

HIRO给目标goal定义了明确的语义：**状态空间中较好的观测**，而high-level的策略表示的则是**状态空间中的变化**。在这样的情况下，模型就无需特意去训练goal的特征表达了，并且能够让low-level即便在没有环境奖励的情况下也能立即收到奖励信号。

### **High-level策略的off-policy修正**

之前我们提到low-level策略的变化导致经验回放池中的样本不再能够准确反映 $g_t$ 与状态转移之间的关系，即在相同goal下，当前low-level策略所产生的动作与过去样本对应的low-level动作是不同的，因而所产生的状态也会是不同的。为此，HIRO提出通过重新产生一个新的high-level动作 $\tilde{g}_t$ 来修正 $\left(s_{t}, g_{t}, \sum {R}_{t: t+c-1}, s_{t+c}\right)$。这里 $\tilde{g}_t$ 通过最大化该样本的条件轨迹出现概率 $\mu^{l o}\left(a_{t: t+c-1} \mid s_{t: t+c-1}, \tilde{g}_{t: t+c-1}\right)$ 产生，其中 $\tilde{g}_{{t}+1: t+c-1}$ 通过固定的目标转移函数生成，概率的计算公式可以计算为



$$
\log \mu^{l o}\left(a_{t: t+c-1} \mid s_{t: t+c-1}, \tilde{g}_{t: t+c-1}\right) \propto-\frac{1}{2} \sum_{i=t}^{t+c-1}\left\|a_{i}-\mu^{l o}\left(s_{i}, \tilde{g}_{i}\right)\right\|_{2}^{2}+\text { const. } \\
$$



为了近似得到一个最大值，可以选定一定数量的 $\tilde{g}_t$ 来计算上述概率然后选择其中概率最大的 $\tilde{g}_t$ 重新标记样本。本文以 $s_{t+c}-s_t$ 为中心随机采样8个候选目标，再加上原始目标 $g_t$ 以及 $s_{t+c}-s_t$ 总共构成10个候选值，然后从中选出使得上述概率最大的作为新的经验。

## **总结与个人思考**

本文的HIRO实际上与FeUdal十分相似，均基于subgoals，且high-level policy产生的goals均包含与状态转移有关的语义特征。所不同的只是FeUdal侧重于网络模型和相应训练方法的设计，而本文则是简明介绍了分层架构，同时提出了在该架构中使用off-policy算法所存在的问题与相应的解决方案。

## **参考文献**

[1] O. Nachum, S. Gu, H. Lee, and S. Levine, ‘Data-Efficient Hierarchical Reinforcement Learning’, *arXiv:1805.08296 [cs, stat]*, Oct. 2018, Accessed: Mar. 23, 2022. [Online]. Available: <http://arxiv.org/abs/1805.08296>
