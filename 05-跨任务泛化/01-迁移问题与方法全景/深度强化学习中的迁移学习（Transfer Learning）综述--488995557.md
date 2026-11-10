# 【迁移强化学习】深度强化学习中的迁移学习（Transfer Learning）综述

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2022-05-21  
来源：[https://zhuanlan.zhihu.com/p/488995557](https://zhuanlan.zhihu.com/p/488995557)

[知识库首页](../../README.md)

---

> 又到了课题找点的阶段，一个好的课题最开始应该是有一个需要解决的问题，然后再找合适的方法，奈何找问题太难，所以我只能另辟蹊径，先看看有没有好的方法，再从这些方法中思考是否有可以应用的场景。这次我想了解的是【迁移学习】，我选了一篇比较新的综述来对这一技术有一个整体的理解，以下内容均提取自论文《Transfer Learning in Deep Reinforcement Learning: A Survey》。对于我而言，每个算法背后的原理并不是很重要，我更需要了解的是每个算法所提出的背景、适用场景，以及它们可以应用在何种DRL框架下。但是说实话这篇综述看完之后，我发现迁移学习和我理解的不大一样，至少这篇论文所综述的算法，大多都是应用在比较局限的场景中，而且很多对应着非常老的RL算法，对于这些问题，我也提出了一些看法，如果有比较了解的同学，希望能在评论区多多交流，让我能够有更多的思路~

**迁移学习（Transfer Learning，TL）**可以将从专家或其他过程中获取的知识迁移到当前任务中，达到加速学习的作用。迁移学习在监督学习（Supervised Learning）领域已经得到了广泛的研究，但在强化学习（Reinforcement Learning，RL）领域，这一技术面临着更为复杂的挑战，且目前仍然处在发展阶段，本文将对深度强化学习领域中的迁移学习技术进行梳理和总结。

## **强化学习场景下的迁移学习**

定义**源域（source domains）**为 $\boldsymbol{\mathcal{M}}_{s}=\left\{\mathcal{M}_{s} \mid \mathcal{M}_{s} \in \boldsymbol{\mathcal{M}}_{s}\right\}$ ，$\boldsymbol{\mathcal{M}}_{s}$ 可以为 **目标域（target domain）** $\mathcal{M}_t$ 提供先验知识，使得目标智能体能够学得更好（相比于不获取任何先验知识）。对于最简单的情况，两个相同域的智能体可以相互迁移知识，此时 $|\boldsymbol{\mathcal{M}}_{s}|=1$, $\mathcal{M}_{s}=\mathcal{M}_{t}$。

- 【**强化学习场景下的迁移学习**】给定一个 **源域（source domains）**的集合 $\boldsymbol{\mathcal{M}}_{s}=\left\{\mathcal{M}_{s} \mid \mathcal{M}_{s} \in \boldsymbol{\mathcal{M}}_{s}\right\}$，以及一个 **目标域（target domain）**$\mathcal{M}_t$，**迁移学习（Transfer Learning，TL）**的目标是通过同时利用 $\boldsymbol{\mathcal{M}}_{s}$ 的外部信息 $\mathcal{D}_s$ 以及 $\mathcal{M}_t$ 的内部信息 $\mathcal{D}_t$，学习到目标域的最优策略 $\pi^*$，即：$\pi^{*}=\arg \max \mathbb{E}_{s \sim \mu_{0}^{t}, a \sim \pi}\left[Q_{\mathcal{M}}^{\pi}(s, a)\right], \\$ 其中策略 $\pi=\phi\left(\mathcal{D}_{s} \sim \mathcal{M}_{s}, \mathcal{D}_{t} \sim \mathcal{M}_{t}\right): \mathcal{S}^{t} \rightarrow \mathcal{A}^{t}$ 是目标域 $\mathcal{M}_t$ 中，状态到动作之间的函数映射关系，该策略从 $\mathcal{D}_s$ 与 $\mathcal{D}_t$ 学习得到。

在上述定义中，我们用 $\phi(\mathcal{D})$ 表示根据信息 $\mathcal{D}$ 学习到的策略。常规强化学习是上述定义的一种特殊情况：$\mathcal{D}_s=\emptyset$, $\pi=\phi(\mathcal{D}_t)$。

### **相关技术**

除了迁移学习之外，还有许多类似的技术用于辅助强化学习，本节将介绍它们与迁移学习之间的区别以及联系。

- **模仿学习（Imitation Learning）**：模仿学习也被称为**学徒学习（Apprenticeship Learning）**，其目标是通过少量的专家示例让策略尽可能地模仿给定专家策略的行为。该方法适用于环境反馈不易得到的序列决策问题。目前主流的模仿学习主要分为两大类，第一类为 **行为克隆（Behavior Clonning）**，此时策略在没有任何奖励信号的情况下，以监督学习的方式训练策略；第二类方法被称为 **反向强化学习（Inverse Reinforcement Learning）**，该方法的目标是还原任务的奖励函数，用于解释专家示例的行为。模仿学习与迁移学习十分接近，因此被进一步改进成了一种迁移学习方法，被称为 **从示例中学习（Learning from Demonstrations，LfD）**。模仿学习与LfD的区别在于，LfD仍然可以从环境中获得奖励信号，该方法只是通过少量的专家示例来辅助优化目标策略，而模仿学习则是假设无法从环境中获取奖励信号。
- **终生学习（Lifelong Learning）**：终生学习也被称为**持续学习（Continual Learning）**，其目标是在给定一系列不平稳信息的情况下，学习多个时间或空间相关的任务。终身学习的关键是在获取新信息与在新任务上保留先前学习到的知识之间取得平衡。相比于迁移学习，终身学习更加具有挑战性，因为它要求智能体在一系列动态变化且无法预测的任务之间迁移知识，而不仅是在固定的任务上进行知识迁移。另外，终身学习还要求智能体具有自动检测新任务的能力，而在迁移学习中，智能体通常会被告知新任务的出现。

### **迁移学习的分类**

迁移学习方法可以根据不同方面进行分类：

1. 根据所迁移知识的种类：从源域所迁移的知识可以有多种形式，例如：专家经验的集合，专家策略的动作概率分布，以及一个用于估计源马尔科夫过程（Markov Decision Process，MDP）或目标MDP中状态-动作对质量好坏的势函数。知识表征的不同从根本上决定了迁移学习的方式，而所迁移知识的质量（最优还是次优）同样会影响到迁移过程。
2. 根据所兼容的RL框架：根据所兼容的RL框架，迁移学习可以分为两类：与策略无关的迁移学习以及针对特定RL框架的迁移学习。这种分类方法与所迁移知识的形式有很大关系，例如，专家示例组成的知识迁移通常是与策略无关的，而策略蒸馏（policy distillation）方法则通常不适合DQN类的RL算法。
3. 根据源域与目标域的差异：一些迁移学习方法适用于 $\mathcal{M}_s$ 与 $\mathcal{M}_t$ 相同的场景，而另外一些迁移学习方法则用于在不同域之间迁移知识，例如观测不同，或者奖励函数不同，等等。
4. 根据目标域所能获取的信息：通常而言，从源域中获取知识的成本较低，相反智能体通常很难从目标域获取知识，或者由于较高的采样成本，智能体只能获得数量有限的环境交互。例如，将仿真平台训练好的自动驾驶智能体投入到实际环境中进行学习，或者将仿真训练的航行机器人放在真实环境中训练。因此，目标域中可获取知识的多少决定了迁移学习的方式。
5. 根据迁移学习的采样效率：相比于从头开始学习，迁移学习依据从源域所迁移的知识优化了智能体的初始化过程，从而帮助智能体在目标域中以更少的交互次数达到收敛。根据所需要的环境交互次数，可以将其分为三类：1）**零次（Zero-shot）迁移**：无需任何环境交互，即可让智能体学习到目标域的策略；2）**小样本（Few-shot）迁移**：只需少量目标域的样本（交互）；3）**采样高效（sample-efficient）迁移**：依然比没有迁移学习的RL所需要的交互数更少。
6. 根据迁移学习的目标：这里可以从两个方面来考虑：1）评价指标；2）目标函数。迁移学习的评价指标可以是收敛后的渐进性能，也可以是达到一定性能门限所需要的迭代次数。另一方面，迁移学习可以依据各种各样包含不同正则化的目标函数组成，这些正则化处理通常与所迁移的知识有关。例如，当所迁移的知识是不完美的专家示例时，可以在原最大化收益目标中增加熵正则来促进智能体的探索过程。

### **举例说明**

本小节以 HalfCheetah 为例来说明迁移学习的适用场景，HalfCheetah 是一种求解物理运动任务的RL基准任务：训练一个两足智能体，使其能够在不失去控制的情况下尽可能地快跑。

### **可能的域差异：**

源域与目标域的差异可以体现在组成MDP的任何一个元素中：

- $\mathcal{S}$ （状态空间）：HalfCheetah 中状态空间的差异可以通过扩展或者限制智能体移动的区域来产生。
- $\mathcal{A}$ （动作空间）：动作空间的差异可以通过改变智能体大腿、胫骨或脚部的可用扭矩范围来产生。
- $\mathcal{R}$（奖励函数）：HalfCheetah 中的奖励既可以用简单的移动距离作为奖励，也可以用每个方向上所增加的速度作为额外的惩罚代价。
- $\mathcal{T}$（动态转移）：源域和任务域所遵循的物理规律可能是不同的，这就导致了相同状态-动作对下不同的转移概率。
- $\mu_0$（初始状态）：源域与目标域可能包含不同的初始状态，具体而言就是智能体开始移动的位置以及初始姿势。
- $\tau$（轨迹）：源域与目标域中，智能体完成任务所需要的步数可能是不同的。

### **可迁移的知识**

- 示例轨迹：目标智能体可以通过训练好的专家行为中进行学习，例如，一系列专家策略所获得的示例轨迹。
- 模型动态规律：智能体可以从源域中学习到环境动态规律的近似模型，然后基于该近似模型，利用动态规划算法对目标域任务进行求解。
- 专家策略：智能体可以直接参照专家策略，即给定状态下不同动作的概率分布。
- 专家价值函数：除了专家策略外，智能体还可以参考专家策略所对应的价值函数，即从专家的视角判断状态-动作对的好坏。

## **迁移学习方法介绍**

本节主要依据所迁移的知识种类来介绍几种不同的迁移学习方法。首先介绍 **奖励设计（Reward Shaping，RS）**方法，该方法对RL基本框架的改变很小，可以适用于不同的RL方法，并且通常可以与其他迁移学习方法结合使用。

### **奖励设计（Reward Shaping）**

**奖励设计（Reward Shaping，RS）**利用外部知识重构目标域的奖励分布从而引导智能体的策略学习。具体来说，除了环境奖励函数之外，RS还会学习一个奖励设计函数：$\mathcal{F}: \mathcal{S} \times \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ 来提供辅助奖励，该奖励设计函数包含了引导智能体做出更好动作选择的外部知识。因此，智能体会利用新设计的奖励 $\mathcal{R}^{\prime}: \mathcal{R}^{\prime}=\mathcal{R}+\mathcal{F}$ 来学习策略，这意味着RS用一个不同的奖励函数替换了目标域：



$$
\mathcal{M}=(\mathcal{S}, \mathcal{A}, \mathcal{T}, \gamma, \mathcal{R})) \rightarrow \mathcal{M}^{\prime}=\left(\mathcal{S}, \mathcal{A}, \mathcal{T}, \gamma, \mathcal{R}^{\prime}\right). \\
$$



- **基于势函数的奖励设计（Potential based Reward Shaping，PBRS）**[2] 是最经典的一种奖励设计方法，该方法定义了一个设计函数 $F$ 来作为两个势函数（ $\phi(\cdot)$ ）之间的差异：$F\left(s, a, s^{\prime}\right)=\gamma \Phi\left(s^{\prime}\right)-\Phi(s), \\$ 其中势函数 $\phi(\cdot)$ 来自外部获取的知识，用于评估给定状态的质量。经证明，PBRS是保持策略不变性的充分必要条件。另外，原MDP与转换后的MDP所对应的最优Q函数有如下关系：$Q_{\mathcal{M}^{\prime}}^{*}(s, a)=Q_{\mathcal{M}}^{*}(s, a)-\Phi(s). \\$

在PBRS的基础上，此后又出现了类似的几种方法：

- **Potential Based state action Advice（PBA）**[3]：势函数为状态与动作的函数。
- **Dynamic Potential Based（DPB）**[4]：势函数是状态和时间的函数，因此是动态的。
- **Dynamic Value-Function Advice（DPBA）**[5]：势函数是状态与动作的动态函数，该函数可以作为一个额外的状态-动作价值函数（与环境的价值函数平行）。

上述的这些方法主要解决的是如何构造不同的设计函数 $F(s,a)$，但是并没有解释如何迁移知识用于得到构造势函数。

- [6] 提出利用RS将源域（$\mathcal{M}_s$）的专家策略迁移到目标域（$\mathcal{M}_t$）。该方法假设两个映射函数的存在：$M_S$ 与 $M_A$ 分别用于将状态和动作从源域转换到目标域，此时势函数为 $\pi_{s}\left(\left(\mathcal{M}_{S}(s), \mathcal{M}_{A}(a)\right)\right)$，即映射之后的状态和动作被源域中的专家策略选取的概率。[7]
- [7] 利用专家策略产生的示例状态-动作样本来设计奖励，该方法包含一个鉴别器（discriminator）用于区分专家策略产生的样本与目标策略产生的样本，鉴别器的损失函数用于设计奖励，以此激励智能体模仿专家行为，该方法是RS方法与LfD方法的结合。

上述RS方法总结如下：

![](../../assets/images/e303b9416900423b758b.jpg)

**一些思考**

从上表可以看出，大多数RS方法并没有注明知识的来源，即如何从源域迁移知识到目标域中，并且所对应场景的目标域与源域相同，那RS所迁移的知识到底是什么呢？是问题设计者所掌握的一些关于状态或动作的知识吗？那为什么我们不可以直接利用这些知识去构造MDP过程呢，又或者说，RS本身就是构造MDP过程的一种方式，毕竟它所做的，实际上就是设计了一种奖励信号，而并非从源域迁移知识到目标域。（我并没有仔细地阅读过上面方法对应的论文，所以如果我思考得不对，请反驳我）

### **从示例中学习（Learning from Demonstrations）**

示例（demonstrations）通常来源于各种不同的源域：它可以通过一个人类专家产生，也可以通过一个之前训练好的专家策略，甚至是次优策略来产生。在下面的介绍中，我们用 $D_E$ 来表示示例的集合，$D_E$ 中的每一个元素都是一次交互，即 $(s,a,s',r)\in D_E$。这一类迁移学习方法通常假设源域与目标域的MDP相同：$\mathcal{M}_s=\mathcal{M}_t$。

根据示例何时用于知识迁移，LfD方法可以分为离线（offline）方法和在线（online）方法。在离线方法中，示例通常用于预训练RL模块，即利用这些示例用监督学习的方式初始化价值函数 $V(s)$，策略 $\pi$，或者是模型的动态转移概率。而在在线方法中，示例则直接用于引导智能体在学习过程中更有效地探索。以下我们讨论的大多数方法均为在线方法。

- [8] 提出了通过示例进行策略迭代（Direct Policy Iteration with Demonstrations，DPID）算法。该方法分别从专家策略 $\pi_E$ 与正在学习的策略中产生样本 $D_E$ 与 $D_\pi$，然后利用 $D_\pi \cup D_\pi$ 对Q值进行蒙特卡洛估计，所学习的策略为：$\pi(s)=\underset{a \in \mathcal{A}}{\arg \max } \hat{Q}(s, a)$。然后该策略进一步通过一个损失函数 $\mathcal{L}(s,\pi_E)$ 最小化与专家策略决策之间的差异：  
   $\mathcal{L}\left(\pi, \pi_{E}\right)=\frac{1}{N_{E}} \sum_{i=1}^{N_{E}} \mathbb{1}\left\{\pi_{E}\left(s_{i}\right) \neq \pi\left(s_{i}\right)\right\}, \\$ 其中 $N_E$ 是专家示例的数量，$\mathbb{1}(\cdot)$ 为指示函数。
- [9] 提出了示例中的深度Q-学习（Deep Q-learning from Demonstration，DQfD）算法，该算法用两个独立的回放池分别用于储存示例数据与自己产生的数据，这样示例中的数据总会以一定的概率被采样到。
- [10] 提出的算法同样基于Q-learning框架，该算法结合了RS与LfD技术，根据专家示例构建了一种势函数，其中任意给定状态-动作的势被定义为其余专家经验之间的最高相似度，这样能够使得接近专家示例的状态-动作对被给予更多的信任，使得智能体趋近于专家策略的行为。
- [11] 提出了适用于策略梯度框架的**生成对抗模仿学习（Generative Adversarial** **Imitation Learning ，GAIL）**算法，GAIL首先定义了 $d_\pi$ 用于表示策略 $\pi$ 的状态-动作分布，基于此，GAIL的目标为最小化当前策略 $\pi$ 与专家策略 $\pi_E$ 所对应的 $d_\pi$ 与 $d_{\pi_E}$ 之间的差异。具体而言，其奖励函数通过对抗训练获得：构建一个鉴别器（discriminator） $D$ 用于区分当前策略 $\pi$ 与专家策略 $\pi_E$ 所产生样本：  
   $J_{D}=\max _{D: \mathcal{S} \times \mathcal{A} \rightarrow(0,1)} \mathbb{E}_{d_{\pi}} \log [1-D(s, a)]+\mathbb{E}_{d_{E}} \log [D(s, a)] \\$ 由于 $\pi_E$ 是未知的，它的状态-动作分布 $d_E$ 从给定的专家示例 $D_E$ 中估计而来。证明可得，对于最优的鉴别器，其输出满足 $D(s, a)=\frac{d_{\pi}}{d_{\pi}+d_{E}}$。鉴别器的输出将作为新的奖励函数 $r^{\prime}(s, a)=-\log (1-D(s, a))$ 用于估计分布匹配。最终，强化学习的目标变为通过优化一下minmax目标达到分布匹配：  
   $\max _{\pi} \min _{D} J(\pi, D):=\mathbb{E}_{d_{\pi}} \log [1-D(s, a)]+\mathbb{E}_{d_{E}} \log [D(s, a)] \\$ GAIL与模仿学习有着很强的联系，只不过它利用了专家示例来进行分布匹配。
- [12] 在 GAIL的基础上提出了 POfD，该算法将鉴别器奖励与环境奖励结合起来，这样智能体可以同时最大化环境的累计奖励（RL目标）以及执行分布匹配（模仿学习）：  
   $\max _{\theta}=\mathbb{E}_{d_{\pi}}[r(s, a)]-\lambda D_{J S}\left[d_{\pi} \| d_{E}\right] \\$ 论文进一步证明了该算法等同于一个动态的RS机制：  
   $\max _{\theta}=\mathbb{E}_{d_{\pi}}\left[r^{\prime}(s, a)\right], \\$ 其中 $r^{\prime}(s, a)=r(s, a)-\lambda \log \left(D_{w}(s, a)\right)$ 为设计之后的奖励。
- [13] 将DQfD算法应用在了DDPG算法中，提出DQfD算法。
- [14] 在DQfD的基础上，在目标函数中增加了一个行为克隆损失用于鼓励智能体模仿示例的行为：  
   $\mathcal{L}_{B C}=\sum_{i=1}^{\left|D_{E}\right|}\left\|\pi\left(s_{i} \mid \theta_{\pi}\right)-a_{i}\right\|^{2} \mathbb{1}\left[Q\left(s_{i}, a_{i}\right)>Q\left(s_{i}, \pi\left(s_{i}\right)\right)\right] \\$ 该函数表明只有Q值更高的示例动作会对损失函数产生影响。

上面所介绍的算法总结在下表中：

![](../../assets/images/0e3d59bafad0bfb375e8.jpg)

以上算法均假设示例是完美的，但实际上示例通常都是不理想的，此时我们可以仅利用示例进行初始化，然后在继续学习，另外LfD还存在过拟合的问题，这些都是未来可以研究的方向。

**一些思考**

我对于这一类方法的问题在于示例从哪里来，如果已经有一个专家策略，那为什么不直接用专家策略，我的想法是这一类方法适用的场景是只有示例，没有专家策略，例如在自动驾驶场景下，我们可以先人为驾驶产生一些安全的示例，但我们并不知道这些示例对应的策略如何用数学表达，此时我们就可以让智能体从这些示例中进行学习，感觉就像是MDP框架下的监督学习。

### **策略转移（Policy Transfer）**

**策略迁移（Policy Transfer）**的外部知识通常为一个或多个源域预先训练好的策略。这类方法的工作通常考虑多对一的问题场景：

- 策略迁移：一个专家策略的集合 $\pi_{E_{1}}, \pi_{E_{2}}, \ldots, \pi_{E_{K}}$ 分别在各自对应源域 $\mathcal{M}_{1}, \mathcal{M}_{2}, \ldots, \mathcal{M}_{K}$ 上进行了训练，学生策略 $\pi$ 则利用从 $\left\{\pi_{E_{i}}\right\}_{i=1}^{K}$ 获取的知识在目标域上学习。

这一类工作分为两大类：策略蒸馏（Policy Distillation）与策略复用（Policy Reuse）。

**策略蒸馏（Policy Distillation）**

**知识蒸馏（knowledge distillation）**指的是将多个教师模型的知识整合到一个学生模型中，此后该概念被扩展到了RL领域，通常假设学生模型比教师模型更为肤浅（swallow，不太不明白这个地方想要表达什么）并且可以处理多个教师任务，因而该方法也经常被用于模型压缩与多任务RL中。

知识蒸馏的概念在RL领域被称之为**策略蒸馏（Policy Distillation）**。

传统策略蒸馏方法以监督学习的方式迁移教师策略，具体来说，学生策略的训练目标是最小化教师策略 $\pi_E$ 与学生策略 $\pi_\theta$ 之间的差异，该差异定义为 $\mathcal{H}^{\times}\left(\pi_{E}\left(\tau_{t}\right) \mid \pi_{\theta}\left(\tau_{t}\right)\right)$，即



$$
\min _{\theta} \mathbb{E}_{\tau \sim \pi_{E}}\left[\sum_{t=1}^{|\tau|} \nabla_{\theta} \mathcal{H}^{\times}\left(\pi_{E}\left(\tau_{t}\right) \mid \pi_{\theta}\left(\tau_{t}\right)\right)\right] \\
$$



这里的期望是对教师策略 $\pi_E$ 采样得到的轨迹计算的，因此被称为**教师蒸馏（teacher distillation）**。

- [15] 正是采用了教师蒸馏，首先在 $N$ 个源任务中训练了 $N$ 个教师策略，每个教师会产生一个数据集 $D^{E}=\left\{s_{i}, \boldsymbol{q}_{i}\right\}_{i=0}^{N}$，该数据集包含状态 $s$ 以及它们对应的Q值 $\boldsymbol{q}$，$\boldsymbol{q}_{i}=\left[Q\left(s_{i}, a_{1}\right), Q\left(s_{i}, a_{2}\right), \ldots \mid a_{j} \in \mathcal{A}\right]$。然后这些教师策略会蒸馏成一个单独的学生策略 $\pi_{\theta}$，目标是最小化每个教师策略 $\pi_{E_i}(a|s)$ 与学生策略 $\pi_{\theta}$ 之间的KL-散度：



$$
\min _{\theta} \mathcal{D}_{K L}\left(\pi^{E} \mid \pi_{\theta}\right) \approx \sum_{i=1}^{\left|D^{E}\right|} \operatorname{softmax}\left(\frac{\boldsymbol{q}_{i}^{E}}{\tau}\right) \ln \left(\frac{\operatorname{softmax}\left(\boldsymbol{q}_{i}^{E}\right)}{\operatorname{softmax}\left(\boldsymbol{q}_{i}^{\theta}\right)}\right) \\
$$



与教师蒸馏相对的是**学生蒸馏（student distillation）**，此时训练阶段的期望值从学生策略产生的轨迹中计算得到：



$$
\min _{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}}\left[\sum_{t=1}^{|\tau|} \nabla_{\theta} \mathcal{H}^{\times}\left(\pi_{E}\left(\tau_{t}\right) \mid \pi_{\theta}\left(\tau_{t}\right)\right)\right] \\
$$



目前大多数工作都采用的都是学生蒸馏，因为这能保证智能体更好的探索，因为教师策略通常是确定的。

从另一个角度来考虑，策略蒸馏的方式同样可以分为两大类：（1）最小化教师与学生策略中动作分布之间的交叉熵损失；（2）最大化教师策略产生学生策略所产生的轨迹的概率，即 $\max_\theta P(\tau\sim\pi_E|\tau\sim\pi_\theta)$。

- [16] 所提出的演员-模仿（Actor-mimic）算法是上述第一种方法，其策略蒸馏的损失函数为：  
   $\mathcal{L}^{i}(\theta)=\sum_{a \in \mathcal{A}_{E_{i}}} \pi_{E_{i}}(a \mid s) \log _{\pi_{\theta}}(a \mid s) \\$ 其中每一个教师智能体都是根据DQN训练的，因此他们的策略根据Q-函数输出的玻尔兹曼分布得到：  
   $\pi_{E_{i}}(a \mid s)=\frac{e^{\tau^{-1} Q_{E_{i}}(s, a)}}{\sum_{a^{\prime} \in \mathcal{A}_{E_{i}}} e^{\tau^{-1} Q_{E_{i}}\left(s, a^{\prime}\right)}} \\$
- [17] 提出的Distral算法采用的是上述第二种方法，该算法基于 $K$ 个教师策略训练一个质心策略 $\pi_\theta$，每个教师策略的源域为 $\left.\mathcal{M}_{i}=\left\{\mathcal{S}_{i}, \mathcal{A}_{i}, \mathcal{T}_{i}, \gamma, \mathcal{R}_{i}\right)\right\}$，假设不同源MDP的动态转移 $\mathcal{T}_i$ 与奖励分布 $\mathcal{R}_i$ 是不同的，学生策略的目标是能够在不同域中执行任务，其目标函数为最大化 $\max _{\theta} \sum_{i=1}^{K} J\left(\pi_{\theta}, \pi_{E_{i}}\right)$，其中  
   $\begin{aligned} J\left(\pi_{\theta}, \pi_{E_{i}}\right)=\sum_{t} \mathbb{E}_{\left(s_{t}, a_{t}\right) \sim \pi_{\theta}}\left[\sum _ { t \geq 0 } \gamma ^ { t } \left(r_{i}\left(a_{t}, s_{t}\right)-\right.\right.\\ \left.\left.\frac{\alpha}{\beta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right)+\frac{1}{\beta} \log \left(\pi_{E_{i}}\left(a_{t} \mid s_{t}\right)\right)\right)\right] \end{aligned} \\$ 该方法与RS方法十分类似，$-\log \left(\pi_{E_{i}}\left(a_{t} \mid s_{t}\right)\right.$ 项引导所学习的策略 $\pi_\theta$ 更加接近于教师策略，而熵项 $\log \pi_{\theta}\left(a_{t} \mid s_{t}\right)$ 则可以用于促进智能体探索。

**策略复用（Policy Reuse）**

**策略复用（Policy Reuse）**直接从源任务中复用策略以构建目标策略。

- [18] 提出以一定的概率分布 $P$ 来学习专家策略，这里每个策略的使用概率与该策略在目标域的期望性能增益有关，记为 $W_i$：$P\left(\pi_{E_{i}}\right)=\frac{\exp \left(t W_{i}\right)}{\sum_{j=0}^{K} \exp \left(t W_{j}\right)} \\$ 这里 $t$ 是随时间增加的动态温度系数。在Q-learning框架下，目标策略的Q函数以迭代的方法学习：在每一个学习episode，每个策略的 $W_i$ 被重新评估，以此得到新的复用概率 $P$，然后根据该概率采样得到一个行为策略。如果采样到的是专家策略，则采用 $\epsilon$-贪心策略做出动作，如果是正在学习的策略，则直接采用贪心策略。

**一些思考**

实际上我并没有理解策略蒸馏的真实操作，当教师策略和学生策略的状态空间和动作空间都不一样时，怎么让学生策略模仿教师策略呢？另外，策略蒸馏的意义是什么，如果是从多个教师策略中蒸馏出一个学生策略，这样可能是希望学生策略执行多任务，但这样状态和动作空间完全不一样，是如何操作的呢？而如果是一对一任务，即一个教师策略对一个学生策略，那这样策略蒸馏的意义又是什么呢？

### **任务间映射（Inter-Task Mapping）**

这一类方法一般都有如下假设：【域间映射】假设在源域 $\mathcal{M}_s=\left(\mu_{0}^{s}, \mathcal{S}^{s}, \mathcal{A}^{s}, \mathcal{T}^{s}, \gamma^{s}, \mathcal{R}^{s}, \mathcal{S}_{0}^{s}\right)$ 与目标域 $\mathcal{M}_t\left(\mu_{0}^{t}, \mathcal{S}^{t}, \mathcal{A}^{t}, \mathcal{T}^{t}, \gamma^{t}, \mathcal{R}^{t}, \mathcal{S}_{0}^{t}\right)$ 之间存在一对一映射。

早期研究需要给定的映射函数。

- [19] 假设分别存在给定的状态空间与动作空间的映射：$X_{S}\left(\mathcal{S}^{t}\right) \rightarrow \mathcal{S}^{s}$, $ X_{A}\left(\mathcal{A}^{t}\right) \rightarrow \mathcal{A}^{s}$，基于此，可以推导出Q值之间的映射：$M\left(Q_{s}\right) \rightarrow Q_{t}$。

后来更多的研究通过自动学习一个映射函数，来处理任务间映射问题，大多数工作会学习状态空间或者状态空间子集上的映射函数，在这些工作中，状态表征会被划分为 **智能体特定的** 以及 **任务特定** 的表征，分别记为 $s_{agent}$ 与 $s_{env}$。

- [20] 在智能体特性的状态子空间中学习了一个映射函数，这个映射函数用于设计奖励，这里从 $s_{agent}$ 中获得的不变的特征空间可以用于动作空间不同但有形态学共同点的智能体上。具体而言，该方法假设智能体训练于同一个代理任务上，基于此可以得到映射函数。映射函数使用一个编码-译码神经网络来学习，以此尽可能地保留源域中的特征。当将源域智能体的知识迁移到新任务的目标智能体上时，会在奖励中增加一项用于鼓励目标智能体在潜在特征空间模仿源智能体：  
   $r^{\prime}(s, \cdot)=\alpha\left\|f\left(s_{a g e n t}^{s} ; \theta_{f}\right)-g\left(s_{a g e n t}^{t} ; \theta_{g}\right)\right\| \\$ 其中 $f(s^s_{agent})$ 是源域中的智能体特定状态，而 $g(s^t_{agent})$ 是目标域中的智能体特定状态。
- [21] 提出可以学习转移动作空间： $\mathcal{S}\times\mathcal{A}\times\mathcal{S}$ 上的任务间映射，该研究假设源域与目标域的转移空间不同。源域与目标域的转移元组： $\left\langle s^{s}, a^{s}, s^{\prime s}\right\rangle$ 与 $\left\langle s^{t}, a^{t}, s^{\prime t}\right\rangle$ 均被映射到隐藏空间 $Z$。给定 $Z$ 上的特征表达，可以度量源域元组与目标域元组之间的相似度，相似度高的对可以用于学习映射关系：$\mathcal{X}:\left\langle s^{t}, a^{t}, s^{\prime t}\right\rangle=\mathcal{X}\left(\left\langle s^{s}, a^{s}, s^{\prime s}\right\rangle\right) $。此后，源域专家策略中采样得到的状态可用于促进目标域中好状态更多的出现，以此得到一个更好的初始化。

**一些思考**

我大致理解了这一类方法的思想，问题只是在于如何具体实施，这需要更加仔细地阅读论文。

### **表征迁移（Representation Transfer）**

**表征迁移（Representation Transfer）**通常将特征表达作为迁移的知识，例如价值函数或Q函数所学习到的表征，本节所介绍的方法主要得益于深度神经网络强大的近似能力，它们都基于如下假设：

【**任务不变空间的存在**】状态空间（$\mathcal{S}$），动作空间（$\mathcal{A}$），以及奖励空间（$\mathcal{R}$）可以被解构为相互正交的子空间，其中一些在不同任务之间是不变的，因此可以被源域与目标域共享，因此这些子空间上的知识可以在不同域之间迁移。

这一类方法可以进一步分为两类：1）直接复用源域的表征；2）将源域表征结构成独立的子特征表达，然后复用其中可以被源域目标域共享的部分。

**表征复用（Reusing Representations）**

- [22] 提出了一种**递进神经网络（progressive neural network）**，用于递进地在多个RL任务上递进任务。一个递进网络包含多列（columns），每一列均为一个用于训练某特定任务的策略网络。网络首先训练第一个任务对应的第一列，随后列数随着任务数增加。训练新的任务时，之前列的神经元系数固定起来，然后这些任务的表征将通过一个特定的联系辅助新的任务学习，这一过程可以用数学表示为：  
   $h_{i}^{(k)}=f\left(W_{i}^{(k)} h_{i-1}^{(k)}+\sum_{j<k} U_{i}^{(k: j)} h_{i-1}^{(j)}\right)$ 其中 $h^{k}_i$ 为任务（列）$k$ 的第 $i$ 个隐藏层，$W_i^{(k)}$ 是其对应的权重矩阵，$U_i^{(k:j)}$ 表示之前任务的第 $i-1$ 层特征对任务 $k$ 当前层的联系。递进神经网络可以有效处理多任务，但它需要巨大的网络结构，该结构随着任务增多而增大。
- [23] 提出了 **PathNet** ，该结构采用固定尺寸的网络解决了网络增大的问题。PathNet包含 **pathways**，即包含之前任务知识的神经元子集，pathway的数目通过锦标赛选择遗传算法（tournament selection genetic algorithm）发展。
- [24] 采用**模块化网络（modular network）**，将策略网络分解为任务特定模块与智能体特定模块，具体而言，$\pi$ 是智能体在任务 $\mathcal{M}_k$ 上的策略，是状态 $s$ 的函数 $\phi$，该函数可以分解为两个模块 $g_k$ 与 $f_r$：  
   $\pi(s):=\phi\left(s_{\text {env }}, s_{\text {agent }}\right)=f_{r}\left(g_{k}\left(s_{\text {env }}\right), s_{\text {agent }}\right) \\$ 其中 $f_r$ 为智能体特定模块，$g_k$ 为任务特定模块，该论文的中心思想是任务特定模块可以作为迁移知识，应用于执行相同任务的不同智能体上。相应地，智能体特定模块可以应用在相同智能体的不同任务上。

**表征解构（Disentangling Representations）**

这一类方法通常会学习解构后的表征，本节介绍其中两个代表性的技术：继承表征（Successor Representation，SR）以及万能价值函数近似（Universal Value Function Approximation，UVFA）。

**继承表征（Successor Representation，SR）**技术用于将一个域中的状态特征从奖励分布中解耦出来，该技术假设**不同域之间的区别仅在于奖励分布**：$\mathcal{R}_i=\mathcal{R}_j$，这样知识就可以在不同域 $\mathcal{M}=\left\{\mathcal{M}_{1}, \mathcal{M}_{2}, \ldots, \mathcal{M}_{K}\right\}$ 之间迁移。标准MDP用v值或Q值描述状态对奖励的依赖关系。而在SR中，通常用后续状态的占用率（occupancy measure）描述一个状态，这里的占用率表示智能体采取策略 $\pi$ 后遇到状态或状态-动作对的概率分布。具体来说，SR将任何策略的价值函数分解成两个独立的部分：$\psi$ 与 $R$：



$$
V^{\pi}(s)=\sum_{s^{\prime}} \psi\left(s, s^{\prime}\right) \mathbf{w}\left(s^{\prime}\right) \\
$$



其中 $\mathbf{w}(s')$ 为奖励映射函数，用于将状态映射成标量奖励，而 $\psi$ 则表示在状态 $s$ 采取策略 $\pi$ 后会遇到某一状态的概率：



$$
\psi\left(s, s^{\prime}\right)=\mathbb{E}_{\pi}\left[\sum_{i=t}^{\infty} \gamma^{i-t} \mathbb{1}\left[S_{i}=s^{\prime}\right] \mid S_{t}=s\right] \\
$$



其中 $\mathbb{1}\left[S=s^{\prime}\right]=1$ 为指示函数。这一继承特性使得该方法可以应用在各类强化学习算法中。

**万能价值函数近似（Universal Value Function Approximation，UVFA）**同样用于解构状态特征表达。与SR一样，UVFA同样允许只有奖励函数不同的域之间的迁移。不同于SR学习一个与奖励无关的状态表征，UVFA的目标是寻找一个可适用于状态与奖励的函数近似。UVFA的问题场景为：

【依赖于目标的RL】任务目标定义于状态上，例如。给定状态空间 $\mathcal{S}$ 与目标空间 $\mathcal{G}$，满足 $\mathcal{G}\subseteq\mathcal{S}$。其中一个示例就是智能体在迷宫中寻找目标，此时目标定义为迷宫内的特定位置。

在该问题场景下，可以通过对奖励矩阵应用矩阵分解技术将一个UVFA模块分解为一个状态嵌入 $\phi(s)$ 与一个目标嵌入 $\psi(g)$，此时就可以在只有目标不同的任务中迁移 $\psi(s)$。然而矩阵分解需要耗费大量的时间，使得该方法在实际场景下难以实现。

上面介绍的方法总结如下：

![](../../assets/images/f5d92a85983db3f63a63.jpg)

**一些思考**

这一类方法通常假设源域和目标域只有奖励分布不同，而其他的MDP成分都是相同的，这样的假设感觉过于理想化了，可以应用的场景比较具有局限性。

## **总结**

![](../../assets/images/83db6e38903c49b1a83b.jpg)

不同的迁移学习方法总结如上，说实话看完这篇综述之后，我完全想不到可以如何将迁移学习应用在我的场景中，所以我打算继续看一些偏应用的论文，另外，在OpenAI的SpinningUp教程中，给出的迁移学习推荐论文如下，其中一些在上文中也有所介绍，例如Progressive Networks、PathNet以及UVFA，但大多数并没有出现在这篇综述中，如果感兴趣的话也可以看一看。

![](../../assets/images/d8776fb67053106415c2.jpg)

## **参考文献**

[1] Z. Zhu, K. Lin, and J. Zhou, ‘Transfer Learning in Deep Reinforcement Learning: A Survey’, *arXiv:2009.07888 [cs, stat]*, Mar. 2021, Accessed: Mar. 23, 2022. [Online]. Available: <http://arxiv.org/abs/2009.07888>

[2] A. Y. Ng, D. Harada, and S. Russell, “Policy invariance under reward transformations: Theory and application to reward shaping,” in ICML, vol. 99, 1999, pp. 278–287.

[3] E. Wiewiora, G. W. Cottrell, and C. Elkan, “Principled methods for advising reinforcement learning agents,” in Proceedings of the 20th International Conference on Machine Learning (ICML-03), 2003, pp. 792–799.

[4] S. M. Devlin and D. Kudenko, “Dynamic potentialbased reward shaping,” in Proceedings of the 11th International Conference on Autonomous Agents and Multiagent Systems. IFAAMAS, 2012, pp. 433–440.。

[5] A. Harutyunyan, S. Devlin, P. Vrancx, and A. Now´e, “Expressing arbitrary reward functions as potentialbased advice,” in Twenty-Ninth AAAI Conference on Artificial Intelligence, 2015.

[6] T. Brys, A. Harutyunyan, M. E. Taylor, and A. Now´e, “Policy transfer using reward shaping,” in Proceedings of the 2015 International Conference on Autonomous Agents and Multiagent Systems. International Foundation for Autonomous Agents and Multiagent Systems, 2015, pp. 181–188.

[7] M. Veˇcer´ık, T. Hester, J. Scholz, F. Wang, O. Pietquin, B. Piot, N. Heess, T. Roth¨ orl, T. Lampe, and M. Riedmiller, “Leveraging demonstrations for deep reinforcement learning on robotics problems with sparse rewards,” arXiv preprint arXiv:1707.08817, 2017.

[8] J. Chemali and A. Lazaric, “Direct policy iteration with demonstrations,” in Twenty-Fourth International Joint Conference on Artificial Intelligence, 2015.

[9] T. Hester, M. Vecerik, O. Pietquin, M. Lanctot, T. Schaul, B. Piot, D. Horgan, J. Quan, A. Sendonaris, I. Osband et al., “Deep q-learning from demonstrations,” in Thirty- Second AAAI Conference on Artificial Intelligence, 2018.

[10] T. Brys, A. Harutyunyan, H. B. Suay, S. Chernova, M. E. Taylor, and A. Now´e, “Reinforcement learning from demonstration through shaping,” in Twenty-Fourth International Joint Conference on Artificial Intelligence, 2015.

[11] J. Ho and S. Ermon, “Generative adversarial imitation learning,” in Advances in neural information processing systems, 2016, pp. 4565–4573.

[12] B. Kang, Z. Jie, and J. Feng, “Policy optimization with demonstrations,” in International Conference on Machine Learning, 2018, pp. 2474–2483.

[13] M. Veˇcer´ık, T. Hester, J. Scholz, F. Wang, O. Pietquin, B. Piot, N. Heess, T. Roth¨ orl, T. Lampe, and M. Riedmiller, “Leveraging demonstrations for deep reinforcement learning on robotics problems with sparse rewards,” arXiv preprint arXiv:1707.08817, 2017.

[14] A. Nair, B. McGrew, M. Andrychowicz, W. Zaremba, and P. Abbeel, “Overcoming exploration in reinforcement learning with demonstrations,” in 2018 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2018, pp. 6292–6299.

[15] A. A. Rusu, S. G. Colmenarejo, C. Gulcehre, G. Desjardins, J. Kirkpatrick, R. Pascanu, V. Mnih, K. Kavukcuoglu, and R. Hadsell, “Policy distillation,” arXiv preprint arXiv:1511.06295, 2015.

[16] E. Parisotto, J. L. Ba, and R. Salakhutdinov, “Actormimic: Deep multitask and transfer reinforcement learning,” arXiv preprint arXiv:1511.06342, 2015.

[17] Y. Teh, V. Bapst, W. M. Czarnecki, J. Quan, J. Kirkpatrick, R. Hadsell, N. Heess, and R. Pascanu, “Distral: Robust multitask reinforcement learning,” in Advances in Neural Information Processing Systems, 2017, pp. 4496–4506.

[18] F. Fern´andez and M. Veloso, “Probabilistic policy reuse in a reinforcement learning agent,” in Proceedings of the fifth international joint conference on Autonomous agents and multiagent systems. ACM, 2006, pp. 720–727.

[19] M. E. Taylor, P. Stone, and Y. Liu, “Transfer learning via inter-task mappings for temporal difference learning,” Journal of Machine Learning Research, vol. 8, no. Sep, pp. 2125–2167, 2007.

[20] A. Gupta, C. Devin, Y. Liu, P. Abbeel, and S. Levine, “Learning invariant feature spaces to transfer skills with reinforcement learning,” International Conference on Learning Representations (ICLR), 2017.

[21] H. B. Ammar, K. Tuyls, M. E. Taylor, K. Driessens, and G. Weiss, “Reinforcement learning transfer via sparse coding,” in Proceedings of the 11th International Conference on Autonomous Agents and Multiagent Systems- Volume 1. International Foundation for Autonomous Agents and Multiagent Systems, 2012, pp. 383–390.

[22] A. A. Rusu, N. C. Rabinowitz, G. Desjardins, H. Soyer, J. Kirkpatrick, K. Kavukcuoglu, R. Pascanu, and R. Hadsell, “Progressive neural networks,” arXiv preprint arXiv:1606.04671, 2016.

[23] C. Fernando, D. Banarse, C. Blundell, Y. Zwols, D. Ha, A. A. Rusu, A. Pritzel, and D. Wierstra, “Pathnet: Evolution channels gradient descent in super neural networks,” arXiv preprint arXiv:1701.08734, 2017.

[24] C. Devin, A. Gupta, T. Darrell, P. Abbeel, and S. Levine, “Learning modular neural network policies for multitask and multi-robot transfer,” in 2017 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2017, pp. 2169–2176.
