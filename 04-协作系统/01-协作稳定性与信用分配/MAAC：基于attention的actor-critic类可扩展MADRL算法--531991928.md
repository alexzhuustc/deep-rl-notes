# 【多智能体强化学习】MAAC：基于attention的actor-critic类可扩展MADRL算法

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2022-10-03  
来源：[https://zhuanlan.zhihu.com/p/531991928](https://zhuanlan.zhihu.com/p/531991928)

[知识库首页](../../README.md)

---

*#碎碎念*

*调研了强化学习的一些其他方向之后，我还是决定要做多智能体了，相对元强化学习和分层强化学习等等有些冷门的方向而言，直觉告诉我做多智能体强化学习痛苦可能会少一点，至少网上资料多，做的人多，我能大致知道要朝着怎样的目标去做，遇到问题了也有可以请教的同学，这一点对于需要完全独立完成课题的人来说简直太重要了~去年的时候自己就看过一波多智能体的论文，那个时候学长极力劝阻我做多智能体，让我做单智能体，于是靠着constrained-RL水了一篇论文，之后就开始调研分层强化学习和元强化学习，但想了好久实在想不出来这俩方向能用在什么场景中，再者分层和元强化学习好冷门，算法感觉也不成熟，大概是不适合基础并没有那么好的交叉学科研究者使用。作为一名不太擅于提创新点的博士，最痛苦的不是具体做什么做不出来，而是实在想不出来自己可以做什么，挣扎了好久，还是决定要做多智能体了，卷就卷吧，难就难吧，至少这样我即将面临的将会是一个一个具体实在的问题，而不是我到底可以做什么这种想破脑袋也想不出来的东西。兜兜转转还是回到了最初的原点。*

*---*
> MAAC是基于actor-critic的[learn to cooperate]类多智能体强化学习算法，算法结合了MADDPG、COMA、VDN以及attention机制，虽然创新点不算很大，但是看这篇论文加深了对[learn to cooperate]类多智能体算法的理解，从这个角度来看也是蛮有意义的。推荐指数：★★★★

首先回顾一下多智能体强化学习~

[XuanAxuan：【多智能体强化学习】新手入门算法论文大总结](新手入门算法论文大总结--432241482.md)

在多智能体系统中，单个智能体的环境包含其他智能体，在学习的过程中，每个智能体的策略都在不断更新，这样就导致每个智能体所处的环境是非平稳的，破坏了强化学习中的马尔科夫条件，因此，独立训练各个智能体的方法在复杂的多智能体场景下可能会难以收敛。另一方面，将所有智能体当做一个智能体，集中学习环境状态到所有智能体动作之间映射的方法可以完全消除不同智能体策略改变所带来的非平稳性，但是这种方法的状态、动作空间会随着智能体数目增多而指数增加，扩展性非常差，通常没有办法学出来。基于上述背景，我们需要寻求**介于完全独立学习与完全集中学习之间**的多智能体强化学习算法。

现有的多智能体强化学习算法大致可以分为两大类：

1. **[learn to cooperate]**算法研究如何通过集中训练的方式得到一系列可分布式执行的智能体策略，这类算法又可以分为下面三类：

- 基于**独立学习**：这类算法在独立学习的基础上利用一些训练技巧来缓解非平稳性的影响，例如**参数共享（parameter sharing**），该方法也是许多多智能体算法的默认设定；
- 基于**值函数分解（value decomposition）**：这类算法以DQN为基础，假定各个智能体的局部Q网络与整个系统的全局Q网络之间存在某种线性/非线性关系，具体做法是首先定义每个智能体的局部Q网络，然后由假设条件推断出全局Q网络，训练时利用全局Q网络的损失函数对局部Q网络进行集中式的端到端训练，最终即可得到各个智能体的局部Q网络用于分布式执行，代表算法有VDN、QMIX；
- 基于**actor-critic**：该类算法以on-policy/off-policy的actor-critic框架为基础，利用全局状态-动作信息对一个或多个critic进行集中式训练，然后利用集中式critic训练每个智能体的actor，actor只能接收单个智能体的局部状态信息作为输入，确保分布式执行，代表算法有MADDPG、COMA。

2. **[learn to communicate]则研究如何利用智能体之间信息的交互**进一步提升算法的性能。

总而言之，[learn to cooperate]类算法都是通过在训练过程中利用全局状态-动作信息训练一个/多个集中式的Q网络/critic，缓解非平稳性的影响，但在分布式执行的过程中，各个智能体的策略是完全独立的，相互之间不会传递任何信息，这里的策略既可以是基于Q网络的离散动作策略，也可以是基于actor的连续动作策略；[learn to communicate]则是在独立学习或[learn to cooperate]算法的基础上通过智能体之间信息的交互进一步缓解非平稳性，提升算法性能。

MAAC是基于actor-critic的[learn to cooperate]算法，该算法利用**attention机制**改善了**MADDPG**中critic输入随智能体数目增大而指数增加的**扩展性**问题，同时还借鉴**COMA**的思想，利用**反事实基线（counterfactual baseline）**来区分单个智能体对系统奖励的贡献，另外，MAAC还借鉴了VDN中**值函数分解**的思想用所有Q网络损失函数之和对每个Q网络进行集中式训练。MAAC的基础算法为**SAC**算法。奇怪的是，既然采用的是actor-critic框架，按理来说MAAC应该很适合连续任务，但作者只对MAAC在离散任务下的性能进行了仿真，并没有对连续任务进行实验，所以我也不确定该算法是否适合连续任务。

## **MADDPG、COMA**

MAAC最相关的两个算法为MADDPG与COMA，因此本小节首先简单地回顾和总结一下这两个算法，也可以去看我之前的笔记：

[【多智能体强化学习】[learn to cooperate] MADDPG 阅读笔记](MADDPG 阅读笔记--436281028.md)

[【多智能体强化学习】[learn to cooperate] COMA 阅读笔记](COMA 阅读笔记--436284848.md)

### **MADDPG（multi-agent deep deterministic policy gradient）**

[XuanAxuan：【多智能体强化学习】[learn to cooperate] MADDPG 阅读笔记](MADDPG 阅读笔记--436281028.md)![](../../assets/images/2898be3b68a072a18e90.jpg)

MADDPG以DDPG算法为基础，每个智能体均需要学习一个critic与一个actor，其中critic接收所有智能体的观测-动作作为输入，而actor则只能以单个智能体的局部观测为输入，输出其动作。由于每个智能体都会单独学习一个Q值，因此每个智能体都**能够设计任意属于自己的奖励**。MADDPG背后的思想是，如果我们知道所有智能体执行的动作，那么即使每个智能体的策略都在发生变化，整个系统的环境也是平稳变化的，因为对于任意 $\mathbf{\pi}_i\ne \mathbf{\pi}_{i^{\cdot}}'$​ ，均有



$$
P\left(s^{\prime} \mid s, a_{1}, \ldots, a_{N}, \pi_{1}, \ldots, \boldsymbol{\pi}_{N}\right)=P\left(s^{\prime} \mid s, a_{1}, \ldots, a_{N}\right)=P\left(s^{\prime} \mid s, a_{1}, \ldots, a_{N}, \pi_{1}^{\prime}, \ldots, \boldsymbol{\pi}_{N}^{\prime}\right) \\
$$



实际上，MADDPG各个智能体的学习过程可以看做是**相互独立**的，只是在学习的过程中需要利用其他智能体的观测-动作信息，也就是说在训练过程中进行了**观测共享**。这样做的好处是，MADDPG可以用在**异构智能体**的场景中，这里的异构指的是不同智能体的观测空间、动作空间或奖励函数可以不同。但是，由于critic需要接收所有智能体的观测-动作作为输入，其输入会随着智能体数目的增加而增加，存在**扩展性**问题。并且，为了保证critic输入节点固定，MADDPG中智能体的**数目还必须是固定**的，不允许智能体数目动态变化。

考虑异构智能体既是MADDPG的优势，同样也使MADDPG存在一些局限。当所有智能体都共用一个系统奖励时，每个智能体所学习的critic实际上是全局critic，评估的是所有智能体的整体观测-动作对的好坏，因此单个智能体无法从中判断自身观测-动作对整个系统的影响，这也就是所谓的**信用分配（credit assignment）**问题。最后一个比较实际的问题是，MADDPG中没有提到参数共享，也就是说每个智能体分别需要学习一套参数，这就给训练带来了很大的难度，但我想如果智能体确实是同质的，加上参数共享应该问题不大。

### **COMA（Counterfactual Multi-Agent Policy Gradients）**

[XuanAxuan：【多智能体强化学习】[learn to cooperate] COMA 阅读笔记](COMA 阅读笔记--436284848.md)![](../../assets/images/6919ecfa7e4120b733f4.jpg)

COMA的思想与MADDPG十分类似，区别在于，MADDPG要求每个智能体学习一个全观测的critic，而在COMA算法中，只需要学习一个集中式的critic即可，该critic接收所有智能体的观测-动作作为输入，类似于MADDPG，每个智能体的actor也只能接收其自身的观测作为输入。另外，MADDPG以DDPG为基础，因此是off-policy算法，而COMA以策略梯度定理为基础，因此是on-policy算法。COMA假设的场景是所有智能体是同质的，观测、动作空间以及奖励均相同，因此用了**参数共享**。COMA算法的另一个重要贡献是利用**反事实基线（counterfactual baseline）**解决了共用一个系统奖励的多智能体系统中的**信用分配（credit assignment）**问题。关于信用分配，我的理解是：在MADDPG与COMA中，critic都是接收所有智能体的观测-动作作为输入，也就是说，此时**critic度量的是全局动作的好坏，单个智能体也就无法从中判断其自身动作对整个系统奖励的影响**。 为了解决该问题，COMA的作者提出为每个智能体计算一个**优势函数（advantage function）**用于评估单个智能体动作的好坏，即用全局Q值减去该智能体动作随机选取求期望之后的值：



$$
A^{a}(s, \mathbf{u})=Q(s, \mathbf{u})-\sum_{u^{\prime a}} \pi^{a}\left(u^{\prime a} \mid \tau^{a}\right) Q\left(s,\left(\mathbf{u}^{-a}, u^{\prime a}\right)\right) \\
$$



这里的 $u$ 表示动作，$a$ 表示智能体索引。对于每个智能体 $a$，可以直接将 $A^{a}(s, \mathbf{u})$ 代入到策略梯度定理中，对策略进行更新（这里应用到了**带基线的策略梯度定理**，即在 $Q(s, \mathbf{u})$ 的基础上减去一个 $b(s)$ 不影响策略梯度的计算）。

由于涉及求期望，按常规方法计算 $A^{a}(s, \mathbf{u})$ 十分麻烦，在具体的实施过程中，针对**离散任务**，作者提出对于单个智能体，将其他智能体的动作作为Q网络的输入，然后输出该智能体每个动作对应的Q值，再根据当前的策略计算 $A^{a}(s, \mathbf{u})$ 即可。对于 **连续任务**，作者提出可以用蒙特卡洛方法计算 $A^{a}(s, \mathbf{u})$ 表达式中的期望，另外，MAAC的作者还提出可以再学一个Q网络用于学习其他智能体的观测-动作价值来计算 $A^{a}(s, \mathbf{u})$（两个Q网络相减？），即此时的输入不包括该智能体自身的动作。可惜的是，COMA和MAAC的源代码均只考虑了离散任务的情况，所以我不清楚反事实基线在连续任务中到底效果如何，是否可行。

关于信用分配，通俗一点的话解释，对于单个智能体而言，系统奖励值很大，可能是因为其他智能体的动作很好，而自己的动作其实不好，所以就需要计算当前动作相对于随机选取动作的优势，这就是COMA的思想。

### **总结**

可以看出，MADDPG与COMA都是基于actor-critic的[learn to cooperate]算法，两者的算法框架十分类似，但所针对的场景不同，其中COMA针对的是同质智能体完全协作（共用同一系统奖励）的场景，只学习一个集中式critic，因此着重考虑了其中所存在的**信用分配**问题；而在MADDPG算法中，每个智能体都需要学一个全观测的critic，因而每个智能体的奖励既可以相同也可以不同，也就没有考虑信用分配问题。

## **MAAC（Multi-Actor-Attention-Critic）**

我个人认为MAAC本身没有提出特别大的创新，它结合了MADDPG、COMA和VDN等算法中的基本思想，与此同时利用attention解决了集中式critic的扩展性问题，由于考虑的东西太多，我对MAAC的实际使用效果是存疑的，即便如此，我仍然觉得将MAAC中的各个操作拆解出来一一分析十分有意义，这样做可以让我们重新回顾MADDPG、COMA、VDN等等优秀的算法，并对其中的思想进行再次梳理。

![](../../assets/images/58fd778af4d26b25881e.jpg)

类似于MADDPG，MAAC算法中，每个智能体均需要训练一个全观测的critic用于克服环境非平稳性的影响，与此同时利用critic训练各自的actor用于策略的分布式执行。在MADDPG中，critic的输入由各个智能体的观测-动作对**堆叠**而成，因此其输入节点会随着智能体数目的增加而增加，当智能体数目增大时，学习难度会越来越大，难以扩展。针对此问题，MAAC算法最重要的思想就是利用attention机制从其他智能体的观测-动作对中选取重要的信息用于构造每个智能体critic的输入，这样做既能解决扩展性问题，还能帮助智能体有选择性地筛选出重要的环境信息，忽略掉不重要的信息。

### **Attention**

Attention机制采用的是key-value记忆模型，即每个智能体向其他智能体询问（query）关于对方观测-动作的信息，然后将这些信息作为自身critic的输入。具体地，智能体 $i$ （$i\in\{1...N\}$）的Q函数可以记为



$$
Q_{i}^{\psi}(o, a)=f_{i}\left(g_{i}\left(o_{i}, a_{i}\right), x_{i}\right) \\
$$



其中 $o=\left(o_{1}, \ldots, o_{N}\right)$ 与 $a=\left(a_{1}, \ldots, a_{N}\right)$ 分别表示所有智能体的观测集合与动作集合。$f_i$ 与 $g_i$ 均表示MLP层，其中 $g_i$ 用于对智能体的观测-动作对进行embedding操作。 $x_i$ 表示来自其他智能体观测-动作对的环境信息，其表达式为



$$
x_{i}=\sum_{j \neq i} \alpha_{ij} v_{j}=\sum_{j \neq i} \alpha_{ij} h\left(V g_{j}\left(o_{j}, a_{j}\right)\right) \\
$$



这个表达式就体现了所谓的attention机制，$\alpha_{ij}$ 是智能体 $i$ 对智能体 $j$ 的注意力权重，$v_j$ 是智能体 $j$ 观测-动作对的embedding，即首先用 $g_j$ 对观测-动作对进行embedding编码，然后利用线性矩阵 $V$ 进行线性变换，最后进行非线性操作 $h$ （例如leaky ReLU）。

注意力权重 $\alpha_{ij}$ 通过比较智能体 $i$ 与智能体 $j$ 的embedding向量 $g_i$ 与 $g_j$ 之间的相似性得到，即



$$
\alpha_{ij} \propto \exp \left(e_{j}^{\mathrm{T}} W_{k}^{\mathrm{T}} W_{q} e_{i}\right) \\
$$



其中 $W_q$ 将 $e_i$ 映射为 query，而 $W_k$ 则将 $e_j$ 映射为 key，另外我们还需要将 $\alpha$ 放缩到合适的值放置梯度消失。

MAAC使用了多个attention heads，每个head使用一套独立的参数 $(W_k,W_q,V)$，然后将每个head得到的 $v_j$ 连接起来作为critic的输入。另外，所有智能体共用参数 $(W_k,W_q,V)$ ，这样可以促使智能体即便在异构的情况下也能学习到共同的embedding空间，这种参数共享背后的思想是，即便智能体的观测空间、动作空间和奖励不同，也总会有一些共同的特征，而参数共享有利于学习到这些共同的特征。

总结来说，MAAC中实际上就是将其他智能体的观测-动作对先进行embedding操作然后利用注意力权重加权相加然后再与自身智能体的观测-动作对连接起来作为critic的输入，注意力权重度量了两个智能体embedding的相似程度。直观上来看，这样做能够让智能体更多地关注与自己相似的智能体，理论上来说对信息的利用率应该更高，另外通过将所有智能体的观测-动作对合并成一个单维度向量，也解决了扩展性问题。

### **Critic的集中式训练**

由于不同智能体的critic之间用到了共享的参数，所以MAAC提出用一个联合的损失函数来训练各个critic，即：



$$
\begin{aligned} \mathcal{L}_{Q}(\psi)=& \sum_{i=1}^{N} \mathbb{E}_{\left(o, a, r, o^{\prime}\right) \sim D}\left[\left(Q_{i}^{\psi}(o, a)-y_{i}\right)^{2}\right],  \\ \text { where }y_{i}=& r_{i}+\gamma \mathbb{E}_{a^{\prime} \sim \pi_{\bar{\theta}}\left(o^{\prime}\right)}\left[Q_{i}^{\bar{\psi}}\left(o^{\prime}, a^{\prime}\right)-\right. \left.\alpha \log \left(\pi_{\bar{\theta}_{i}}\left(a_{i}^{\prime} \mid o_{i}^{\prime}\right)\right)\right] \end{aligned} \\
$$



作者把多智能体的值函数近似比作**多任务回归（multi-task regression）**问题，通过参数共享来提取各个智能体的共同特征。

各个智能体的actor策略则通过下面的式子进行更新



$$
\begin{gathered} \nabla_{\theta_{i}} J\left(\pi_{\theta}\right)=  \mathbb{E}_{o \sim D, a \sim \pi}\left[\nabla _ { \theta _ { i } } \log ( \pi _ { \theta _ { i } } ( a _ { i } | o _ { i } ) ) \left(-\alpha \log \left(\pi_{\theta_{i}}\left(a_{i} \mid o_{i}\right)\right)+\right.\right.  \left.\left.Q_{i}^{\psi}(o, a)-b\left(o, a_{\backslash i}\right)\right)\right] \end{gathered} \\
$$



这个地方我觉得作者写得十分不严谨，这里用的是**随机策略梯度**，但作者用到的是SAC算法，而SAC的策略梯度是在**确定性策略梯度**的基础上推导得到的，显然不是这个表达式，也不知道作者的源代码里是怎么写的........根据SAC的策略梯度，上面的式子应该修改为：



$$
\begin{gathered} \nabla_{\theta_{i}} J\left(\pi_{\theta}\right)=  \mathbb{E}_{o \sim D, a \sim \pi}\left[\alpha\nabla _{\theta_{i}}\log(\pi_{\theta_{i}}(a_{i}|o_{i}))-\nabla _ {\theta_{i}}\left(Q_{i}^{\psi}(o, a)-b\left(o, a_{\backslash i}\right)\right) \right] \end{gathered} \\
$$



这里的 $b(o,a_{\backslash i})$ 是用于计算优势函数的反事实基线（counterfactual baseline），与COMA类似，下节介绍。另外，在MADDPG中，其他智能体的动作都是直接从经验回放池中采样的，这样可能会导致过采样，而在MAAC中，所有智能体的动作都从当前策略中采样得到，保证了智能体能够在当前策略上进行协同。

### **Counterfactual Baselines**

MAAC还借鉴了COMA中用counterfactual baseline计算优势函数的方法来解决信用分配问题。优势函数的计算公式如下：



$$
\begin{aligned} A_{i}(o, a) &=Q_{i}^{\psi}(o, a)-b\left(o, a_{\backslash i}\right), \text { where } \\ b\left(o, a_{\backslash i}\right) &=\mathbb{E}_{a_{i} \sim \pi_{i}\left(o_{i}\right)}\left[Q_{i}^{\psi}\left(o,\left(a_{i}, a_{\backslash i}\right)\right)\right] \end{aligned} \\
$$



跟COMA一样，MAAC同样只针对离散任务，提出用一个神经网络输出智能体每一个动作的 $Q_i(o,(a_i,a_{\backslash i}))$，此时 $a_i$ 从输入端移除到输出端，因此需要额外为每个智能体学习一个观测的embedding $e_i=g_i^o(o_i)$ 来替代之前推导中的 $e_i=g_i(o_i,a_i)$，然后修正 $f_i$ 使其能够输出每个可能动作的Q值。得到 $Q_i(o,(a_i,a_{\backslash i}))$ 之后便可计算 $b(o,a_{\backslash i})$：



$$
\begin{gathered} \mathbb{E}_{a_{i} \sim \pi_{i}\left(o_{i}\right)}\left[Q_{i}^{\psi}\left(o,\left(a_{i}, a_{\backslash i}\right)\right)\right]= \sum_{a_{i}^{\prime} \in A_{i}} \pi\left(a_{i}^{\prime} \mid o_{i}\right) Q_{i}\left(o,\left(a_{i}^{\prime}, a_{\backslash i}\right)\right) \end{gathered} \\
$$



对于连续任务，作者同样只说明了一下可以通过对智能体策略采样计算上面的优势函数，或者学习一个额外的网络用于拟合 $b(o,a_{\backslash i})$，源代码中没有考虑这一情况。

实际上，这一部分我不确定是不是多此一举，首先COMA是on-policy方法，MAAC是off-policy方法，把随机策略梯度里边的东西用在确定性策略梯度里边真的有效吗？另外，上面提到MAAC采用了联合损失函数训练各个critic，而且每个critic已经使用了attention机制来筛选其他智能体的信息，我认为这样应该已经能够从一定程度上解决信用分配问题了。此时还在算法上增加counterfactual baseline操作，一来增加算法难度，二来还把算法限制在了离散任务中。作者并没有在实验中对这一操作进行消融实验，所以我也不确定这个操作是否有用，反正我个人觉得是没有太大必要的。

### 算法流程

![](../../assets/images/9634964950e1d840236a.jpg)![](../../assets/images/d665b4ee7e4b2539d061.jpg)

## **仿真实验**

![](../../assets/images/f023ba07f4b543066e9a.jpg)![](../../assets/images/4a356f2d455ff85c4949.jpg)

在作者设计的仿真实验中，MAAC的性能相较而言是最好的，并且随着智能体数目的增加，MAAC的优势越来越明显。

## **个人思考与总结**

我个人认为MAAC最大的创新点就是利用attention解决了MADDPG中critic集中式学习的扩展性问题，实际上很多[learn to communicate]算法中都用到了attention，这样看MAAC其实就是把一系列经典算法中的创新点结合在了一起，但无论如何应当都能算是一个比较不错的工作吧~在写作方面，感觉introduction写得挺好的，把各个算法都解释得比较清楚，但是后面的MAAC算法介绍部分，很多地方都写得不太严谨，不知道在代码中有没有体现。

另外，有没有同学在评论下讲一讲MAAC跟MADDPG相比到底有没有明显的优势呢~

### 参考

S. Iqbal and F. Sha, ‘Actor-Attention-Critic for Multi-Agent Reinforcement Learning’. arXiv, May 27, 2019. Accessed: Jun. 16, 2022. [Online]. Available: <http://arxiv.org/abs/1810.02912>
