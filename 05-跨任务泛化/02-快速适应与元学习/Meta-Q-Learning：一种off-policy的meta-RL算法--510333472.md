# 【元强化学习】Meta-Q-Learning：一种off-policy的meta-RL算法

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2022-05-06  
来源：[https://zhuanlan.zhihu.com/p/510333472](https://zhuanlan.zhihu.com/p/510333472)

[知识库首页](../../README.md)

---

今天介绍论文《Meta-Q-Learning》。

> 与PEARL一样，本文同样提出了一种off-policy的meta-RL算法，该算法将context类的meta-RL算法与off-policy方法结合了起来，另外还提出了用新老数据同时对新任务策略进行更新的adaptation过程，但实验结果表明该过程对性能提升作用不大。这篇论文给我的感觉就是算法部分写得比较复杂，但实验结果好像并没有体现出算法有很大优势，总体来说，可以稍微看看吧。推荐指数：★★★。

![](../../assets/images/b5d81ad0daad0904f4ac.jpg)

作者通过实验对比了两个典型meta-RL算法（MAML与PEARL）与一个非meta-RL算法（TD3-context）的性能，这里的TD3-context是在标准TD3算法的基础上增加了一个context变量作为对任务信息的推断，实验结果表明**meta-RL算法相较非meta-RL算法实际上并没有产生非常明显的优势**，这就不仅让我们产生疑问：meta-RL算法真的有用吗？

另外，现有的meta-RL算法大多十分复杂，且难以调参，为此，作者另辟蹊径，从**多任务目标（multi-task objective）**的角度设计了一种新的算法——meta-Q-learning（MQL），这种算法直接通过在大量任务上训练得到一个模型的初始参数，然后利用该初始参数去学习（adapt）新的任务，另外，与PEARL类似，MQL同样设计了一个基于**context**的隐层变量用于对新任务进行推断，但是相较于PEARL，MQL除了利用context推断新任务之外还会在adaptation阶段对策略参数进行更新，因而能够更好地适应新任务。MQL还有一个重要特点：不同于大多数meta-RL只适用于on-policy算法，MQL以 **value-based（off-policy）**类的RL算法为基础，样本效率高。本算法属于基于context的meta-RL算法。

## **背景知识**

将MDP表示为 $x_{t+1}=f^{k}\left(x_{t}, u_{t}, \xi_{t}\right) \quad x_{0} \sim p_{0}^{k}$，其中 $x_t$ 表示状态，$u_t$ 表示动作，$\xi_t$ 表示噪声，$f^k$ 表示环境动态。用 $\mu_{\theta}(x_t)$ 表示参数为 $\theta$ 的确定性策略，此时对应的奖励函数记为 $r_{t}^{k}:=r^{k}\left(x_{t}, u_{\theta}\left(x_{t}\right)\right)$。不同任务的 $f^k$ 与 $r^k$ 不同，因此我们可以将任务记为



$$
D^{k}(\theta)=\left\{x_{t}, u_{\theta}\left(x_{t}\right), r^{k}, x_{t+1}=f^{k}\left(x_{t}, u_{\theta}\left(x_{t}\right), \xi_{t}\right)\right\}_{t \geq 0, x(0) \sim p_{0}^{k}, \xi(\cdot)} \\
$$



给定一个特定的任务 $k\in\{1,...,n\}$，标准 RL 的目标是得到最优的策略参数 $\theta$，使得该任务下的期望收益最大，即：



$$
\widehat{\theta^{k}}=\underset{\theta}{\arg \max } \ell^{k}(\theta) \quad \text { where } \ell^{k}(\theta)=\underset{x \sim p_{0}}{\mathbb{E}}\left[q^{k}\left(x, u_{\theta}(x)\right)\right] \\
$$



其中 $q^{k}(x, u)=\underset{\xi_{(\cdot)}}{\mathbb{E}}\left[\sum_{t=0}^{\infty} \gamma^{t} r_{t}^{k} \mid x_{0}=x, u_{0}=u, u_{t}=u_{\theta}\left(x_{t}\right)\right]$。

不同于标准RL，Meta-RL的目标是通过在大量服从同一分布的任务上对策略模型 进行训练得到一个初始参数 $\hat{\theta}_{\text{meta}}$ ，该套参数在新的任务上能够通过少量的更新迅速获得较好的性能，具体而言，meta-RL在 $\mathcal{D}_{\text {meta }}=\left\{D^{k}\right\}_{k=1, \ldots, n}$ 的训练过程可以记为



$$
\widehat{\theta}_{\mathrm{meta}}=\arg \max _{\theta} \frac{1}{n} \sum_{k=1}^{n} \ell_{\mathrm{meta}}^{k}(\theta) \\
$$



其中 $\ell_{\mathrm{meta}}^{k}(\theta)$ 为meta-training的损失函数，其表达式取决于具体所使用的算法。

### **MAML**

**MAML**是一种gradient-based的meta-RL算法，其对应的 $\ell_{\mathrm{meta}}^{k}(\theta)$ 为



$$
\ell_{\text {meta }}^{k}(\theta)=\ell^{k}\left(\theta+\alpha \nabla_{\theta} \ell^{k}(\theta)\right) \\
$$



这里 $\alpha$ 为学习率，$\ell^k(\theta)$ 为inner problem中的标准RL目标函数。这里 $\ell_{\mathrm{meta}}^{k}(\theta)$ 表示策略参数 $\theta$ 在特定任务 $k$ 上进行一次或多次梯度更新之后能够获得的目标函数。MAML的背后思想是：meta-training所得到的初始策略参数 $\hat{\theta}_{\text{meta}}$ 不能在 $\mathcal{D}_{\text{meta}}$ 的所有任务上都取得较好的性能，但是当面对新的任务时，它只需要几次梯度更新就能得到快速地性能提升。

## **MQL**

MQL是off-policy的meta-RL算法，其算法分为两个阶段：meta-training与adaptation。与大多meta-RL算法不同，MQL中meta-training与adaptation的具体实施过程有较大的差异。

1. **Meta-Training**

在meta-training阶段，MQL的目标是通过off-policy的更新方式使得智能体的策略在各个任务上的平均收益最大，即



$$
\widehat{\theta}_{\text {meta }}=\arg \max _{\theta} \frac{1}{n} \sum_{k=1}^{n} \underset{\tau \sim D^{k}}{\mathbb{E}}\left[\ell^{k}(\theta)\right] \\
$$



由于off-policy算法通常都是基于value值对策略进行更新，因此这里的 $\ell^k(\theta)$ 用的是值函数更新中的temporal-difference，即 $\ell^{k}(\theta)=\mathrm{TD}^{2}(\theta ; \tau)$。

【**MQL与MAML**】对比一下MQL与MAML的meta-training，可以看出两者是完全不同的，MAML中meta-training的目标函数是**策略参数在各个任务下几次梯度更新之后所能获得的RL目标**，而MQL中meta-training的目标函数就是**策略参数本身在各个任务下所能获得的RL目标**，无需经历任何梯度更新，实际上MQL的meta-training目标函数就是一个多任务目标函数，相较而言更为简单。那么为什么MQL要采用这样的方式进行更新呢？让我们首先对MAML中损失函数的梯度进行泰勒展开：



$$
\nabla \ell_{\text {meta }}^{k}(\theta)=\nabla \ell^{k}(\theta)+2 \alpha(m-1)\left(\nabla^{2} \ell^{k}(\theta)\right) \nabla \ell^{k}(\theta)+\mathcal{O}\left(\alpha^{2}\right) \\
$$



可以看出，$\nabla \ell_{\text {meta }}^{k}(\theta)$ 同样也是下面式子的梯度：



$$
\ell^{k}(\theta)+\alpha(m-1)\left\|\nabla \ell^{k}(\theta)\right\|_{2}^{2} \\
$$



这个式子给MAML的原理提供了新的解读：MAML实际上under-fit了单个任务，当 $\left\|\nabla \ell^{k}(\theta)\right\|_{2}^{2}$ 较大时，结果就离 $\ell^{k}(\theta)$ 的局部最优越远。因此，作者得出结论：**MAML的adaptation速度是以单个任务的under-fitting为代价的**。

我个人不是很理解这个解释，如果要想迅速适应各种不同的任务，那在训练过程中必然要under-fit各个任务呀，不可能训练得到一个策略参数能够对每个任务都适用吧，所以这又有什么问题呢？

1. **Adaptation**

在adaptation阶段，MQL同样使用off-policy的方式对策略参数进行更新，但由于新任务在训练刚开始时经验回访池中的样本数量通常较少，不利于对新任务的快速适应，为了解决这个问题，作者提出在使用新任务样本对策略进行更新的同时，也可以部分使用meta-training的经验回访池样本对策略进行更新，由于meta-training中的经验样本中大多都不是新任务所产生的，作者提出引入一个 **propensity estimation（倾向估计）**来修正老数据的偏差。具体而言，adaptation阶段的更新表达式可以写为



$$
\underset{\theta}{\arg \max }\left\{\underset{\tau \sim D^{\text {new }}}{\mathbb{E}}\left[\ell^{\text {new }}(\theta)\right]+\underset{\tau \sim \mathcal{D}_{\text {meta }}}{\mathbb{E}}\left[\beta\left(\tau ; D^{\text {new }}, \mathcal{D}_{\text {meta }}\right) \ell^{\text {new }}(\theta)\right]-(1-\widehat{\mathrm{ESS}})\left\|\theta-\widehat{\theta}_{\text {meta }}\right\|_{2}^{2}\right\} \\
$$



其中 $\mathcal{D}_{\text{meta}}$ 为meta-training的经验回放池，$D^{new}$ 为adaptation过程中新任务的经验回放池，propensity得分 $\beta\left(\tau ; D^{\text {new }}, \mathcal{D}_{\text {meta }}\right) $ 表示 transition $\tau$ 来源于 $D^{new}$ 而非 $\mathcal{D}_{\text{meta}}$ 的概率，$\widehat{\mathrm{ESS}}$ 为**有效样本尺寸（effective sample size）**，用于度量新任务与meta-training任务之间的相似度。在adaptation的策略更新表达式中，第一项是新任务上的off-policy更新，第二项是老数据上的off-policy更新，权重因子为 $\beta$，第三项用于防止策略在adaptation过程中性能下降。

以上就是MQL的具体算法，接下来我们将对算法中所存在的具体细节进行一一分析。

### **引入context变量**

作者表示在meta-RL中，每个task的信息可以看做是POMDP中不能被观测到的状态，因此作者设计了一个context变量 $z_t$ 用于表示智能体所学习到的关于任务的推断，$z_t$ 取决于 $\{(x_i,u_i,r_i\}_{i\le t}$，具体而言，作者将 $z_t$ 设计为一个 GRU模型（Gated Recurrent Unit）的隐层状态。引入了 $z_t$ 之后，策略网络 $u_{\theta}(x)$ 与值函数网络 $q_{\phi}(x,u)$ 分别变为 $u_{\theta}(x,z)$ 与 $q_{\phi}(x,u,z)$。

【**MQL与PEARL**】回顾一下在PEARL中同样有一个context变量，但是PEARL中的context变量是随机变量，服从一定的分布，该分布具有置换不变性（permutation invariance），即产生context变量时无需考虑transitions的次序。在MQL中，作者认为任务的动态变化与时间的变化方向是有关系的有关，因此直接用GRU模型生成一个确定性的context变量，该变量取决于当前时刻之前的整个trajectory，很显然此时context变量不具备置换不变性。另外，这种直接用GRU生成context变量的方法比PEARL中的方法要简单很多。

这个地方我不明白的是，作者是将整个策略网络以及值函数网络构建成了GRU，还是构建了一个额外的GRU网络用于生成 $z_t$，感觉讲得不是很清楚。

### **Adaptation**

这一节将对MQL中的adaptation过程（用新任务的少量经验 $D^{\text{new}}$ 对策略参数 $\widehat{\theta}_{\text{meta}}$ 进行更新）进行重点分析。作者提出MQL中的adaptation实际上分为两个步骤：

1. **Vanilla off-policy adaptation**

Adaptation过程的第一步为使用新任务所产生的新数据样本对初始策略参数 $\widehat{\theta}_{\text{meta}}$ 进行更新：



$$
\arg \max \left\{\underset{\theta}{\underset{\tau}{\mathbb{E}}} \underset{D_{\text {new }}}{\mathbb{E}}\left[\ell^{\text {new }}(\theta)\right]-\frac{\lambda}{2}\left\|\theta-\widehat{\theta}_{\text {meta }}\right\|_{2}^{2}\right\} \\
$$



这里惩罚项 $\left\|\theta-\widehat{\theta}_{\text {meta }}\right\|_{2}^{2}$ 让更新后的策略参数尽量接近 $\widehat{\theta}_{\text{meta}}$，这对于减少模型的方差至关重要。

1. **Importance-ratio corrected off-policy updates**

Adaptation过程的第二步是利用meta-training的经验回放池进一步对参数进行更新。然而，meta-training过程中所产生的transitions并不全部来自于新任务，因此我们必须对一个系数对数据进行修正，具体更新表达式为



$$
\underset{\theta}{\arg \max }\left\{\underset{\tau \sim \mathcal{D}_{\text {meta }}}{\mathbb{E}}\left[\beta\left(\tau ; D^{\text {new }}, \mathcal{D}_{\text {meta }}\right) \ell^{\text {new }}(\theta)\right]-\frac{\lambda}{2}\left\|\theta-\widehat{\theta}_{\text {meta }}\right\|_{2}^{2}\right\} \\
$$



这里的 $\beta\left(\tau ; D^{\text {new }}, \mathcal{D}_{\text {meta }}\right) $ 为倾向得分（propensity score）/ 重要性比值（importance ratio），另外，这里我们同样使用惩罚项 $\left\|\theta-\widehat{\theta}_{\text {meta }}\right\|_{2}^{2}$ 来限制参数的更新幅度，$\lambda=1-\widehat{\text{ESS}}$。该步骤让我们能够充分利用大量过去产生的数据样本，作者提出在具体实验过程中，第二个步骤的更新频率是第二个步骤的100倍。

### **重要性比值的估计**

在adaptation的过程中，为了修正用老数据样本更新新任务策略所产生的偏差，我们用到了重要性比值（importance ratio）作为权重系数，那么重要性比值究竟有什么具体含义，以及如何计算呢？

考虑监督学习的场景，给定训练数据分布 $q(x)$ 以及测试数据分布 $p(x)$，我们希望估计模型的预测值 $\hat{y}(x)$ 如何根据这两个分布发生变化，这可以通过**重要性采样（importance sampling）**实现：



$$
\underset{x \sim p(x)}{\mathbb{E}} \underset{y \mid x}{\mathbb{E}}[\ell(y, \hat{y}(x))]=\underset{x \sim q(x)}{\mathbb{E}} \underset{y \mid x}{\mathbb{E}}[\beta(x) \ell(y, \hat{y}(x))] \\
$$



其中 $y|x$ 表示数据的真实标签，$\hat{y}(x)$ 为模型预测值，$\ell(y,\hat{y}(x))$ 表示每组数据 $(x,y)$ 的损失函数。这里的 $\beta(x)=\frac{dp}{dq}(x)$ 即为**重要性比值（importance ratio）/ 倾向得分（propensity score）**，它是两个数据分布的Radon-Nikodym derivative，度量了样本 $x$ 来源于 $p$ 而非 $q$ 的概率。

在实际场景中，我们并不知道 $q(x)$ 与 $p(x)$ 的具体分布，只能够利用分别从 $q$ 以及 $p$ 采样得到的有限数据样本 $X_q=\{x_1,...,x_m\}$ ，$X_p=\{x'_1,...,x'_m\}$ 中对 $\beta(x)$ 进行估计。根据文献 [2]，这能够通过**逻辑回归（logistic regression）**的方法实现。首先将 $X_q$ 中所有样本的标签记为 $z_k=1$，$X_p$ 中所有样本的标签记为 $z_k=-1$，然后利用这 $2m$ 个样本求解下列问题：



$$
w^{*}=\min _{w} \frac{1}{2 m} \sum_{(x, z)} \log \left(1+e^{-z w^{\top} x}\right)+c\|w\|^{2} \\
$$



此时，$\beta(x)$ 的值就可以估计为



$$
\beta(x)=\frac{\mathbb{P}(z=-1 \mid x)}{\mathbb{P}(z=1 \mid x)}=e^{-w^{* \top} x} . \\
$$



因此，在上述adaptation的过程中，作者提出用meta-training经验回访池取出的样本以及新任务产生的数据样本来拟合这样一个分类器计算 $\beta(x)$（大约200个新数据，200~400个老数据），另外，作者提出context变量 $z_t$ 是该分类器的feature（没明白是什么意思）。

在具体实现过程中，作者提出 $c$ 的值要取得稍微大一些，以免出现过多错误的正值。另外，作者还提出需要对 $\beta$ 进行一个clip操作，防止过大的 $\beta$ 带来过大的方差。

### **$\widehat{\text{ESS}}$ 的估计**

在adaptation的过程中，还有一个惩罚项 $\left\|\theta-\widehat{\theta}_{\text {meta }}\right\|_{2}^{2}$ 用于限制策略更新的幅度，这一项的权重系数为 $\lambda=1-\widehat{\text{ESS}}$。这里的 $\widehat{\text{ESS}}$ 为**归一化有效样本尺寸（Normalized Effective Sample Size）** ，其含义为从目标分布 $p(x)$ 直接采样得到一个与用重要性采样方式性能（方差）相同的估计器所需要的相对数量。$\widehat{\text{ESS}}$ 的值可以通过蒙特卡洛的方法计算得到：



$$
\widehat{\mathrm{ESS}}=\frac{1}{m} \frac{\left(\sum_{k=1}^{m} \beta\left(x_{k}\right)\right)^{2}}{\sum_{k=1}^{m} \beta\left(x_{k}\right)^{2}} \in[0,1] \\
$$



$\widehat{\text{ESS}}$ 度量了两个分布之间的相似程度，当 $p$ 与 $q$ 之间很相近时，$\widehat{\text{ESS}}$ 就趋近于1，相反当 $p$ 与 $q$ 差别很大时，$\widehat{\text{ESS}}$ 就趋近于0。

将惩罚项的权重系数设定为 $\lambda=1-\widehat{\text{ESS}}$ 意味着当新任务与元训练的任务很相近时，权重系数就小，此时就可以放宽惩罚项，相反就需要加紧惩罚项。

## **实验结果**

![](../../assets/images/789165b1989102404564.jpg)

- 普通的带有context变量的off-policy算法就能达到与PEARL算法差不多的性能；

![](../../assets/images/f2c2ca2776f26389abd1.jpg)

- MQL中的adaptation过程能够进一步提升算法性能（但感觉好像提升并不大）；
- 引入meta-training的经验回访池进行训练也能提升算法性能（感觉提升好像也不大）；

## **个人思考与理解**

这篇论文乍一看感觉作者提出了一个很不错的创新点，但是结合实验结果仔细想想就会发现，作者不过是把meta-RL的问题当做成一个POMDP问题，即把任务信息当做了观测不到的状态，用RNN类的模型（GRU）去得到一个context变量用于从历史经验中学习任务信息，因此本质上该算法还是context类的算法，至于作者所提出的adaptation过程，看似很复杂，其实从实验结果来看，对整个算法性能的提升并不明显，因此实际上MQL就是一个使用了off-policy训练方法的context类meta-RL算法。

## **参考文献**

[1] R. Fakoor, P. Chaudhari, S. Soatto, and A. J. Smola, ‘Meta-Q-Learning’, *arXiv:1910.00125 [cs, stat]*, Apr. 2020, Accessed: Apr. 26, 2022. [Online]. Available: <http://arxiv.org/abs/1910.00125>

[2] Deepak Agarwal, Lihong Li, and Alexander Smola. Linear-time estimators for propensity scores. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pp. 93–100, 2011.
