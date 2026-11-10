# 【元强化学习】PEARL：一种off-policy的meta-RL算法

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2022-05-06  
来源：[https://zhuanlan.zhihu.com/p/509225503](https://zhuanlan.zhihu.com/p/509225503)

[知识库首页](../../README.md)

---

今天介绍论文《Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables》。

> 这篇论文提出了一种适用于off-policy训练的meta-RL算法，想法还挺新颖的，但是算法有些复杂，里边有些具体细节也比较难理解，总体来说我觉得是一篇比较不错的论文。推荐指数：★★★★

本文考虑的是meta-RL算法中的**采样效率（sample efficiency）**问题，这里的采样效率既包括元训练的采样效率（**meta-training efficiency**），也包括适应新任务的采样效率（**adaptation efficiency**）。作者指出在此之前的meta-RL算法均依赖于**on-policy**策略，采样效率很低，而在元学习的训练中，需要从大量不同任务中收集经验对模型进行训练，这进一步加剧了样本效率低下的问题。为了解决这一问题，作者提出了**PEARL（probabilistic embeddings for actor critic）**算法，该算法借鉴了[MAESN](MAESN：一种让智能体学习如何高效探索（explanation）的meta-RL算法--508845258.md)中用隐层特征对任务进行推断这一思想，并通过**将任务推断（task inference）以及策略更新这两个过程分离开来，使得用off-policy算法更新策略成为可能**。

作者把meta-RL分为两大类，一类是基于context（我理解为任务/环境）的，另一类是基于gradient。PEARL属于基于context的meta-RL算法，[RL^2](RL^2：meta-RL算法的开山之作--508387238.md)同样属于基于context的算法，不同的是， RL$^2$ 将过去的经验整合在RNN的内部特征向量中，利用RNN中的内部特征向量构建对任务环境的认知，而PEARL则是用一个额外的神经网络训练得到一个可用于描述context的隐层变量中，也就是说，在PEARL中，智能体需要学习如何将对任务的推断整合在一个隐层变量中，同时让策略根据该隐层变量迅速地适应新任务。实验表明RL$^2$ 很难与off-policy算法结合。[MAML](经典meta-RL算法之MAML--508512375.md)是典型的基于gradient的算法，这一类算法需要用on-policy数据进行训练，样本效率比较低。相较而言，本文提出的PEARL不但具有更好的采样效率，而且在实验中也表现出了更好的性能。

另外，作者还提出，meta-RL所解决的问题实际上可以看做是POMDP（partially observable Markov Decision Process）的一种特殊情况，其中未知的任务信息就是状态中不能被观测到的部分。

## **关于为什么MAML只适用于on-policy算法**

![](../../assets/images/b01f6cfafe158957be09.jpg)

这一段是作者对于为什么MAML类算法必须采用on-policy方式训练的解释，按照作者的意思，原因有两个：

1. Meta-training和meta-testing部分的数据分布（我的理解是数据类型）必须相同，由于meta-RL中必须用on-policy数据来适应新任务，因此meta-training部分也必须用on-policy算法。（对于这个解释，我是不理解的，为什么必须用on-policy适应新任务，谁能给我解释一下？）
2. Meta-RL要求策略能够推断出任务的分布，然后优化其探索策略。然而off-policy类的算法本质上都是基于value的算法，目标是减小temporal-difference error，学习到准确的值函数，也就是说它们并不能直接优化智能体的trajectory。相反，on-policy算法是基于policy gradient的算法，它们直接控制策略中动作的分布，从而能够根据当前数据所反映出来的任务分布对策略进行调整，符合meta-RL的要求。

我对第二个理由的理解是，meta-RL中必须有一个对新任务进行推断的过程，很明显，新任务的信息会暗含在根据其MDP所产生的trajectory中，on-policy算法中使用大量完整的rajectories来对策略进行优化，这样便于智能体从这些trajectories中对任务进行推断，而off-policy算法是将每个时刻的transition作为独立的样本储存在经验回放池中，训练的时候从中随机采样相互独立的transitions优化值函数，然后间接优化策略。我猜测之所以不能用off-policy算法，就是因为智能体很难从大量相互独立的transitions中推断出一个任务MDP的信息，如果非要用的话也需要数量非常庞大的样本，显然不切实际。而且off-policy的本质是优化值函数，而非根据当前trajectory优化策略，回放池中的样本也不能准确反映当前策略特征，这和meta-RL的目标显然是不太一样的。

我其实不是很理解第一个理由，为什么meta-RL必须用on-policy数据适应新任务，还是因为必须进行任务推断吗？如果大家有什么见解的话，可以在评论区讨论讨论~

## **问题描述**

在meta-RL中，考虑服从一定分布 $p(\mathcal{T})$ 的一系列任务，每一个任务都是一个完整的MDP。为了区分不同任务，我们可以将每个任务表示成 $\mathcal{T}=\left\{p\left(\mathbf{s}_{0}\right), p\left(\mathbf{s}_{t+1} \mid \mathbf{s}_{t}, \mathbf{a}_{t}\right), r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)\right\}$，其中 $p\left(\mathbf{s}_{0}\right)$ 为初始状态分布，$p\left(\mathbf{s}_{t+1} \mid \mathbf{s}_{t}, \mathbf{a}_{t}\right)$ 表示状态转移分布，$r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)$ 表示奖励函数。给一个用于训练的任务集合，元训练的过程就是学习得到一个策略，这个策略能够通过对历史transition经验的推理迅速适应新的任务。我们将这种历史信息称为**contex**，$\mathbf{c}$。具体来说，$\mathbf{c}_{n}^{\mathcal{T}}=\left(\mathbf{s}_{n}, \mathbf{a}_{n}, r_{n}, \mathbf{s}_{n}^{\prime}\right)$ 表示任务 $\mathcal{T}$ 中的一个transition，$\mathbf{c}^{\mathcal{T}}_{1:N}$ 则包含了至今为止所有的经验。在测试阶段，智能体的策略必须能够迅速适应从 $p(\mathcal{T})$ 所采样得到的新任务。

（作者用so far表示至今为止，我不太明白这是什么意思，意思是任意时刻的所有之前历史经验吗？不用考虑每个trajectory单独算吗？）

## **PEARL**

为了能够高效地迅速适应不同的任务，智能体必须能够根据历史经验对当前出现的任务做出准确地推断，在PEARL中，作者提出将任务推断这一过程整合在一个隐层context变量 $Z$ 的分布中，然后使用所学习到的隐层变量分布辅助智能体在不同任务中进行决策，此时智能体的策略可以写为 $\pi_{\theta}(\mathbf{a}|\mathbf{s},\mathbf{z})$。因此，PEARL元学习过程中需要学习的内容可以总结为两个部分：

1. 通过元训练得到一个 $\mathbf{z}$ 后验分布参数，在adaption过程中，智能体能够根据新任务中最近的历史经验对当前任务信息进行推断，从而更新得到 $\mathbf{z}$ 的后验分布；
2. 通过元训练得到一个最优策略，该策略能够根据从$Z$ 后验分布中采样得到的隐层变量迅速适应新的任务。

### **对context变量进行建模与学习**

在问题描述中我们已经提到本文用 $\mathbf{c}^{\mathcal{T}}_{1:N}$ 表示至今为止的所有历史经验，为了方便描述，下面都用 $\mathbf{c}$ 替代 $\mathbf{c}^{\mathcal{T}}_{1:N}$。作者提出用**变分推理（variational inference ）**[2] 的方法来学习推断 $Z$，即训练一个**推理网络（inference network）**$q_{\phi}(\mathbf{z}|\mathbf{c})$ （encoder）用于估计后验分布 $p(\mathbf{z}|\mathbf{c})$，$\phi$ 为模型参数。如果用常规生成模型的方法训练，就需要学习环境模型的动态变化让 $q_{\phi}(\mathbf{z}|\mathbf{c})$ 能够尽可能地重构MDP，但本文采用的是一种model-free的方式，通过估计值函数或者最大化收益的方式来优化 $q_{\phi}(\mathbf{z}|\mathbf{c})$（后面作者又提到估计值函数的效果最好）。假设目标函数是对数似然，那么对应的变分下限为



$$
\mathbb{E}_{\mathcal{T}}\left[\mathbb{E}_{\mathbf{z} \sim q_{\phi}\left(\mathbf{z} \mid \mathbf{c}^{\mathcal{T}}\right)}\left[R(\mathcal{T}, \mathbf{z})+\beta D_{\mathrm{KL}}\left(q_{\phi}\left(\mathbf{z} \mid \mathbf{c}^{\mathcal{T}}\right) \| p(\mathbf{z})\right)\right]\right] \\
$$



其中 $p(\mathbf{z})$ 表示 $Z$ 的先验单位高斯分布，$R(\mathcal{T},\mathbf{z})$ 可以是各种各样的目标函数，例如期望收益或者时间差分等。这里的KL散度项可以解释为信息瓶颈（information bottleneck）的变分近似，用于约束 $Z$ 与 $\mathbf{c}$ 之间的互信息，让 $\mathbf{z}$ 尽可能地只包含适应任务所必需的context信息。在meta-training阶段，我们将对 $q_{\phi}$ 的参数进行优化，而在meta-testing阶段，我们则可以通过所收集到的新任务的历史经验直接对 $\mathbf{z}$ 进行推断。

（这一段关于互信息的部分有些不太理解，有时间去看一下VAE的原文吧。）

![](../../assets/images/4c836766fb012a11ce25.jpg)

关于如何构建 $q_{\phi}(\mathbf{z}|\mathbf{c})$ 的网络模型，作者提出，MDP的编码需要具备**置换不变性（permutation invariant）**的特点：即如果我们希望推断当前任务是什么，识别其MDP，或者训练一个合适的值函数，只需要收集一系列transitions $\left\{\mathbf{s}_{i}, \mathbf{a}_{i}, \mathbf{s}_{i}^{\prime}, r_{i}\right\}$ 即可，无需考虑这些transitions被观察的次序。 为此，作者将 $q_{\phi}(\mathbf{z}|\mathbf{c}_{1:N})$ 建模为：



$$
q_{\phi}\left(\mathbf{z} \mid \mathbf{c}_{1: N}\right) \propto \Pi_{n=1}^{N} \Psi_{\phi}\left(\mathbf{z} \mid \mathbf{c}_{n}\right) \\
$$



其中 $\Psi_{\phi}\left(\mathbf{z} \mid \mathbf{c}_{n}\right)=\mathcal{N}\left(f_{\phi}^{\mu}\left(\mathbf{c}_{n}\right), f_{\phi}^{\sigma}\left(\mathbf{c}_{n}\right)\right)$， $f_{\phi}$ 是参数为 $\phi$ 的神经网络，用于估计均值 $\mu$ 和方差 $\sigma$。

### **算法的具体实施**

![](../../assets/images/980dd80542d2522126d6.jpg)

PEARL的特点之一是可以与off-policy的算法结合，原因是用于训练encoder的数据与用于训练策略网络的数据可以不同。我们可以将context得到的隐层变量 $\mathbf{z}$ 当做是off-policy RL中状态的一部分。Actor与critic总是用完整经验回访池 $\mathcal{B}$ 中采样得到的off-policy数据训练，定义采样器 $\mathcal{S}_c$ 用于采样训练encoder需要的context数据，如果让 $\mathcal{S}_c$ 从完整的回放池中采样，会造成训练数据与真实on-policy数据之间严重的分布不匹配。但实际上，context数据不需要严格地一定是on-policy的，作者表示可以从经验回访池中最近所收集的数据中采样训练数据，这样可以在维持on-policy性能的同时得到更好的采样效率。

PEARL做使用的基础RL算法为SAC（soft actor-critic）算法。根据上面的描述，我们需要联合优化 interference网络 $q_{\phi}(\mathbf{z}|\mathbf{c})$，策略网络 $\pi_{\theta}(\mathbf{a}|\mathbf{s},\mathbf{z})$ 以及 $Q_{\theta}(\mathbf{s},\mathbf{a},\mathbf{z})$。

[XuanAxuan：【深度强化学习】最大熵 RL：从Soft Q-Learning到SAC](../../03-决策与学习机制/01-策略优化与连续控制/最大熵 RL：从Soft Q-Learning到SAC--444441890.md)

Inference网络使用来自critic贝尔曼方程的梯度进行训练，因为作者通过实验发现通过估计值函数来训练encoder比通过最大化actor收益以及恢复MDP的效果更好。具体而言，Critic的损失函数为



$$
{\mathcal{L}}_{ critic }={\mathbb{E}_{\left(\mathbf{s}, \mathbf{a}, r, \mathbf{s}^{\prime}\right) \sim \mathcal{B}，\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{c})}\left[Q_{\theta}(\mathbf{s}, \mathbf{a}, \mathbf{z})-\left(r+\bar{V}\left(\mathbf{s}^{\prime}, \overline{\mathbf{z}}\right)\right)\right]^{2}} \\
$$



其中 $\bar{V}$ 为target网络，$\bar{\mathbf{z}}$ 表明此时没有计算其梯度。Actor的损失函数几乎与原SAC的相同，只是增加了 $\mathbf{z}$ 作为输入：



$$
\mathcal{L}_{ {actor }}=\underset{\substack{\mathbf{s} \sim \mathcal{B}, \mathbf{a} \sim \pi_{\theta} \\ \mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{c})}}{\mathbb{Z}_{\theta}}\left[D_{\mathrm{KL}}\left(\pi_{\theta}(\mathbf{a} \mid \mathbf{s}, \overline{\mathbf{z}}) \| \frac{\exp \left(Q_{\theta}(\mathbf{s}, \mathbf{a}, \overline{\mathbf{z}})\right)}{\mathcal{Z}_{\theta}(\mathbf{s})}\right)\right] \\
$$



作者提醒用于推断 $q_{\phi}(\mathbf{z}|\mathbf{c})$ 的context以及用于构成critic loss的context不同，我的理解是，在adaptation过程中，context是on-policy的历史数据，但是在训练过程中，context是从经验回放池中采样得到的。具体而言，$\mathcal{S}_c$ 会从最近所收集到的数据中均匀采样，每1000个meta-training步重新采样一次，而actor和critic的数据则是从完整的经验回放池中均匀采样。

总结来说，在PEARL的meta-training阶段，目标是用大量的任务集合训练得到 $q_{\phi}(\mathbf{z}|\mathbf{c})$ 的参数，与此同时还需要训练最优的策略模型，该策略模型具有利用历史经验迅速推断并适应新任务的能力。在meta-test阶段，首先根据事先给定的先验分布（单位高斯）采样得到一个 $\mathbf{z}$，然后根据 $\mathbf{z}$ 执行策略直至一个episode结束，这样就能够产生时间上扩展的探索行为。紧接着我们就可以利用所收集到的trajectory作为经验来更新 $Z$ 的后验分布，然后用后验分布探索。通过不断地探索、收集经验、更新后验分布，最终我们就能获得对新任务越来越准确地推断，从而得到最优的策略。完整的meta-training和meta-testing过程如下：

![](../../assets/images/ebbc654afcbf2e74eb77.jpg)![](../../assets/images/37d54ffc1cbc473765c4.jpg)

## **实验结果**

![](../../assets/images/3afa3eef9ac19cf70dcc.jpg)

- 结果表明PEARL的性能确实优于其他meta-RL算法，不管是sample efficiency还是最后的收敛结果。

![](../../assets/images/d7683e963b4eef1f5414.jpg)

- PEARL的性能比MAESN的好

## **PEARL 与 MAESN**

[XuanAxuan：【元强化学习】MAESN：一种让智能体学习如何高效探索（explanation）的meta-RL算法](MAESN：一种让智能体学习如何高效探索（explanation）的meta-RL算法--508845258.md)

PEARL与MAESN同样都利用隐层变量对任务进行推断，在MAESN中，$\mathbf{z}\sim\mathcal{N}(\mu_i,\sigma_i)$ ，首先通过元训练得到每个任务的初始隐层变量分布，该分布接近单位高斯分布，也就是先验分布，在adaptation过程中，首先用先验分布采样，然后利用梯度上升的方法对 $\mathbf{z}$ 分布的均值和方差进行更新；在PEARL中，$\mathbf{z}\sim q_{\phi}(\mathbf{z}|\mathbf{c})$，也就是说构建了一个神经网络用于从历史经验context中推断 $\mathbf{z}$，此时我们需要学的是这个interference network的参数 $\phi$，另外PEARL中同样将单位高斯作为先验分布，在adaptation过程中，首先用先验分布采样，然后利用收集到的context信息更新 $\mathbf{z}$ 的后验分布，此时我们不需要利用梯度更新，只需要将收集到的context信息输入到interference network里就可以了。可以看出，PEARL和MAESN虽然都利用了隐层变量作为对任务的推断，但是MAESN是用基于gradient直接对 $\mathbf{z}$ 的分布进行学习，而PEARL则是建立了context与 $\mathbf{z}$ 之间的关系，因而在adaptation过程中，不需要再更新任何参数了，两个方法是截然不同的。

## **个人总结与思考**

这篇论文看得我好费劲，一方面我觉得想法很棒，另一方面又有很多看不懂的地方，看了两三遍稍微有了一些自己的理解，但可能还是会有一些错漏之处，有感兴趣的同学欢迎和我交流交流。

![](../../assets/images/c99754b96c81f82bceae.jpg)

关于整篇文章的思想，我认为最精华的部分就是用先验分布和后验分布来表示历史经验和新任务的经验，具体的过程就是，在每一个任务开端，我们对当前任务一无所知，只能加强探索，尝试不同的策略，所以首先用先验分布生成隐层变量作为对任务的初始推断，积累了一定的trajectories作为经验之后，就可以根据这些经验更新隐层变量的分布作为后验分布，也就是更新我们对于新任务的认识，随着积累的经验越来越多，后验分布越来越能够反映任务的信息，此时我们也就能够做出更加好的决断了。对于这个做法，作者在她写的博客中给出了非常生动地解释，当我们第一次看到火龙果的时候，我们联想到了曾经吃芒果的经历，于是拿了一把刀子去切火龙果，这就是一个非常好的利用历史经验作为先验信息进行探索的过程，随后当我们切开火龙果之后发现它更像猕猴桃而非芒果，于是这个时候我们根据对猕猴桃的经验去拿了一把勺子继续吃火龙果，这就是根据所收集到的经验更新对新任务认知的一个过程。另外，相较于MAESN利用gradient去学隐层变量，PEARL是直接利用智能体所生成的transition来构建隐层变量，我个人认为这样能够更好地建立起任务信息与隐层变量之间的关系，更具有解释性。

另外一个我决定好好研究这篇论文的原因就是，我个人非常喜欢SAC这个算法，而这篇论文就是用的这个算法，而且最后的实验结果也很好，但是另一方面，从我对这篇论文的理解来看，里面涉及到许多实现细节，这一点可能需要研读作者给出的源代码才能慢慢搞懂。

## **参考文献**

[1] K. Rakelly, A. Zhou, D. Quillen, C. Finn, and S. Levine, ‘Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables’, *arXiv:1903.08254 [cs, stat]*, Mar. 2019, Accessed: Apr. 25, 2022. [Online]. Available: <http://arxiv.org/abs/1903.08254>

[1] D. P. Kingma and M. Welling, ‘Auto-Encoding Variational Bayes’, *arXiv:1312.6114 [cs, stat]*, May 2014, Accessed: Jul. 09, 2020. [Online]. Available: <http://arxiv.org/abs/1312.6114>
