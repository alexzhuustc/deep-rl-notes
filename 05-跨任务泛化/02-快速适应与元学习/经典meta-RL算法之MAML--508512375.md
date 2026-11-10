# 【元强化学习】经典meta-RL算法之MAML

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2022-05-03  
来源：[https://zhuanlan.zhihu.com/p/508512375](https://zhuanlan.zhihu.com/p/508512375)

[知识库首页](../../README.md)

---

今天介绍《Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks》。

> MAML是当前用得最多的元学习算法之一，它适用于小样本学习（few-shot learning）、元强化学习（meta-RL）等问题，近年来也出现了许多MAML算法的各类花式变体，可以说，了解元强化学习就不得不了解MAML。推荐指数：★★★★★

之前我们已经介绍过，元学习的目标是让模型能够从少量的样本中快速学习到新的任务，为了达到这个目标，需要用大量不同的任务来训练模型，使其获得快速学习的能力。MAML的背后思想十分简单：训练得到一套模型的**初始参数（initial parameters）**，使得该模型仅通过少量样本上的一次或几次梯度更新就能够最大化新任务的性能。从特征学习的观点来看，这一过程可以看做是构建一个适用于许多任务的**内部表征（internal representation）**，此时只需要简单地微调参数（例如仅修改最上层的神经网络系数）就能够在某一特定任务上获得很好的结果。而从动态系统观点来看，这一过程又可以看做是最大化新任务损失函数对于模型参数的敏感度：当敏感度比较高时，模型参数很细微的改变就能造成任务损失函数很大的进步。

MAML就是这样一种算法，它能够训练得到一套模型参数，只需对这套模型参数进行少量的梯度更新就能让其迅速地迁移至新的任务。在强化学习中使用MAML算法，能够在多任务场景下加速智能体学习新任务的速度。

目前的元强化学习算法主要分为两大类，第一类是基于recurrence的算法，即利用RNN结构获取经验，例如[RL^2](RL^2：meta-RL算法的开山之作--508387238.md)；另一类是基于gradient的算法，MAML属于后者。

## **算法介绍**

Meta-RL的主要目标就是**实现快速适应（adaptation）**，这通常可以被构建成**小样本学习（few-shot learning）问题**。由于MAML既可以用于监督学习也可以用于强化学习，因此下面的介绍均使用通用的表示形式。

### **元学习问题建模**

小样本元学习的目标是训练模型使其能够在数据样本和迭代次数均很少的情况下快速适应新的任务。为了实现这个目标，我们需要在一个元学习阶段，用一系列任务对模型进行训练，使得训练好的模型可以仅使用很少的示例就能迅速适应新的任务。因此，元学习问题**将完整的任务当做是训练样本**。

考虑一个模型 $f$ 用于将观测 $\mathrm{x}$ 映射为输出 $\mathrm{a}$，元学习中的模型需要适应大量不同的任务，这里的每个任务 $\mathcal{T}=\left\{\mathcal{L}\left(\mathrm{x}_{1}, \mathbf{a}_{1}, \ldots, \mathrm{x}_{H}, \mathbf{a}_{H}\right), q\left(\mathrm{x}_{1}\right), q\left(\mathrm{x}_{t+1} \mid \mathrm{x}_{t}, \mathbf{a}_{t}\right), H\right\}$ 包含一个损失函数 $\mathcal{L}$，一个初始观测的分布 $q(\mathrm{x}_1)$，一个转移分布 $q(\mathrm{x}_{t+1}|\mathrm{x}_{t},\mathrm{a}_t)$，以及episode的长度 $H$。对于独立同分布的监督学习问题，$H=1$。模型可以通过在每个时刻 $t$ 选择 $a_t$ 产生长度为 $H$ 的样本。$\mathcal{L}\left(\mathrm{x}_{1}, \mathbf{a}_{1}, \ldots, \mathrm{x}_{H}, \mathbf{a}_{H}\right)\rightarrow\mathbb{R}$ 提供了特定任务的反馈，它可以是分类问题中的交叉熵损失函数，也可以是MDP中的代价函数（奖励函数）。

在元强化学习中，假设模型需要适应的不同任务均服从同一分布 $p(\mathcal{T})$。考虑一个 $K$ 样本学习场景，在元训练过程中，首先从 $p(\mathcal{T})$ 中采样得到一个任务 $\mathcal{T}_i$，然后用 $K$ 个样本以及这些样本的损失函数 $\mathcal{L}_{\mathcal{T}_i}$ 对模型进行训练，并将训练之后的模型用 $\mathcal{T}_i$ 产生的新样本进行测试。最终，**采样任务** $\mathcal{T}_i$ **上的测试误差将作为元学习过程的训练损失函数**。

总结来说，这一过程实际上也包含一个 **inner loop** 与一个 **outer loop**。在inner loop中，采样新的任务，对模型训练，用更新后的参数对模型测试；在outer loop中，将各采样任务在测试集上的误差作为损失函数对模型进行训练，注意这里的模型是inner loop中更新前的模型，每个inner loop中对模型的更新只会用于产生该任务的测试误差，最终元学习所更新的模型是初始模型。

### **算法模型**

![](../../assets/images/97e14f439cd612d11ae6.jpg)

将模型用参数为 $\theta$ 的参数化函数 $f_{\theta}$ 表示，当转移到新的任务 $\mathcal{T}_i$ 时，模型的参数 $\theta$ 通过一次或几次梯度上升更新为 $\theta'_i$，当更新次数为1时，表达式为



$$
\theta_{i}^{\prime}=\theta-\alpha \nabla_{\theta} \mathcal{L}_{\mathcal{T}_{i}}\left(f_{\theta}\right) \\
$$



更新步长 $\alpha$ 可以是固定的超参数，也可以通过元学习获得。

模型的参数 $\theta$ 以最大化不同任务的性能 $f_{\theta'_i}$ 为目标进行更新，该目标可以写为



$$
\min _{\theta} \sum_{\mathcal{T}_{i} \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_{i}}\left(f_{\theta_{i}^{\prime}}\right)=\sum_{\mathcal{T}_{i} \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_{i}}\left(f_{\theta-\alpha} \nabla_{\theta} \mathcal{L}_{\mathcal{T}_{i}}\left(f_{\theta}\right)\right) \\
$$



可以看书，元学习在模型参数 $\theta$ 上进行，而目标函数则使用更新后的模型参数 $\theta'$ 计算。实际上，MAML的目标就是优化模型参数，使得新任务上的少量梯度更新就能产生最有效的行为。

不同任务上的元学习过程可以通过随机梯度上升（stochastic gradient descent，SGD）进行，因此，$\theta$ 的更新准则为



$$
\theta \leftarrow \theta-\beta \nabla_{\theta} \sum_{\mathcal{T}_{i} \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_{i}}\left(f_{\theta_{i}^{\prime}}\right) \\
$$



$\beta$ 为元更新步长。

MAML的算法流程如下：

![](../../assets/images/ea5d1ca4323410685ff0.jpg)

### **FO-MAML（First-Order MAML）**

MAML原论文没有展开对 $\theta$ 更新公式的具体推导，为了方便理解，我自己稍微把公式（8）推导了一下。首先根据矩阵求导的链式法则得到以下公式：



$$
\begin{aligned} \theta^{[m]}&=\theta^{[m]}-\sum_{\mathcal{T}_{i} \sim p(\mathcal{T})}{\frac{\partial \mathcal{L}_{\mathcal{T}_{i}}\left(f_{\theta_{i}^{\prime}}\right)}{\partial \theta^{[m]}}}\\ &=\theta^{[m]}-\sum_{\mathcal{T}_{i} \sim p(\mathcal{T})}\sum_{k}{\frac{\partial \mathcal{L}_{\mathcal{T}_{i}}\left(f_{\theta_{i}^{\prime }}\right)}{\partial \theta'^{[k]}_i}}\frac{\partial \theta_i'^{[k]}}{\partial \theta^{[m]}} \end{aligned} \\
$$



这里，$\theta^{[m]}$ 指的是参数 $\theta$ 中的第 $m$ 个元素，同理，$\theta'^{[k]}_j$ 是 $\theta'_j$ 的第 $k$ 个元素。根据（1），$\theta'^{[k]}_j$ 的计算公式为：



$$
\theta'^{[k]}_i=\theta^{[k]}-\alpha \frac{\partial \mathcal{L}_{\mathcal{T}_{i}}\left(f_{\theta}\right)}{\partial \theta^{[k]}} \\
$$



将（8）代入（7）的最后一项可以得到：



$$
\frac{\partial \theta_i'^{[k]}}{\partial \theta^{[m]}}=\frac{\partial \theta^{[k]}}{\partial \theta^{[m]}}-\alpha \frac{\partial^2 \mathcal{L}_{\mathcal{T}_{i}}\left(f_{\theta}\right)}{\partial \theta^{[k]}\theta^{[m]}}= \begin{cases} -\alpha \frac{\partial^2 \mathcal{L}_{\mathcal{T}_{i}}\left(f_{\theta}\right)}{\partial \theta^{[k]}\theta^{[m]}} & \text{if } k\ne m,\\ 1-\alpha \frac{\partial^2 \mathcal{L}_{\mathcal{T}_{i}}\left(f_{\theta}\right)}{\partial \theta^{[k]}\theta^{[m]}} & \text{if } k=m. \end{cases} \\
$$



可以看出，为了更新 $\theta$，我们需要计算Hessian-vector product，即包含二次求导，这会造成很高的计算复杂度。为了解决该问题，作者提出可以采用一阶近似（first-order approximation）的方法，忽略（9）中的二阶求导项，即



$$
\frac{\partial \theta_i'^{[k]}}{\partial \theta^{[m]}}\approx \begin{cases} 0 &\text{if } k\ne m,\\ 1 &\text{if } k=m. \end{cases} \\
$$



此时（6）也就简化成了



$$
\theta \leftarrow \theta-\beta \nabla_{\theta’_i} \sum_{\mathcal{T}_{i} \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_{i}}\left(f_{\theta_{i}^{\prime}}\right) \\
$$



![](../../assets/images/519e860f0de5e2b9460e.jpg)

作者在分类问题的实验中发现，FO-MAML可以获得与MAML几乎一样的性能，这说明MAML中性能的提升主要来源于损失函数对于后更新参数 $\theta'_i$ 的梯度，而非二阶梯度。

## **MAML-DRL**

MAML既可以用于监督学习，也可以用于强化学习。在强化学习中，元学习的目标是让智能体仅通过测试阶段少量的经验样本就能够快速获得新任务的策略，此时一个任务就是一个MDP，$p(\mathcal{T})$ 可以造成MDP中任意要素的改变。此时 $f_\theta$ 表示的即为智能体的策略（状态 $x_t$ 到动作 $a_t$ 之间的映射），而每个任务 $\mathcal{T}_i$ 的损失函数为



$$
\mathcal{L}_{\mathcal{T}_{i}}\left(f_{\phi}\right)=-\mathbb{E}_{\mathbf{x}_{t}, \mathbf{a}_{t} \sim f_{\phi}, q \mathcal{T}_{i}}\left[\sum_{t=1}^{H} R_{i}\left(\mathbf{x}_{t}, \mathbf{a}_{t}\right)\right] \\
$$



在 $K$ 样本强化学习中，根据 $f_\theta$ 与任务 $\mathcal{T}_i$ 产生的 $K$ 个rollouts $\left(\mathrm{x}_{1}, \mathrm{a}_{1}, \ldots \mathrm{x}_{H}\right)$ 以及它们所对应的奖励 $R(\mathrm{x}_t,\mathrm{a}_t)$ 将用于适应新的任务 $\mathcal{T}_i$。

由于在强化学习中，期望收益是不可导的（环境动态未知），因此算法中的模型梯度更新部分以及元优化部分均采用策略梯度方法估计梯度。由于策略梯度是 **on-policy** 算法，在 $f_{\theta}$ 适应新任务的过程中，每一次更新都需要当前策略 $f_{\theta'_i}$ 所产生的新样本。算法流程如下：

![](../../assets/images/eb92dc50230cbfbcee0c.jpg)

## **实验结果**

![](../../assets/images/88bbbf6a43c946eb36c2.jpg)![](../../assets/images/095cea131dba5fa2f7b0.jpg)

在MAML-DRL的实验中，智能体的任务是找到不同位置的目标，除了MAML外，作者还给出了三种对比算法：（1）pretrained：用所有任务对策略进行预训练，然后在测试阶段微调；（2）random：用随机初始化的参数训练策略；（3）oracle：将任务参数（目标位置、方向、速度）当做已知信息添加到状态中。另外，在具体实验中，inner loop中参数的更新使用的是REINFORCE算法，而outer loop中参数的更新则使用的是TRPO算法。

实验结果表明，MAML确实学习到了有效的初始参数，可以达到接近oracle的性能（但似乎不会超过）。

## **思考与总结**

这篇论文给我一种大道至简的感觉，算法的推导以及使用都十分简单，但另一方面算法的思路又真的很巧妙，很符合元学习的逻辑，作者对于算法背景的描述更是让我感受到了她对元学习这一领域深刻的理解。重要的是，MAML的性能得到了大量的检验，一直到现在，在我阅读过的交叉学科论文里，几乎就没有看到过除了MAML以外的其他元强化学习算法，这足以说明MAML的价值。

## **参考文献**

[1] C. Finn, P. Abbeel, and S. Levine, ‘Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks’, *arXiv:1703.03400 [cs]*, Jul. 2017, Accessed: Apr. 12, 2022. [Online]. Available: <http://arxiv.org/abs/1703.03400>
