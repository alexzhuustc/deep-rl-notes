# 【多智能体强化学习】[learn to cooperate] Stabilising Experience Replay阅读笔记

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2021-12-15  
来源：[https://zhuanlan.zhihu.com/p/432262311](https://zhuanlan.zhihu.com/p/432262311)

[知识库首页](../../README.md)

---

> 我的其他论文笔记汇总：

[XuanAxuan：【多智能体强化学习】新手入门算法论文大总结](新手入门算法论文大总结--432241482.md)
> 经验回放池是DQN算法的关键技术之一，它不仅能够打破样本之间的时间关联性从而稳定深度神经网络的训练（DNN的训练要求样本近似独立同分布），防止DNN过拟合最近经验，降低泛化误差，还能够通过重复利用经验样本提升样本效率。然而，经验回访池与独立学习的结合会产生一定的问题：经验回访池的使用要求环境不随时间而变化，而独立学习产生的非平稳性会导致经验回访池中的样本不再能反映当下环境的动态性特点，也就是说由于环境动态变化规律在不停地变化，回放池中的样本是过时的，不再适合用于更新网络。另一方面，如果取消经验回访池，采样效率会变低从而影响MADRL的稳定性。在这样的背景下，一些工作开始研究如何改进经验回访池来使其适应多智能体系统。

为了解决独立学习中存在的环境非平稳问题，本文利用 **重要性采样（importance sampling）**使得过时数据逐渐衰减，同时利用 **指纹（fingerprint）** 来鉴别样本年龄。

- 研究点：研究如何解决多智能体独立学习中的环境非平稳性问题
- 场景设定：局部可观、全部观测历史
- 训练方法：独立学习

### **模型方法**

在全局可观的情况下，多智能体系统中单个智能体 $a$ 的贝尔曼最优方程可以写为



$$
\begin{gathered} Q_{a}^{*}\left(s, u_{a} \mid \boldsymbol{\pi}_{-a}\right)=\sum_{\mathbf{u}_{-a}} \boldsymbol{\pi}_{-a}\left(\mathbf{u}_{-a} \mid s\right)\left[r\left(s, u_{a}, \mathbf{u}_{-a}\right)+\right.  \left.\gamma \sum_{s^{\prime}} P\left(s^{\prime} \mid s, u_{a}, \mathbf{u}_{-a}\right) \max _{u_{a}^{\prime}} Q_{a}^{*}\left(s^{\prime}, u_{a}^{\prime}\right)\right] \end{gathered} \\
$$



这里 $\boldsymbol{\pi}_{-a}$ 与 $\mathbf{u}_{-a}$ 分别表示除了 $a$ 以外其余所有智能体的联合策略与动作。从该公式可以看出，由于其他智能体的策略 $\boldsymbol{\pi}_{-a}\left(\mathbf{u}_{-a} \mid s\right)=\Pi_{i \in-a} \pi_{i}\left(u_{i} \mid s\right)$ 会随时间而变化，导致此时的环境不再平稳，经验回访池中的样本不能再准确反映当下环境特征。为此，作者提出了两个方法来解决环境的非平稳性：

- **重要性采样（importance sampling）**：在单智能体强化学习中，重要性采样可以用于**离策略（off-policy）**学习，即从行为策略产生的数据中学习目标策略，另外，它还能够从不同环境采集的数据中进行**离环境（off-environment）**学习。重要性采样的思想是：在训练阶段，我们能够获取每个智能体的策略，也就是说我们知道环境的变化规律，因此我们能够通过一个 **重要性权重因子（importance weighting）** 来修正环境的动态变化。  
   为了实现重要性采样，每一次收集样本时，我们需要额外将当前时刻 $t_c$ 所有智能体的策略 $\boldsymbol{\pi}_{-a}^{t_{c}}\left(\mathbf{u}_{-a} \mid s\right)$ 与其他元素共同组成元组 $\left\langle s, u_{a}, r, \pi\left(\mathbf{u}_{-a} \mid s\right), s^{\prime}\right\rangle^{\left(t_{c}\right)}$ 放入经验回放池中。在训练时刻 $t_r$​，损失函数的计算公式变为  
   $\mathcal{L}(\theta)=\sum_{i=1}^{b} \frac{\boldsymbol{\pi}_{-a}^{t_{r}}\left(\mathbf{u}_{-a} \mid s\right)}{\boldsymbol{\pi}_{-a}^{t_{i}}\left(\mathbf{u}_{-a} \mid s\right)}\left[\left(y_{i}^{D Q N}-Q(s, u ; \theta)\right)^{2}\right]$ 这里 $\frac{\boldsymbol{\pi}_{-a}^{t_{r}}\left(\mathbf{u}_{-a} \mid s\right)}{\boldsymbol{\pi}_{-a}^{t_{i}}\left(\mathbf{u}_{-a} \mid s\right)}$ 是重要性权重因子。\
   注意：上面的推导是基于全观测的假设，在部分可观的条件下，环境的非平稳特性不仅取决于 $\boldsymbol{\pi}_{-a}\left(\mathbf{u}_{-a} \mid s\right)$，同时还会间接依赖于其他智能体的策略，这些项通常难以处理，此时 $\frac{\boldsymbol{\pi}_{-a}^{t_{r}}\left(\mathbf{u}_{-a} \mid s\right)}{\boldsymbol{\pi}_{-a}^{t_{i}}\left(\mathbf{u}_{-a} \mid s\right)}$ 只能作为一种近似。  
   重要性采样为真实的目标提供了无偏估计，但它产生的重要性比值通常具有较大的方差，对重要性权重因子进行截断或调整可以减小方差但同时也会引入偏差，为此作者又提出了另一种解决非平稳性的方法。
- **多智能体指纹（fingerprint）**：指纹的基本思想是通过对其他智能体策略进行估计的基础上求解Q函数来减少非平稳性，实践表明，智能体的探索率以及训练迭代次数均可以作为指纹加入到智能体观测中，从而减少环境的非平稳性。

### 参考文献

J. Foerster *et al.*, ‘Stabilising Experience Replay for Deep Multi-Agent Reinforcement Learning’, *arXiv:1702.08887 [cs]*, May 2018, Accessed: Oct. 25, 2021. [Online]. Available: [Stabilising Experience Replay for Deep Multi-Agent Reinforcement Learning](http://arxiv.org/abs/1702.08887)
