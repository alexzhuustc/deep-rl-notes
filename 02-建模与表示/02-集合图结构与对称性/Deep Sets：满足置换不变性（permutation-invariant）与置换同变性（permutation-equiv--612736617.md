# 【神经网络】Deep Sets：满足置换不变性（permutation-invariant）与置换同变性（permutation-equivariant）的通用网络结构

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2023-03-09  
来源：[https://zhuanlan.zhihu.com/p/612736617](https://zhuanlan.zhihu.com/p/612736617)

[知识库首页](../../README.md)

---

> **置换不变性（permutation-invariant）**与**置换同变性（permutation-equivariant）**是处理动态输入/输出的基础，这篇论文针对这两个特性，提出了通用模型，给出了相应的理论证明，对于理解后续的许多网络结构很有帮助，推荐指数：★★★★★

[[1703.06114] Deep Sets (arxiv.org)](https://arxiv.org/abs/1703.06114)

传统的机器学习算法，例如回归算法和分类算法，都是为固定维度的数据样本设计的。在某些问题中，问题的输入和输出是满足**置换不变性（permutation-invariant）**的**集合（sets）**，而非固定维度的**向量（vectors）**。本文针对这一特殊的问题设计了一种通用的网络结构——deep sets，该结构具有**置换不变性（permutation-invariant）**与**置换同变性（permutation-equivariant）**两大特点，可用于处理输入输出是集合的问题。

## **置换不变性（permutation-invariant）**

假设函数 $f$ 将 $\mathcal{X}$ 域映射到 $\mathcal{Y}$ 域。通常来说， $\mathcal{X}$ 域是一个向量空间 $\mathbb{R}^d$，而 $\mathcal{Y}$ 域则是一个离散空间，例如 $\{0,1\}$，或连续空间 $\mathbb{R}$。现在，考虑输入是一个集合 $X=\{x_1,...,x_M\}$，$x_m \in \mathfrak{X}$，我们希望 $X$ 中元素的次序对 函数 $f$ 的响应没有影响，下面定理给出了函数满足置换不变性的充分必要条件。

【**Property**】作用于**集合**上的函数 $f:\mathcal{X}\rightarrow \mathcal{Y}$ 必须对集合中元素的次序具有**置换不变性**，即：对于任何置换 $\pi$ 有



$$
f\left(\left\{x_1, \ldots, x_M\right\}\right)=f\left(\left\{x_{\pi(1)}, \ldots, x_{\pi(M)}\right\}\right) \\
$$



【**Therorem**】函数 $f:\mathcal{X}\rightarrow \mathcal{Y}$ 具有置换不变性，当且仅当该函数能够分解成 $\rho\left(\sum_{x \in X} \phi(x)\right)$ 的形式，这里 $\rho$ 和 $\phi$ 表示任意函数。

证明见论文

### **网络结构（Deep Sets）**

![](../../assets/images/a76fd1cfb5b770a06ae7.jpg)

上述定理给出了置换不变函数的通用结构，我们可以用通用逼近器（例如神经网络）来替换 $\rho$ 和 $\phi$，此时我们只需学习这些逼近器的参数即可。具体的网络结构可以表述为：

- 首先将输入 $X$ 中的每个元素 $x_m$，$\forall 1\le m\le M$，转换为嵌入特征 $\phi(x_m)$；
- 将 $\phi(x_m)$ 加起来得到 $\sum_m{\phi(x_m)}$，然后用 $\rho$ 函数处理，这里的 $\rho$ 可以是全连接层、非线性激活函数等等；
- 如果有额外信息 $z$，则可以将 $z$ 作为条件得到 $\phi(x_m|z)$。

简单来说，deep sets的关键就是将所有元素的嵌入特征相加起来然后应用非线性转换。这种网络结构兼具通用性和正确性。

## **置换同变性（permutation-equivariant）**

现在考虑置换同变性场景，此时 $\mathcal{X}=\mathcal{Y}=\mathbb{R}^M$，将 $\mathbf{f}$ 限制为神经网络。标准的神经网络可以表示为 $\mathrm{f}_{\Theta}(\mathbf{x})=\sigma(\Theta \mathbf{x})$，这里 $\Theta\in\mathbb{R}^{M\times M}$ 表示权重矩阵，$\boldsymbol{\sigma}:\mathbb{R}^M \rightarrow \mathbb{R}^M$ 表示逐点非线性激活函数。下面的定理给出了这类型函数满足置换同变性的充分必要条件。

![](../../assets/images/63ddca4987e5263ea9af.jpg)

【**Lemma**】函数 $\mathbf{f}_{\Theta}: \mathbb{R}^M \rightarrow \mathbb{R}^M$ 具有置换同变性，当且仅当 $\Theta$ 的所有对角元素相等，非对角元素也相等，即



$$
\Theta=\lambda \mathbf{I}+\gamma\left(\mathbf{1 1}^{\top}\right) \quad \lambda, \gamma \in \mathbb{R} \quad \mathbf{1}=[1, \ldots, 1]^{\top} \in \mathbb{R}^M \quad \mathbf{I} \in \mathbb{R}^{M \times M} \text { is the identity matrix } \\
$$



另外，我们还可以通过改变神经网络层的参数得到其他变体，例如



$$
\mathbf{f}(\mathbf{x}) \doteq \boldsymbol{\sigma}(\lambda \mathbf{I x}+\gamma \operatorname{maxpool}(\mathbf{x}) \mathbf{1}) \\
$$



和求和操作一样，最大值池化同样可交换。在某些情况下，最大池化比求和池化效果更好。

上面考虑的是单输入输出的情况，即 $x_m\in\mathbb{R}$。对于多输入多输出的情况，即 $\mathbf{x} \in \mathbb{R}^{M \times D}, \mathbf{y} \in \mathbb{R}^{M \times D^{\prime}}$，该函数满足



$$
f(\mathbf{x})=\sigma\left(\mathbf{x} \Lambda-11^{\top} \mathbf{x} \Gamma\right) \\
$$



这里 $\Lambda,\Gamma\in\mathbb{R}^{D\times D'}$ 是模型参数 。同样，最大池化对应的函数形式为



$$
f(\mathbf{x})=\sigma(\mathrm{x} \Lambda-1 \operatorname{maxpool}(\mathbf{x}) \Gamma) \\
$$



这里 $\operatorname{maxpool}(\mathbf{x})=\left(\max _m \mathbf{x}\right) \in \mathbb{R}^{1 \times D}$ 。该模型还可以进一步简化为



$$
f(\mathbf{x})=\sigma(\beta+(\mathbf{x}-1 \operatorname{maxpool}(\mathrm{x})) \Gamma) \\
$$



### **网络结构**

![](../../assets/images/4d554433f49f45996b9b.jpg)

满足上述定理的神经网络层实际上只需将非线性函数应用于下列元素的加权求和：1）输入 $\mathbf{Ix}$；2）输入元素的和 $\left(\mathbf{1}\mathbf{1}^{\top}\right) \mathrm{x}$ 。由于求和项是置换不变的，所以最终输出是置换同变的。

![](../../assets/images/0988412b33f7425f6f87.jpg)

另外，置换同变函数的组合依然是置换同变的，所以我们可以通过堆叠置换同变网络层来构造DeepSets网络。

## **总结与个人思考**

放在2023年来看，deepsets的表征能力或许不能与许多更高级的结构（GNN、Attention、HPN等等）相比，但这篇论文系统性地对置换不变性与置换同变性进行了总结，个人感觉还是蛮有意义的~

## **参考文献**

[1] M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. Salakhutdinov, and A. Smola, ‘Deep Sets’. arXiv, Apr. 14, 2018. Accessed: Mar. 07, 2023. [Online]. Available: <http://arxiv.org/abs/1703.06114>
