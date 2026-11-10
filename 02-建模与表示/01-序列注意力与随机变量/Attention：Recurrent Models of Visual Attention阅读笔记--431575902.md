# 【神经网络】Attention：Recurrent Models of Visual Attention阅读笔记

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2021-11-10  
来源：[https://zhuanlan.zhihu.com/p/431575902](https://zhuanlan.zhihu.com/p/431575902)

[知识库首页](../../README.md)

---

> 在多智能体强化学习中，有一些论文将attention框架作为通信的基础框架，因此我们需要了解什么是attention，但看了许多多智能体论文之后我发现他们用的attention和这个attention似乎有些区别，但anyway，这篇论文还是让我理解了为什么会出现attention这种结构。  
> 推荐指数：★★★★★  
> 本文适合阅读人群：熟悉RNN与强化学习

本论文提出了一种基于注意力机制的任务驱动图像处理框架，将图像基于注意力机制的处理看作是控制问题，结合RNN和RL模型对该问题进行了解决，无论是需要学习的模型参数还是处理图像所需要的计算都独立于输入图像的尺寸。

### **模型框架**

![](../../assets/images/6a940001e56c2e9e393f.jpg)

本模型基于RNN，每一时刻，智能体首先处理感知到的数据，然后整合随时间获取的所有信息，最后选择 **环境动作** 以及下一时刻感知器要放置的位置。

- 感知器： 每一时刻智能体只能观测到环境（图像 $x_t$ ）的部分信息，本论文假设智能体使用一个带宽有限的感知器提取出图像 $x_t$上位置 $l_{t-1}$附近的一个类似于视网膜的表征数据。感知器以较高的分辨率编码 $l$ 附近的区域，然后以递减的分辨率编码远离 $l$ 的像素，最终能够产生一个维度远小于原图像 $x$ 的向量。这种低分辨率的表征方法被称为 **glimpse**。glimpse感知器被用于 **glimpse network**中，最终产生一个glimpse特征向量 $g_{t}=f_{g}\left(x_{t}, l_{t-1} ; \theta_{g}\right)$，其中 $\theta_g=\{\theta^0_1,\theta^0_2, \theta^0_3\}$。
- 内部特征：智能体会保存一个内部特征，该内部特征能够总结从历史观测中提取到的所有信息，它对智能体对于环境的知识进行了编码，从而能够指导智能体下一步的动作。内部特征 $h_t$ 依据核心网络（core network）更新： $h_{t}=f_{h}\left(h_{t-1}, g_{t} ; \theta_{h}\right)$。核心网络的输入是glimpse特征向量 $g_t$。
- 动作：每一时刻，智能体执行两个动作：1）决定下一个时刻如何放置感知器，即决定 $l_t$；2）决定能够影响环境的动作 $a_t$。
- 奖励：根据任务类型来确定

这一过程是一个典型的部分观测马尔科夫过程（POMDP）。环境的真实状态无法观测，智能体需要学习一个策略 $\pi\left(\left(l_{t}, a_{t}\right) \mid s_{1: t} ; \theta\right)$，在每一个时刻 $t$，将于环境的历史交互 $s_{1: t}=x_{1}, l_{1}, a_{1}, \ldots x_{t-1}, l_{t-1}, a_{t-1}, x_{t}$ 映射到下一个时刻的动作。在本论文中，策略用RNN来刻画，而历史 $s_t$ 则总结在隐层特征 $h_t$ 中。

### 参考文献

[1] V. Mnih, N. Heess, A. Graves, and K. Kavukcuoglu, ‘Recurrent Models of Visual Attention’, *arXiv:1406.6247 [cs, stat]*, Jun. 2014, Accessed: Oct. 11, 2021. [Online]. Available: <http://arxiv.org/abs/1406.6247>
