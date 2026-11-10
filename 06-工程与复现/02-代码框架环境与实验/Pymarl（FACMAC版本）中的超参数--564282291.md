# 【多智能体强化学习】Pymarl（FACMAC版本）中的超参数

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2022-09-14  
来源：[https://zhuanlan.zhihu.com/p/564282291](https://zhuanlan.zhihu.com/p/564282291)

[知识库首页](../../README.md)

---

> 我目前使用的是[FACMAC](http://github.com/)配套的代码库，这套代码库在Pymarl的基础上编写，包含了VDN、QMIX、MADDPG以及FACMAC四个算法的连续版本和离散版本，因此可以说涵盖了最基础的几类off-policy MADRL算法，比原始的Pymarl库更为全面。

[oxwhirl/facmac (github.com)](https://github.com/oxwhirl/facmac)

关于Pymarl的介绍请见

[XuanAxuan：【多智能体强化学习】Pymarl代码分析](Pymarl代码分析--550191778.md)

关于FACMAC算法的介绍请见

[XuanAxuan：【多智能体强化学习】FACMAC/DOP：MADDPG与QMIX的结合](../../04-协作系统/01-协作稳定性与信用分配/FACMAC-DOP：MADDPG与QMIX的结合--552612045.md)

## **代码框图**

![](../../assets/images/fd8988f89d03dc061aad.jpg)

FACMAC代码库在Pymarl的基础上增加了off-policy的PG算法，但代码组织形式和Pymarl完全一样，因此可以先从Pymarl算法开始入手（详见[Pymarl代码分析](Pymarl代码分析--550191778.md)）。我把FACMAC代码库中相较原始Pymarl增加的部分框了起来，这些部分都是可以自定义修改的，蓝色框对应的是离散算法（环境），而黄色框对应的是连续算法（环境）。

## **Pymarl的经验回放池**

Pymarl（FACMAC）代码库中所有算法都用的是RNN智能体，由于RNN必须使用时间上连续的样本进行训练，因此Pymarl中的经验回放池和常规的经验回放池组织形式完全不同。常规的经验回放池将每个step的样本（transition）存储起来，然后每次训练随机取出1个batch size的样本对网络进行更新，这里batch size的单位是step，也就是说常规经验回放池中的样本是以 **transition** 为单位存储的。而在Pymarl中，受到RNN的局限，样本是以 **episode** 为单位进行存储的，因此采集样本时每次都得run完一整个episode，才能将这一个episode的样本同时存储到经验回放池中，在此过程中不会进行任何网络更新。在学习过程中，每次采集1个batch size的样本，这里batch size的单位是episode，也就是随机采样多个episodes的样本对网络进行更新。总结来说，在常规off-policy算法的经验回放池中，不同transition样本是完全打乱顺序采样的，而在基于RNN的经验回放池中，同一episode的transition样本是按顺序存取的，但具体采样哪些episodes的样本是随机的。

我的疑惑是Pymarl的这种经验回放池限制了必须得使用RNN，如果要使用MLP这种经验回放池效率就很低了，所以使用RNN就一定比MLP好吗？考虑了一下改经验回放池的话是一个非常巨大的工程，很麻烦，有没有大佬能提供一些建议呢~~

## **Pymarl超参数**

暂时先记录下有哪些超参数，但目前还没有什么调参经验，大家有什么建议欢迎评论区留言呀~~

### **通用超参数**

- `gamma`：折扣因子
- `batch_size`：用于网络更新的采样样本数量（单位：episode）
- `buffer_size`：经验回放池的大小（单位：episode）
- `lr`：agent的学习率（QMIX/VDN：Q网络，MADDPG/FACMAC：actor网络）
- `critic_lr`：critic网络的学习率
- agent网络参数（QMIX/VDN：Q网络，MADDPG/FACMAC：actor网络）

- 网络大小：1层MLP+1层RNN+1层MLP
- `rnn_hidden_dim`：RNN隐层特征维度

  
- `optimizer`：优化器类型
- `td_lambda`：$\text{TD}(\lambda)$ 中的 $\lambda$

### **QMIX**

- `epsilon_start`：$\epsilon$ 初值
- `epsilon_finish`：$\epsilon$ 结束值
- `epsilon_anneal_time`：$\epsilon$ 衰减的步长
- `target_update_interval`：目标Q网络的更新频率
- `mixing_embed_dim`：mixing网络的隐层特征维度（默认一个输入层一个输出层）
- `hypernet_layers`：hyper网络的层数
- `hypernet_embed`：hyper网络的隐层特征维度

### **MADDPG**

- `start_steps`：在训练开始之前完全随机探索的步数
- `act_noise`：动作噪声
- `target_update_tau`：目标网络软更新参数

### **MADDPG-discrete**

- `epsilon_start`：$\epsilon$ 初值
- `epsilon_finish`：$\epsilon$ 结束值
- `epsilon_anneal_time`：$\epsilon$ 衰减的步长
- `target_update_tau`：目标网络软更新参数

### **FACMAC**

FACMAC就是QMIX+MADDPG

- `mixing_embed_dim`：mixing网络的隐层特征维度（默认一个输入层一个输出层）
- `hypernet_layers`：hyper网络的层数
- `hypernet_embed`：hyper网络的隐层特征维度
- `start_steps`：在训练开始之前完全随机探索的步数
- `act_noise`：动作噪声
- `target_update_tau`：目标网络软更新参数

### **FACMAC-discrete**

- `mixing_embed_dim`：mixing网络的隐层特征维度（默认一个输入层一个输出层）
- `hypernet_layers`：hyper网络的层数
- `hypernet_embed`：hyper网络的隐层特征维度
- `epsilon_start`：$\epsilon$ 初值
- `epsilon_finish`：$\epsilon$ 结束值
- `epsilon_anneal_time`：$\epsilon$ 衰减的步长
- `target_update_tau`：目标网络软更新参数

## **Pymarl2的调参建议**

Pymarl2提供了一些QMIX的code-level implementations。

[RETHINKING THE IMPLEMENTATION TRICKS AND MONOTONICITY CONSTRAINT IN COOPERATIVE MULTI-AGENT REINFORCEMENT LEARNING](https://arxiv.org/abs/2102.03479)[GitHub - hijkzzz/pymarl2: Fine-tuned MARL algorithms on SMAC (100% win rates on most scenarios)github.com/hijkzzz/pymarl2](https://github.com/hijkzzz/pymarl2)

- 用Adam优化器替代RMSProp优化器
- 使用资格迹方法，例如 $\text{TD}(\lambda)$
- 减小经验回放池的尺寸，即`buffer_size`
- 增大局部Q网络的hidden size，即`rnn_hidden_dim`
- 增大`epsilon_anneal_time`

## **写在最后**

开始做实验之后发现很多问题，网上关于implementations的资料又很少很少，特别是针对自定义场景的调参建议，就更少了。日常吐槽读博真是太难了，哭~
