# 【多智能体强化学习】多智能体环境MPE（multiagent particle environment）介绍与代码分析

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2022-08-19  
来源：[https://zhuanlan.zhihu.com/p/555515701](https://zhuanlan.zhihu.com/p/555515701)

[知识库首页](../../README.md)

---

**MPE（multiagent particle environment）**是由OpenAI开发的一套时间离散、空间连续的二维多智能体环境，该环境通过控制二维空间中不同角色粒子（particle）的运动来完成一系列任务，使用方法与gym十分类似，目前被广泛用于各类MARL算法的仿真验证。

我的研究方向是多无人机协同控制，相关场景和MPE十分类似，因此我花了两天的时间研究MPE的整套代码，为日后自己搭建环境作参考。这套代码虽然刚开始读的时候会比较绕，但捋清楚了之后就会发现逻辑还是蛮清楚的，并且模块化程度高修改起来也方便，因此我将在这篇笔记中记录下我所整理的MPE代码框架，欢迎大家交流讨论~~

[GitHub - openai/multiagent-particle-envs: Code for a multi-agent particle environment used in the paper "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"](https://github.com/openai/multiagent-particle-envs)

p.s. MPE中粒子的运动既可以是离散的（前后左右四个方向），也可以是连续的（x轴和y轴的移动距离），因此既可以用于离散MARL算法的验证（QMIX），也可以用于离散MARL的验证（MADDPG）。最初在文献[1]中被提出时，MPE粒子的运动是连续的，但是在被人所熟知的MADDPG实验中，粒子的运动确是离散的。我不太理解为什么MADDPG作为连续算法，要用来验证离散环境，是因为MPE中智能体之间要交互离散通信信息，为了不出现混合动作空间，所以要把物理运动也设置成离散的吗？

## **MPE代码分析**

![](../../assets/images/86a4b7bab1afeb899e25.jpg)

MPE中包含三个最为重要的对象，`scenario`，`world`和`env`，其中`scenario`是一个静态的场景对象，用于构造环境模拟器以及计算不同状态下环境模拟器内的参数，`world`是动态的环境模拟器对象，`env`则通过接收算法提供的智能体控制环境模拟器，三者之间的关系大致可以用这张图表示，接下来我将一一介绍这三个对象。

### **scenario对象**

`Scenario(BaseScenario)`类的`scenario`对象的作用是**提供一系列方法用于构造特定场景下的环境模拟器以及计算不同场景参数下的强化学习元素（观测、奖励、done、info）**，每个`scenario`对象对应一个特定的场景，该场景下`Scenario`类的定义在对应场景的`.py`文件中完成。MPE提供了一系列不同的粒子场景，这些场景全部被放置在`scearios`文件夹下：

![](../../assets/images/8107ea48446585139cc2.jpg)

一个`scenario`对象包含如下方法：

- `def make_world(self)`：构建一个由**实体`entities`组成的世界`world`对象，定义各个实体的属性**，例如名字、尺寸、能否运动等，返回值即为`world`。
- `def reset_world(self, world)`：在每一个episode之前通过初始化各个实体的属性，也即**重置reset**世界。
- 以及根据当前`world`内各个`entities`属性计算单个智能体`agent`观测、奖励、done和info的方法：

- `def observation(self, agent, world)`
- `def reward(self, agent, world)`
- `def done(self, agent, world)`
- `def info(self, agent, world)`

总结来说，`scenario`对象是完全**静态**的，其本身不会发生任何改变，作用是根据`world`和`agent`对象的属性计算相关值。

### **world对象**

`World(object)`类的`world`对象相当于**环境模拟器**，该对象是对特定场景下真实环境的模拟，通过`scenario.make_world()`方法构造。`world`对象由一系列不同角色的`entities`对象组成，这些`entites`对象分为以下两大类：

- `Landmark`类，属于该类的`entites`对象属性不会发生变化，相当于物理世界中的**静物**，例如地标`landmark`对象，草`grass`对象，食物`food`对象等等；
- `Agent`类，属于该类的`entites`对象属性会不断发生变化，相当于物理世界中的**动物**，在这里也即我们所关注的智能体`agent`对象，另外，`agent`对象还分为两类：

- `policy_agent`：动作由强化学习算法所控制的智能体；
- `scripted_agent`：动作由自定义脚本所控制的智能体。

`agent`对象除了一些固有属性外，还包含两个非常重要的动态属性：

- `action`，即智能体的动作，动作又包含物理动作`action.u`与通信信息`action.c`，在MPE中，`action.u`实际上就是智能体的移动情况；
- `state`，即智能体的状态，在MPE中，`state`包含位置`state.p_pos`，速度`state.p_vel`以及通信信息`state.c`。

所有上述`entites`对象都是`world`的属性，因此调用`world.对象名s`就可以返回该类别的所有对象，例如`world.entities`返回`world`内所有`entities`，而`world.policy_agents`则只会返回所有由强化学习控制的智能体。另外，`World`类和所有以上实体对应的类均定义在`core.py`文件中。

`world`对象最关键的方法是：

- `def step(self)`：该方法根据`world`内各个`agent`所采取的动作更新当前`world`内所有`entites`的属性（实际上只有`agents`）。

总结来说，`world`对象由一系列`entites`组成，其中`agents`的状态会不断发生改变，因此`world`是**动态**的。

### **env对象**

`MultiAgentEnv(gym.Env)`类的`env`对象是**强化学习算法与环境模拟器之间的桥梁**，主要作用就是将由强化学习控制的智能体`agents`与环境模拟器中的`agents`连接起来，实现对环境模拟器的控制。因此，我们对于环境模拟器的控制最终都是通过`env`对象完成。

`env`对象由`env.world`对象与`env.agents`对象组成，这里的`world`对象就是上面所介绍的环境模拟器，而`agents`对象是`world`中由强化学习所控制的所有智能体`world.policy_agents`。

`env`对象最为关键的三个方法是：

- `def reset(self)`：重置环境模拟器`world`；
- `def step(self, action_n)`：更新环境模拟器`world`；
- `def render(self, mode='human')`：渲染，显示当前环境模拟器状态。

另外，我们还可以调用以下方法来获取单个智能体的观测、奖励、info和done:

- `def _get_obs(self, agent)`；
- `def _get_reward(self, agent)`；
- `def _get_done(self, agent)`；
- `def _get_info(self, agent)`。

总结来说，`env`对象的作用就是连接强化学习算法和环境模拟器，上述方法的任务最终还是需要靠调用`world`对象的方法来完成。

## **使用方法**

MPE的代码接口与大部分MADRL代码兼容，因此使用方法基本上都是统一的。

### **创建env对象**

```
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

# load scenario from script
scenario = scenarios.load(args.scenario_name + ".py").Scenario()

# create world
world = scenario.make_world()
# create multiagent environment
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
```

### **控制env对象**

- 重置环境

```
obs = env.reset(
```

- 更新环境

```
obs_next, r, done, info = self.env.step(actions
```

- 渲染

```
env.render()
```

## **针对Pymarl代码的修改**

为了兼容Pymarl代码的接口，需要对上述的`scenario`，`env`以及`world`进行进一步封装，具体地，定义一个`class MultiAgentEnv(gym.Env)`类的`env`对象，然后将原始MPE中的`scenario`，`env`以及`world`对象赋给新定义的`env`对象作为属性。然后再次定义`env.reset()`，`env.step()`，`env.render()`等方法，这些方法中的具体操作都是直接调用`env.env`已有的方法，另外，还需要定义`env.get_obs()`，`env.get_state()`等方法配合Pymarl的算法。对应的框架图如下，具体可参考FACMAC代码。

[GitHub - oxwhirl/facmac](https://github.com/oxwhirl/facmac)![](../../assets/images/ed0712a65865b0ba2328.jpg)

## 局限

我在准备编写环境的时候发现MPE有个很大的局限性——单个智能体的状态空间和动作空间都是完全一致的，也就是说在MPE中，`agent`的状态`state`只有位置`p_pos`，速度`p_vel`和通信信息`c`，而动作`action`只有代表移动的`u`和代表通信的`c`，不同场景的区别只是在于智能体种类、数目以及固有属性有所差异。但对于做交叉学科的人来说，实际的任务肯定不可能这么简单，所以还需要自行考虑如何更好地组织相应代码。

## **总结**

我认为学习这样一套代码对我来说是很有收获的，一方面我可以了解一个好的代码应当怎样组织才能做到使用最优化，另一方面，以后编写自己的环境时，只要严格参考官方代码的接口连接关系，就可以直接调用现有的各种算法库，减小由于接口不统一需要改算法代码所导致的错误。这是我个人的想法，仅供参考。

## **参考文献**

[1] I. Mordatch and P. Abbeel, ‘Emergence of Grounded Compositional Language in Multi-Agent Populations’. arXiv, Jul. 24, 2018. Accessed: Jun. 22, 2022. [Online]. Available: <http://arxiv.org/abs/1703.04908>

[2] R. Lowe, Y. Wu, A. Tamar, J. Harb, P. Abbeel, and I. Mordatch, ‘Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments’, *arXiv:1706.02275 [cs]*, Mar. 2020, Accessed: Oct. 05, 2021. [Online]. Available: <http://arxiv.org/abs/1706.02275>
