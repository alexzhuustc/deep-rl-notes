# 【多智能体强化学习】Pymarl虚拟环境安装步骤（Windows/Ubuntu）

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2023-03-16  
来源：[https://zhuanlan.zhihu.com/p/542727892](https://zhuanlan.zhihu.com/p/542727892)

[知识库首页](../../README.md)

---

> 终于把pymarl的虚拟环境安好啦！这个过程遇到好些问题，还好有超级热心超级nice的同学/学长给我提供建议/帮助，让我这个计算机小菜鸡不至于在代码跑通前就因为心态崩溃直接跑路，真的非常感恩~  
> 为了防止以后踩相同的坑，我决定把整个过程记录一下，也希望能够帮助到有需要的同学~

P.S 整个过程需要耗费非常多流量，建议大家在有WIFI且网速快的地方搞定~

## 一、安装StarCraft II

1.Windows

直接在官网下载安装最新版就好

[《星际争霸II》官方网站_暴雪出品，史诗级竞技大作 (blizzard.cn)](https://sc2.blizzard.cn/landing)

2. Ubuntu

在Ubuntu系统下，安装SC2.4.6.2.69232版本的SC2环境。具体步骤是先安装下面的压缩包，然后解压到用户名文件夹中。

<https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.6.2.69232.zip>

这个安装包有3G，我在实验室用流量怎么都下载不下来，所以直接找学长要了安装包上传在百度云里，有需要的同学可以私聊我~

## 二、创建pymarl虚拟环境

具体的Anaconda虚拟环境是什么以及怎么安装可以看一下下面这篇文章

[宁萌时光：Anaconda 中使用 conda 配置虚拟环境与管理安装包](https://zhuanlan.zhihu.com/p/508480163)

1. 创建虚拟环境

```
conda create -n pymarl python=3.7 -y
conda activate pymarl
```

2.安装pytorch

直接在pytorch官网上根据电脑情况找相应的安装语句

```
# CUDA 10.2
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
```

3.安装下面这些pymarl运行所依赖的包

```
pip install sacred numpy scipy matplotlib seaborn pyyaml pygame pytest probscale imageio snakeviz tensorboard-logger
```

如果报错`the read operation timed out`，就说明超时了，可以把命令改成：

```
 pip --default-timeout=100 install sacred numpy scipy matplotlib seaborn pyyaml pygame pytest probscale imageio snakeviz tensorboard-logger
```

这一堆包安装完之后会有两个包需要改成低版本，一个是pyyaml，还有一个是protobuf

```
pip uninstall pyyaml
pip install pyyaml==3.13
pip uninstall protobuf
pip install protobuf==3.19.1
```

（可能每个人遇到的情况不同，根据报错的情况随机应变就好！）

## 三、安装SMAC

```
pip install git+https://github.com/oxwhirl/smac.git
```

## 四、添加环境变量

1. Windows

在Windows里，增加SC2PATH变量，值为Starcraft II安装的位置，

![](../../assets/images/4966b8ca580430164a4f.jpg)

2. Ubuntu

在Ubuntu中，用命令行添加SC2PATH变量，首先用vim打开bashrc：

```
vim ~/.bashrc
```

在最后一行添加环境变量，键和值同样分别是SC2PATH和StarCraft II在Ubuntu系统中用户名文件夹下的位置

```
export SC2PATH=~/StarCraftII/
```

（P.S 打开bashrc之后，先按i键进入insert模式，然后在最后一行输入上面的代码，随后按Esc键退出，之后按Shift+：，然后输入wq加回车退出，会用Ubuntu的忽略哈哈）

添加完环境变量后更新环境

```
 source ~/.bashrc
```

然后将安装的ScarcraftII文件夹复制到src的3dparty文件夹下。

## 五、添加地图

1. Windows

先下载下面的SMAC_MAPS压缩包，然后解压到StarCraft II下的Maps文件夹中（没有就自己建立）。

<https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip>

2. Ubuntu

Ubuntu在下载SC2环境时就自带了地图，所以不需要额外操作。

## 六、测试SMAC环境

```
python -m smac.examples.random_agents
```

此时命令行会出现以下信息，windows还会弹出游戏界面

![](../../assets/images/98f643052251312809ad.jpg)

## 七、测试Pymarl代码

Windows下把main函数第16行的SETTINGS['CAPTURE_MODE'] = "fd"改成SETTINGS['CAPTURE_MODE'] = "sys"，否则会报错。

![](../../assets/images/5566e836ed28536c90d7.png)

命令行进入pymarl-master文件夹，运行src/main.py文件

```
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z
```

此时终端会出现训练信息

![](../../assets/images/6a64dd75eb2687190506.jpg)

至此就完成安装啦！！！

## 八、pycharm传参

因为pymarl是采用命令行传参的方式，如果要用pycharm的话，可以直接在edit configuration里传参：

![](../../assets/images/a083ef31a2fa96ecb959.jpg)

---

天知道，这些看似很简单的东西，对一个不会多少命令行指令也不怎么会用Ubuntu系统的人来说有难，不过有痛苦就会有成长哈哈哈，接下来就是研究代码啦！！！
