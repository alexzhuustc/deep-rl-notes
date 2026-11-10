# Pytorch笔记（3）——变换（Transform）

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2021-08-19  
来源：[https://zhuanlan.zhihu.com/p/401135685](https://zhuanlan.zhihu.com/p/401135685)

[知识库首页](../../README.md)

---

数据集中的数据往往不是训练机器学习算法所需要的数据形式，因此我们需要在训练之前使用`transform`对数据进行一些处理。

所有的Torchvision数据集均包含两个参数——`transform`用于修改特征，`target_transform`用于修改标签，它们可以接受包含转换逻辑的可调用对象。[torchvision.transforms](https://pytorch.org/vision/stable/transforms.html)模块提供了许多常用的转换。

FashionMINIST的特征为PIL图像格式，而标签则是整数。为了训练，我们需要将特征转换成归一化的张量，标签转换为one-hot编码张量。为了实现这两种转换，我们需要利用`ToTensor`与`Lambda`。

```
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
​
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```

### **ToTensor()**

`ToTensor`可以将PIL图像或Numpy数组转换成`FloatTensor`，并将图像的像素密度值压缩到[0.,1.]区间。

### **Lambda Transforms**

Lanbda转换可以应用于任何用户自定义的lambda函数。这里，我们定义的函数可以将一个整数转换成one-hot编码张量，它首先穿件一个尺寸为10的零值张量，然后调用`scatter__`将label`y`数值对应索引处的0改为1.

```
target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
```

---

**任何笔记都具有著作权，不可随意转载和剽窃**。
