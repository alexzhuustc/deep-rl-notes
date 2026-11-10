# Pytorch笔记（5）——利用TORCH.AUTOGRAD进行自动微分

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2021-08-19  
来源：[https://zhuanlan.zhihu.com/p/401323737](https://zhuanlan.zhihu.com/p/401323737)

[知识库首页](../../README.md)

---

在训练神经网络时，最常用的算法是**反向传播算法**，在这个算法中，参数（模型权重）会根据损失函数对于给定参数的梯度进行调整。为了计算这些梯度，Pytorch中有一个名为`torch.autograd`的内置微分引擎，它能够支持任何计算图的梯度自动计算。

考虑一个简单的1层神经网络，输入`x`，参数`w`与`b`，以及一些损失函数，它们定义如下：

```
import torch
​
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
```

### **张量，函数与计算图**

上述代码定义了一下**计算图（computational graph）**：

![](../../assets/images/29beaec65fd6f7282c30.jpg)

在这个网络中，`w`与`b`都是需要优化的参数。因此，我们需要计算损失函数对于这些变量的梯度，为此，我们需要将这些张量的`requires_grad`属性。

---

注意：我们既可以在创建张量时设置`requires_grad`的值，也可以之后使用`x.requires_grad_(True)`方法。

---

在张量上计算计算图的函数实际上是`Function`类的一个对象，这个对象知道如何前向计算函数以及如何后向计算导数，后向传播的引用存储在张量的`grad_fn`属性中。

```
print('Gradient function for z =', z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)
```

Out:

```
Gradient function for z = <AddBackward0 object at 0x7fafdf903048>
Gradient function for loss = <BinaryCrossEntropyWithLogitsBackward object at 0x7fafdf903048>
```

### **计算梯度**

为了优化神经网络的权重参数，我们需要计算损失函数对于参数的导数，也就是说我们需要在`x`与`y`固定的情况下计算\frac{\partial loss}{\partial \omega}​ 与\frac{\partial loss}{\partial b}​​​ 。为了计算这些导数，我们需要调用`loss.backward`，然后从`w.grad`与`b.grad()`中获得这些值：

```
loss.backward()
print(w.grad)
print(b.grad)
```

Out:

```
tensor([[0.0837, 0.1592, 0.0930],
        [0.0837, 0.1592, 0.0930],
        [0.0837, 0.1592, 0.0930],
        [0.0837, 0.1592, 0.0930],
        [0.0837, 0.1592, 0.0930]])
tensor([0.0837, 0.1592, 0.0930])
```

---

注意：

- 我们只能获得计算图中叶子节点的`grad`特性。
- 对于给定的图，我们只能用`backward`计算一次梯度，，如果我们需要对同一张图多次进行反向计算，则需要向`backward`的调用传递`retain_graph=True`。

### **停止梯度追踪（Disabling Gradient Tracking）**

在默认情况下，所有`require_grad=True`的张量都会追踪它们的计算历史同时支持梯度计算。然而， 在很多情况下我们并不需要这样，例如，训练结束之后，我们只需要网络的前向计算功能，此时我们可以通过`torch.no_grad()`板块停止跟踪计算：

```
z = torch.matmul(x, w)+b
print(z.requires_grad)
​
with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
```

Out:

```
True
False
```

另一种方法是对张量执行`detach()`方法：

```
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)
```

Out:

```
False
```

停止梯度追踪的原因主要有以下两种：

- 在对预训练网络进行微调时，我们通常会固定神经网络的某些参数。
- 当只需要进行前向传递时，停止梯度追踪可以加速计算。

### **有关计算图**

理论上，`autograd`会在一个由Function对象组成的有向无环图（directed acyclic graph，DAG）中记录所有的数据以及执行过的操作。在DAG中，输入张量为叶子节点，输出张量为根节点，通过从根节点到叶子节点的顺序追踪这个图，我们可以利用链式法则自动计算梯度。

在前向传递中，`autograd`会同时做两件事：

- 执行计算结果张量所需要的操作
- 在DAG中保存操作的梯度函数

当`.backward()`被调用时，后向传播算法被执行，此时`autograd`会：

- 从每个 `.grad_fn`中计算梯度
- 将它们累积在各自张量的`.grad`属性中
- 使用链式法则，将所有的梯度传播给叶子节点

---

注意

**DAGs在Pytorch中是动态的。**每当`.backward()`被调用时，`autograd`就会开始填充一个新的图，这让我们能够在模型中使用控制流语句，在每一次迭代时，我们可以更改形状、尺寸以及操作。

---

### ***选读：张量梯度与雅克比矩阵（Tensor Gradients and Jacobian Products）**

在许多情况下，损失函数为标量，因此我们只需要计算一些参数的梯度即可。然而，在某些情况下，损失函数同样为张量，此时Pytorch要求我们计算**向量-雅克比乘积（vector-Jacobian product）**，而非真实的梯度。

对于向量函数\vec{y}=f(\vec{x})，其中\vec{x}=\left\langle x_{1}, \ldots, x_{n}\right\rangle，\vec{y}=\left\langle y_{1}, \ldots, y_{n}\right\rangle，\vec{y} 关于\vec{x}​ 的梯度由​**雅克比矩阵（Jacobian matrix）**计算：

$$  
J=\left(\begin{array}{ccc} \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}} \\ \vdots & \ddots & \vdots \\ \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}} \end{array}\right)  
$$

通常而言，`torch.autograd`是一个用于计算向量-雅克比乘积的引擎。具体来说，给定任意向量\vec{v}=\left\langle v_{1}, \ldots, v_{n}\right\rangle，计算\vec{v}\cdot J^{T} ，如果\vec{v} 刚好是标量函数 l=g(\vec{y}) 的梯度，那么\vec{v} =(\frac{\partial l}{\partial y_1},\ldots, \frac{\partial l}{\partial y_n})，根据链式法则，向量-雅克比乘积即为l对于 \vec{x} 的梯度：​

$$  
J \cdot v^{T}=\left(\begin{array}{ccc} \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{1}} \\ \vdots & \ddots & \vdots \\ \frac{\partial y_{1}}{\partial x_{n}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}} \end{array}\right)\left(\begin{array}{c} \frac{\partial l}{\partial y_{1}} \\ \vdots \\ \frac{\partial l}{\partial y_{m}} \end{array}\right)=\left(\begin{array}{c} \frac{\partial l}{\partial x_{1}} \\ \vdots \\ \frac{\partial l}{\partial x_{n}} \end{array}\right)  
$$

所以Pytorch的函数`backward()`进行的操作就是向量-雅克比乘积操作，`backward`的参数即为`v`，调用者为`y`。由于`v`就是`l`关于`y`的导数，所以`v`和`y`的形状要相同，若`y`为标量，则不需要加参数.

```
inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print("First call\n", inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nSecond call\n", inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nCall after zeroing gradients\n", inp.grad)
```

Out:

```
First call
 tensor([[4., 2., 2., 2., 2.],
        [2., 4., 2., 2., 2.],
        [2., 2., 4., 2., 2.],
        [2., 2., 2., 4., 2.],
        [2., 2., 2., 2., 4.]])
​
Second call
 tensor([[8., 4., 4., 4., 4.],
        [4., 8., 4., 4., 4.],
        [4., 4., 8., 4., 4.],
        [4., 4., 4., 8., 4.],
        [4., 4., 4., 4., 8.]])
​
Call after zeroing gradients
 tensor([[4., 2., 2., 2., 2.],
        [2., 4., 2., 2., 2.],
        [2., 2., 4., 2., 2.],
        [2., 2., 2., 4., 2.],
        [2., 2., 2., 2., 4.]])
```

注意当我们多次调用`backward`时，梯度的值会发生变化，这是因为当我们执行后向传播操作时，Pytorch会积累梯度，也就是说梯度的计算值会添加到计算图中所有叶子节点的`grad`特性中，如果我们想计算合适的梯度，我们清空之前的`grad`特征。在实际的训练中，优化器会帮助我们实现这个过程。

```
x = torch.randn(3, requires_grad=True)
​
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
​
print(y)
tensor([-1864.6609, -473.0628, 259.2955], grad_fn=<MulBackward0>)
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
​
print(x.grad)
```

Out:

```
tensor([1.0240e+02, 1.0240e+03, 1.0240e-01])
y.backward()
```

Out:

```
RuntimeError: grad can be implicitly created only for scalar outputs
```

---

注意：

当我们调用无参数的`backward)`时，这等效于调用`backward(torch.tensor(1.0))`，这在标量值函数的情况下很有用，例如神经网络训练过程中的损失函数。

---

**任何笔记都具有著作权，不可随意转载和剽窃**。
