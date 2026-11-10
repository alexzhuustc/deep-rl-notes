# 【深度学习】关系归纳偏置（Relational Inductive Bias）与图神经网络（GNN）

作者：XuanAxuan · 保存：2026-11-10 · 原文更新：2023-03-28  
来源：[https://zhuanlan.zhihu.com/p/617556272](https://zhuanlan.zhihu.com/p/617556272)

[知识库首页](../../README.md)

---

> 这篇论文首先从关系归纳偏置的角度对不同网络结构的设计进行了分析，然后重点介绍了图网络。全文都是干货，没有废话，通俗易懂。看完之后，我弄懂了许多之前比较模糊的概念，非常推荐大家精读一下~~推荐指数：★★★★★

原文：《Relational inductive biases, deep learning, and graph networks》

[Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261)

人类智能的关键在于使用有限的工具产生无数可能性，例如将有限的文字组成无限的句子，这种利用已知模块构造新的推理、预测和行为的方法被称作**组合泛化（combinatorial generalization）**。人类之所以具有组合泛化的能力，是因为我们的认知机制能够很好地表征结构从而推理出不同事物之间的关系，例如，将复杂系统表示成不同实体以及实体之间关系的组合。受到人类这一处理机制的启发，如何构造具有组合泛化能力的智能系统（**结构化方法，structured approach**）成为了人工智能发展的核心。在机器学习发展的最初几十年，数据和计算资源都十分稀缺与昂贵，在这种情况下，结构化方法强大的**关系归纳偏置（relational inductive bias）**所带来的高样本效率就十分重要。然而，不同于机器学习的早期方法，当今的**深度学习**更多地遵循一种**端到端**的设计方案，这种方案强调尽可能少地对表征方式和计算方法做出先验假设，即**避免人工设计（hand-engineering）显式结构（explicit structure）**，这是因为当前的数据和计算资源成本很低，因此可以适当地降低样本效率以此换来更加灵活的网络结构。尽管深度学习已经取得了巨大的成功，但其在一些特定任务下仍然面临着巨大的挑战，例如对复杂语言与场景的理解，对结构化数据的推理，训练条件以外的迁移学习，以及小样本学习等等。这些场景无一例外地要求算法具有组合泛化能力，不具有显式结构的模型很难处理这些任务。需要指出的是，结构化（structure）与灵活性（flexibility）并不是相互矛盾的，合理的思路应当是结合结构化方法和深度学习的优势设计出更好的具有组合泛化能力的方案。近年来已经涌现出了许多深度学习与结构化方法相结合的方法，这些方法通过对网络结构进行先验假设，引入了强大的关系归纳偏置，能够在离散的实体（entities）与实体间的关系（relations）上进行计算，因而具备对显式结构化数据进行推理的能力。接下来本文将从关系归纳偏置的角度对一些重要的深度学习方法进行剖析，然而重点介绍图神经网络。

## **关系归纳偏置（Relational Inductive Bias）**

### **关系推理**

定义**结构（structure）**为一系列已知**基础模块（building blocks）**组合而成的产物，结构表征（structured representations）对这种组合进行表征（例如元素的排列），而结构计算（structured computation）则表示将这些元素和其组合当做是一个整体进行处理。**关系推理（Relational reasoning）**表示利用规则（rules）推导实体与实体间关系的结构表征，接下来对实体、关系与规则进行明确的定义：

- **实体（entity）**表示具有某种**属性（attribute）**的元素（element），例如具有尺寸和质量的物体。
- **关系（relation）**表示两个实体之间的性质（property），例如与...一样，比...重，以及距离等等。关系同样具有属性，该属性可能与全局状态有关，例如石头比羽毛重这一关系属性是否成立取决于空气的状态。
- **规则（rule）**表示实体与关系到其他实体与关系之间的函数映射关系。

### **关系归纳偏置**

**学习（Learning）**是指通过观察与交互获取有用知识的过程，这一过程需要在一个解空间中搜索到最优解，该最优解能够对数据提供更好的解释或者实现更高的奖励。在许多情况下，存在多个同样好的解。**归纳偏置（inductive bias）**的作用就是让学习算法更加倾向于选择某一类解，并且这种倾向性与观测数据无关。例如，在贝叶斯模型中，归纳偏置可以表示成对先验分布参数的选择，而在一些其他的场景下，归纳偏置则可以表示成目标函数中用于防止过拟合的正则项（regularization term），又或者是对算法框架的先验假设。归纳偏置以灵活性为代价提高了样本效率，这可以看做是**偏差-方差（bias-variance）**的折中。理想情况下，归纳偏置可以帮助算法找到具有泛化能力的解，并且不会损失性能。然而，不合理的归纳偏置可能会通过引入过强的约束从而导致性能下降。归纳偏置可以通过对数据产生过程或者解空间进行假设得到，通过增加不同解之间的偏向性，搜索过程会变得更加简单。进一步的，对实体与实体间关系所做出的约束被称为**关系归纳偏置（relational inductive bias）**。

近年来，深度学习领域出现了许多新的网络结构，这些网络结构的设计原则是**利用基础模块构建更加复杂的、深层次的计算层级和图（graph）**。例如，通过堆叠全连接（fully conected）层则产生了**多层感知机（multilayer perceptions，MLPs）**，通过堆叠卷积（convolutional）层则产生了**卷积神经网络（convolutional neural networks，CNNs）**，而更加标准的图像处理网络则是通过一系列CNN和一个MLP组合而成。这些通过层的堆叠设计网络的方式提供了一种特殊的关系归纳偏置——**分层处理（hierarchical processing）**——即计算分阶段进行，这可以增加输入信号中信息之间的远程交互（long range interactions）。实际上，这些基础模块本身也包含了不同的关系归纳偏置，接下来我们将举例介绍不同深度学习网络模块中的关系归纳偏置。

### **深度学习中不同网络结构的关系归纳偏置**

在深度学习中，实体和关系通常表示成分布式的表征（**distributed representations**），规则表示神经网络函数近似器。实际上，不同网络结构中有关实体、关系和规则的具体定义有所不同。为了更加深入地理解不同网络结构的区别，在分析每一种网络结构时，首先考虑以下问题：

- 规则函数的实参（arguments），即将什么样的实体与关系作为网络输入；
- 参数如何复用，例如在实体/关系中共享，或者在不同时间步共享；
- 该网络结构如何定义表征之间的交互（interactions）与隔离（isolation）。

![](../../assets/images/0999022c98b496e3e1d2.jpg)![](../../assets/images/187d0c183037babf7576.jpg)

### **全连接层（Fully Connected Layers）**

全连接层是最常见的深度学习基础模块，该模块的输入是向量，而规则函数则是一个非线性向量函数，具体地，输出向量中的每一个元素或单元（units）是输入向量与一个权重（weight）向量点乘，然后增加一个偏置（bias）项，最后经过一个非线性函数得到的结果。此时，**实体**表示网络中的**单元**，**关系**是**所有对所有**（all to all，即第 $i$ 层的所有单元会与第 $j$ 层的所有单元相连），规则函数由权重与偏置表示。规则函数的实参是全输入信号，另外，整个网络模块不存在任何复用，信息之间也不存在隔断。因此，全连接层的关系归纳偏置是非常微弱的：所有输入单元共同决定任何一个输出的单元值，不同输出单元值之间是相互独立的。

### **卷积层（Convolutional Layers）**

卷积层是另一个最常见的深度学习基础模块，该模块利用一个与输入相同秩的核对输入向量进行卷积，增加一个偏置项，然后进行逐点的非线性操作得到最终的输出。卷积层中的实体依然是单个单元（或者网格单元，例如像素），但是此时实体间的关系更加稀疏。卷积层包含两种关系归纳偏置：局部性（locality）和平移不变性（translation invariance）。**局部性**表明关系规则的输入是输入信号的坐标空间中相近的实体，距离较远的实体之间是相互隔离的。**平移不变性**则表示相同的规则可以在输入的不同区域之间复用。在自然图像中，邻近区域的像素具有很高的协方差，该协方差会随着距离的增加消失，这表明了局部性在CNN中的合理性；另外，一副图像中的统计特性沿着不同区域大致是平稳的，因此平移不变性也非常有效。

### **循环层（Recurrent Layers）**

循环层通常用于处理一系列时间步的序列数据。在循环层中，可以将每个时间步的输入和隐层特征当作实体，然后将当前隐层特征以及输入与前一个时间步隐层特征之间的马尔科夫依赖看作关系。此时规则函数通过将每个时间步的输入和隐层特征作为实参更新隐层特征，规则函数可以在不同时间步之间复用，这反映了循环层的关系归纳偏置——**时间不变性（temporal invariance）**（与CNN的空间平移不变性对应）。例如，一些物理事件的规律在时间上具有不变性。另外，通过马尔科夫结构，RNN同样具有**局部性**的关系归纳偏置。

### **集合（sets）与图（graphs）上的计算**

尽管目前深度学习领域已经有各种包含不同关系归纳偏置的方法，但目前为止依然没有一种能够适用于任意关系结构的深度学习模块。一个很重要的问题是，世界上的实体通常没有固定的顺序；又或者是，实体的顺序取决于实体之间的关系。因此，在设计用于关系推理的深度学习模块时，我们必须考虑排序不变性（invariance to ordering）（在没有关系的情况下）。

**集合（Sets）**可用于描述没有固定顺序的实体系统。举例而言，考虑一个任务，预测太阳系系统的质量，该系统包含 $n$ 个行星，这些行星的属性（质量，位置，速度等等）可以表示为 $\{\mathbf{x}_1,\mathbf{x}_2,...,\mathbf{x}_n\}$。如果在该任务中采用MLP，那么输入 $(\mathbf{x}_1,\mathbf{x}_2,...,\mathbf{x}_n)$ 所得到的结果则可能和另一种排序 $(\mathbf{x}_n,\mathbf{x}_2,...,\mathbf{x}_1)$ 不同。$\{\mathbf{x}_1,\mathbf{x}_2,...,\mathbf{x}_n\}$ 总共有 $n!$ 种可能的排列方式，在最坏的情况下，MLP需要单独考虑每一种排列，为了得到一个近似函数，所需要的输入/输出训练样本将指数级增加。为了处理这种**组合式爆炸（combinatorial explosion）**，一种可能的方式是让最终的预测结果取决于所有输入元素属性的**对称函数**，也就是首先利用共享的函数计算每个实体的特征 $\{f(\mathbf{x}_1),f(\mathbf{x}_2),...,f(\mathbf{x}_n)\}$，然后用对称的方式将其聚合起来（例如求平均），这就是**DeepSets**的基本原理。

然而，在许多问题中，集合中的每个实体还受到其他实体的影响，此时置换不变性（permutation invariance）并不是唯一需要考虑的因素。例如，假设需要预测单个行星 $\Delta t$ 时刻之后的位置，此时，单靠聚合的实体信息已经无法得到准确的结果，因为每个行星的运动还取决于其他行星所施加的外力。这个时候我们可以将每个实体的更新状态记为 $\mathbf{x}_i^{\prime}=f\left(\mathbf{x}_i, \sum_j g\left(\mathbf{x}_i, \mathbf{x}_j\right)\right)$，这里 $g$ 用来计算行星 $j$ 施加给行星 $i$ 的外力，$f$ 则用于计算外力和环境动态变化所导致的行星 $i$ 的下一个状态。之所以可以在不同实体之间的关系中共用 $g$ 依然是考虑到全局系统的置换不变性。另外，这里只考虑了两个实体之间的关系，在更加复杂的系统中，还可以用同样的方式考虑更多实体之间的关系。

上述的两个例子描述了两种关系结构：第一个系统实体之间没有任何关系；第二个系统每个实体之间都存在成对的关系。在现实世界中，大多系统所包含的关系结构处于这两种极端的中间状态，即有些实体之间有关系，而有些实体之间并没有关联。也就是说，成对的交互关系只存在于某些实体之间，此时每个实体的更新状态可以修改为 $\mathbf{x}_i^{\prime}=f\left(\mathbf{x}_i, \sum_{j\in\delta(i)} g\left(\mathbf{x}_i, \mathbf{x}_j\right)\right)$，$\delta(i) \subseteq\{1, \ldots, n\}$ 表示与实体 $i$ 有关联的实体集合。这种组织形式实际上就对应着一张**图（graph）**，在这个图中，每个实体 $i$ 只与一小部分其他实体交互，这一小部分实体可以被称为 $i$ 的**邻居（neighborhood）**。注意此时状态的更新对于邻居节点而言仍然具有置换不变性。

**图（Graphs）**用于表征任意关系结构，相较于CNN与RNN，图上的计算提供了更强的关系归纳偏置，因而接下来将具体介绍图网络模块。

## **图网络（Graph Network，GN）模块**

![](../../assets/images/43cda88d82ca02f2e991.jpg)

### **图（graph）的定义**

在GN框架下，图（graph）可以用 $G=(\mathbf{u},V,E)$ 表示，这里 $\mathbf{u}$ 表示全局属性，$V=\{\mathbf{v}_i\}_{i=1:N^v}$ 表示节点的集合（尺寸为 $N^v$），这里的节点也即上文所描述的实体，$\mathbf{v}_i$ 代表每个节点的属性， $E=\left\{\left(\mathbf{e}_k, r_k, s_k\right)\right\}_{k=1: N^e}$ 表示边的集合（尺寸为 $N^e$），$\mathbf{e}_k$ 代表每条边的属性，$r_k$ 是接收节点的索引， $s_k$ 则表示发送节点的索引。

### **GN模块的内部结构**

![](../../assets/images/6583a0d4638d2d623448.jpg)

一个GN模块包含三个更新函数，$\phi$，以及三个聚合函数，$\rho$，



$$
\begin{array}{rlrl} \mathbf{e}_k^{\prime} & =\phi^e\left(\mathbf{e}_k, \mathbf{v}_{r_k}, \mathbf{v}_{s_k}, \mathbf{u}\right) & \overline{\mathbf{e}}_i^{\prime} =\rho^{e \rightarrow v}\left(E_i^{\prime}\right) \\ \mathbf{v}_i^{\prime} & =\phi^v\left(\overline{\mathbf{e}}_i^{\prime}, \mathbf{v}_i, \mathbf{u}\right) & \overline{\mathrm{e}}^{\prime}=\rho^{e \rightarrow u}\left(E^{\prime}\right) \\ \mathbf{u}^{\prime} & =\phi^u\left(\overline{\mathbf{e}}^{\prime}, \overline{\mathbf{v}}^{\prime}, \mathbf{u}\right) & \overline{\mathbf{v}}^{\prime}=\rho^{v \rightarrow u}\left(V^{\prime}\right) \end{array} \\
$$



这里 $E_i^{\prime}=\left\{\left(\mathbf{e}_k^{\prime}, r_k, s_k\right)\right\}_{r_k=i, k=1: N^e}, V^{\prime}=\left\{\mathbf{v}_i^{\prime}\right\}_{i=1: N^v}, \text { and } E^{\prime}=\bigcup_i E_i^{\prime}=\left\{\left(\mathbf{e}_k^{\prime}, r_k, s_k\right)\right\}_{k=1: N^e}$。

$\phi^e$ 用于计算每条边的更新，$\phi^v$ 用于计算每个节点的更新，$\phi^u$ 则用于处理全局状态的更新。$\rho$ 函数将集合作为输入，然后得到一个单独的元素用于表示整个集合的聚合信息。$\rho$ 对于其输入具有置换不变性的特点，并且能够接收任意数量的输入，常用的 $\rho$ 有逐元素（elementwise）的求和，求平均以及求最大值。当图 $G$ 输入到GN模块后，整张图的计算从边开始，到节点，最后到全局状态。

### **图网络的关系归纳偏置**

当GN模块作为网络的组成部分时可以引入强大的关系归纳偏置，原因如下：

- 首先，图可以用于表征实体之间的任意关系，这意味着GN的输入可以精准地表征实体之间的交互与隔断关系，而不是简单由固定结构组成，具体的，两个实体之间的交互关系可以用边来表示，如果两个实体之间没有任何关系，则这两个实体对应的节点间没有边。
- 其次，图用**集合**来表示节点以及节点之间的关系，因此具有**置换不变性**。
- 另外，GN模块中每条边和表格节点的更新函数在所有边和节点中是**共享**的，这意味着GN模块支持某种形式的**组合泛化**：由于图由边、节点和全局特征组成，单个GN模块可以作用于不同尺寸（节点和边的数目）和形状（边的连接性）的图上。

## **图网络的设计**

![](../../assets/images/2b1ea1be3e3d9d7da52b.jpg)

### **可配置的块内结构**

GN模块中的结构和函数可以通过不同的方式进行配置，这为函数的输入输出形式以及产生方式提供了灵活性。具体的，每个 $\phi$ 都可以用某种特定的函数 $f$ 表示，这里函数 $f$ 的实参形式决定了需要什么信息作为输入。在全连接的GN（full GN）模块中，$\phi$ 用神经网络表征（分别记为 $NN_e$，$NN_v$ 以及 $NN_u$），$\rho$ 则通过逐元素的求和实现（也可求平均或求最大值）：



$$
\begin{aligned} \phi^e\left(\mathbf{e}_k, \mathbf{v}_{r_k}, \mathbf{v}_{s_k}, \mathbf{u}\right)&:=f^e\left(\mathbf{e}_k, \mathbf{v}_{r_k}, \mathbf{v}_{s_k}, \mathbf{u}\right)&=\mathrm{NN}_e\left(\left[\mathbf{e}_k, \mathbf{v}_{r_k}, \mathbf{v}_{s_k}, \mathbf{u}\right]\right) \\ \phi^v\left(\overline{\mathbf{e}}_i^{\prime}, \mathbf{v}_i, \mathbf{u}\right)&:=f^v\left(\overline{\mathbf{e}}_i^{\prime}, \mathbf{v}_i, \mathbf{u}\right) \quad&=\mathrm{NN}_v\left(\left[\overline{\mathbf{e}}_i^{\prime}, \mathbf{v}_i, \mathbf{u}\right]\right) \\ \phi^u\left(\overline{\mathbf{e}}^{\prime}, \overline{\mathbf{v}}^{\prime}, \mathbf{u}\right)&:=f^u\left(\overline{\mathbf{e}}^{\prime}, \overline{\mathbf{v}}^{\prime}, \mathbf{u}\right) \quad&=\mathrm{NN}_u\left(\left[\overline{\mathbf{e}}^{\prime}, \overline{\mathbf{v}}^{\prime}, \mathbf{u}\right]\right) \\ \rho^{e \rightarrow v}\left(E_i^{\prime}\right)&:=\quad&=\sum_{\left\{k: r_k=i\right\}} \mathbf{e}_k^{\prime} \\ \rho^{v \rightarrow u}\left(V^{\prime}\right)&:=\quad&=\sum_i \mathbf{v}_i^{\prime} \\ \rho^{e \rightarrow u}\left(E^{\prime}\right)&:=\quad&=\sum_k \mathbf{e}_k^{\prime} \\ \end{aligned} \\
$$



这里 $[\mathbf{x},\mathbf{y},\mathbf{z}]$ 表示向量/张量的拼接。对于向量形式的属性，MLP通常用于表示 $\phi$，而对于张量例如图像，则CNN更加合适。

RNN也可用于表示 $\phi$，此时需要一个额外的隐层特征作为输入和输出，上图（b）展示了没有任何信息传递过程的GN模块，这里的RNN可以用于处理动态图的时序特征，RNN也可以作为 $\phi$ 直接用于全连接GN中。

GN模块还有许多不同的结构，接下来将介绍集中常见的GN结构。

### **消息传递神经网络（Message-passing neural network，MPNN）**

消息传递网络（MPNN）的包含如下几个部分：

- 消息函数（message function），$M_t$，即GN模块中的 $\phi_e$，但该函数的输入不包含 $\mathbf{u}$；
- 逐元素的求和用于表示GN模块中的 $\rho^{e\rightarrow v}$；
- 更新函数（update function），$U_t$，用于表示GN模块中的 $\phi_v$；
- 读出函数（readout function），$R$，用于表示GN模块中的 $\phi_u$，但没有将 $\mathbf{u}$ 或 $E'$ 作为输入，因此标准GN模块中的 $\rho^{e\rightarrow u}$ 在MPNN中并不需要；
- $d_{master}$ 的功能与GN模块中的 $\mathbf{u}$ 类似，但是被定义为与所有节点连接的一个额外节点，该节点不直接影响边和全局状态的更新，它可以被表示在GN模块的 $V$ 中。

### **非局部神经网络（Non-local Neural Networks，NLNN）**

非局部神经网络（NLNN）用于表示一系列“intra-/self-/vertex-/graph-attention”方法，这些方法也可以转换成GN模块的表现形式。这里的注意力attention表明了节点的更新方式：每个节点的更新基于其邻居节点特性的加权和，每个节点与任意邻居节点的权重通过这两个节点属性的成对函数计算得到（然后在所有邻居节点中进行归一化）。NLNN的组成中并没有显式地包含进边，而是对所有节点之间的成对权重进行了计算，但NLNN依然可以处理显式的边结构，只需要将没有边的节点之间的权重设为0即可。

在NLNN中，$\phi_e$ 可用于计算两个节点之间的注意力值，即 $\alpha^e\left(\mathbf{v}_{r_k}, \mathbf{v}_{s_k}\right)=a'_k$，以及发送节点的特征值，$\beta^e\left(\mathbf{v}_{s_k}\right)=\mathbf{b}'_k$。在 $\rho^{e\rightarrow v}$ 的聚合函数中，$a'_k$ 在所有边中归一化，然后加权相加所有的 $\mathbf{b}'_k$:



$$
\begin{aligned} \phi^e\left(\mathbf{e}_k, \mathbf{v}_{r_k}, \mathbf{v}_{s_k}, \mathbf{u}\right)&:=f^e\left(\mathbf{v}_{r_k}, \mathbf{v}_{s_k}\right) \quad=\left(\alpha^e\left(\mathbf{v}_{r_k}, \mathbf{v}_{s_k}\right), \beta^e\left(\mathbf{v}_{s_k}\right)\right)=\left(a_k^{\prime}, \mathbf{b}_k^{\prime}\right)=\mathbf{e}_k^{\prime}\\ \phi^v\left(\overline{\mathbf{e}}_i^{\prime}, \mathbf{v}_i, \mathbf{u}\right) &:=f^v\left(\overline{\mathbf{e}}_i^{\prime}\right) \\ \rho^{e \rightarrow v}\left(E_i^{\prime}\right) & :=\frac{1}{\sum_{\left\{k: r_k=i\right\}} a_k^{\prime}} \sum_{\left\{k: r_k=i\right\}} a_k^{\prime} \mathbf{b}_k^{\prime} \end{aligned} \\
$$



NLNN通过在所有节点之间构造一条边，为输入的集合自主构成了图。多头注意力机制将 $\phi_e$ 与 $\rho^{e\rightarrow v}$ 表示成一系列函数的集合，然后将这些函数的结果在 $\rho^{e\rightarrow v}$ 的最后一步拼接起来。这可以被看做是使用不同的边特征，每一个特征对应一个不同的 $\phi_e$ 函数组成部分。

### **其他图网络**

有些图网络，例如CommNet，structure2vec，忽略了 $\phi_e$ 中的接收节点，而是只考虑发送节点以及边属性：



$$
\begin{aligned} \phi^e\left(\mathbf{e}_k, \mathbf{v}_{r_k}, \mathbf{v}_{s_k}, \mathbf{u}\right) & :=f^e\left(\mathbf{v}_{s_k}\right) \\ \text { or } \quad \phi^e\left(\mathbf{e}_k, \mathbf{v}_{r_k}, \mathbf{v}_{s_k}, \mathbf{u}\right) & :=\mathbf{v}_{s_k}+f^e\left(\mathbf{e}_k\right) \\ \text { or } \quad \phi^e\left(\mathbf{e}_k, \mathbf{v}_{r_k}, \mathbf{v}_{s_k}, \mathbf{u}\right) & :=f^e\left(\mathbf{e}_k, \mathbf{v}_{s_k}\right) \end{aligned} \\
$$



关系网络（Relation Networks）忽略了节点的更新，直接根据聚合的边信息预测全局输出：



$$
\begin{aligned} \phi^e\left(\mathbf{e}_k, \mathbf{v}_{r_k}, \mathbf{v}_{s_k}, \mathbf{u}\right)&:=f^e\left(\mathbf{v}_{r_k}, \mathbf{v}_{s_k}\right)&=\mathrm{NN}_e\left(\left[\mathbf{v}_{r_k}, \mathbf{v}_{s_k}\right]\right) \\ \phi^u\left(\overline{\mathbf{e}}^{\prime}, \overline{\mathbf{v}}^{\prime}, \mathbf{u}\right)&:=f^u\left(\overline{\mathbf{e}}^{\prime}\right) \quad&=\mathrm{NN}_u\left(\overline{\mathbf{e}}^{\prime}\right) \\ \rho^{e \rightarrow u}\left(E^{\prime}\right)&:=\quad&=\sum_k \mathbf{e}_k^{\prime} \\ \end{aligned} \\
$$



Deep Sets忽略了边的更新，直接根据聚合的节点信息预测全局输出



$$
\begin{aligned} & \phi^v\left(\overline{\mathbf{e}}_i, \mathbf{v}_i, \mathbf{u}\right):=f^v\left(\mathbf{v}_i, \mathbf{u}\right)=\mathrm{NN}_v\left(\left[\mathbf{v}_i, \mathbf{u}\right]\right) \\ & \phi^u\left(\overline{\mathbf{e}}^{\prime}, \overline{\mathbf{v}}^{\prime}, \mathbf{u}\right):=f^u\left(\overline{\mathbf{v}}^{\prime}\right)=\mathrm{NN}_u\left(\overline{\mathbf{v}}^{\prime}\right) \\ & \rho^{v \rightarrow u}\left(V^{\prime}\right):=\quad=\sum_i \mathbf{v}_i^{\prime} \\ & \end{aligned} \\
$$



PointNet的更新规则与Deep Sets类型，只是 $\rho^{v\rightarrow u}$ 采用的是最大值聚合，然后采用了两步节点更新。

## **可组合的多模块结构**

![](../../assets/images/d68347c52269f22235e5.jpg)

图网络的设计原则是组合使用GN模块构建复杂结构。GN模块的输入是由边、节点以及全局元素组成，输出与输入的组成部分相同。这种图对图的输入/输出接口保证了一个GN模块的输出可以作为另一个GN模块的输入，即使两个GN模块的架构不同也可以。一般来说，两个GN模块，$\text{GN}_1$ 和 $\text{GN}_2$ 可以通过将第一个GN模块的输出作为第二个GN模块的输入组合成 $\mathrm{GN}_1 \circ \mathrm{GN}_2$：$G'=\text{GN}_2(\text{GN}_1(G))$。

任意数目的GN模块可以组合起来，不同模块之间可以类似CNN一样不共享参数/结构，即 $\text{GN}_1\ne\ldots\ne\text{GN}_M$，也可以类似RNN一样共享参数/结构，即 $\text{GN}_1=\text{GN}_2=\ldots=\text{GN}_M$。共享的架构类似于消息传递，同样的局部更新机制迭代进行使得信息在整个结构内传递。如果不考虑全局信息 $\mathbf{u}$，那么一个节点经过 $m$ 步消息传递所能获得的信息取决于 $m$ 跳之外的节点和边的集合。这种消息传递机制可以看做是将复杂的计算分解成小的基础单元。

## **GNN的优势与局限**

GNN天然地适合组合泛化，因为GNN不是在整个系统层面上进行计算，而是在不同节点和边之间进行共享的计算操作，这不正是使用有限的工具产生无限的可能性。然而，GNN并不适合解决置换可变问题问题，这是GNN的局限。

## **总结与个人思考**

这篇论文将我之前所了解到的许多知识完全串联在了一起：

- 设计网络结构以及算法的首要原则是引入合理的（关系）归纳偏置，无论是比较新的GNN、Attention，还是已经非常经典的MLP、CNN、RNN都是如此；
- 经典的网络结构（例如MLP）将输入信号作为一个整体进行处理，所能处理的输入结构是固定的，因此不具有组合泛化能力，相反GNN将整个输入信号分解成边、节点的集合，然后对每个节点和每条边进行局部处理，即将复杂的整体计算分布到局部，因而可以处理不同尺寸的输入/输出，具有组合泛化能力；
- Attention，Deep Sets等结构实际上都是简化的GNN，其中attention自动计算两个节点之间的注意力权重作为边的属性，可以用于处理无显式边特征的图；而Deep Sets则忽略了边特征，只考虑节点，更加适合于没有边特征的集合输入；

看完这篇论文我终于理解了GNN为什么这么重要hhh~~~

## **参考文献**

[1] P. W. Battaglia *et al.*, ‘Relational inductive biases, deep learning, and graph networks’. arXiv, Oct. 17, 2018. Accessed: Mar. 23, 2023. [Online]. Available: <http://arxiv.org/abs/1806.01261>
