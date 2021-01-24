## 介绍

假设您要求一群用户以0-100的比例对一组电影进行评分。在经典因素分析中，您可以尝试根据一组潜在因素来解释每部电影和每个用户。例如，像《星球大战》和《指环王》这样的电影可能与潜在的科幻小说和幻想因素有很强的联系，而喜欢瓦力和《玩具总动员》的用户可能与**潜在的皮克斯因素**有很强的联系。

受限玻尔兹曼机实质上执行因子分析的*二进制*版本。（这是考虑RBM的一种方法，当然，还有其他方法以及许多使用RBM的方法，但是本文中将采用这种方法）而不是用户连续地对一组电影进行评分规模，**他们只是告诉您是否喜欢电影**，RBM会尝试发现可以解释这些电影选择的激活的潜在因素。

从技术上讲，受限玻尔兹曼机是一个**随机神经网络**（*神经网络*意味着我们有类神经元的单元，其二进制激活取决于它们所连接的邻居；随机意味着这些激活具有一个概率元素），它包括：

- 一层**可见单位**（我们知道并设置其状态的用户的电影首选项）；
- 一层**隐藏的单元**（我们尝试学习的潜在因素）；
- **偏差单元**（其状态始终为开，是一种针对每部电影的不同固有流行度进行调整的方式）。

此外，每个可见单元都连接到所有隐藏单元（此连接是无向的，因此每个隐藏单元也都连接到所有可见单元），并且偏置单元连接到所有可见单元和所有隐藏单元。为了使学习变得更容易，我们限制了网络，以便没有可见单元连接到任何其他可见单元，也没有隐藏单元连接到任何其他隐藏单元。

例如，假设我们有一套六部电影（《哈利·波特》，《阿凡达》，《 指环王3》，《角斗士》，《泰坦尼克号》和《闪闪发光》），并要求用户告诉我们他们想看哪部电影。如果我们想学习电影偏好的两个潜在单元。例如，我们六部电影中的两个自然类别似乎是科幻/幻想（包含哈利·波特，阿凡达和指环王3）和奥斯卡获奖者（包含指环王3，角斗士和泰坦尼克号），因此我们希望我们的潜在单元可以对应这些类别，那么我们的RBM如下所示：

![RBM](C:\Users\ASUS\Desktop\Learning_notes\深度学习\images\RBM.png)

（请注意与因子分析图形模型的相似之处。）

## 状态激活

受限的玻尔兹曼机和一般的神经网络通过给定其他神经元的状态来更新某些神经元的状态来工作，因此让我们来谈谈单个单元的状态如何变化。假设我们知道我们的RBM中的连接权重（我们将在下面说明如何学习），以更新单位$ i $的状态：

- 计算单位$ i $的**激活能量**$ a_i = \sum_j w_ {ij} x_j $，其中总和遍及与该单位$ i $连接的所有单位$ j $，$ w_ {ij} $是权重$ i $和$ j $之间的连接，而$ x_j $是单元$ j $的0或1状态。换句话说，单元$ i $的所有邻居都向它发送一条消息，然后我们计算所有这些消息的总和。
- 设$ p_i = \sigma(a_i)$，其中$ \sigma(x)=1/(1+exp(-x))$是逻辑函数。注意，对于大的正激活能，$ p_i $接近1，而对于负激活能，$ p_i $接近0。
- 然后，我们以概率$ p_i $打开单位$ i $，并以概率$ 1-p_i $将其关闭。
- （用外行的话说，彼此**正连接**的单位会试图让彼此共享同一状态（即，两者都处于打开或关闭状态），而彼此**负连接**的单位则是倾向于处于不同状态的敌人状态。）

例如，假设我们的两个隐藏单位确实与科幻/幻想和奥斯卡奖得主相对应。

- 如果爱丽丝已经告诉我们她在这组电影中的六个二进制偏好，那么我们可以询问我们的RBM她的偏好激活了哪些隐藏单元（即，要求RBM用潜在因素解释她的偏好）。因此，这六部电影向隐藏的单元发送消息，告诉它们进行自我更新。（请注意，即使爱丽丝已经宣布希望观看哈利·波特，阿凡达和指环王3，这也不能保证科幻 /幻想隐藏单元会打开，而只能保证它以高概率打开。这使得有点道理：在现实世界中，爱丽丝想看完所有这三部电影，这使我们高度怀疑她总体上喜欢科幻/幻想，但由于其他原因，她很少有机会看它们，因此，RBM允许我们产生 混乱的现实世界中的人物模型。）
- 相反，如果我们知道某个人喜欢科幻/幻想（这样就打开了科幻/幻想单元），则可以询问RBM隐藏单元打开了哪个电影单元（即，要求RBM生成一个电影推荐）。因此，隐藏的单元会向电影单元发送消息，告诉它们更新状态。（同样，请注意，科幻/幻想单元的启用并不能保证我们会始终推荐《哈利·波特》，《阿凡达》和《指环王3》这三部影片，因为并非每个喜欢科幻小说的人都喜欢《阿凡达》）

## 学习权重

那么，我们如何学习网络中的连接权重？假设我们有一堆训练示例，其中每个训练示例都是一个二进制向量，其中六个元素对应于用户的电影首选项。然后针对每个时期执行以下操作：

- 以一个培训示例（一组六个电影首选项）为例。将可见单位的状态设置为这些首选项。
- 接下来，使用上述**逻辑激活规则**更新隐藏单元的状态：对于第$ j $个隐藏单元，计算其激活能量$ a_j = \sum_i w_ {ij} x_i $，然后将$ x_j $以$ \sigma(a_j)$的概率设置为1并以$ 1-\sigma(a_j)$的概率降为0。然后，对于每个边$ e_ {ij} $，计算$ Positive(e_ {ij})= x_i * x_j $（即，对于每对单元，测量它们是否都打开）。
- 现在以类似的方式**重建**可见单元：对于每个可见单元，计算其激活能量$ a_i $，并更新其状态。（请注意，此重构可能与原始首选项不匹配。）然后再次更新隐藏的单位，并为每个边计算$ Negative（e_ {ij}）= x_i * x_j $。
- 通过设置$ w_ {ij} = w_ {ij} + L *(Positive(e_ {ij})-Negative(e_ {ij}))$来更新每个边$ e_ {ij} $的权重，其中$ L $是学习率。
- 重复所有训练示例。

继续直到网络收敛（即，训练示例与其重构之间的误差降至某个阈值以下），或者我们达到某个最大时期。

为什么此更新规则有意义？注意

- 在第一阶段，$ Positive(e_ {ij})$衡量我们希望网络从我们的训练示例中学习的$ i $ th和$ j $ th单位之间的关联；
- 在“重构”阶段，RBM仅基于其关于隐藏单元的假设生成可见单元的状态，$ Negative(e_ {ij})$衡量网络*自身*生成的关联（或“白日梦”）当没有单位固定于训练数据时。

因此，通过向每个边缘权重添加$ Positive(e_ {ij})-Negative(e_ {ij})$，我们正在帮助网络的白日梦更好地匹配我们训练示例的实际情况。 

（您可能会听到这个更新规则被称为“**对比散度”**更新规则，它基本上是“近似梯度下降”的时髦术语）

## 实例解析

按照原文作者给出的代码（https://github.com/echen/restricted-boltzmann-machines），让我们来演示一些例子。

首先，我使用一些造的数据来训练RBM：

- Alice: (Harry Potter = 1, Avatar = 1, LOTR 3 = 1, Gladiator = 0, Titanic = 0, Glitter = 0). Big SF/fantasy fan（科幻超级迷妹）
- Bob: (Harry Potter = 1, Avatar = 0, LOTR 3 = 1, Gladiator = 0, Titanic = 0, Glitter = 0). SF/fantasy fan, but doesn’t like Avatar（科幻迷，但不喜欢阿凡达）
- Carol: (Harry Potter = 1, Avatar = 1, LOTR 3 = 1, Gladiator = 0, Titanic = 0, Glitter = 0). Big SF/fantasy fan（科幻超级迷）
- David: (Harry Potter = 0, Avatar = 0, LOTR 3 = 1, Gladiator = 1, Titanic = 1, Glitter = 0). Big Oscar winners fan（奥斯卡超级迷）
- Eric: (Harry Potter = 0, Avatar = 0, LOTR 3 = 1, Gladiator = 1, Titanic = 0, Glitter = 0). Oscar winners fan, except for Titanic（奥斯卡迷，但不喜欢泰坦尼克号（译者注：原文此处Titanic = 1有误））
- Fred: (Harry Potter = 0, Avatar = 0, LOTR 3 = 1, Gladiator = 1, Titanic = 1, Glitter = 0). Big Oscar winners fan（奥斯卡超级迷）

该网络学习到了以下权重：

![学习权重](C:\Users\ASUS\Desktop\Learning_notes\深度学习\images\学习权重.png)

请注意，第一个隐藏单元看起来对应“奥斯卡获得者”，第二个隐藏单元看起来对应“科幻小说/魔幻”，跟我们预期一致。
如果给RBM一个新的用户输入会发生什么呢？（可视层到隐藏层）比如George, 他的偏好是(Harry Potter = 0, Avatar = 0, LOTR 3 = 0, Gladiator = 1, Titanic = 1, Glitter = 0) ，结果是“奥斯卡获得者”对应的隐藏单元被置1（而不是“科幻小说/魔幻”），正确地猜出乔治可能更喜欢“奥斯卡获得者”类型的电影。
如果我们只激活“科幻小说/魔幻”对应的单元，并且让RBM训练一段时间会发生什么呢？（隐藏层到可视层）在我的试验中，有三次哈利·波特、阿凡达和指环王3被置1；有一次阿凡达和指环王3被置1，而哈利波特没有；有两次，哈利·波特和指环王3被置1，而阿凡达没有。注意，给予我们的训练集，这些生成的偏好的确匹配了我们觉得真正的科幻迷想要看的电影。