# 受限玻尔兹曼机详解（RBM）

## 受限玻尔兹曼机的基本模型

​		受限波尔兹曼机(Restricted Boltzmann Machines，RBM)是一类具有两层结构、对称连接且无自反馈的随机神经网络模型，**层间全连接，层内无连接**。如下图所示。$v$为**可见层**，用于表示观测数据，$h$为**隐层**，可视为一些特征提取器，$W$为两层之间的**连接权重**。这里，为了讨论方便起见，我们假设所有的可见单元和隐单元均为二值变量，即$\forall i,j \Rightarrow v_i \in \lbrace 0,1 \rbrace,h_i \in \lbrace 0,1 \rbrace$。

![20181120115904193](https://raw.githubusercontent.com/red-fox-yj/MarkDownPic/master/typora/20210215095339.png)

​		如果一个RBM有$n$个可见单元和$m$个隐单元，用向量$v$和$h$分别表示可见单元和隐单元的状态，如上图所示，其中，$v_i$表示第$i$个可见单元的状态，$h_j$表示第$j$个隐单元的状态。那么，对于一组给定的状态$(v,h)$，RBM作为一个系统所具备的能量定为：
$$
E(v,h|\theta)=-\sum_{i=1}^{n}{a_iv_i}-\sum_{j=1}^{m}{b_jv_j}-\sum_{i=1}^{n}{\sum_{j=1}^{m}{v_iW_{ij}h_j}} \tag 1
$$
上式中，$\theta= \lbrace W_j,a_i,b_j \rbrace$是RBM的参数，他们均为实数，其中$W_{ij}$表示可见层单元$i$与隐单元$j$的神经元**连接权重**，$a_{i}$表示可见单元神经元$i$的**偏置**（bias），$b_j$表示隐层单元$j$的**偏置**。

​		之所以能量函数可以写成这个样子，是因为能量函数是符合**玻尔兹曼分布**的，因此可以写成玻尔兹曼分布的形式，又因为是两层即可见层和隐层，按照我们上面的学习过程，可知我们需要不断的计算这里两层的概率，又因为两层是有联系的因此需要写成联合概率密度，这个概率密度可以解释为在$\theta$（其中$\theta$就是待调整的权值和偏置值$w,a,b$）的条件下$v$和$h$的联合概率密度。当参数确定时，基于该能量函数，我们可以得到$(v,h)$的**联合概率密度**分布： 

$$
P(v,h|\theta)=\frac{e^{-E(v,h|\theta)}}{Z(\theta)},Z(\theta)=\sum_{v,h}e^{-E(v,h|\theta)} \tag 2
$$
其中$Z(\theta)$为**归一化因子**即**所有可能情况下的能量和**，概率的形成就是某一个状态的能量除以总的可能状态能量和。

​		对于一个实际问题，我们关心的是由RBM所定义的关于观测数据$v$的分布$P(v|\theta)$，即在训练好的权值的情况下在可见层识别出内容的概率分布，如何求这个概率分布呢？很简单，对（2）式求**边缘分布**即可，如下：
$$
P(v|\theta)=\frac{1}{Z(\theta)}\sum_{h}{e^{-E(v,h|\theta)}} \tag 3
$$
为了确定该分布，需要计算归一化因子$Z(\theta)$，这需要计算$2^{m+n}$次计算（因为可见单元和隐层单元他们是**全连接**的，又因为可见单元为$n$个，隐单元为$m$个），计算量很大，因此，即使通过训练可以得到模型的参数$W_{ij},a_i,b_j$，我们任然无法计算出这些参数所确定的分布。

​		但是，由于RBM的特殊结构（即层间无连接，层内连接）可知，当给定可见单元的状态时，各隐单元的激活状态之间是条件独立的。此时，第$j$个隐单元的激活概率为：
$$
P(h_j=1|v,\theta)=\sigma(b_j+\sum_i{v_iW_{ij}}) \tag 4
$$
 其中$\sigma(x)=\frac{1}{1+exp(-x)}$为激活函数。
由于RBM的结构是对称的，当给定隐单元的状态时，各可见单元的激活状态之间也是条件独立的，即第$i$个可见单元的激活概率为：
$$
P(v_i=1|h,\theta)=\sigma(a_i+\sum_j{h_jW_{ij}}) \tag 5
$$
这里大家需对（4）、（5）两式有清晰的认识，其中$b_j$，$a_{i}$为对应的偏置值，如（4）式对隐层的某个神经元j的状态等于1的概率大小就等于把所有可见层的单元和权值相乘在相加然后加上偏置值取$\sigma$函数就是概率了，（5）式类似。

## RBM学习算法

​		学习RBM的任务是求出参数$\theta$的值，$\theta= \lbrace W_j,a_i,b_j \rbrace$。使用训练数据去训练$\theta$，然后参数$\theta$可以通过**最大化**RBM在训练集（假设包含$T$个样本）上的对数**似然函数**学习得到，如下：
$$
\theta^{*}=\mathop{argmax}_{\theta}L(\theta)=\mathop{argmax}_{\theta}\sum_{t=1}^T{P(v^{(t)},h|\theta)} \tag 6
$$
所谓似然函数，即在给定输出$x$时，关于参数$\theta$的似然函数$L(\theta|x)$（在数值上）等于给定参数$\theta$后变量$X$恢复原值的概率。而（6）式代表了我们在求得最大似然函数时的参数$\theta$取值。在训练样本的情况下，我们调整参数使其最大概率的复现这个训练数据，根据我们上面的模型图我们可以知道从输入数据层（可见层）到隐层，在从隐层到可见层这是一次训练，此时的可见层尽可能的和输入时相同，获取到一个无限逼近样本的概率，这就是学习原则，当有新的数据时就可以进行识别了。至于为什么（6）式是这样的一个形式，这里进行一下说明：
		假若我们有$T$个样本，我们都尽可能的去复现每个样本，即调整参数$\theta$使得每个样本的似然函数概率值都尽量最大化，这里我们称这些样本的**最大化似然函数概率值**为**子似然函数概率值**。因此，一次迭代的**最大似然函数概率值**就是**所有的子似然函概率值的乘积**，但是乘积不容易后面的计算，而且我们对概率的最大值不感兴趣，我们只对**使似然函数概率值最大化**的参数$\theta$感兴趣，因此为了方便求导，我们取对数，这时子似然函数概率值的乘积就是求和了，（6）式只是简单的代表，这里把上式具体化一下：
$$
\begin{aligned}
L(\theta)&=\sum_{t=1}^T{logP(v^{(t)}|\theta)}=\sum_{t-1}^Tlog{\sum_hP(v^{(t)},h|\theta)}\\
&=\sum_{t=1}^T log{\frac{\sum_h exp[-E(v^{(t)},h|\theta)]}{\sum_v\sum_h exp[-E(v,h)|\theta]}}\\
&=\sum_{t=1}^T \left\{{log\sum_h exp[-E(v^{(t)},h|\theta)]-log \sum_v\sum_h exp[-E(v,h)|\theta]}\right\}
\end{aligned}
$$
把（6）式展开再把（3）式带进去即可得到（7）式：
$$
L(\theta)=\sum_{t=1}^T \left\{{log\sum_h exp[-E(v^{(t)},h|\theta)]-log \sum_v\sum_h exp[-E(v,h)|\theta]}\right\} \tag{7}
$$
​		为了获得最优的$\theta ^*$参数，我们使用**梯度上升**进行求$L(\theta ) =\sum_{t=1}^{T}logP(v^{(t)}|\theta )$的最大值，其中关键步骤是计算$logP(v^{(t)},h|\theta)$关于各个模型参数的偏导数，下面的求导中的$\theta$是代表$W、a、b$，因为他们的求导形式一样的，因次我们使用$\theta$代替进行求导，求出后使用对于的$w、a、b$替换$\theta$就好了，求偏导前先回顾一下期望的定义：
定义：设离散型随机变量$X$的分布律为：
$$
P(X=x_k) = p_k,k=1,2,3,4,...
$$
则随机变量的**数学期望**为记为$E(X)$。即：
$$
E(X) = \sum_{k=1}^{\infty }x_kp_k
$$


好了我们现在开始求偏导。对$L(\theta)$关于$\theta$求偏导可得：
$$
\begin{aligned}
\frac{\partial{L}}{\partial{\theta}}&=\sum_{t=1}^{T}{\frac{\partial}{\partial{\theta}}\left\{{log\sum_h{exp[-E(v^{(t)},h|\theta]}-log\sum_v\sum_h exp[-E(v,h|\theta)]} \right\}}\\

&=\sum_{t=1}^T \left\{{\sum_h \frac{exp[-E(v^{(t)},h|\theta)]}{\sum_h exp[-E(v^{(t)},h|\theta)]}×\frac{\partial (-E(v^{(t)},h|\theta))}{\partial \theta}-\sum_v \sum_h \frac{exp[-E(v,h|\theta)]}{\sum_v \sum_h exp[-E(v,h|\theta)]}×\frac{\partial(-E(v,h|\theta))}{\partial \theta}}\right\}\\

&=\sum_{t=1}^T \left\{ {\left \langle \frac{\partial(-E(v^{(t)},h|\theta))}{\partial\theta}_{P(h|v^{(t)},\theta)} \right \rangle}-\left \langle \frac{\partial(-E(v,h|\theta))}{\partial\theta} \right \rangle_{P(v,h|\theta)} \right\}
\end{aligned}
$$
简要介绍一下推导过程。第二行$\left\{\right\}$里面，对于**左边乘积**，对比数学期望的公式，乘号左边的是概率的表达式（因为分母是对所有$h$的能量求和而分子是其中某一个$h$的能量，他们相比就是概率），乘号右边的就是对应的随机变量$h$的能量值，我们知道此时左边乘积是$v$已知的情况下对$h$求和，因此正好符合期望公式，同时对应的分布概率为$P(h|v^{t},\theta)$，即他求的是边缘的期望。**右边乘积**求的是联合的期望，因此需要全部求和，且分布概率为$P(v,h|\theta)$。最后我们得到第三行的公式，代表的就是均值且分别符合各自分布的均值：

$$
\frac{\partial{L}}{\partial{\theta}}=\sum_{t=1}^T \left\{ {\left \langle \frac{\partial(-E(v^{(t)},h|\theta))}{\partial\theta}_{P(h|v^{(t)},\theta)} \right \rangle}-\left \langle \frac{\partial(-E(v,h|\theta))}{\partial\theta} \right \rangle_{P(v,h|\theta)} \right\} \tag{8}
$$

$\left \langle \bullet \right \rangle_P$表示求$\bullet$关于分布P的数学期望。$P(h|v^{(t)},\theta )$表示在可见单元限定为己知的训练样本$v^{(t)}$时：隐层的概率分布，故式（8）中的前一项比较容易计算。$P(v,h|\theta)$表示可见单元与隐单元的联合分布，由于归一化因子$Z(\theta)$的存在，该分布很难取，导致我们无法直接计算式（8）中的第二项，只能通过一些采样方法（如Gibbs采样）获取其近似值。值得指出的是，在最大化似然函数的过程中：为了加快计算速度，上述偏导数在每一迭代步中的计算一般只基于部分而非所有的训练样本进行，关于这部分内容我们将在后面讨论RBM的参数设置时详细阐述。

​		接下来我们把（1）式关于能量的定义带入（8）式然后对$W_{ij}$求偏导。（8）式第一项我们发现前两项为0，后面只剩下关于$v_i,h_j$的参数，这时是针对$P(h|v^{(t)},\theta)$概率分布来求的期望，同样第二项的结果也是只剩下关于$v_i,h_j$的参数，此时是针对$P(v,h|\theta)$概率分布来的。然后对另外两个参数求偏导，可分别得到对应的偏导。假设只有一个训练样本，我们分别用$data$和$model$来简记$P(h|v^{(t)},\theta )$和$P(v,h|\theta)$这两个概率分布，则对数释然函数关于连接权重$W_{ij}$、可见层单元的偏置$a_i$和隐层单元的偏置$b_j$的偏导数分别为：
$$
\begin{aligned}
\frac{\partial{logP(v|\theta)}}{\partial{W_{ij}}}&=\left\langle v_i h_j\right\rangle_{data}-\left\langle{v_i h_j}\right\rangle_{model}\\

\frac{\partial{logP(v|\theta)}}{\partial{a_i}}&=\left\langle v_i\right\rangle_{data}-\left\langle{v_i}\right\rangle_{model}\\

\frac{\partial{logP(v|\theta)}}{\partial{b_j}}&=\left\langle h_j\right\rangle_{data}-\left\langle{h_j}\right\rangle_{model}

\end{aligned}
$$
通过上面我们知道了学习函数，以及学习函数的难点在哪里，我们知道学习函数的难点在求均值时归一化因子$Z(\theta)$的计算量很大即整个模型的期望，计算量是$2^{m+n}$,无法求解，但是我们引入了$Gibbs$采样解决这个问题，即我通过采样去逼近这个均值，因为分布函数我们知道。只是计算量很大，因此使用统计的方法进行处理可以很好的解决问题，但是问题是计算$Gibbs$采样的计算量还是很大，如下图，需要迭代很多次才能达到热平衡状态，效率还是很低，这时候的问题是$\left \langle \bullet \right \rangle_{model}$通过$Gibbs$采样还是很大  怎么办呢？所以引入快速学习算法。

![Gibbs](https://raw.githubusercontent.com/red-fox-yj/MarkDownPic/master/typora/20210215095328.png)

## 基于对比散度的快速学习算法