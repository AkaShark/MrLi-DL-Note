# Gradient Descent 

目标
$\theta$ 是参数 找到一个最好的参数使得loss func 最小 

$$\theta^* = \arg\min_{\theta}L(\theta)$$

假设$\theta$是参数的集合：Suppose that $\theta$ has two variables 

$$\left\{ \theta_{1}, \theta_{2}
\right\}
$$

随机选取一组起始的参数：Randomly start at $\theta^{0}=\left[\begin{array}{l}{\theta_{1}^{0}} \\ {\theta_{2}^{0}}\end{array}\right] \quad$

计算$\theta$处的梯度gradient：$\nabla L(\theta)=\left[\begin{array}{l}{\partial L\left(\theta_{1}\right) / \partial \theta_{1}} \\ {\partial L\left(\theta_{2}\right) / \partial \theta_{2}}\end{array}\right]$

$\left[\begin{array}{l}{\theta_{1}^{1}} \\ {\theta_{2}^{1}}\end{array}\right]=\left[\begin{array}{l}{\theta_{1}^{0}} \\ {\theta_{2}^{0}}\end{array}\right]-\eta\left[\begin{array}{l}{\partial L\left(\theta_{1}^{0}\right) / \partial \theta_{1}} \\ {\partial L\left(\theta_{2}^{0}\right) / \partial \theta_{2}}\end{array}\right] \Rightarrow \theta^{1}=\theta^{0}-\eta \nabla L\left(\theta^{0}\right)$

$\left[\begin{array}{c}{\theta_{1}^{2}} \\ {\theta_{2}^{2}}\end{array}\right]=\left[\begin{array}{c}{\theta_{1}^{1}} \\ {\theta_{2}^{1}}\end{array}\right]-\eta\left[\begin{array}{c}{\partial L\left(\theta_{1}^{1}\right) / \partial \theta_{1}} \\ {\partial L\left(\theta_{2}^{1}\right) / \partial \theta_{2}}\end{array}\right] \Rightarrow \theta^{2}=\theta^{1}-\eta \nabla L\left(\theta^{1}\right)$

![gradient_descent](/img/gradient_descent.jpg)

上图是将gradient descent在投影到二维坐标系可视化的样子，图上的每个点都是$\left(\theta_1, \theta_2\right), loss$在该平面的投影

红色箭头是指在$(\theta_1, theta_2)$这点的梯度，梯度方方向即箭头方向，梯度大小即箭头长度

蓝色曲线代表实际情况下参数$\theta_1$和$\theta_2$的更新过程图，每次更新沿着蓝色箭头方向loss会减小，蓝色箭头方向和红色箭头刚好相反，代表着梯度下降的方向 


因此，在整个gradient descent的过程中，梯度不一定是递减的(红色箭头的长度可以长短不一)，但是沿着梯度下降的方向，函数值loss一定是递减的，且当gradient=0时，loss下降到了局部最小值，总结：梯度下降法指的是函数值loss随梯度下降的方向减小

参数的更新是沿着梯度的反方向，步长是由梯度大小和学习率共同决定的，当某次gradient = 0 说明了达到了局部最小值

gradient更重要的是他的方向而不是他的大小

## Learning rate 存在的问题

![learning_rate](/img/learning_rate.jpg)

* 如果learning rate刚刚好，就可以像下图中红色线段一样顺利地到达到loss的最小值

* 如果learning rate太小的话，像下图中的蓝色线段，虽然最后能够走到local minimal的地方，但是它可能会走得非常慢，以至于你无法接受

* 如果learning rate太大，像下图中的绿色线段，它的步伐太大了，它永远没有办法走到特别低的地方，可能永远在这个“山谷”的口上振荡而无法走下去

* 如果learning rate非常大，就会像下图中的黄色线段，一瞬间就飞出去了，结果会造成update参数以后，loss反而会越来越大(这一点在上次的demo中有体会到，当lr过大的时候，每次更新loss反而会变大)

> gradient descent一个很重要的事情是，要把不同的learning rate下，loss随update次数的变化曲线给可视化出来，它可以提醒你该如何调整当前的learning rate的大小，直到出现稳定下降的曲线


## Adaptive Learning rates

自动去调节learning rate 最简单 最基本的方法是learning rate 通常是随着参数的update越来越小的

例如可以这样设计learning rate

$\eta^{t} = \frac{\eta^{t-1}}{\sqrt{t+1}}$

> Vanilla Gradient descent 

这种方法使得所有参数以同样的方式同样的learning rate进行update，而最好的状况是每个参数都给他不用的learning rate 去做更新 

### Adagrad 

> Divide the learning rate of each parameter by the root mean square(方均根) of its previous derivatives

![Adagrad](/img/Adagrad.jpg)

$\sigma^t$表示之前所有loss对w偏微分的root mean squart 这个值对于每个参数来说都是不一样的

![Adagrad_1](/img/Adagrad_1.jpg)


由于$\eta^t$和$\sigma^t$中都有一个$\sqrt{\frac{1}{1+t}}$的因子，两者相消，即可得到adagrad的最终表达式：

$$w^{t+1}=w^t-\frac{\eta}{\sum\limits_{i=0}^t(g^i)^2}\cdot g^t$$

$g^i$ 是第i次对于w计算的梯度 

其实就是学习率变为
$$
\frac{\eta}{\sum\limits_{i=0}^t(g^i)^2}
$$
相当于原来的学习率除以了之前所有梯度的平方和 



![comparsion](/img/comparsion.jpg)

> gradient越大，离最低点越远这件事情在有多个参数的情况下是不一定成立的

可以看出，比起a点，c点距离最低点更近，但是它的gradient却越大

对于一个二次函数$y=ax^2+bx+c$来说，最小值点的$x=-\frac{b}{2a}$，而对于任意一点$x_0$，它迈出最好的步伐长度是$|x_0+\frac{b}{2a}|=|\frac{2ax_0+b}{2a}|$(这样就一步迈到最小值点了)，联系该函数的一阶和二阶导数$y'=2ax+b$、$y''=2a$，可以发现**the best step is $|\frac{y'}{y''}|$**，也就是说他不仅跟一阶导数(gradient)有关，还跟二阶导数有关，因此我们可以通过这种方法重新比较上面的a和c点，就可以得到比较正确的答案

其中$g^t$就是一次微分，而分母中的$\sum\limits_{i=0}^t(g^i)^2$反映了二次微分的大小，所以Adagrad想要做的事情就是，在不增加任何额外运算的前提下，想办法去估测二次微分的值

![second](/img/second.jpg)


## Stockastic Gradicent Descent

随机梯度下降的方法可以让训练更加迅速，传统gradient descent思路是看完所有的样本点之后再去构建loss func
然后去update参数，而stocastic gradient descent 的做法是，看到一样本点就update一次，因此他的loss func不是所有样本点的error平方和，而是这个随机样本点的error平方


![stochastic_Gradient](/img/stochastic_Gradient.jpg)

SGD 计算的会更快速 faster

## Frature Scaling

特征缩放 

![stochastic_Gradient](/img/stochastic_Gradient.jpg)

特征缩放，当多个特征的分布范围很不一样时，最好将这些不同feature的范围缩放成一样

![feature_scaling](/img/feature_scaling.jpg)

左边的error surface表示，w1对y的影响比较小，所以w1对loss是有比较小的偏微分的，因此在w1的方向上图像是比较平滑的；w2对y的影响比较大，所以w2对loss的影响比较大，因此在w2的方向上图像是比较sharp的
如果x1和x2的值，它们的scale是接近的，那么w1和w2对loss就会有差不多的影响力，loss的图像接近于圆形，那这样做对gradient descent有什么好处呢？

**对gradient descent的帮助**

之前我们做的demo已经表明了，对于这种长椭圆形的error surface，如果不使用Adagrad之类的方法，是很难搞定它的，因为在像w1和w2这样不同的参数方向上，会需要不同的learning rate，用相同的lr很难达到最低点

如果有scale的话，loss在参数w1、w2平面上的投影就是一个正圆形，update参数会比较容易

而且gradient descent的每次update并不都是向着最低点走的，每次update的方向是顺着等高线的方向(梯度gradient下降的方向)，而不是径直走向最低点；但是当经过对input的scale使loss的投影是一个正圆的话，不管在这个区域的哪一个点，它都会向着圆心走。因此feature scaling对参数update的效率是有帮助的


![feature_scaling](/img/feature_scaling_1.jpg)


假设有R个example(上标i表示第i个样本点)，$x^1,x^2,x^3,...,x^r,...x^R$，每一笔example，它里面都有一组feature(下标j表示该样本点的第j个特征)

对每一个demension i，都去算出它的平均值mean=$m_i$，以及标准差standard deviation=$\sigma_i$

对第r个example的第i个component，减掉均值，除以标准差，即$x_i^r=\frac{x_i^r-m_i}{\sigma_i}$

实际上就是==将每一个参数都归一化成标准正态分布，即$f(x_i)=\frac{1}{\sqrt{2\pi}}e^{-\frac{x_i^2}{2}}$==，其中$x_i$表示第i个参数

对于每一个example的每一个对应的feature 做 标准化 减去均值 除以方差 以达到标准化的地步 （服从正太分布 ）

## 标记这个将的gradient Descent 的原理 没有看 等回头补下看下对应的ref 

## Gradient Descent的限制 

![More_Limitation_of_Gradient_Descent](/img/More_Limitation_of_Gradient_Descent.jpg)


gradient descent的限制是，它在gradient即微分值接近于0的地方就会停下来，而这个地方不一定是global minima，它可能是local minima，可能是saddle point鞍点，甚至可能是一个loss很高的plateau平缓高原


