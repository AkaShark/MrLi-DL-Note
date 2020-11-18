# Regression
[视频地址](https://www.bilibili.com/video/BV1JE411g7XF?p=3)

Regression的显示应用

输入一种情况得到输出预测

* 股票的预测
* 自动驾驶的使用
* 推荐系统的使用

![Regression](/img/Regression.png)

### 问题引入

> 预测宝可梦CP值

通过宝可梦当前的信息预测他的进化后cp值的大小已确定是要进化

> 确定场景任务和模型(Scenario Task Model)

Scenario(根据已有的数据来确定)
我们拥有宝可梦进化前后的数据既拥有标注的数据集，所以使用的是监督学习

Task(根据想要的输出类型来确定)
我们想要进化后的cp是一个数值类型的因此使用的是Regression

Model(很多种选择，可以多次试验挑选交好的)
采用Non-linear Model

>参数说明
$X$: 表示一只宝可梦
$X_{cp}$: 表示宝可梦进化前的cp值
$f()$: 要找找到的func
$y$: func的输出进化后的cp是一个数值scalar

![Example](/img/Example.png)
![Estimating](/img/Estimating%20CP.png)

### ML的基本步骤

* step1 Model的选择
* step2 Loss Func 确定
    * 确定损失函数
    * 损失函数可视化
    * 选择最好的函数
* step3 Gradient Descent 寻找最好的func
* Back step1 Redesign the Model 重新设计model
* Back step2 Regularization
* for loop 

>Step1 Model

![Step1](/img/Step1.png)

linear Model 
$ y=b+w \cdot X_{cp} $

![linearModel](/img/linermodel.png)

$X_{i}$  an attribute of input X ( xi is also called feature，即特征值)
$w_{i}$：weight of xi
$b$ 是偏置 bias
![Xi](/img/Xi.png)

根据不同的w和b，可以确定不同的无穷无尽的function，而$y=b+w \cdot X_{cp}$这个抽象出来的式子就叫做model，是以上这些具体化的function的集合，即function set

可以将这个model写为
==$y=b+ \sum w_ix_i$==


> Step2 Goodness of Functuon

![Step2](/img/Step2.png)

参数说明

$X^i$: 表示第i只宝可梦(下标表示该object中的component)

$\widehat{y}^i$：用$\widehat{y}$表示一个实际观察到的object输出，上标为i表示是第i个object

注：由于regression的输出值是scalar，因此$\widehat{y}$里面并没有component，只是一个简单的数值；但是未来如果考虑structured Learning的时候，我们output的object可能是有structured的，所以我们还是会需要用上标下标来表示一个完整的output的object和它包含的component

![step2_1](/img/step2_1.png)

为了衡量function set中的某个function的好坏，我们需要一个评估函数，即==Loss function==，损失函数，简称L；loss function是一个function的function(输入是一个func，输出这个func的好坏程度)

由于$f:y=b+w \cdot x_{cp}$，即f是由b和w决定的，因此input f就等价于input这个f里的b和w，因此==Loss function实际上是在衡量一组参数的好坏==

loss function，最常用的方法就是采用类似于方差和的形式来衡量参数的好坏，即预测值与真值差的平方和；这里真正的数值减估测数值的平方，叫做**估测误差**，Estimation error，将10个估测误差求和起来就是loss function 的大小

$$ L(f)=L(w,b)=\sum_{n=1}^{10}(\widehat{y}^n-(b+w \cdot {x}^n_{cp}))^2$$
![loss func](/img/loss%20func.png)


如果$L(f)$越大，说明该function表现得越不好；$L(f)$越小，说明该function表现得越好，loss func要找的就是最优的参数也就是最优的model
![lossfunc_1](/img/lossfunc_1.png)

> loss func可视化

![lossfunc_2](/img/lossfunc_2.png)

> pick the best func

我们已经确定了loss function，他可以衡量我们的model里面每一个function的好坏，接下来我们要做的事情就是，从这个function set里面，挑选一个最好的function

挑选最好的function这一件事情，写成formulation/equation的样子如下：

$$f^*={arg} \underset{f}{min} L(f)$$，或者是

$$w^* ,b^*={arg}\ \underset{w,b}{min} L(w,b)={arg}\ \underset{w,b}{min} \sum\limits^{10}_{n=1}(\widehat{y}^n-(b+w \cdot x^n{cp}))^2$$

也就是那个使$L(f)=L(w,b)=Loss$最小的$f$或$(w,b)$，就是我们要找的$y^* $或$(w^* ,b^*)$(有点像极大似然估计的思想)

### Gradient Descent 

上面的例子比较简单，用线性代数的知识就可以解；但是对于更普遍的问题来说，==gradient descent的厉害之处在于，只要$L(f)$是可微分的，gradient descent都可以拿来处理这个$f$，找到表现比较好的parameters==

> 只带一个参数的问题

以只带单个参数w的Loss Function L(w)为例，首先保证$L(w)$是可微的 $$w^*={arg}\ \underset{w}{min} L(w) $$ 我们的目标就是找到这个使Loss最小的 $w^ * $，实际上就是寻找切线L斜率为0的**global minima**最小值点(注意，存在一些local minima极小值点，其斜率也是0)

有一个暴力的方法是，穷举所有的w值，去找到使loss最小的$w^*$，但是这样做是没有效率的；而gradient descent就是用来解决这个效率问题的

* 首先随机选取一个初始的点$w^0$ (当然也不一定要随机选取，如果有办法可以得到比较接近$w^*$的表现得比较好的$w^0$当初始点，可以有效地提高查找$w^ *$的效率)

* 计算$L$在$w=w^0$的位置的微分，即$\frac{dL}{dw}|_{w=w^0}$，几何意义就是切线的斜率

w的改变量step size的大小取决于两件事

    一、是现在的微分值$\frac{dL}{dw}$有多大，微分值越大代表现在在一个越陡峭的地方，那它要移动的距离就越大，反之就越小；

    二、是一个常数项$η$，被称为==learning rate==，即学习率，它决定了每次踏出的step size不只取决于现在的斜率，还取决于一个事先就定好的数值，如果learning rate比较大，那每踏出一步的时候，参数w更新的幅度就比较大，反之参数更新的幅度就比较小

如果learning rate设置的大一些，那机器学习的速度就会比较快；但是learning rate如果太大，可能就会跳过最合适的global minima的点


* 因此每次参数更新的大小是 $η \frac{dL}{dw}$，为了满足斜率为负时w变大，斜率为正时w变小，应当使原来的w减去更新的数值，即 $$ w^1=w^0-η \frac{dL}{dw}|_{w=w^0} \ w^2=w^1-η \frac{dL}{dw}|_{w=w^1} \ w^3=w^2-η \frac{dL}{dw}|_{w=w^2} \ ... \ w^{i+1}=w^i-η \frac{dL}{dw}|_{w=w^i} \  if\ \ (\frac{dL}{dw}|_{w=w^i}==0) \ \ then \ \ stop; $$ 此时$w^i$对应的斜率为0，我们找到了一个极小值local minima，这就出现了一个问题，当微分为0的时候，参数就会一直卡在这个点上没有办法再更新了，因此通过gradient descent找出来的solution其实并不一定是最佳解global minima

![Gradient Descent](/img/Gradient%20Descent.png)
![learning rate](/img/Learing%20rate.png)
![gradent descent_1](/img/gradent%20descent_1.png)
![gradient descent_2](/img/gradient%20descent_2.png)

> ps : 沿着梯度的负方向更新参数

两个参数的问题

今天要解决的关于宝可梦的问题，是含有two parameters的问题，即$(w^ *, b^ *)=arg\ \underset{w,b} {min} L(w,b)$

* 首先，也是随机选取两个初始值，$w^0$和$b^0$
* 然后分别计算$(w^0,b^0)$这个点上，L对w和b的偏微分，即$\frac{\partial L}{\partial w}|_{w=w^0,b=b^0}$ 和 $\frac{\partial L}{\partial b}|_{w=w^0,b=b^0}$
* 更新参数，当迭代跳出时，$(w^i,b^i)$对应着极小值点 
$$ w^1=w^0-η\frac{\partial L}{\partial w}|_{w=w^0,b=b^0} \\ b^1=b^0-η\frac{\partial L}{\partial b}|_{w=w^0,b=b^0} \\w^2=w^1-η\frac{\partial L}{\partial w}|_{w=w^1,b=b^1} \\ b^2=b^1-η\frac{\partial L}{\partial b}|_{w=w^1,b=b^1} \ \\... \\ w^{i+1}=w^{i}-η\frac{\partial L}{\partial w}|_{w=w^{i},b=b^{i}}  \\b^{i+1}=b^{i}-η\frac{\partial L}{\partial b}|_{w=w^{i},b=b^{i}} 
\\ if(\frac{\partial L}{\partial w}==0 \ and \ \frac{\partial L}{\partial b}==0) \ \ \ then \ \ stop $$

实际上，L 的gradient就是微积分中的那个梯度的概念，即 $$ \nabla L= \begin{bmatrix} \frac{\partial L}{\partial w} \ \frac{\partial L}{\partial b} \end{bmatrix}_{gradient} $$ 可视化效果如下：(三维坐标显示在二维图像中，loss的值用颜色来表示)

横坐标是b，纵坐标是w，颜色代表loss的值，越偏蓝色表示loss越小，越偏红色表示loss越大

每次计算得到的梯度gradient，即由$\frac{\partial L}{\partial b}和\frac{\partial L}{\partial w}$组成的vector向量，就是该等高线的法线方向(对应图中红色箭头的反方向)；而$(-η\frac{\partial L}{\partial b},-η\frac{\partial L}{\partial w})$的作用就是让原先的$(w^i,b^i)$ **朝着gradient的反方向**即等高线法线方向前进，其中η(learning rate)的作用是每次更新的跨度(对应图中**红色箭头的长度**)；经过多次迭代，最终gradient达到极小值点

![gradient descent_3](/img/gradient%20descent_3.png)

### Gradient Descent的缺点

gradient descent有一个令人担心的地方，它每次迭代完毕，寻找到的梯度为0的点必然是极小值点，local minima；却不一定是最小值点，global minima, 有可能就在极值点处停止了计算

这会造成一个问题是说，如果loss function长得比较坑坑洼洼(极小值点比较多)，而每次初始化$w^0$的取值又是随机的，这会造成每次gradient descent停下来的位置都可能是不同的极小值点；而且当遇到梯度比较平缓(gradient≈0)的时候，gradient descent也可能会效率低下甚至可能会stuck卡住；也就是说通过这个方法得到的结果，是看人品的

![gradent descent_4](/img/gradent%20descent_4.png)


现在我们来求具体的L对w和b的偏微分 $$ L(w,b)=\sum\limits_{n=1}^{10}(\widehat{y}^n-(b+w\cdot x_{cp}^n))^2 \ \\ \frac{\partial L}{\partial w}=\sum\limits_{n=1}^{10}2(\widehat{y}^n-(b+w\cdot x_{cp}^n))(-x_{cp}^n) \ \\ \frac{\partial L}{\partial b}=\sum\limits_{n=1}^{10}2(\widehat{y}^n-(b+w\cdot x_{cp}^n))(-1) $$

我们需要有一套评估系统来评价我们得到的最后这个function和实际值的误差error的大小；这里我们将training data里每一只宝可梦 $i$ 进化后的实际cp值与预测值之差的绝对值叫做$e^i$，而这些误差之和Average Error on Training Data为$\sum\limits_{i=1}^{10}e^i=31.9$

当然我们真正关心的是generalization的case，也就是用这个model去估测新抓到的pokemon，误差会有多少，这也就是所谓的testing data的误差；于是又抓了10只新的pokemon，算出来的Average Error on Testing Data为$\sum\limits_{i=1}^{10}e^i=35.0$；可见training data里得到的误差一般是要比testing data要小，这也符合常识

##### How can we do better?
我们有没有办法做得更好呢？这时就需要我们重新去设计model；如果仔细观察一下上图的data，就会发现在原先的cp值比较大和比较小的地方，预测值是相当不准的

实际上，从结果来看，最终的function可能不是一条直线，可能是稍微更复杂一点的曲线

![x2](/img/x2.png)
![x3](/img/x3.png)
![x4](/img/x4.png)
![x5](/img/x5.png)

###### 5个model的对比
这5个model的training data的表现：随着$(x_{cp})^i$的高次项的增加，对应的average error会不断地减小；实际上这件事情非常容易解释，实际上低次的式子是高次的式子的特殊情况(令高次项$(X_{cp})^i$对应的$w_i$为0，高次式就转化成低次式)

也就是说，在gradient descent可以找到best function的前提下(多次式为Non-linear model，存在local optimal局部最优解，gradient descent不一定能找到global minima)，function所包含的项的次数越高，越复杂，error在training data上的表现就会越来越小；但是，我们关心的不是model在training data上的error表现，而是model在testing data上的error表现


在training data上，model越复杂，error就会越低；但是在testing data上，model复杂到一定程度之后，error非但不会减小，反而会暴增，在该例中，从含有$(X_{cp})^4$项的model开始往后的model，testing data上的error出现了大幅增长的现象，通常被称为**overfitting过拟合**

![overfitting](/img/overfitting.png)

因此model不是越复杂越好，而是选择一个最适合的model，在本例中，包含$(X_{cp})^3$的式子是最适合的model

#### 重新设计model
![xs](/img/xs.png)

$x_{s}$ 表示物种

$$ if \ \ x_s=Pidgey: \ \ \ \ \ \ \ y=b_1+w_1\cdot x_{cp} \\ if \ \ x_s=Weedle: \ \ \ \ \ \ y=b_2+w_2\cdot x_{cp} \\ if \ \ x_s=Caterpie: \ \ \ \ y=b_3+w_3\cdot x_{cp} \\ if \ \ x_s=Eevee: \ \ \ \ \ \ \ \ \ y=b_4+w_4\cdot x_{cp} $$ 也就是根据不同的物种，设计不同的linear model(这里$x_s=species \ of \ x$


考虑所有可能设计出更复杂的model
![modle_2](/img/model_2.png)
这样大概率会产生过拟合

#### 重新设计损失函数

原来的loss function只考虑了prediction的error，即$\sum\limits_i^n(\widehat{y}^i-(b+\sum\limits_{j}w_jx_j))^2$；而regularization则是在原来的loss function的基础上加上了一项$\lambda\sum(w_i)^2$，就是把这个model里面所有的$w_i$的平方和用λ加权(其中i代表遍历n个training data，j代表遍历model的每一项)

因为需要的
$$L = \sum\limits_{n}(\widehat{y}^n - (b + \sum w_ix_i)) $$ 损失函数的值最小，接近于零所以加上$$\lambda\sum(w_i)^2$$则是希望$w$越小参数的值接近于0这样的话保证函数是平滑的

$$L = \sum\limits_{n}(\widehat{y}^n - (b + \sum w_ix_i))+\lambda\sum(w_i)^2 $$


也就是说，我们期待参数$w_i$越小甚至接近于0的function，为什么呢？

因为参数值接近0的function，是比较平滑的；所谓的平滑的意思是，当今天的输入有变化的时候，output对输入的变化是比较不敏感的

举例来说，对$y=b+\sum w_ix_i$这个model，当input变化$\Delta x_i$，output的变化就是$w_i\Delta x_i$，也就是说，如果$w_i$越小越接近0的话，输出对输入就越不sensitive敏感，我们的function就是一个越平滑的function；说到这里你会发现，我们之前没有把bias——b这个参数考虑进去的原因是bias的大小跟function的平滑程度是没有关系的，bias值的大小只是把function**上下移动**而已

如果我们有一个比较平滑的function，由于输出对输入是不敏感的，测试的时候，一些noises噪声对这个平滑的function的影响就会比较小，而给我们一个比较好的结果


![regularization](/img/regularization.png)

**注：这里的λ需要我们手动去调整以取得最好的值**

λ值越大代表考虑smooth的那个regularization那一项的影响力越大，我们找到的function就越平滑

观察下图可知，当我们的λ越大的时候，在training data上得到的error其实是越大的，但是这件事情是非常合理的，因为当λ越大的时候，我们就越倾向于考虑w的值而越少考虑error的大小；但是有趣的是，虽然在training data上得到的error越大，但是在testing data上得到的error可能会是比较小的

下图中，当λ从0到100变大的时候，training error不断变大，testing error反而不断变小；但是当λ太大的时候(>100)，在testing data上的error就会越来越大

==我们喜欢比较平滑的function，因为它对noise不那么sensitive；但是我们又不喜欢太平滑的function，因为它就失去了对data拟合的能力；而function的平滑程度，就需要通过调整λ来决定==，就像下图中，当λ=100时，在testing data上的error最小，因此我们选择λ=100

注：这里的error指的是$\frac{1}{n}\sum\limits_{i=1}^n|\widehat{y}^i-y^i|$


![regularization_!](/img/regularization_1.png)


#### 总结

* 根据已有的data特点(labeled data，包含宝可梦及进化后的cp值)，确定使用supervised learning监督学习

* 根据output的特点(输出的是scalar数值)，确定使用regression回归(linear or non-linear)

* 考虑包括进化前cp值、species、hp等各方面变量属性以及高次项的影响，我们的model可以采用这些input的一次项和二次型之和的形式，如： 
$$ if \ \ x_s=Pidgey: \ \ \ \ \ \ \ y=b_1+w_1\cdot x_{cp} \\ if \ \ x_s=Weedle: \ \ \ \ \ \ y=b_2+w_2\cdot x_{cp} \\ if \ \ x_s=Caterpie: \ \ \ \ y=b_3+w_3\cdot x_{cp} \\ if \ \ x_s=Eevee: \ \ \ \ \ \ \ \ \ y=b_4+w_4\cdot x_{cp} $$ 

* 而为了保证function的平滑性，loss function应使用regularization，即$$L=\sum\limits_{i=1}^n(\widehat{y}^i-y^i)^2+\lambda\sum\limits_{j}(w_j)^2$$，注意bias——参数b对function平滑性无影响，因此不额外再次计入loss function(y的表达式里已包含w、b)

* 利用gradient descent对regularization版本的loss function进行梯度下降迭代处理，每次迭代都减去L对该参数的微分与learning rate之积，假设所有参数合成一个vector：$[w_0,w_1,w_2,...,w_j,...,b]^T$，那么每次梯度下降的表达式如下： $$ 梯度:\ \ \nabla L= \begin{bmatrix} \frac{\partial L}{\partial w_0} \ \frac{\partial L}{\partial w_1} \ \frac{\partial L}{\partial w_2} \ ... \ \frac{\partial L}{\partial w_j} \ ... \ \frac{\partial L}{\partial b} \end{bmatrix}_{gradient} \ \ \ \\ gradient \ descent: \begin{bmatrix} w'_0\ w'1\ w'2\ ...\ w'j\ ...\ b' \end{bmatrix}{L=L'} = \ \begin{bmatrix} w_0\ w_1\ w_2\ ...\ w_j\ ...\ b \end{bmatrix}{L=L_0} -\ \ \ \ \eta \begin{bmatrix} \frac{\partial L}{\partial w_0} \ \frac{\partial L}{\partial w_1} \ \frac{\partial L}{\partial w_2} \ ... \ \frac{\partial L}{\partial w_j} \ ... \ \frac{\partial L}{\partial b} \end{bmatrix}{L=L_0} $$ 
也就是
$$ w' = w - \eta \nabla{L} $$
当梯度稳定不变时，即$\nabla L$为0时，gradient descent便停止，此时如果采用的model是linear的，那么vector必然落于global minima处(凸函数)；如果采用的model是Non-linear的，vector可能会落于local minima处(此时需要采取其他办法获取最佳的function),假定我们已经通过各种方法到达了global minima的地方，此时的vector：$[w_0,w_1,w_2,...,w_j,...,b]^T$所确定的那个唯一的function就是在该λ下的最佳$f^*$，即loss最小

* 这里λ的最佳数值是需要通过我们不断调整来获取的，因此令λ等于0，10，100，1000，...不断使用gradient descent或其他算法得到最佳的parameters：$[w_0,w_1,w_2,...,w_j,...,b]^T$，并计算出这组参数确定的function——$f^*$对training data和testing data上的error值，直到找到那个使testing data(验证集)的error最小的λ，(这里一开始λ=0，就是没有使用regularization时的loss function)


>注：引入评价$f^* $的error机制，令error=$\frac{1}{n}\sum\limits_{i=1}^n|\widehat{y}^i-y^i|$，分别计算该$f^* $对training data和testing data(more important)的$error(f^*)$大小

先设定λ->确定loss function->找到使loss最小的$[w_0,w_1,w_2,...,w_j,...,b]^T$->确定function->计算error->重新设定新的λ重复上述步骤->使testing data上的error最小的λ所对应的$[w_0,w_1,w_2,...,w_j,...,b]^T$所对应的function就是我们能够找到的最佳的function

这里的测试集指的是验证集 
不应该在测试集上进行先关的操作

