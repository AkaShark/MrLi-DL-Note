
## Where dose the error come from ?

* error due to Bias
* error due to Variance

误差是来着这两个方面 
Bias：偏差
Variance： 方差 


### 抽样分布
实际上对应着物理实验中系统误差和随机误差的概念，假设有n组数据，每组数据都会产生一个相应的$f^*$, 此时**biase**表示所用$f^*$的平均落靶位置和真值靶心的距离, **variance**表示这些&f^*&集中程度

### 抽样分布的理论(概率论与数理统计)

看不懂 这是什么东西！！

$f^*$的个数取决于model的复杂程度以及data的数量

$f^*$的variance取决于model的复杂程度和data数量 

$f^*$的variance是由model决定的，一个简单的model在不同的training data下可以获得比较稳定分布的$f^*$，而复杂的model在不同的training data下的分布比较杂乱(如果data足够多，那复杂的model也可以得到比较稳定的分布)

复杂的model (model设置的比较复杂参数比较多) 的话$f^*$的分布就比较不均匀相反的分布就比较均匀

> 为什么会这样呢

原因其实很简单，其实前面在讲regularization正规化的时候也提到了部分原因。简单的model实际上就是没有高次项的model，或者高次项的系数非常小的model，这样的model表现得相当平滑，受到不同的data的影响是比较小的

$f^*$的bias只取决于model的复杂程度 

bias是表示，
![bias](/img/bias_loss.png)


![loss_variance](/img/loss_variance.png)
ps：上图分别是含有一次项，三次项和五次项的model试验的结果，

根据上图我们发现，当model比较简单的时候，每次试验等到的$f^*$之间的variance会比较小，这些$f^*$会稳定在一个范围内，但他们呢的平均值距离真实值会有比较大的偏差，相反的当modle比较复杂的时候，每次试验等到的$f^*$之间的variance会比较大，但实际体现出来的就是每次重新试验得到的$f^*$都会与之前得到的有比较大的差距，但是这些差距比较大的结果的平均值与真实值之间的偏差却很小 

复杂的model 需要的参数会更多


## Bias VS Variance

![Bias_VC_Variance](/img/bias_vs_variance.png)

上图是由上面的试验最高此项从一次到五次的loss展示图其中，绿色的线代表variance造成的error，红色的线代表bias造成的error，蓝色的线代表这个model实际观测到的error

$$error_{实际} = error_{variance} + error_{bias}$$

随着model逐渐复杂:

* bias逐渐减小
* variance逐渐变大
* 实际表现的error先是减小后是增大，因此error为最小值的那个点，即为bias和variance的error之和最小的地方，model表现的最好 

* 如果实际error来自variance很大，这状况就是overfitting过拟合，如果实际error来自于bias很大，这种情况就是欠拟合（这里的过拟合和欠拟合和之前理解的不同 ）

## Bias大还是Variance大

> 确定自己的error来自哪里

* 如果model没有办法fit trining data则表示bias比较大这时候是underfitting（整体的error还在一直的下降）需要调整model

* 如果model可以fit training data 在training data上得到了比较小的error 但是在testing data 或者 验证数据集上得到了比较大的error表示的是variance比较大，这时候是overfitting

## Feature Work

### bias比较大的情况
bias大表示现在的model没有需要的，问题来自于本身的model就不是很fit这个问题

* redesign 重新设计model
    * 增加更多的feature作为model的输入变量

    * 让model变得更加复杂化

### variance 比较大

* 增加data
    * 增大data是一个万能丹药
    * 没有足够的data可以采用自己去generate假数据 比如图片的旋转等

* Regularizer（正则化）

让你的model更加的平滑

Regularization 就是在loss func里面再加入一个与model高次项相关的term，他会希望你的model里面的高次项的参数越小越好，这个新加入的term前面可以有个weight，代表你希望你的曲线有多平滑

![regularization](/img/regularization.png)

加入regularization后如上图，regularization让你的model更加的平滑对于一些异常的数据更加不敏感，这样会使得model的variance变小但是相应的可能会伤害你的bias，因为他实际上调整了function set的sapce范围，变成他这能包含一些比较平滑的曲线，因此需要调整regularization的weight在variance和bias之前取得平衡这个过程如下图所示的样子

![regularization_weight](/img/regularization_weight.png)

1. 蓝色区域代表最初的情况，此时model比较复杂，func set的sapace的范围比较大，包含了target靶心，但由于data不够$f^*$比较分散，variance比较大

2. 红色区域表示通过regularization之后的情况，此时modle 的func set范围被缩小成只能包含平滑的曲线，space减小，variance也跟着变小，但是这个缩小后的space实际上并没有包含到原先的靶心位置，因此model的bias变大

3. 橙色区域表示增大了regularization的weight的情况，增大weight就是放大func set的space，慢慢调节到包含整个靶心，此时model的bias变小，而对于一开始的case，由于限定了曲线的平滑程度（weight控制）所以model的variance也比较小

通过Regularization优化model的过程就是上述的1 2 3 步骤， 不断地调整regularization的weight，使model的bias和variance达到一个最佳平衡的状态


## Model Selection
>ps: 要动用手里的testing data 去做数据的验证（Select Model 阶段）将training data 分成training set和validation set

* train set 用来训练model 
* validation set 用来验证model

用training set 去寻找func $f^*$用validation set 来选择最好的model

当你得到public set上的error的时候(尽管它可能会很大)，不建议回过头去重新调整model的参数，因为当你再回去重新调整什么东西的时候，你就又会把public testing set的bias给考虑进去了，这就又回到了第一种关系，即围绕着有偏差的testing data做model的优化

这样的话此时你在public set上看到的performance就没有办法反映实际在private set上的performance了，因为你的model是针对public set做过优化的，虽然public set上的error数据看起来可能会更好看，但是针对实际未知的private set，这个“优化”带来的可能是反作用，反而会使实际的error变大

### N-flod Cross Validation

![N-flod](/img/N-flod_CrossValidation.png)

## Conclusion

1. 一般来说，error是bias和variance共同作用的结果

2. model比较简单和比较复杂的情况：

* 当model比较简单的时候，variance比较小，bias比较大，此时$f^*$会比较集中，但是function set可能并没有包含真实值；此时model受bias影响较大

* 当model比较复杂的时候，bias比较小，variance比较大，此时function set会包含真实值，但是$f^*$会比较分散；此时model受variance影响较大
必须要调平滑的原因是train data 的数量不够

3. 区分bias大 or variance大的情况

* 如果连采样的样本点都没有大部分在model训练出来的$f^*$上，说明这个model太简单，bias比较大，是欠拟合

* 如果样本点基本都在model训练出来的$f^*$上，但是testing data上测试得到的error很大，说明这个model太复杂，variance比较大，是过拟合(仅仅只是在train data 上表现的很优秀)

4. bias大 or variance大的情况下该如何处理

* 当bias比较大时，需要做的是重新设计model，包括考虑添加新的input变量，考虑给model添加高次项；然后对每一个model对应的计算出error，选择error值最小的model(随model变复杂，bias会减小，variance会增加，因此这里分别计算error，取两者平衡点)

* 当variance比较大时，一个很好的办法是增加data(可以凭借经验自己generate data)，当data数量足够时，得到的实际上是比较集中的；如果现实中没有办法collect更多的data，那么就采用regularization正规化的方法，以曲线的平滑度为条件控制function set的范围，用weight控制平滑度阈值，使得最终的model既包含，variance又不会太大

5. 如何选择model

选择model的时候呢，我们手头上的testing data与真实的testing data之间是存在偏差的，因此我们要将training data分成training set和validation set两部分，经过validation挑选出来的model再用全部的training data（training data ，validation data合在一起）再次训练一遍参数，最后用testing data去测试error，这样得到的error是模拟过testing bias的error，与实际情况下的error会比较符合







