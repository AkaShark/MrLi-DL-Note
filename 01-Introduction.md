# 01 课程介绍

## 总览

介绍了机器学习的相关概念，同时直白的介绍了机器学习的一些方法，以及分类，和一些前沿的领域，同时介绍了机器学习的一些任务，以及这些任务的关系，十分的通俗易懂，既包含了一些基础的内容，也囊括了前沿的一些内容。

总体来说这节课程是入门介绍课程，让你知道机器学习是什么，了解机器学习的一些任务，前沿知识等。

### Learning Map

介绍机器学习的相关概念，给出了一张学习图，其中蓝色快代表着场景，也就是我要做任务的大背景，比如我要处理一个没有标注数据的任务，我们可能会选择无监督学习，又如我们的数据量很少的情况下我们会选择强化学习(Reinforcement Learning)。

红色快是task也就是我们要处理的任务，根据我们需求对应着不同的任务如要做分类的话就是classification的task，如果要做预测输出是数值的话需要的就是regression

绿色块是我们选择的model，用来解决问题的模型（func set）

根据下图，可发现Learing Theory对应着最大的块，蓝色的场景是第二大的块，往下逐步细分为任务和解决方法。同样可以看到在一种场景下可以有多个任务，一个任务下可以有多个解决task的model。

![Learning Map](./img/Learning%20Map.png)


### Supervised Learing （监督学习）

supervised Learing 需要大量的标注数据集，这里的标注意思是我们给定一个model输入的同时给出正确的输出，也就是说我们要找到一个func，他的input和output之间的关系，在给出input的同时给出output

### Regression (回归)
回归model是通常解决预测任务的，他的一个特点是model输出的是连续的数值，当然Regression属于监督学习

### Classification(分类)
CLassification解决的任务是分类问题，比如新闻类型的分类，垃圾邮件的分类，classification的特点是输出的是一个数，或者一个向量或者一些不等的概率。

Classification有分为两种:

* Binary Classification（二分类）
在二分类中，我们需要model输出的是yes or no的回答，也就是0或1，常见的问题是垃圾邮件的分类

* Multi Classifiction(多分类)
多分类是让model做选择题，给定了选择项（类别），让model选择一个可能性最大的输出（类别），常见的问题是新闻文章分类，输入一个新闻输出他属于那种类型


### Semi-supervised Learning(半监督学习)

手头上有少量的labeled data，它们标注了图片上哪只是猫哪只是狗；同时又有大量的unlabeled data，它们仅仅只有猫和狗的图片，但没有标注去告诉机器哪只是猫哪只是狗

在Semi-supervised Learning的技术里面，这些没有labeled的data，对机器学习也是有帮助的

### Transfer Learning(迁移学习)
假设一样我们要做猫和狗的分类问题

我们也一样只有少量的有labeled的data；但是我们现在有大量的不相干的data(不是猫和狗的图片，而是一些其他不相干的图片)，在这些大量的data里面，它可能有label也可能没有label

Transfer Learning要解决的问题是，这一堆不相干的data可以对结果带来什么样的帮助


### Unsupervised Learning(无监督学习)
区别于supervised learning，unsupervised learning希望机器学到无师自通，在完全没有任何label的情况下，机器到底能学到什么样的知识。

举例来说，如果我们给机器看大量的文章，机器看过大量的文章之后，它到底能够学到什么事情？它能不能学会每个词汇的意思或者说机器能学到什么知识。

### Structured Learning(结构化学习)

在structured Learning里，我们要机器输出的是，一个有结构性的东西

在分类的问题中，机器输出的只是一个选项；在structured类的problem里面，机器要输出的是一个复杂的组件

举例来说，在语音识别的情境下，机器的输入是一个声音信号，输出是一个句子；句子是由许多词汇拼凑而成，它是一个有结构性的object

比如GAN也是structured Learning的一种方法

![Structured Learning](./img/Structured%20Learing.png)

### Reinforcement Learning(强化学习)

Reinforcement Learning：我们没有告诉机器正确的答案是什么，机器最终得到的只有一个分数，就是它做的好还是不好，但他不知道自己到底哪里做的不好，他也没有正确的答案；很像真实社会中的学习，你没有一个正确的答案，你只知道自己是做得好还是不好。其特点是Learning from critics(循环一只学不断试错不断改进)

比如训练一个聊天机器人，让它跟客人直接对话；如果客人勃然大怒把电话挂掉了，那机器就学到一件事情，刚才做错了，它不知道自己哪里做错了，必须自己回去反省检讨到底要如何改进，比如一开始不应该打招呼吗？还是中间不能骂脏话之类的

![Reinforcement Learning](./img/Reinforcement%20Learing.png)

再拿下棋这件事举例，supervised Learning是说看到眼前这个棋盘，告诉机器下一步要走什么位置；而reinforcement Learning是说让机器和对手互弈，下了好几手之后赢了，机器就知道这一局棋下的不错，但是到底哪一步是赢的关键，机器是不知道的，他只知道自己是赢了还是输了

其实Alpha Go是用supervised Learning+reinforcement Learning的方式去学习的，机器先是从棋谱学习，有棋谱就可以做supervised的学习；之后再做reinforcement Learning，机器的对手是另外一台机器，Alpha Go就是和自己下棋，然后不断的进步(对抗学习)

### Generation(生成式学习)
给定modle 一定的数据让机器去自己创造新的内容知识

### Explainable AI
可解释的Ai 很热门的领域

### Adversarial Attack
新方向 对抗攻击
[参考](https://zhuanlan.zhihu.com/p/49755857)

### Network Compression 

### Meta Learning 
元学习

### life-long Learning
终生学习



