---
title: 基于主题的情感分析
date: 2021-11-26 22:12:40
tags: [nlp]
---


# **基于主题的情感分析**

[基于主题情感分类Github](git@github.com:ppphhhleo/SentiClassify.git)

aspect-based sentiment
analysis 是文本分类的一个子任务。文本分类是 NLP
的基础任务，旨在对给定文本预测其类别，应用场景广泛，比如垃圾邮件分类，微博情感分类，外卖评论，电影评论分类等。

**输入**：一段文本； **输出**：主题-极性，比如：服务-积极（ positive
）、味道-消极（ negative ）、价格-中立（ neutral ）

<br />

---


## **QUICK START**

-   **文件**

    -   **/Data，测试集 训练集**

    -   **/Emb，已训练好的词向量，词典**

-   **main.py ，可选择模型** 

**LSTM，AELSTM（添加aspect embedding），ATLSTM（添加attention机制），ATAELSTM**  
</br>
选择完成后，运行main.py即开始训练模型

![](media/e64c9609bfeab6835dfb862b97ae4272.png)

-   **model.py 定义以上模型**

 包括如何实现aspect 嵌入、注意力机制等 



-   **数据记录和作图，保存在record.ipynb文件中。**

---
<br />

## **详细描述**
### **0 数据集**

**SemEval2014 task4**

数据集主要用于细粒度情感分析，包含“Laptop”和“Restaurant”两个领域。

单条数据集，形式如：{**"sentence"**: "desserts are almost incredible: my
personal favorite is their tart of the day.", **"aspect"**: "food",
**"sentiment"**:
"positive"}，包含文本内容**sentence**、主题**aspect**和情感极性**sentiment**，其中，情感极性分为积极**positive、消极negative**和**中立neutral**三个水平。我们的目标是根据给定主题，来识别句子在该主题下包含的情感极性。

| Aspect            | Positive | Negative | Neural |      |       |      |
|-------------------|----------|----------|--------|------|-------|------|
|                   | Train    | Test     | Train  | Test | Train | Test |
| **Food**          | 867      | 302      | 209    | 69   | 90    | 31   |
| **Price**         | 179      | 51       | 115    | 28   | 10    | 1    |
| **Service**       | 324      | 101      | 218    | 63   | 20    | 3    |
| **Ambience**      | 263      | 76       | 98     | 21   | 23    | 8    |
| **Miscellaneous** | 546      | 127      | 199    | 41   | 357   | 51   |
| **Total**         | 2179     | 657      | 839    | 222  | 500   | 94   |

 
<br />
<br />


### **1 情感分析任务描述**

基于主题的情感分析（aspect-based sentiment analysis）有两个子任务：

-   **ATSA (aspect-term sentiment analysis)**

这个任务的目的是预测与文本中出现的**目标实体相关的情感极性**。如：“这个餐厅的味道很不错，但是太贵了”，对于这个句子，我们对“味道”
进行情感分析，并返回结果“{味道: positive}”。

-   **ACSA (aspect-category sentiment analysis)**

这个任务的目的是预测与**给定的主题相关的情感极性**。如：“这个餐厅的味道很不错，但是太贵了”，对于这个句子，我们对“价格”
进行情感分析，并返回结果“{价格: negative}”。

---
<br />
<br />

### **2 复现流程**

#### **a. 安装并导入相关的深度学习库。**

**Pytorch**

#### **b. 访问原始数据集，加载和处理数据**

data_process.py
中定义MyDataset类：定义三个列表，分别用于存储数据集中单条数据的**“sentence”**内容、**“aspect”**主题和**“sentiment”**情感极性，其中情感极性用**0、1、2**分别表示**negative、positive、neutral**；获取数据长度；定义根据index获取数据。

![](media/c8c6a1710ca55363c542c03ae0c34ac8.png)

#### **c. 数据加载**

#### **d. 定义模型**

**model.py 内定义了LSTM、Aspect Embedding-LSTM、Attention Aspect
Embedding-LSTM、AT-LSTM**

#### **e. 定义损失函数和优化器**

![](media/d13576500c9259dc704036c15c016aba.png)

#### **f. 训练和测试模型**


</br>

---

</br>


### **2 模型分析**

#### **（1）LSTM（Long Short-term Memory）**

循环神经网络
(RNN)，是传统前馈神经网络的扩展。为解决标准RNN存在梯度消失或梯度爆炸问题，长短期记忆神经网络架构被提出，并取得更佳的表现结果。

在LSTM架构中，有记忆状态cell state和三道控制门。

**状态向量**cell state表示LSTM本身的状态信息，一步一步往下传。

**遗忘门forget gate**，ft，将上一轮输出和本轮x混合，再Sigmoid激活函数处理。

**输入门 input gate**，it，将上一轮输出和本轮x混合，再利用tanh函数处理。

**![图1-1 遗忘门](media/2cff921d72e7bcf2f8a0ccc433e3c727.png) 
![图1-2 输入门](media/a5c295b0f49be7685ede0503b3a4bdca.png)**



**更新cell
state**，遗忘门控制上一个历史状态（做点乘），输入门控制本轮次的状态Ct（做点乘，上一轮的输出和本轮x混合的tanh），再把遗忘门和输入门做加法。

**输出门output gate**，**ot**，ht输出由ot控制（0\~1），更得到怎样的状态向量。

**![图1-3 更新](media/fd8391bffde64ff2a0dc09f0870c2a95.png) 
![图1-4 输出门](media/48830b3b8c1b5c8412ebb8829ab91c64.png)**


**每一步的LSTM可以按如下公式计算：**

![](media/0d5a561d876cf4a5a08821dec1ac207a.png)

**总而言之，LSTM可以整体架构如下图所示：**

**![图1-5 标准LSTM架构](media/245879326d998b05fca24db3001a58bc.png)**



#### **（2）AE-LSTM （LSTM with Aspect Embedding）**

在对一个句子给定的主题进行分类时，数据集中的主题内容是最重要的。如果考虑多个**不同**的主题，我们可能会得到**相反的情感极性**。为了充分利用主题信息，提出对每个主题都构造嵌入向量。

向量vai ∈ Rda 代表主题（aspect）的嵌入，其中da是主题嵌入（aspect
embedding）的维度。A ∈ Rda×\|A\|
由所有主题嵌入组成。本次实验复现的论文，据其团队所了解，是首次提出主题方面嵌入（aspect
embedding）。

#### **（3）AT-LSTM （Attention-based LSTM）**

对于基于主题的情感分析，标准LSTM模型无法检测文本中的最重要部分。为了能获取最重要的内容主题，引入基于LSTM模型的注意力机制。

基于注意力机制的LSTM模型架构，如图1-6所示。主题嵌入决定注意力权重和句子表示，{w1,
w2，…，
wN}表示所在句子长度为N的词向量；vα代表主题嵌入，注意力机制会生成一个注意力权重向量α和一个加权隐藏结果γ；{h1,
h2，…， hN}为隐藏向量。

![图1-6 Attention-based LSTM架构](media/3f13c2943347db957da4839003966dbc.png)



除了LSTM模型的基本处理，之后，**注意力机制**还会完成以下处理：

![](media/6e51c2ba2bf20eb9e8fd13ff48188179.png)

其中，**M ∈ R(d+da)×N , α ∈ RN , r ∈ Rd**，且**Wh ∈ Rd×d , Wv ∈ Rda×da 和** **w
∈ Rd+da**属于投影参数（projection parameters）。α 是一个包含注意力权重的向量，γ
是一个具有给定主题句子的加权表达（weighted representation of sentence with given
aspect）。算式7中，包含**va ⊗ eN = [v; v; . . . ; v]**运算，⊗
操作符意味着重复连接v向量N次，运算中的eN是一个N维列向量，**Wvva ⊗
eN**重复线性变换**va（linea**，句子次数就是变换次数。

最终的句子表示为下式（10），其中**h∗ ∈ Rd**，Wp 和 Wx
是训练时要学习的投影参数。添加**WxhN**到最终句子表示中，运行效果更好。

![](media/bc1bd0e106eb43f666f38fb8724d133f.png)

当需要考虑不同的主题aspect，注意力机制能让模型捕捉到句子中最重要的部分。

经过Attention-Based
LSTM运算处理得到的最终句子表达**h\***，是一个**给定输入主题的句子的特征表达。**我们增加一个**线性层**，将句子向量转换至e，是一个长度等于
\|C\| 的实值向量。接着，再连接**softmax层**归一化，将e转换至条件概率分布。

![](media/6e608a007f67a0ddfd67ffa5a0dbab5e.png)

其中Ws 和 bs 是softmax层参数。

#### **（4）ATAE-LSTM （Attention-based LSTM with Aspect Embedding）**

在AE-LSTM中利用主题信息（aspect information）的方法，是让主题嵌入（aspect
embedding）在注意力权重（attention
weight）的计算中发挥作用。为了更好地利用句子中所包含的主题信息，我们将**输入主题嵌入**（input
aspect embedding），附加到**每个词的输入向量**（input
vector）中。如此，通过LSTM模型，得到的输出隐藏表示（h1，h2，…,hn）将包含输入层的主题输入vα（input
aspect）。因此，在之后计算注意力权重的步骤中

![图1-7 Attention-based LSTM with Aspect Embedding (ATAE-LSTM)](media/b360c90a56134d2b5d0e5354a8fecf4e.png)



本次实验在model.py文件中import
torch.nn，定义了LSTM、AE-LSTM、ATAE-LSTM三个模型，均使用预训练词向量，基于单向LSTM模型。**并且，词向量维度、主题嵌入（aspect
embedding）和隐藏层（hidden
layer）大小均为300**；对于AE-LSTM和ATAE-LSTM，因经过LSTM模型训练前，要嵌入aspect，因此对应的词向量输入维度扩大一倍为600。根据上文介绍的模型流程，注意力机制处理后，还需将attention后的表示与LSTM的hidden层作拼接，**用到torch.cat函数**。

![](media/3cde31ae701ec61e36e1a553ad58f16f.png)


</br>

---

</br>

### **3 结果分析**

-   **Long Short-term Memory（LSTM）**

训练结果为：**test_loss = 0.6819，test_accuracy = 0.7415**

训练、测试函数的模型输入为：output = model (input_)，训练过程loss曲线如图：

![图2-1 LSTM模型损失曲线](media/0b3d3f088e6274651a627f1d61d11677.png)



-   **LSTM with Aspect Embedding （AE-LSTM）**

训练结果为：**test_loss = 0.8327，test_accuracy = 0.6281**

训练、测试函数的模型输入为：output = model
(input_，aspect)，训练过程loss曲线如图：

**![图2-2 AE-LSTM模型损失曲线](media/195646fc495a2224f7e9f3eaa2b09e10.png)**



该模型训练过程，损失曲线无大变化，学习遇到瓶颈，相比于LSTM模型，该模型训练效果较差。推测对于单条数据文本，可能存在涉及主题的多个表达，且没有注意力机制，模型受到混淆和干扰。

-   **Attention-based LSTM（AT-LSTM）**

训练结果为：**test_loss = 0.6104，test_accuracy = 0.7893**

训练、测试函数的模型输入为：output = model
(input_，aspect)，训练过程loss曲线如图：

**![图2-3 AT-LSTM模型损失曲线](media/724475af42199fcc2680275d8cea0d6a.png)**


模型训练过程中，损失曲线下降平滑。

-   **Attention-based LSTM with Aspect Embedding（ATAE-LSTM）**

训练结果为：**test_loss = 0.5950，test_accuracy = 0.8025**

训练、测试函数的模型输入为：output = model
(input_，aspect)，训练过程loss曲线如图：

**![图2-4 ATAE-LSTM模型损失曲线](media/e6ae8fcebe5f5b9d3ead8258189ffe9d.png)**



-   **对比模型**

| **Models**    | **Test Accuracy** | **Test Loss** |
|---------------|-------------------|---------------|
| **LSTM**      | 0.6228            | 0.8974        |
| **AE-LSTM**   | 0.6241            | 0.8357        |
| **AT-LSTM**   | 0.7893            | 0.6104        |
| **ATAE-LSTM** | **0.8025**        | **0.5950**    |

本次实验中，注意力机制和主题嵌入的提出，都一定程度提高了分类准确率，AE-LSTM、AT-LSTM模型的表现均高于LSTM模型。

ATAE-LSTM模型，输入部分将主题嵌入到词向量中，如此，训练过程可以很好地学习到主题内容，且主题能参与到注意力权重计算中。该模型不仅解决了词向量和主题嵌入地不一致性，还能根据给定主题捕获句子中最重要的信息，因此测试准确率最高，**达0.8025**。

