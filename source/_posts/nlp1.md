---
title: NER 命名实体识别  LSTM CNN CRF
date: 2021-11-01 01:46:07
tags: [nlp]
---
# **NER 命名实体识别  LSTM CNN CRF**

[**github 代码仓库**](https://github.com/ppphhhleo/nlp_NER) 


## **相关论文**

1.  [**Bidirectional LSTM-CRF models for sequence tagging**](https://arxiv.org/pdf/1508.01991.pdf)

2.  [**A survey on deep learning for named entity recognition**](https://ieeexplore.ieee.org/abstract/document/9039685/)

数据集用的是论文ACL 2018Chinese NER using Lattice LSTM(<https://github.com/jiesutd/LatticeLSTM>)中收集的简历数据，数据的格式如下，它的每一行由一个字及其对应的标注组成，标注集采用BME，B表示实体开头，M表示实体中间，E表示实体结尾，句子之间用一个空行隔开。该数据集就位于目录下的ResumeNER文件夹里。如下为数据及标签示例：

![dataset](media/bb54f4c318200718bea37e5fc0f7351c.png)

## **Quick Start**

- 实验环境  
> python3 + jieba + pytorch1.0.0 + numpy  
  pip install -r requirement.txt  


-   配置模型相关超参数及预训练词向量路径：

    /models/config.py

![配置文件](media/03c896105edfba85e4e491be3a5a6b49.png)


-   模型训练及测试：

![](media/4d17f4773c60c20fe3335818573433ee.png)

其中，可根据实际需求调整

**--model [模型]：cnn / lstm / bilstm**

**--crf ：增加crf层**

**--use_w2v ：使用预训练词向量**

-   **数据记录**

    **Loss图，测试结果，均保存在：**

    **Record.ipynb**

----
---  
## 详细描述
### **一、基本流程**

**1、安装并导入相关的深度学习库。**

PyTorch， Sklearn

**2、访问原始数据集，加载和处理数据**

data.py
中定义build_corpus构建语料库：定义2个列表，分别用于存储数据集中单条数据每个字**word**和**对应的词性标注tag**。同时加载已有的word2id序列，单个字映射为数字序列；如设置使用预训练词向量，则还需加载词典。

如果make_vocab为True，还需要重新构建词汇库，在已有word2id基础上完善词汇表，返回新的word2id、tag2id和word_embedding_init_matrix。在本次实验测试中，make_vocab均设置为False，即使用已获取的word2id。

**3、训练准备通道。**

utils.py
中定义加载模型和保存模型函数。对于LSTM模型训练时，还需要在word2id和tag2id加入PAD和UNK，如果是加了CRF的LSTM
/
BiLSTM模型，还需要加入\<start\>和\<end\>，用于解码。而对于加了CRF的模型测试数据，就无需加\<end\>**。**

**4、定义神经网络模型**

**多种基于长短期记忆**（Long Short-Term Memory）**LSTM**，bidirectional
LSTM**（BI-LSTM）**，LSTM with a Conditional Random
Field（CRF）layer**（LSTM-CRF）**，bidirectional LSTM with a CRF
layer**（BI-LSTM-CRF）**，Convolutional neural network**（CNN）**，CNN with a
CRF layer**（CNN-CRF）。**

**base_model.py**定义基础的LSTM模型，init初始化参数，包括字典大小、词向量维数、隐藏层向量的维数、标注种类、是否bidirectional，forward向前处理；也定义了CNN模型，init初始化隐藏层数量、隐藏层向量维数、输入维度、kernel
size。

针对添加了CRF的模型，在model_crf.py中封装了完整模型结构及训练测试函数。

**5、定义损失函数、优化器和评价模型的函数，设置超参数**

在**util.py**中定义计算损失的函数，这里使用交叉熵损失函数，针对LSTM-CRF模型额外定义计算方式。在config.py中设置超参数（epoch
= 20，learning rate = 0.001，embedding size = 100，bath size =
64，LSTM模型hidden size，CNN模型layer numbers，hidden size）。

**6、定义评估结果的函数**

在evaluating.py中定义Metrics类，计算精确率、召回率、F1分数；并对所有标注实体类别的预测结果计算平均值，计算混淆矩阵。

**7、训练和测试**

**以下为使用神经网路模型进行命名实体识别的实验流程总结：**

![流程](media/b15a2c6f230f4e2343aa5e306ca5f7a3.png)

---

### **二、模型分析**

本次实验，我们使用到LSTM、BI-LSTM、CRF、LSTM-CRF、BI-LSTM、BI-LSTM-CRF、CNN、CNN-CRF模型，以下将介绍相关模型。

**1、LSTM**

循环神经网络
(RNN)，是传统前馈神经网络的扩展，能够保留基于历史信息的记忆，对于较多的特征，能更好地预测当先输出特征表达。

图2-1展示了RNN模型的框架，输入层x，隐藏层h和输出层y，在命名实体识别中，x代表输入特征（features），y代表实体识别标注（tags）。图2-1演示了一个命名实体识别系统中，每个字（word）都会标注为other（O）或者其他四种实体类别：人物（Person，PER），地点（Location，LOC），组织（Organization，ORG）和混合（Miscellaneous，MISC）。例句EU
rejects German call to boycott British lamb.
中的每个词，依次被标注为B-ORG，O，B-MISC，O，O，O，B-MISC，O，O，其中B-和I-前缀标注说明了实体的起始和中间的位置。

![**图2-1 简略RNN模型**](media/48a726e9989e7adf2c64cc4a29a13823.png)



RNN模型中，输入层描述特征，特征能独热编码为词特征、密集矢量特征和稀疏特征。输入层与特征大小在维度上相同。输出层，代表一个标签的概率分布，和标签大小有相同的维度。与前馈神经网络相比，RNN构建了历史隐藏状态和当前隐藏状态（包括循环层权参数）的联系。循环层将保存历史信息，会有如下的计算：

![](media/5364255eb165a92fdf2edbc14091d96d.png)

其中U，W，V是在训练时计算连接的权值；f(z) 和 g(z)
是sigmoid和softmax函数，有如下计算公式：

![](media/aba3a901290f9fb3e2d67089aa038210.png)

为解决标准RNN存在梯度消失或梯度爆炸问题，长短期记忆神经网络架构被提出，并取得更佳的表现结果。LSTM模型与RNN模型大致相同，除了隐藏层被记忆cell所取代。因此，模型能更好地探索到数据中有较长周期的依赖关系。图2-2演示了一个LSTM
memory cell处理流程，包括遗忘门、输入门、更新、输出门

![**图2-2 LSTM Memory Cell**](media/1457862c5db7cba5109ef48163a18870.png)



LSTM Memory Cell 可以按照以下步骤计算：

![](media/8d852d85b230cc6fa6ee26e28502ec16.png)

其中，σ是logistic
sigmoid处理；i，f，o和c分别是输入门、遗忘门、输出门和cell的向量，这四组向量的维度与隐藏向量h大小相同。权矩阵的下标含义由以上的字母含义表示，如Whi是隐藏-输入矩阵，Wxo是输入-输出矩阵。从cell到gate向量的权矩阵是对角矩阵，因此在每个gate向量的元素m，只接收cell向量的m元素输入。

图2-3展示了LSTM模型的序列标注，应用了Memory cell，取代了RNN模型对应的隐藏层。

![**图2-3 LSTM模型**](media/617173009261ec1ad9c8c4d835cd4a9c.png)



**2、Bi-LSTM**

在序列标注任务中，我们可以在给定的时间内访问过去和未来的输入特征，因此我们可以利用双向LSTM网络，如图2-4所示。如此，在特定的时间范围内，**我们能充分利用过去的特征（通过向前的状态，forward
states），以及未来的特征（通过向后的状态，backward
states）。**我们使用了反向传播，来训练双向LSTM网络。在展开的网络上，随着时间的推移，向前和向后传递的方式与规则网络的向前和向后传递的方式类似，只是我们需要展开**所有时间步骤的隐藏状态**。我们还需要对数据在起始和结束点做特殊处理。在本次实验中，我们对整个句子执行向前和向后，仅需要在每个句子执行时将隐藏状态设置为0。我们进行批处理实现，可以在同一时间内处理多个句子。

![**图2-4 BiLSTM模型**](media/734115428493b33f1c4b6b10298420ee.png)



**3、CRF**

有两种方法，可以充分利用相邻标注的信息，来预测当前序列标注。第一类是预测每个时间步的标签分布，然后使用射线编码来找到最优标注序列，**最大熵分类和最大熵马尔可夫模型**属于这一类别。第二类是关注句子级别而非单个词语，这是**条件随机场模型**所属类别。条件随机场模型的**输入和输出**是直接相连，与LSTM和BiLSTM模型（中间使用了memory
cell或循环结构）不同，如图2-5所示。CRF模型在整体上可以获得更高的序列标注准确度。

![**图2-5 CRF结构**](media/136286eda3ec95bb57da3c159f3a75c6.png)



**4、LSTM-CRF**

我们将LSTM网络和CRF网络组合，便构成了LSTM-CRF网络，如图2-6所示。这个模型网络可以通过LSTM层有效使用过去的输入特征，以及通过CRF层使用句子级标注信息。CRF层由连接连续输出层的行表示。CRF层以状态转移矩阵作为参数。有了CRF层，我们可以有效使用过去和未来的标注来预测当前的标注，这与BiLSTM模型利用过去和未来输入特征的方法相似。我们将矩阵得分fθ([x]T1)作为网络的输出，为了简化标注，我们省略输入[x]T1。

**5、CNN**

卷积神经网络，**模型由输入层、卷积层、非线性激活函数、池化层和全连接层组成。**

![**图2-6 含有两个通道的模型架构**](media/e9d14d3b060b9d317a8d917b370279dd.png)



以图2-7为例，**第一层embedding
layer**，是图中最左侧7\*5的句子矩阵，每行是词向量，维度为5；**第二层convolutional
layer**，经过kernel_sizes = (2, 3,
4)的一维卷积层，每个kernel_sizes有两个输出channel；**第三层max-pooling layer**，
MaxPolling最大池化，抽取特征向量的最大值作为最重要特征；**第四层fully connected
layer**，将池化后的所有特征值连接；**第五层softmax层**，输出每个类别的概率。

![**图2-7 TextCNN**](media/efe039009ab9c2999d35e50ba587c5b0.png)



以下具体展示了如何使用CNN进行句子分类。

![**图2-8  CNN 序列标注**](media/3f387a4cd6db5686bb6b03978badf640.png)



由图2-8，具体而言，模型首先将所有输入的句子padding成相同长度，数据在模型中的流动，按照如下顺序：

**① 模型输入**，[batch_size, seq_len]

**② 经过embedding层**：加载预训练词向量或者随机初始化,
词向量维度为**embed_size：[batch_size, seq_len, embed_size]**

**③ 卷积层**：NLP中卷积核宽度与embed-size相同，相当于一维卷积。

3个尺寸的卷积核：(2, 3,
4)，每个尺寸的卷积核有100个。卷积后得到三个特征图：[batch_size, 100,
seq_len-1]、[batch_size, 100, seq_len-2]、[batch_size, 100, seq_len-3]

**④ 池化层**：对三个特征图做最大池化

[batch_size, 100]、[batch_size, 100]、[batch_size, 100]

**⑤ 拼接**：[batch_size, 300]

**⑥ 全连接**：num_class是预测的类别数：[batch_size, num_class]

**⑦ 预测**：softmax归一化，将num_class个数中最大的数对应的类作为最终预测：

[batch_size, 1]

**卷积操作**相当于提取了句中的2-gram，3-gram，4-gram信息，多个卷积是为了提取多种特征，**最大池化**将提取到最重要的信息保留。

**6、CNN + CRF**

我们将CNN和CRF结合，便构成了CNN-CRF网络

**7、BILSTM + CRF**

我们将BILSTM和CRF结合，便构成了BILSTM-CRF网络

### **三、模型对比**

模型训练超参数，batch size = 64，learning rate = 0.001，epoches = 20，print_step
= 5，embedding size = 100，pretrained embedding
vectors：‘./ResumeNER/pretrained_word_emb/word2vec.txt’，**hidden size =
100，不使用预训练词向量**

**对于LSTM，BiLSTM，CNN模型，三组validation loss曲线如图。**

**![](media/a3bee9fdecfd456b9d6690ae8cfc1ac0.png)**

**对于LSTM-CRF，BiLSTM-CRF，CNN-CRF，三组validation loss如下图。**

**![](media/3a8297b2fff14ea8dc6c22e97d550ba7.png)**

以上三组模型，在添加条件随机场后，模型表现都有所提升，**CRF能充分利用相邻标注的信息，来预测当前序列标注，获得更高的序列标注准确度。**

NER
任务，对比不同模型的表现，所有实体识别的f1-score和训练时间，对比如下表，**在当前实验条件下，LSTM-CRF表现较佳**，与论文得出结论（BiLSTM-CRF
模型能有效利用过去和未来的标注信息，预测当前标注，表现最佳）**稍有偏差**，**可能与数据集、学习率、LSTM隐藏层向量维数等其他变量有关。**

| Model        | F1_score   | Time    |
|--------------|------------|---------|
| LSTM         | 0.9094     | 524     |
| BiLSTM       | 0.9141     | 547     |
| CNN          | 0.8501     | 72      |
| **LSTM-CRF** | **0.9264** | **697** |
| BiLSTM-CRF   | 0.9221     | 701     |
| CNN-CRF      | 0.9001     | 216     |

### **四、是否使用预训练词向量对比**

模型训练超参数，batch size = 64，learning rate = 0.001，epoches = 20，print_step
= 5，embedding size = 100，pretrained embedding
vectors：‘./ResumeNER/pretrained_word_emb/word2vec.txt’，**hidden size =
100（CNN输出维度，LSTM隐向量维数），layer number =
3（CNN层数），使用预训练词向量**

使用已有的预训练词向量，对应模型的f1_score均降低。

| Model                       | F1_score   | Time    |
|-----------------------------|------------|---------|
| **BiLSTM-CRF-pretrain w2v** | **0.9177** | **694** |
| CNN-CRF- pretrain w2v       | 0.8827     | 201     |

### **五、CNN模型调参对比**

**1、kernel size**

模型训练超参数，batch size = 64，learning rate = 0.001，epoches = 20，print_step
= 5，embedding size = 100，pretrained embedding
vectors：‘./ResumeNER/pretrained_word_emb/word2vec.txt’，**hidden size =
100（CNN输出维度），layer number = 3（CNN层数），不使用预训练词向量。**

对kernel size分别取**[3，5，7，9，11]**进行测试，对比标注f1_score.

五组kernel size，validation loss 曲线：

![](media/8cb4f8551dd847b65f4b1c3f7b8f358d.png)

F1_score对比如下表，在本次实验中，kernel size = 7, CNN模型表现较佳。

| Size  | F1_score   | Time    |
|-------|------------|---------|
| 3     | 0.9001     | 216     |
| 5     | 0.9142     | 202     |
| **7** | **0.9239** | **217** |
| 9     | 0.9222     | 239     |
| 11    | 0.9150     | 243     |

**2、hidden size**

模型训练超参数，batch size = 64，learning rate = 0.001，epoches = 20，print_step
= 5，embedding size = 100，pretrained embedding
vectors：‘./ResumeNER/pretrained_word_emb/word2vec.txt’，**kernel size =
7，layer number = 3（CNN层数），不使用预训练词向量。**

对kernel size分别取[100，200，300]进行测试，对比标注f1_score.

三组kernel size，validation loss 曲线：

![](media/624d362ab57d03d16acc6f31493aceec.png)

F1_score对比如下表，在本次实验中，hidden size = 100, CNN模型表现较佳。

| Size    | F1_score   | Time    |
|---------|------------|---------|
| **100** | **0.9239** | **217** |
| 200     | 0.9116     | 301     |
| 300     | 0.9190     | 406     |


----



### **六、总结**

**1、结果**

（1）对LSTM，BiLSTM，CNN三组模型分别加上CRF，命名实体识别准确度均有所提高，**CRF能充分利用相邻标注的信息，来预测当前序列标注，获得更高的序列标注准确度**。

（2）LSTM-CRF，BiLSTM-CRF，CNN-CRF三组模型中，本次实验LSTM-CRF的标注准确度更高，**f1_score高达0.9264，略高于BiLSTM-CRF模型**。但**按照实验所复现的论文**，BiLSTM-CRF模型能充分利用过去和未来的输入特征和句子级别的标注信息，在不同标注任务中表现更佳。本次实验结果**并未达到论文所呈现的效果**，分析可能与数据集（论文使用三种不同任务的数据集）、学习率（原文为0.1，本次实验为0.001）、LSTM隐藏层维数（原文为300且模型效果对隐藏层维数并不敏感，本次实验为100）、batch
size选取等因素相关，后续可以进行更完善的复现测试。

（3）CNN的kernel size 一般选取较小的奇数，本次实验测试CNN模型在kernel size =
7时取得较佳标注准确度，f1_score 达0.9239；在**kernel size = 7**
条件下进一步测试CNN **hidden size = 100**时，表现较佳，**f1_score = 0.9239**

**2、precision、recall、f1_score**

**Accuracy = TP+TN/TP+FP+FN+TN**

**Precision = TP/TP+FP**

**Recall = TP/TP+FN**

**F1 Score = 2\*(Recall \* Precision) / (Recall + Precision)**

**3、kernel size 选择，分析大小影响**

（1）kernel

Kernel size
可以控制转换的速度，可以通过连接最优的核密度估计，来达到对数据集的最佳拟合。在全连接网络中，输入中的每个向量元素都连接到第一层的每个隐藏单元。因此，每个单元将连接到所有其他单元。对于有三个通道的图像而言，如果是11\*11的图像，那么单个单元将有121个连接，而实际上将有121\*3
= 363个连接.
在局部连接网络中，我们使用内核/过滤器。内核是一个单元可以看到的图像的一部分。所以我们可以说内核就像一个窗口。但是这里的窗口可能不同，即内核的权重可能不同。

![](media/4f10bb52a3935f91deb30798262c3255.png)

卷积网络与局部连接网络类似，只是在内核中有权重共享。

![](media/0b1360e26cd31b6aebb7c24d66fbc427.png)

Kernel不断在图像上滑动，生成输出，为特征表达图。

![](media/c474e9b5284b55be2587bdceb6ce711d.png)

**（2）奇数大小的kernel size**

对于奇数大小的滤波器，所有前一层像素都将围绕输出像素对称。如果没有这种对称性，我们将不得不考虑使用均匀大小的内核时发生的跨层失真。因此，即使大小的内核过滤器也大多被跳过以提高实现的简单性。如果将卷积视为从给定像素到中心像素的插值，我们无法使用偶数大小的过滤器对中心像素进行插值。

**（3）一般的选择**

参数数量随内核大小呈二次方增长，这使得大卷积核的效率不够高。限制kernel
size，可以限制了可能的不相关特征的数量。这迫使机器学习算法学习不同情况下的共同特征，从而更好地概括。因此，常见的选择是将内核大小保持在
3x3 或 5x5。

当然，较小的kernel size
虽然可以限制提取不相关特征，但如果文本特征较丰富，较小的kernel size
也会限制提取到相关特征，当然卷积用时随之增大。本次实验，经过测试CNN的kernel
size，发现在7\*7内核下的命名实体识别准确度较佳。

**4、复现论文阅读。**

**为了体现模型的鲁棒性，论文在两方面进行测试**，一是设计三个不同任务：Part Of
Speech tagging（POS），Chunking和Named-entity
Recognition（NER）；二是设计两种初始化词嵌入方式：Random，Senna。三个实体识别标注任务，来检测模型在不同场景的适应和表现；两种词嵌入方式，检测模型对词嵌入的依赖程度。模型在越多的场景表现佳、对词嵌入依赖程度越小，则鲁棒性越高。此外，论文为了预估模型关于工程特征(拼写spelling和上下文特征context
features)的鲁棒性，仅使用词特征（word
features）对所有模型进行一组测试，与使用拼写和上下文特征的相同模型对比性能下降程度，来判断模型对工程特征的依赖程度。**实验结果表明，
Bi-LSTM-CRF模型鲁棒性最高、标注准确度也最高，CRF模型对工程特征依赖程度最大、鲁棒性最低。**

论文还将模型与其他已有的模型进行系统性测试，BI-LSTM-CRF模型都有最佳的准确度（state
of the art，or close
to）。**总结，论文首次提出将BI-LSTM-CRF模型应用于NLP基准序列标记数据，在POS、chunking、NER数据集中有极好的出色表现、较高的鲁棒性和较少的词嵌入依赖。**

---

## References 
[1] POS Chunking NER
<https://medium.com/greyatom/learning-pos-tagging-chunking-in-nlp-85f7f811a8cb>

[2] BiLSTM - CNNs <https://arxiv.org/pdf/1511.08308v5.pdf>

[3] CNN kernel size
<https://medium.com/analytics-vidhya/significance-of-kernel-size-200d769aecb1>

[4] CNN <https://en.wikipedia.org/wiki/Convolutional_neural_network>

[5] kernel size
<https://blog.csdn.net/weixin_42490152/article/details/100160864>

