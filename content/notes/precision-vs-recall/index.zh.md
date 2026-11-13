---
title: "混淆矩阵：precision、recall，以及每种错误由谁买单"
date: 2026-11-12
lastmod: 2026-11-12
description: "precision 和 recall 是同一张 2×2 表的两种读法。拿癌症筛查、银行卡欺诈和虚假账户走一遍：三个场景最后都不是二选一，而是在一个分数上切几个阈值，配上分级的动作。"
topics: [data-science, healthcare]
---
做二分类，迟早会碰到同一个问题：该偏向 precision 还是 recall？常见的回答是一句口诀（医疗要 recall，垃圾邮件要 precision），可一旦摆出真实数字，口诀就不够用了。这篇笔记想讲清楚四件事：混淆矩阵到底说了什么；阳性一少，precision 为什么会塌；代价怎么变成阈值；以及癌症筛查、银行卡欺诈、短视频平台的虚假账户这三个场景，各自是怎么处理这个取舍的。

> **先说结论。** recall 是模型和阈值的性质；precision 还要再加上“模型跑在什么人群上”；该偏向哪个，取决于判为阳性之后要做什么动作。下面三个场景最后都不是二选一，而是在一个分数上切几个阈值，配上分级的动作。

关于用词：指标名保留英文（precision、recall、specificity、NPV、accuracy），因为中文译名容易混，precision 常译精确率或查准率，recall 是召回率或查全率，accuracy 是准确率。“阳性”指我们要找的那一类（癌症、欺诈、虚假账户）；“阳性占比”就是 prevalence，医学上叫患病率。

## 问题在哪

模型给出的是一个分数，产品要的是一个决定。于是有人在分数上划一条阈值，从这一刻起，模型会犯两种错：把没问题的标出来，把有问题的放过去。这两种错通常不是同一个人买单，代价也很少相等。

“precision 还是 recall”其实压着三个问题：

1. **模型上线的地方，阳性有多稀少？**
2. **每种错的代价是多少，谁来付？**
3. **模型判了阳性之后会发生什么？** 再做一次检查，交给人审，还是直接执行一个撤不回的动作？

这三个问题都能在混淆矩阵里看到，所以先从它开始。

## 一张表，三种读法

![二乘二的混淆矩阵：recall 和 specificity 横着读，precision 和 NPV 竖着读](confusion-matrix.svg "图 1。四个格子和两种读法。recall 和 specificity 沿着行除，precision 和 NPV 沿着列除。")

### 四个格子

行是真实情况，列是模型的判定。读任何数字之前，先确定哪一类是“阳性”。习惯上是那个稀少、会触发动作的类别：癌症、欺诈、虚假账户。这个选择一换，下面所有名字都跟着换。

下面是一张真实的混淆矩阵：2007 到 2013 年美国社区医疗机构的 1,682,504 次乳腺 X 线筛查（计数单位是“次检查”，不是“人”），数据来自 [Breast Cancer Surveillance Consortium](https://www.bcsc-research.org/statistics/screening-performance-benchmarks-archive/screening-sens-spec-false-negative-archive)（BCSC）。“阳性”指检查后一年内确诊癌症。

| | 叫回复查 | 判为正常 | 合计 |
|---|---:|---:|---:|
| 有癌症 | 8,529（TP） | 1,283（FN） | 9,812 |
| 没有癌症 | 186,139（FP） | 1,486,553（TN） | 1,672,692 |
| 合计 | 194,668 | 1,487,836 | 1,682,504 |

### 横着读：模型对每一类做了什么

沿着行除，分母是真实阳性或真实阴性的数量。

- **Recall**（也叫 sensitivity，敏感度）：$8{,}529 / 9{,}812 = 86.9\%$ 的癌症被叫回复查。
- **Specificity**（特异度）：没有癌症的检查里，$1{,}486{,}553 / 1{,}672{,}692 = 88.9\%$ 被判为正常。剩下的 11.1% 是 **false positive rate**（假阳性率，FPR）。

这两个数描述的是检查本身，和受检人群里有多少人患癌无关。

### 竖着读：一个结果意味着什么

沿着列除，分母是拿到某个结果的人数。

- **Precision**（阳性预测值，PPV）：叫回复查的检查里，$8{,}529 / 194{,}668 = 4.4\%$ 确实是癌症。
- **NPV**（阴性预测值）：判为正常的结果里，$1{,}486{,}553 / 1{,}487{,}836 = 99.91\%$ 是对的。

拿到结果的人关心的是这两个数。它们取决于检查本身，也同样取决于阳性占比。

### 整张表：accuracy

accuracy 把对角线加起来：88.9%。一个把所有片子都判成正常的医生能拿 99.4%，同时一例癌症也找不到。accuracy 按数量给每个格子加权，阳性只占 0.58% 时，它几乎只在描述健康人。

## 四个比率，一个一个看

![四个小矩阵，各自标出一个比率用到的格子和它所在的行或列](four-rates.svg "图 2。每个比率都是拿一个判对的格子，除以它所在的行或列；下面一行是它的反面，即同一行或列里的另一个格子除以同一个总数。布局和图 1 相同，数值来自上面的筛查矩阵。")

矩阵有四个格子，也就有四个最自然的比率：各取一个判对的格子，除以它所在的行或者列。四个比率，其实是四种人在问的四个问题。

### Recall：真实阳性里，我抓到了多少？

$$
\text{recall} = \frac{TP}{TP + FN} = \frac{8{,}529}{9{,}812} = 86.9\%
$$

问这个问题的，是要对阳性负责的人：筛查项目，或者背着欺诈损失的风控团队。它的反面是 miss rate（漏报率）：这里有 13.1% 的癌症被判成了正常。

- **看不见什么：** 误报。公式里没有 FP，所以 recall 不告诉你抓到这些阳性花了多大代价。
- **怎么白拿 100%：** 全部判阳性。
- **什么时候它最重要：** 漏掉的代价高，而判阳性之后的动作便宜。筛查、安全过滤、给病历去标识化都是。
- **要留意：** recall 是对模型见到的那批阳性取的平均。不明显的病例比明显的更容易漏，所以换一批难易构成不同的人群，同一个模型的 recall 也会变。

### Specificity：真实阴性里，我放过了多少？

$$
\text{specificity} = \frac{TN}{TN + FP} = \frac{1{,}486{,}553}{1{,}672{,}692} = 88.9\%
$$

问这个问题的，是要为大量误报买单的人。它的反面是假阳性率，11.1%。

- **看不见什么：** 漏报。
- **怎么白拿 100%：** 一个都不判阳性。
- **什么时候它最重要：** 判阳性之后的动作代价高，或者难以撤销：确诊检查、拒绝一笔交易、封掉一个账号。
- **要留意：** 百分比会让它显得比实际好。阴性占绝大多数，这里 specificity 每差一个百分点就是 16,727 次检查。99% 的 specificity 听上去近乎完美；但如果每一千笔交易里只有一笔欺诈，它意味着每出现一笔欺诈就伴随十次误报。阳性稀少的时候，改用“每千个或每百万个阴性里有几次误报”来数。

### Precision：我报警的时候，有多大可能是真的？

$$
\text{precision} = \frac{TP}{TP + FP} = \frac{8{,}529}{194{,}668} = 4.4\%
$$

问这个问题的，是收到警报的人：被叫回复查的人，处理告警队列的分析师，账号被标记的用户。

- **看不见什么：** 漏报。模型只要足够保守，precision 就可以很高。
- **怎么白拿 100%：** 只标最有把握的那一个。
- **什么时候它最重要：** 每一次警报都要有人接：容量固定的人工审核队列，或者直接作用在客户身上的动作。
- **两种读法：** 和基数比，以及倒过来读。被叫回复查之后，患癌的可能从 0.58% 升到 4.4%，是原来的 7.5 倍。而 $1/\text{precision}$ 就是工作量：每查出一例癌症，要叫回复查 23 次。

### NPV：我说没事的时候，有多大可能是对的？

$$
\text{NPV} = \frac{TN}{TN + FN} = \frac{1{,}486{,}553}{1{,}487{,}836} = 99.91\%
$$

问这个问题的，是被放行的人：被告知结果正常的人，顺利通过的那笔交易，没被处理的账号。

- **看不见什么：** 误报。
- **怎么白拿高分：** 阳性稀少时，全部放行。这里一张片子都不看就有 99.42%。
- **什么时候它最重要：** 阴性结果就是流程的终点：排除性检查、分诊、自动放行。
- **怎么读：** 看它的反面，再和基数比。读片之前，一年内确诊癌症的可能是 171 分之一；结果正常之后，是 1,160 分之一。这次检查把风险降到了原来的七分之一左右，这才是“99.91%”真正的含义。

NPV 的反面叫 false omission rate，也就是系统漏在后面的那部分：放行的人里的癌症，通过的交易里的欺诈，还活跃着的虚假账户。“平台上还剩多少”就是这一类数字。

### 四个放在一起看

| | Recall | Specificity | Precision | NPV |
|---|---|---|---|---|
| 分母 | 真实阳性 | 真实阴性 | 报警的 | 放行的 |
| 谁在问 | 要把阳性抓住的人 | 为误报买单的人 | 收到警报的人 | 被放行的人 |
| 筛查的数值 | 86.9% | 88.9% | 4.4% | 99.91% |
| 看不见 | 误报 | 漏报 | 漏报 | 误报 |
| 怎么白拿满分 | 全判阳性 | 全判阴性 | 只标最有把握的一个 | 阳性稀少时全判阴性 |
| 随阳性占比变化吗 | 不变 | 不变 | 占比越高，它越高 | 占比越高，它越低 |
| 反面（1 − 它） | 漏报率 13.1% | 假阳性率 11.1% | 错误发现率 95.6% | 错误遗漏率 0.086% |

有两组搭配值得记住。

**沿着阈值，recall 和 specificity 此消彼长。** 阈值调低，recall 升、specificity 降，模型的其他方面都没变。后面讲阈值的一节会展开。

**沿着对角线，recall 保护 NPV，specificity 保护 precision。** recall 高，漏报就少，放行的人里就混不进多少阳性；specificity 高，误报就少，警报就不会被冲淡。医生把它记成 SnNOut 和 SpPIn：sensitivity 高的检查，结果阴性可以排除（rule out）；specificity 高的检查，结果阳性可以确诊（rule in）。[Pewsner 等人](https://doi.org/10.1136/bmj.329.7459.209)提醒过这条口诀的边界：能不能排除也取决于 specificity，能不能确诊也取决于 sensitivity。阳性占比同样有影响：只有 0.58% 的检查是阳性时，88.9% 的 specificity 什么也确诊不了。

四个比率没有一个可以单独汇报。每一个都有办法白拿满分，每一个都看不见一种错误。一份有用的报告，会把一个看不见误报的比率和一个看不见漏报的比率配在一起（机器学习习惯报 recall 和 precision，医学习惯报 sensitivity 和 specificity），再加上阳性占比。有这三个数，整张矩阵都能还原出来。

### 八个比率，一堆名字

每个比率都有一个“反面”：同一行或同一列里的另一个格子，除以同一个总数。所以一张矩阵里一共有八个比率，两两一对，每对加起来是 100%。医学、统计、信号检测和机器学习各自给它们起过名字，所以同一个数常常有好几个叫法，中文译名又多一层。

| 比率 | 公式 | 筛查的数值 | 英文别名 | 中文叫法 |
|---|---|---:|---|---|
| **Recall** | TP&nbsp;/&nbsp;(TP&nbsp;+&nbsp;FN) | 86.9% | sensitivity、TPR、hit rate、detection rate、统计功效 power（1&nbsp;−&nbsp;β） | 召回率、查全率、灵敏度、敏感度、真阳性率 |
| **Miss rate** | FN&nbsp;/&nbsp;(TP&nbsp;+&nbsp;FN) | 13.1% | false negative rate（FNR）、第二类错误率（β） | 漏报率、漏诊率、假阴性率 |
| **Specificity** | TN&nbsp;/&nbsp;(TN&nbsp;+&nbsp;FP) | 88.9% | true negative rate（TNR）、selectivity | 特异度、特异性、真阴性率 |
| **False positive rate** | FP&nbsp;/&nbsp;(TN&nbsp;+&nbsp;FP) | 11.1% | FPR、fall-out、false alarm rate、第一类错误率（α） | 假阳性率、误报率、误诊率 |
| **Precision** | TP&nbsp;/&nbsp;(TP&nbsp;+&nbsp;FP) | 4.4% | positive predictive value（PPV）；乳腺筛查审计里叫 PPV1 | 精确率、查准率、精准率、阳性预测值 |
| **False discovery rate** | FP&nbsp;/&nbsp;(TP&nbsp;+&nbsp;FP) | 95.6% | FDR，Benjamini–Hochberg 方法控制的就是它 | 错误发现率、伪发现率 |
| **NPV** | TN&nbsp;/&nbsp;(TN&nbsp;+&nbsp;FN) | 99.91% | negative predictive value | 阴性预测值 |
| **False omission rate** | FN&nbsp;/&nbsp;(TN&nbsp;+&nbsp;FN) | 0.086% | FOR | 错误遗漏率 |

还有两个比率描述的是这张表本身，而不是模型准不准：**阳性占比**（prevalence），(TP + FN) / N = 0.58%；以及**标记率**（flag rate），(TP + FP) / N = 11.6%，也就是被判为阳性的样本占多少，也叫告警率、阳性率。

### 同一个词，不同的数

这些名字会撞车。读别人报的指标之前，下面几处值得先知道。

- **乳腺筛查里的 “recall rate” 不是 recall。** 放射科审计用它表示被叫回复查的检查占比：[BCSC 的定义](https://www.bcsc-research.org/statistics/screening-performance-benchmarks-archive/abnormal-interpretations-archive)是初次判读为异常的检查所占的百分比。这里是 11.6%，也就是上面的标记率。机器学习里的 recall，在那里叫 sensitivity：86.9%。中文里更绕，因为 recall 本身就译作“召回率”。
- **“误诊率”在医学统计教材里是假阳性率**，即 1 − 特异度；“漏诊率”是假阴性率，即 1 − 灵敏度（例如[这篇整理](https://damodev.csdn.net/68a6b7b64e4959284dabbe8a.html)）。按字面很容易把“误诊率”读成“诊断为阳性的人里有多少是错的”，那其实是错误发现率，1 − precision，筛查里两者差得很远：11.1% 对 95.6%。
- **准确率和精确率。** “准确率”通常指 accuracy，precision 一般译作精确率或查准率；但也有资料把 precision 译成准确率。看到“准确率”，先看公式。
- **“检出率”** 在风控、安全领域通常指 recall；在筛查里，[癌症检出率](https://pmc.ncbi.nlm.nih.gov/articles/PMC2714544/table/T1)是每 1,000 次检查查出几例癌症，即 TP / N：这里是 5.1。
- **“Hit rate”** 在信号检测里是 recall。风控团队也拿它衡量告警质量，包括[被标记的正常客户和抓到的欺诈之比](https://www.sardine.ai/learn/false-positive-rate)。
- **假阳性“率”（false positive rate）和假阳性“比”（false positive ratio）。** 前者除以真实阴性。后者在反欺诈运营里很常见，[数的是每抓到一笔真欺诈伴随几次误报](https://socure.com/glossary/false-positive-ratio)。换到筛查上是 22 比 1。
- **“我们的假阳性率是 90%”**，如果说的是告警队列，通常意思是 90% 的告警是误报。这是错误发现率，1 − precision。同一个系统真正的假阳性率往往不到 1%。

最稳妥的习惯：问清公式，最好直接要四个计数。

### 用一个数来概括

accuracy 是最早把整张表压成一个数的办法，阳性一少它就失灵。其他常见的几个：

- **Balanced accuracy** 是 recall 和 specificity 的平均，这里是 87.9%。它不受阳性占比影响，补上了 accuracy 最大的毛病，但仍然把漏报和误报看得一样重。
- **$F_1$** 是 precision 和 recall 的调和平均，这里是 0.08。调和平均总是贴近两者中较小的那个，所以只有两个都高，$F_1$ 才高。它不看 TN，还取决于把哪一类叫作阳性：同一批医生、同一批片子，把“正常”当作阳性类来算，$F_1$ 是 0.94。
- **$F_\beta$** 让 recall 的分量是 precision 的 $\beta$ 倍，但不说明 $\beta$ 从哪来。
- **约登指数**（Youden's J），recall + specificity − 1，这里是 0.76。它是工作点在 ROC 图上高出对角线的距离，代价未知时常用它来选阈值。它的另一个名字叫 informedness；竖着读的对应量是 precision + NPV − 1，叫 markedness，这里是 0.043。
- **Matthews 相关系数**（MCC）四个格子都用上，这里是 0.18。它等于 informedness 和 markedness 的几何平均，所以横读、竖读只要有一边弱，它就低。
- **诊断比值比**（diagnostic odds ratio），LR+ / LR−，这里是 53：有癌症时读片为阳性的几率，是没有癌症时的 53 倍。

这些数都回答不了“筛查值不值得做”，因为它们都不知道一次漏报、一次误报各要付出什么。我更愿意把代价直接写出来，后面有一节专门做这件事。

## 阳性一少，precision 为什么会塌

![一千个点：116 次被叫回复查，其中 5 例是癌症；884 次被放行，其中漏掉 1 例癌症](thousand-screens.svg "图 3。把 BCSC 的比率放到 1,000 次筛查上。被叫回复查的几乎都是健康人，因为来做筛查的几乎都是健康人。")

上面这项检查能查出 87% 的癌症，没有癌症的检查里也有 89% 直接过关，可每 100 次叫回复查里仍有 96 次是误报。检查本身没有问题。每一次有癌症的检查背后，大约有 170 次没有癌症的检查，11% 的假阳性率乘在大的那一堆上，就盖过了 87% 的 recall 乘在小的那一堆上。

用几率（odds）形式的贝叶斯公式，这就是一次乘法：

$$
\underbrace{\frac{P(\text{阳性} \mid \text{被标记})}{P(\text{阴性} \mid \text{被标记})}}_{\text{标记之后的几率}}
= \underbrace{\frac{\pi}{1-\pi}}_{\text{之前的几率}} \times \underbrace{\frac{\text{recall}}{\text{FPR}}}_{\text{似然比}}
$$

对筛查来说，之前的几率是 1 比 170，似然比是 $0.869 / 0.111 = 7.8$。一次阳性结果让患癌的几率变成原来的 7.8 倍，这已经是很强的检查了，可之后的几率仍然只有 1 比 22。写成概率：

$$
\text{precision} = \frac{\text{recall} \cdot \pi}{\text{recall} \cdot \pi + \text{FPR} \cdot (1 - \pi)}
$$

![recall 为 87% 时，precision 随阳性占比变化的三条曲线，假阳性率分别为 11%、1% 和 0.1%](precision-vs-prevalence.svg "图 4。recall 固定在 87% 时，precision 随阳性占比的变化，三条线对应三个假阳性率。最浅的那条是筛查的工作点。竖线是本文三个场景大致的阳性占比：卡欺诈按金额算，筛查取自 BCSC，虚假账户取自 Meta 对 Facebook 的估计。")

由此得到三点。

**假阳性率得和阳性占比差不多小。** 只有当 $\text{FPR} \le \text{recall} \cdot \pi / (1 - \pi)$ 时，precision 才到得了 50%。阳性占 0.1%、recall 为 80% 时，这要求被误标的阴性不到 0.08%。1% 的假阳性率听起来很出色，在这里只换来 7% 的 precision。

**在重新配平的测试集上量出来的 precision 带不走。** 一个 recall 90%、假阳性率 5% 的模型，在阴阳各半的测试集上 precision 是 94.7%，到了阳性只占 1% 的地方是 15.4%。只要病例的类型没变，recall 和 specificity 可以原样带过去；precision 必须用上面的公式，按上线环境的阳性占比重算。

**同一个模型，换一个人群就是另一个产品。** 把这项筛查挪到阳性占 4% 的人群，precision 是 25%；挪到卡欺诈那样的占比，是 0.5%。

## 混淆矩阵只是某一个阈值下的快照

![两条分数分布，分别是真实阴性和真实阳性，被一条阈值切开；落在错误一侧的两条尾巴就是漏报和误报](threshold.svg "图 5。两条分数分布和一条阈值，画的是筛查的工作点。移动阈值，就是拿漏报换误报，模型的其他方面都不变。")

一张混淆矩阵属于“某个阈值下的模型”。移动阈值，样本只在同一行里换格子：真阳性变成假阴性，或者假阳性变成真阴性。所以 recall 和假阳性率同升同降，precision 是这两者落在哪里、再加上阳性占比之后的副产品。

这也是为什么描述同一个模型的两种曲线，讲出来的故事可以差那么多。

![左边是一条面积为 0.95 的 ROC 曲线；右边是同一个模型在阳性占 50%、5% 和 0.58% 时的三条 PR 曲线](roc-vs-pr.svg "图 6。同一个模型（两条方差为 1 的高斯分布，穿过筛查的工作点）的两种画法。ROC 曲线看不出阳性占比，PR 曲线几乎只在说阳性占比。")

ROC 曲线画的是两个横着读的比率，所以它看不见阳性占比。不管阳性占一半还是 170 个里才有一个，这里的面积都是 0.95。PR 曲线画的是一个横着读的比率对一个竖着读的比率。同一个工作点上，阳性从 50% 降到 5% 再降到 0.58%，precision 依次是 89%、29% 和 4.4%。

[Saito 和 Rehmsmeier](https://doi.org/10.1371/journal.pone.0118432) 认为，这使得 ROC 图在不平衡数据上很容易被误读：阴性的基数一大，specificity 看上去就让人放心。[Davis 和 Goadrich](https://doi.org/10.1145/1143844.1143874) 证明了两个空间在“谁压过谁”上是一致的（一条曲线在一个空间里处处占优，在另一个空间里也处处占优），同时也证明了优化 ROC 面积并不等于优化 PR 面积。我的做法是：比较模型用 ROC；判断模型能不能用，看上线环境阳性占比下的 PR 曲线。

## 从代价到阈值

给每种错误定一个代价，每个样本的期望代价是

$$
\mathbb{E}[\text{代价}] = c_{FN} \cdot \pi \, (1 - \text{recall}) \; + \; c_{FP} \cdot (1 - \pi) \, \text{FPR}
$$

如果模型输出的是校准过的概率 $p$（这个样本是阳性的概率），判阳性平均要付 $(1-p)\,c_{FP}$，不判要付 $p\,c_{FN}$。前者更小时就判阳性，于是得到 [Elkan 那篇代价敏感学习论文](https://cseweb.ucsd.edu/~elkan/rescale.pdf)里的阈值：

$$
p^* = \frac{c_{FP}}{c_{FP} + c_{FN}}
$$

| 漏报的代价是误报的 | 概率超过多少就判阳性 |
|---|---:|
| 1,000 倍 | 0.1% |
| 100 倍 | 1% |
| 10 倍 | 9% |
| 1 倍（一样贵） | 50% |
| 十分之一 | 91% |
| 百分之一 | 99% |

这样一来，口诀就有了新的读法。“偏向 recall”的意思是漏报相对误报更贵，所以阈值要低；“偏向 precision”则相反。而且代价是跟着**动作**走的，不是跟着行业走的：叫回来加做影像和做活检都属于“医疗”，两者的阈值却差得很远。

实际用起来还有四个麻烦：

- **这个阈值假设概率是校准过的。** 在重采样的数据上训练会破坏校准。如果阴性只保留了比例 $\beta$，模型给出的几率就被放大了 $1/\beta$ 倍，修正后的概率是 $p = \beta p_s / (\beta p_s + 1 - p_s)$。
- **0.5 只是一个特例。** 只有两种错误代价相同、概率又校准过，它才是对的。
- **代价可以每个样本都不一样。** 漏掉一笔 \$2,000 的欺诈和漏掉一笔 \$20 的不是一回事。这时每个样本都有自己的阈值。
- **阈值也可能由产能决定。** 如果每条告警都要人审，而一天只审得完 500 条，那么工作点就是“分数最高的 500 条”，该看的指标是这 500 条里的 precision，以及它换来了多少 recall。

## 三个场景

| | 乳腺筛查 | 银行卡欺诈 | 虚假账户 |
|---|---|---|---|
| 打分的对象 | 一次检查 | 一笔交易 | 一个账号，反复打分 |
| 阳性占比 | 检查的 0.58% | 不到交易额的 0.1% | 活跃账号的百分之几，没有确切数字 |
| 漏报的代价 | 癌症晚发现；病人承担 | 被盗刷的金额；银行或商户承担 | 垃圾内容和虚假互动；所有人各摊一点 |
| 误报的代价 | 多做影像，有时要活检，还有担惊受怕；健康人承担 | 交易被拒，一通客服电话，可能丢掉一个客户 | 一个真人被锁在门外，要走申诉 |
| 真相什么时候来 | 一年之内，来自活检和癌症登记 | 几天到几周之后；被拒的交易永远没有 | 很少；靠人工抽样审核 |
| 判阳性之后 | 再做一次更特异的检查 | 加一步验证、人工审核，或拒绝 | 验证、限流、送审，或封号 |
| 大家汇报的数字 | 每一步的 sensitivity、specificity、PPV | 拦下的欺诈金额、告警的 precision、误拒率 | 各个动作的抽样 precision、申诉成功率、估计的剩余占比 |

### 医疗筛查：先要 recall，再做第二次检查

在 [BCSC 的基准数据](https://doi.org/10.1148/radiol.2016161174)里，11.6% 的筛查被叫回复查，每 1,000 次检查查出 5.1 例癌症。这一步的 precision 是 4.4%。到了被建议做活检的那批人，是 25.6%。

4.4% 的 precision 之所以可以接受，是因为阳性触发的动作是加做影像，不是治疗。误报的代价是多跑一趟、担心几天；漏报的代价是癌症晚一年甚至更久才被发现。代价比悬殊到这个程度，阈值就该放低。

更要紧的一点是，筛查不是一次决定。每一步都在抬高下一步的阳性占比：全部筛查里是 0.58%，叫回复查的是 4.4%，建议活检的大约是 26%。先用一个敏感的检查，再接一个特异的检查，整条流程就同时拿到了 recall 和 precision，这是任何单个阈值都做不到的。

recall 也并没有被推到 100%。每一轮筛查都会重复产生误报，后续检查本身也有伤害，读片医生的时间又有限。在同一份 BCSC 数据里，只有大约六成放射科医生的复查率落在推荐范围内；这个行业同样在考核误报这一头。

### 银行卡欺诈：两种错误都按钱算

欺诈更稀少。[Nilson Report](https://nilsonreport.com/articles/card-fraud-losses-worldwide-in-2023/) 统计 2023 年全球银行卡欺诈损失为 338.3 亿美元，它的新闻稿把这个数表述为每 100 美元交易额里 6.58 美分，按金额算不到 0.1%。

假设有 1,000,000 笔交易，其中 1,000 笔是欺诈，模型的 recall 是 80%。假阳性率 0.5% 时，它抓到 800 笔欺诈，同时产生 4,995 次误报：precision 13.8%，每抓一笔欺诈要打扰六个正常客户。把假阳性率压到 0.1%，precision 升到 44.5%。这还是前面那条关于阳性占比的规律，只不过这里每一次误报都是一个站在收银台前的客户。

这里两种错误都以钱计，代价公式几乎可以照字面套用，而且是一笔交易一笔交易地套。举个例子，数字是我假设的，但量级合理：漏报的代价是交易金额 $A$，误拒的代价是 $A$ 的 2%（损失的利润）加上 \$3 的摩擦成本。那么 \$20 的交易 $p^*$ 是 14.5%，\$200 的是 3.4%，\$2,000 的是 2.1%。金额越大，阈值应该越低。这也是风控团队除了报抓到多少笔，还要报按金额加权的 recall（拦下了多少欺诈金额）的原因。

和医疗相比，欺诈的混淆矩阵有三处更难量准：

- **标签来得晚。** 一笔欺诈要等客户报案，或者 chargeback（拒付退单）到了才算确认。上周的矩阵现在还没填完。
- **标签有选择性。** 被拒的交易永远不会告诉你它是不是欺诈。只有被告警的交易会很快有人核查，所以来得最快的反馈，恰恰来自模型本来就怀疑的那些。[Dal Pozzolo 等人](https://doi.org/10.1109/TNNLS.2017.2736643)把这两件事叫作 verification latency 和 alert-feedback interaction，并指出调查员能核查的告警数量有限，对他们来说真正要紧的是这些告警的 precision。
- **阳性会反击。** 欺诈手法会针对模型调整，一月份选好的工作点会慢慢漂走。

和筛查一样，答案也不止一个阈值：放行，要求一次性验证码，转人工审核，拒绝。验证码的角色相当于第二次乳腺检查：一次便宜的追加检查，真客户几秒钟就能通过，所以它可以放在一个很低的、对 recall 友好的阈值上。

### 虚假账户：规模把很小的比率变成一大群人

在平台上，阳性没那么稀少，却更难数清。Meta 通过[对月活用户抽样并人工标注](https://transparency.meta.com/policies/improving/prevalence-metric/)来估计 Facebook 上虚假账户的占比，[近期给出的数字是 4% 左右](https://www.socialmediatoday.com/news/meta-outlines-latest-data-on-content-removals-and-fake-accounts/807727/)。TikTok 公布的是处置量，而不是占比估计：[2026 年第二季度全球移除了约 1.468 亿个虚假账户](https://www.dhakatribune.com/feature/tech/423261/tiktok-releases-q2-2026-community-guidelines)。

假设一个平台有十亿个账号，其中 4% 是假的，检测器的 recall 是 90%，假阳性率是 0.1%。算出来 precision 是 97.4%，放在哪个分类器上都值得骄傲。可它同时意味着 96 万个真人被误伤。用百分比表示的 precision，把这一大群人藏起来了。Meta 的 [Deep Entity Classification](https://www.usenix.org/conference/usenixsecurity21/presentation/xu-teng) 系统的作者在解释滥用检测为什么需要极高的 precision 时，说的也是这一点。

两种错误落下来的方式也不同。漏掉一个虚假账户，代价是薄薄地摊开的：排序被稍微带偏，粉丝数有水分，广告指标没那么可信。一次误报却落在一个具体的人身上，他可能因此丢了观众，甚至丢了收入。所以分级动作在这里比在哪里都重要：

- **安静、可撤销的动作，按高 recall 来跑。** 不计入一个可疑账号的点赞，或者让它过一次验证码，对真用户几乎没有代价。
- **封号要按极高的 precision 来跑，** 通常还配有申诉通道，因为这是用户自己唯一撤不回的动作。

三个场景里，这里的测量最难，因为没有人会把真相交到你手上。precision 靠人工审核一批被处置账号的样本来估计；申诉成功的数量给出了误报的下限。recall 则根本观察不到，只能从“还留在平台上的账号里抽样抽出多少假的”反推。

矩阵里还有一个没有格子可放的维度：**时间**。一个账号发完垃圾内容之后才被封，在表里算真阳性，在实际效果上是失败。在注册环节拦住和事后清理，即使 precision 和 recall 完全相同，也是两种产品。

### 三个场景的共同点

![三条分数条，分别是乳腺筛查、刷卡支付和平台账号，每条被几个阈值切成分级的动作](graded-actions.svg "图 7。一个分数，几个阈值。低阈值管便宜、可撤销的动作，高阈值管代价高的动作。各段的宽度只是示意。")

三个场景都没有一次性地在 precision 和 recall 之间选边。它们都是阳性稀少、代价悬殊，也都给出了同一种设计：在一个分数上切几个阈值，分数越高，动作越贵、越难撤销。靠前的、便宜的那一步按 recall 来调，precision 再难看也能忍。靠后的、昂贵的那一步按 precision 来调，漏掉一些也能忍，因为前一步已经把这些样本送到了人或者第二次检查面前。取舍在每一个阈值上都真实存在；整个系统合起来，两头都拿到了。

## 汇报一个分类器之前，我会检查什么

1. **说清阳性类和计数单位。** 按检查还是按病人，按交易还是按卡，按账号还是按动作。
2. **用模型上线环境的阳性占比，** 而不是数据集里的。
3. **说清判阳性之后做什么，每种错误谁买单。** 这决定了代价比，代价比决定了阈值。
4. **报工作点上的混淆矩阵，而且报原始计数。** 只报比率会把小样本藏起来：30 个阳性里对了 27 个，recall 是 90%，95% 区间是 74% 到 97%。
5. **和“什么都不做”比一比。** 上面“全判阴性”的 accuracy 是 99.4%。
6. **给出上线阳性占比下的 PR 曲线，** 对概率设阈值之前先检查校准。
7. **问清楚标签从哪来、晚了多久。**
8. **上线之后重新量。** 阳性占比会变，在欺诈和滥用场景里，阳性还会适应你。

## 回头看我自己的项目

带着这些再看我以前的三个项目，读法不一样了。

**RadQNLI：accuracy 掉了，recall 没动。** 在 [RadQNLI](https://github.com/HenryVarro666/RadQNLI) 里，我用四个放射学测试集测 ChatGPT，四个集合里的负例和问题的词汇重合度一个比一个高。accuracy 从 0.81 掉到 0.62，看上去是题目变难了。但阳性类的 recall 在每个集合上都在 0.52 到 0.53 之间；变的是阳性的占比，从 21% 升到 70%，precision 也跟着从 0.56 升到 0.89。换成同样数量的随机负例做对照，accuracy 几乎掉得一样多，从 0.83 到 0.64。用矩阵可以把两种效应分开：在词汇重合的负例上，specificity 确实从 0.89 滑到了 0.84，而在随机负例上它一直在 0.92 附近，这一部分才是真的变难了。其余的是类别比例造成的：accuracy 等于 $\pi \cdot \text{recall} + (1-\pi) \cdot \text{specificity}$，recall 停在 0.53，$\pi$ 从 21% 升到 70%，accuracy 自然被拉向 recall。

**心脏病预测：哪一类才是阳性？** 在我的[心脏病课程项目](https://github.com/HenryVarro666/heart-disease-prediction)里，数据文件把 `target = 1` 编码成“没有病”。于是类别 1 的每一个指标描述的其实是健康人，这一类的“recall”在医生嘴里叫 specificity。61 个测试病人里 36 个没病、25 个有病，所以有病这一类的 recall 只建立在 25 个样本上：25 个里对 22 个是 88%，95% 区间是 70% 到 96%。而当时放在最前面的数字是 ROC AUC，它说明模型排序排得好不好，却不说明阈值该切在哪。

**去标识化：一个 recall 任务，和一个会骗人的下限。** 在 [open-LLM de-identification](https://github.com/HenryVarro666/open-llm-deid) 里，代价高的错误是漏掉一个名字，所以这是 recall 优先的任务；多遮掉一些只是影响可读性。我用的分数统计的是受保护的字符串被去掉了多少，而由于匹配的计数方式，一份完全没动过的病历就已经能拿 0.597。每个结果都得对着这个下限来读。这和“全判阴性也有 99.4%”是同一个教训：相信一个指标之前，先看看什么都不做能得几分。

## 参考文献

1. Lehman CD, Arao RF, Sprague BL, et al. [National Performance Benchmarks for Modern Screening Digital Mammography: Update from the Breast Cancer Surveillance Consortium](https://doi.org/10.1148/radiol.2016161174). *Radiology* 2017;283(1):49–58. 计数取自 [BCSC 基准表](https://www.bcsc-research.org/statistics/screening-performance-benchmarks-archive/screening-sens-spec-false-negative-archive)；假阳性数由“非癌症数减去真阴性数”得到。
2. Saito T, Rehmsmeier M. [The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets](https://doi.org/10.1371/journal.pone.0118432). *PLOS ONE* 2015;10(3):e0118432.
3. Davis J, Goadrich M. [The Relationship Between Precision-Recall and ROC Curves](https://doi.org/10.1145/1143844.1143874). ICML 2006.
4. Elkan C. [The Foundations of Cost-Sensitive Learning](https://cseweb.ucsd.edu/~elkan/rescale.pdf). IJCAI 2001.
5. Dal Pozzolo A, Boracchi G, Caelen O, Alippi C, Bontempi G. [Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy](https://doi.org/10.1109/TNNLS.2017.2736643). *IEEE Transactions on Neural Networks and Learning Systems* 2018;29(8):3784–3797.
6. The Nilson Report. [Card Fraud Losses Worldwide in 2023](https://nilsonreport.com/articles/card-fraud-losses-worldwide-in-2023/). 2024 年 12 月；每 100 美元的数字出自它 2025 年 1 月 6 日的新闻稿。
7. Meta Transparency Center. [Prevalence](https://transparency.meta.com/policies/improving/prevalence-metric/)：虚假账户占比的估计方法。4% 这个数字见 [Social Media Today](https://www.socialmediatoday.com/news/meta-outlines-latest-data-on-content-removals-and-fake-accounts/807727/) 的报道。
8. TikTok Community Guidelines Enforcement Report，2026 年第二季度，见 [Dhaka Tribune](https://www.dhakatribune.com/feature/tech/423261/tiktok-releases-q2-2026-community-guidelines) 的报道。
9. Xu T, Goossen G, Cevahir HK, et al. [Deep Entity Classification: Abusive Account Detection for Online Social Networks](https://www.usenix.org/conference/usenixsecurity21/presentation/xu-teng). USENIX Security 2021.
10. Pewsner D, Battaglia M, Minder C, Marx A, Bucher HC, Egger M. [Ruling a diagnosis in or out with "SpPIn" and "SnNOut": a note of caution](https://doi.org/10.1136/bmj.329.7459.209). *BMJ* 2004;329:209–213.

本文的图和所有计算出来的数字都来自同一个脚本：[`tools/notes/precision_vs_recall_figs.py`](https://github.com/HenryVarro666/henryvarro666.github.io/blob/master/tools/notes/precision_vs_recall_figs.py)。
