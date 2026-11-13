---
title: "ROC-AUC 和 PR-AUC：两块面积，两个问题"
date: 2026-11-12T20:00:00-06:00
lastmod: 2026-11-12T20:00:00-06:00
description: "ROC-AUC（ROC 曲线下面积）是随机一个阳性排在随机一个阴性前面的概率；PR-AUC（precision-recall 曲线下面积）是沿着列表往下走、遇到每个阳性时 precision 的平均。ROC-AUC 一样的两个模型，PR-AUC 可以差四倍；哪个更好，取决于列表会在哪里截断。"
topics: [data-science]
---
[上一篇](/zh/notes/precision-vs-recall/)是一个阈值一个阈值地看分类器。大多数模型比较会跳过阈值，直接报一块面积：ROC-AUC，即 ROC 曲线（receiver operating characteristic，受试者工作特征曲线）下的面积；PR-AUC，即 PR 曲线（precision-recall curve）下的面积；或者两个都报。这篇讲清楚每块面积量的是什么，为什么它们会对同一对模型给出相反的结论，以及我自己会报什么。

> **先说结论。** ROC-AUC 问的是：随机抽一个阳性、一个阴性，模型有多大概率把阳性排在前面。它不受阳性占比影响，对排序的每一段一视同仁。PR-AUC（按 average precision 计算）问的是：沿着列表往下走，一路上有多干净。它随阳性占比变化，而且主要由列表最前面那一段决定。两者都不选阈值。报哪个，看它回答的问题是否和上线场景一致；同时报它的基线和置信区间，旁边再放上工作点的数字。

关于用词：AUC 是 area under the curve（曲线下面积）。average precision 下文简称 AP，中文有时叫平均精度。

## 从一个阈值到所有阈值

分数上的每一个阈值都对应一张混淆矩阵，里面是真阳性（TP）、假阳性（FP）、真阴性（TN）和假阴性（FN）的计数。把阈值从最高分扫到最低分，就得到一串矩阵，每切一刀一张。两种曲线画的是同一串矩阵，只是坐标轴不同：

| | ROC 曲线 | PR 曲线 |
|---|---|---|
| 横轴 | 假阳性率，FP&nbsp;/&nbsp;(FP&nbsp;+&nbsp;TN) | recall，TP&nbsp;/&nbsp;(TP&nbsp;+&nbsp;FN) |
| 纵轴 | recall，TP&nbsp;/&nbsp;(TP&nbsp;+&nbsp;FN) | precision，TP&nbsp;/&nbsp;(TP&nbsp;+&nbsp;FP) |
| 随机排序 | 对角线 | 一条高度等于阳性占比的水平线 |
| 完美排序 | 左上角 | 右上角 |
| 看得见阳性占比吗 | 看不见：两个轴都是横着读的比率 | 看得见：precision 是竖着读的比率 |

最后一行就是上一篇的核心。曲线下面积把所有阈值压成一个数，这很方便，麻烦也从这里开始：这个数不再告诉你它在哪些阈值上表现好。

## ROC-AUC：把一对样本排对的概率

![十五个样本的排序列表，旁边是全部五十个“阳性—阴性”配对的网格，网格上画着 ROC 阶梯](pairs.svg "图 1。按分数排好的十五个样本。网格里每一格是一个“阳性—阴性”配对，阳性排在阴性前面就是蓝色。阶梯就是 ROC 曲线，蓝色格子正好是它下面的面积。")

把每一个阳性和每一个阴性两两配对。ROC-AUC 就是这些配对里“阳性分数更高”的比例，分数相同算半个：

$$
\text{ROC-AUC} = P(s^+ > s^-) + \tfrac{1}{2}\,P(s^+ = s^-)
$$

图 1 里有 5 个阳性、10 个阴性，一共 50 对，其中 39 对是阳性在前：ROC-AUC = 0.78。沿着列表往下走，边走边画 ROC 曲线：遇到阳性往上走一格，遇到阴性往右走一格。阶梯下面的格子，恰好就是阳性排在前面的那些配对，所以面积和配对计数是同一个数。它也等于 Mann–Whitney U 统计量除以配对总数；[Hanley 和 McNeil](https://doi.org/10.1148/radiology.143.1.7063747) 把“面积、随机一对排对的概率、Wilcoxon 统计量”三者相等这件事，变成了放射学里的标准读法。

同一个量也叫 AUROC（area under the ROC curve）、AUC，医学里叫 c 统计量（c-statistic，一致性指数）。信用评分领域报的是基尼系数，等于 2 × AUC − 1；AUC 为 0.90 时，基尼系数是 0.80。

从“配对”这个读法能直接得到：

- **0.5 等于抛硬币。** 低于 0.5 说明分数整个反了，把分数取反就得到 1 − AUC。
- **只看顺序。** 对分数做任何单调变换，它都不变，所以它完全不反映校准。一个把每个概率都高估了的模型，只要顺序不变，ROC-AUC 就和校准好的模型一样。
- **阳性占比不进来。** 把每个阴性复制十份，没有任何一对的顺序会变。
- **两类可以互换。** 把另一类当阳性、分数取反，配对还是那些配对。
- **不确定性主要由少的那一类决定。** 用 Hanley–McNeil 公式算：AUC 0.90，如果是在 50 个阳性和 50,000 个阴性上量的，95% 区间是 0.84 到 0.96；在 250 个阳性和 360 个阴性上量的，是 0.87 到 0.93。阴性多了一百多倍，也补不上阳性只有五分之一。

## PR-AUC：一路往下走时遇到的 precision

![同一个排序列表，每个阳性旁边写着到此为止的 precision；右边是阶梯形的 PR 曲线，阴影面积就是 average precision](average-precision.svg "图 2。同样十五个样本。每遇到一个阳性，记下到此为止的 precision；average precision 就是这五个数的平均，也等于阶梯形 PR 曲线下的阴影面积。虚线是随机排序。")

沿着同一个列表往下走，每遇到一个阳性，就记下到此为止的 precision。average precision（AP）就是这些数的平均：

$$
\text{AP} = \sum_n (R_n - R_{n-1})\,P_n = \frac{1}{n^+}\sum_{k=1}^{n^+} \big(\text{第 } k \text{ 个阳性处的 precision}\big)
$$

图 2 里五个 precision 依次是 1.00、0.67、0.75、0.57 和 0.45，所以 AP = 0.69。它也等于画成阶梯的 PR 曲线下的面积。这和图 1 是同一个列表，ROC-AUC 是 0.78；两个数回答的是关于它的不同问题。

从这个读法能得到：

- **基线是阳性占比，不是 0.5。** 随机排序一路上遇到的 precision 都约等于 π。这里是 0.33。同样是 0.30 的 AP，阳性占 1% 时是很好的成绩，阳性占一半时比随机还差。
- **它随阳性占比变化。** 下面会展开。
- **两类不能互换。** 同一组预测，“有病”类的 AP 和“没病”类的 AP 是两个不同的数。
- **怎么算很重要。** scikit-learn 的 `average_precision_score` 用的是上面的阶梯求和，[文档里提醒](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)用梯形法则（线性插值）算会偏乐观。在这个小例子上，用同样几个点做梯形得到 0.74，而不是 0.69。[Davis 和 Goadrich](https://doi.org/10.1145/1143844.1143874) 解释了原因：PR 空间里两点之间连一条直线，会高估中间任何一个阈值实际能达到的表现。
- **“PR-AUC” 不是一种算法。** 论文里同一个名字下面，可能是阶梯求和、梯形，或者 Davis–Goadrich 插值。[Boyd 等人](https://doi.org/10.1007/978-3-642-40994-3_29)比较了这几种估计和它们的置信区间，推荐了两种简单的区间。我会把用的函数写在数字旁边。
- **它用的是算术平均。** [Flach 和 Kull](https://papers.nips.cc/paper/2015/file/33e8075e9970de0cfea955afd4644bb2-Paper.pdf) 指出，这块面积对 precision 做的是算术平均，而 F 分数用的是调和平均；他们提出 precision-recall-gain 曲线，来修正尺度和基线。

## ROC-AUC 一样，模型不一样

![两条面积相同的 ROC 曲线，左边普通坐标，右边假阳性率取对数](same-auc.svg "图 3。两个 ROC-AUC 都是 0.90 的模型。普通坐标下看起来差不多；假阳性率取对数后，在误报很少的区域 B 远远领先，而误报超过五分之一后 A 反超。")

这里有两个 ROC-AUC 完全相同（0.90）的模型，都是双正态模型：模型 A 的阳性和阴性分数分布宽度相同；模型 B 的阳性分数宽了一倍，于是有一部分阳性分数远远高于所有阴性，另一部分则沉到阴性中间。

普通坐标下，两条 ROC 曲线几乎分不开。把假阳性率换成对数坐标，差别就出来了：

| 假阳性率 | 模型 A 的 recall | 模型 B 的 recall |
|---:|---:|---:|
| 0.1% | 10% | 46% |
| 1% | 30% | 61% |
| 10% | 70% | 79% |
| 30% | 90% | 88% |

两条曲线在假阳性率 22% 处相交，此时两者都能抓到 85% 的阳性。过了这一点 A 更好：要抓到 95% 的阳性，A 要标记 43% 的阴性，B 要标记 66%。

![同样两个模型在阳性占 0.5% 时的 PR 曲线：B 在 recall 0.3 之前 precision 接近 1，A 一开始就低于 0.9 并迅速下降](same-auc-pr.svg "图 4。同样两个模型，阳性占 0.5%。B 的 average precision 是 0.52，A 是 0.13，而两者的 ROC-AUC 完全一样。")

阳性占 0.5% 时，PR 曲线把差距摆得很清楚。recall 为 30% 时，B 的告警里 97% 是阳性，A 只有 13%。average precision：B 是 0.52，A 是 0.13。ROC-AUC 说它们一样，AP 却差了四倍。

哪个模型更好，取决于列表会在哪里截断：

- **如果只处理列表最前面的一段**（容量固定的告警队列，平台直接封掉的账号），B 好得多。
- **如果几乎每个阳性都必须找到**（初筛，把所有可疑的都送去做第二次检查），A 更好。

ROC-AUC 判平手，是因为它把假阳性率从 0 到 1 的 recall 全部平均了，其中包括 22% 以后 A 占优、而几乎没有哪个场景会用到的那一段。AP 偏向 B，是因为它主要由列表最前面决定。[McDermott 等人](https://arxiv.org/abs/2401.06091)把这一点说得很精确：修正排序里的一个错误，不管它在什么位置，ROC-AUC 给的奖励都一样；而 AP 对排在前面的修正奖励得多得多。

[Hand](https://doi.org/10.1007/s10994-009-5119-5) 说得更重：ROC 曲线一旦相交，AUC 更大的那个模型，可能在几乎所有实际会用的阈值上都更差。他把原因归到 AUC 对误分类代价做了隐含的加权，而且这个加权随模型而变；他提出 H-measure，用一个固定的加权来代替。

更简单的办法是保留 ROC 曲线，只在会用到的那一段上积分。[McClish](https://doi.org/10.1177/0272989X8900900307) 提出了这种 partial AUC（部分 AUC），并做了标准化：随机为 0.5，完美为 1。只看假阳性率 1% 以内，A 是 0.60，B 是 0.77；放宽到 10% 以内，A 是 0.75，B 是 0.85。部分面积选出的赢家和 AP 一样，但理由写在了明面上。

## 阳性占比会改变 PR-AUC，不会改变 ROC-AUC

![三个 ROC-AUC 固定的模型，average precision 随阳性占比变化的曲线](ap-vs-prevalence.svg "图 5。上一篇的筛查模型和模型 A、B，average precision 随阳性占比的变化。它们的 ROC-AUC（0.95、0.90、0.90）不动，average precision 随阳性变少而下降。")

模型不变，只改变阳性的比例：

| 阳性占比 | 筛查模型（ROC-AUC 0.95） | 模型 A（0.90） | 模型 B（0.90） |
|---:|---:|---:|---:|
| 50% | 0.95 | 0.90 | 0.93 |
| 5% | 0.67 | 0.46 | 0.71 |
| 1% | 0.41 | 0.20 | 0.58 |
| 0.58% | 0.33 | 0.14 | 0.54 |
| 0.1% | 0.14 | 0.04 | 0.42 |

筛查模型就是上一篇里穿过乳腺筛查工作点拟合出来的那个双正态模型。它的 ROC-AUC 是 0.95，在筛查的阳性占比下 average precision 是 0.33。两个数都对：排序排得很好，但列表里大部分仍然是没有癌症的检查。

由此有三点实际影响：

- **阳性占比不同的数据集，AP 不能直接比。** 模型换到一个阳性更少的人群，它本身并没有变差，变的是它的 AP。
- **测试集里欠采样了阴性，AP 会虚高。** 欠采样不影响 ROC-AUC，AP 则必须按上线环境的阳性占比重算。
- **报 AP 时把阳性占比写在旁边。** 除以阳性占比读起来更直观：模型 A 在 0.5% 时的 0.13，是随机排序的 26 倍。

## 不平衡数据就该用 PR-AUC 吗？

这是常见的建议。广泛使用的[信用卡欺诈数据集](https://api.openml.org/d/42175)来自 ULB（布鲁塞尔自由大学，Université libre de Bruxelles）的机器学习组，欺诈占交易的 0.17%；它的说明以类别不平衡为由，推荐用 AUPRC（area under the precision-recall curve，即 PR-AUC）。

McDermott 等人反驳了这个笼统的说法。在他们的分析里，决定性的因素根本不是不平衡，而是模型怎么用：要紧的错误是集中在排序的前段，还是分布在整条排序里。他们还说明，AP 会偏向于奖励阳性更常见的那些子群体上的改进，从而拉大群体之间的差距。

上面的例子和他们的结论一致。阳性占比决定的是 A 和 B 的差距在 AP 上看起来有多*大*，决定不了哪个模型才对。那要看列表在哪里截断。我的经验法则：

| 分数会怎么用 | 我报的汇总指标 |
|---|---|
| 只处理列表最前面：容量固定的告警队列、检索、直接封号 | AP 加阳性占比、审核容量内的 precision、要求 precision 下的 recall |
| 几乎每个阳性都要找到：初筛、排除性检查 | ROC-AUC、可用假阳性率范围内的 partial AUC、要求 sensitivity 下的 specificity |
| 比较阳性占比不同的群体 | 每个群体各自的 ROC-AUC；AP 必须配上各自的阳性占比 |
| 比较不同数据集 | ROC-AUC；AP 只在阳性占比相同时比 |
| 要选阈值 | 两个都不是：选定阈值下的混淆矩阵，加上代价 |

## 两块面积都说不了的事

- **在哪里截断。** 两者都在所有阈值上取平均，上线只用一个或几个。
- **概率有没有意义。** 两者都只看分数的顺序。校准要单独检查：可靠性图，或者 Brier 分数。
- **一次错误的代价。** 这是上一篇的主题；Hand 的批评正是 ROC-AUC 暗中带进了一个随模型而变的代价假设。
- **差别是不是真的。** 在同一批样本上打分的两个模型是相关的；[Hanley 和 McNeil 1983 年的方法](https://doi.org/10.1148/radiology.148.3.6878708)考虑了这种配对，能把差值的区间收窄。对 AP，按样本做 bootstrap 也能达到同样的目的。
- **取的是哪种平均。** 多于两类，或者标签是 one-hot 时，“AUC”可能是每类各自的、micro 平均的或 macro 平均的，三者并不相等（下面就有一个例子）。

## 回头看我自己的项目

带着这些重新跑了一遍我的[心脏病课程项目](https://github.com/HenryVarro666/heart-disease-prediction)，一张图里就碰到了上面三点。

**一张图，四个 AUC。** 支持向量机的 ROC 面板上印着四个数：“False”类 0.97，“True”类 0.97，micro 平均 0.96，macro 平均 0.97。两个类别各自的值相同，因为 ROC-AUC 不在乎哪一类叫阳性。micro 平均把 one-hot 标签的两列合成一条排序来算，是另一个量，碰巧是最低的那个。

**PR 曲线画的是健康人那一类。** PR 面板上写着 average precision 0.98。它是对类别 1 算的，而这份数据里 `target = 1` 表示*没有病*：61 个测试病人里有 36 个，基线是 0.59。重新跑 notebook 能复现图上的数（ROC-AUC 0.97，AP 0.98）。换成“有病”这一类（61 个里 25 个，基线 0.41），同一组预测的 AP 是 0.95，随机森林是 0.96。ROC-AUC 怎么算都是 0.97。

**这些差别都在噪声之内。** 只有 25 个有病的病人时，AUC 0.97 的 Hanley–McNeil 区间大约是 0.92 到 1.00。支持向量机（0.966）、随机森林（0.968）和逻辑回归（0.96）在这个测试集上分不出高下，这也是项目 README 里已经写下的结论。

## 我会报什么

1. **两条曲线，带上各自的基线：** ROC 图上的对角线，PR 图上的阳性占比线。ROC 曲线相交时，再加一张假阳性率取对数的图。
2. **ROC-AUC 和它的区间；** 多个模型共用一个测试集时，做配对比较。
3. **AP，以及算它时的阳性占比、区间和所用的函数**（`average_precision_score`，而不是在 PR 点上做梯形）。
4. **工作区间上的数字：** 审核容量内的 precision、要求 precision 下的 recall，或者实际会用的假阳性率范围内的 partial AUC。
5. **单独检查校准。**
6. **跨数据集、跨群体比较 AP 时，只在阳性占比相同时比，** 否则把各自的阳性占比写出来。
7. **哪一类是阳性。** 它会改变 AP，也会改变“recall”指的是什么。

## 参考文献

1. Hanley JA, McNeil BJ. [The meaning and use of the area under a receiver operating characteristic (ROC) curve](https://doi.org/10.1148/radiology.143.1.7063747). *Radiology* 1982;143(1):29–36.
2. Hanley JA, McNeil BJ. [A method of comparing the areas under receiver operating characteristic curves derived from the same cases](https://doi.org/10.1148/radiology.148.3.6878708). *Radiology* 1983;148(3):839–843.
3. McClish DK. [Analyzing a portion of the ROC curve](https://doi.org/10.1177/0272989X8900900307). *Medical Decision Making* 1989;9(3):190–195.
4. Davis J, Goadrich M. [The relationship between precision-recall and ROC curves](https://doi.org/10.1145/1143844.1143874). ICML 2006.
5. Hand DJ. [Measuring classifier performance: a coherent alternative to the area under the ROC curve](https://doi.org/10.1007/s10994-009-5119-5). *Machine Learning* 2009;77:103–123.
6. Boyd K, Eng KH, Page CD. [Area under the precision-recall curve: point estimates and confidence intervals](https://doi.org/10.1007/978-3-642-40994-3_29). ECML PKDD 2013.
7. Saito T, Rehmsmeier M. [The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets](https://doi.org/10.1371/journal.pone.0118432). *PLOS ONE* 2015;10(3):e0118432.
8. Flach P, Kull M. [Precision-recall-gain curves: PR analysis done right](https://papers.nips.cc/paper/2015/file/33e8075e9970de0cfea955afd4644bb2-Paper.pdf). NeurIPS 2015.
9. McDermott MBA, Zhang H, Hansen LH, Angelotti G, Gallifant J. [A closer look at AUROC and AUPRC under class imbalance](https://arxiv.org/abs/2401.06091). NeurIPS 2024.
10. scikit-learn. [`average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)，API 文档。
11. Machine Learning Group, ULB. [Credit card fraud detection dataset](https://api.openml.org/d/42175)，数据集说明。

本文的图和所有计算出来的数字都来自同一个脚本：[`tools/notes/roc_auc_vs_pr_auc_figs.py`](https://github.com/HenryVarro666/henryvarro666.github.io/blob/master/tools/notes/roc_auc_vs_pr_auc_figs.py)。心脏病项目的数字来自重新运行该项目的 notebook。
