---
title: "XGBoost：从一棵树到几百棵树"
date: 2026-11-12T21:30:00-06:00
lastmod: 2026-11-12T21:30:00-06:00
description: "XGBoost（eXtreme Gradient Boosting，极端梯度提升）一次加一棵小决策树，每棵树都按损失函数的斜率和曲率来拟合，再用惩罚项把它压住。这篇手算一次切分，讲清楚算法、真正要调的参数、阳性很少时和超出训练范围时会出什么问题，以及它都用在哪里。"
topics: [data-science]
---
[第一篇](/zh/notes/precision-vs-recall/)和[第二篇](/zh/notes/roc-auc-vs-pr-auc/)讲的是怎么评判模型给出的分数。这一篇讲实际中产出很多这类分数的那个模型。XGBoost 是 eXtreme Gradient Boosting（极端梯度提升）的缩写，是一个做梯度提升决策树的库：它用几百棵小树拼出一个预测，一次加一棵，每棵都去修正前面那些树还没做对的地方。陈天奇 2014 年在华盛顿大学把它作为研究项目开始写。同年的 Higgs Boson Machine Learning Challenge（希格斯玻色子机器学习挑战赛）让它出了名，他和 Tong He 拿到了特别奖“High Energy Physics meets Machine Learning”；之后的比赛让它更出名：2015 年 Kaggle 博客上发布的 29 个获胜方案里，有 17 个用了 XGBoost（[Chen and Guestrin, 2016](https://doi.org/10.1145/2939672.2939785)）。下文的数字都来自 2026 年 8 月发布的 3.4.1 版。

> **先说结论。** 每棵新树都按当前预测处损失的梯度（斜率）和 hessian（曲率）来拟合。每个叶子取牛顿步 −G/(H + λ)，每次切分都得用损失的下降“付得起”它的代价。算法就这么多，剩下的是快速找切分点和好用的默认值。实际用的时候：树的数量在留出数据上选；阳性很少时小心学习率；与其给类别加权，不如挪阈值；缺失值就让它缺着；别让树去预测训练范围以外的地方。

## 问题：表格数据

企业和医院手里的数据大多是表：一行一笔交易、一个病人或一个账户；列的类型五花八门（金额、次数、类别、化验值），量纲不同，还有空缺。信号常常藏在阈值和交互里：一笔消费金额大，*而且*这张卡一小时前刚在另一个国家用过，才可疑。

树正适合这种数据。一次切分问的是“这一列是不是小于某个值”，所以列的量纲无所谓，对它做任何单调变换，长出来的树都一样。切分一层套一层，就有了交互。缺失值可以分流，不必填补。而且树训练和预测都便宜。

基准测试也这么说，只是附带的条件越来越多。在 45 个中等规模（约一万行）的数据集上，[Grinsztajn 等人](https://arxiv.org/abs/2207.08815)发现树模型领先深度网络，并把差距归因到目标函数不规则、无用列多这两点。更新的 [TabArena](https://arxiv.org/abs/2506.16791) 基准做了细致的调参和集成，发现最好的深度表格模型已经追平甚至超过 boosting 树；表格基础模型 [TabPFN](https://doi.org/10.1038/s41586-024-08328-6)（Tabular Prior-data Fitted Network）在一万行以内的数据集上领先。boosting 树依然是很强的默认选择，但已经不是自动的赢家。

## 一棵树

![一棵深度为 2 的回归树，旁边是它在 40 个点上算出的阶梯函数](one-tree.svg "图 1。一棵深度为 2 的回归树。三个关于 x 的问题把每个点送进四个叶子之一；每个叶子预测它那些点的均值。画成函数，这棵树就是一段阶梯。")

决策树就是一组是非题，把每一行送到一个叶子，每个叶子里存一个数。图 1 是一棵在 40 个点上拟合出来的深度为 2 的树：三个问题，四个叶子，一段阶梯。深的树什么阶梯都拟合得出来，包括噪声；浅的树又太粗。把很多树合起来有两种办法：

- **把很多深树平均起来**，每棵在重抽样的数据上独立地长。这是随机森林。平均能降低每棵都过拟合的树的方差。
- **把很多浅树一棵接一棵加起来**，每棵都去拟合前面加起来还错的部分。这是 boosting。相加能降低每棵都欠拟合的树的偏差。

## Boosting：每棵树修正剩下的部分

![四格图：boosting 模型在 1、5、30、300 棵树之后的样子；下面是训练误差和测试误差随树的数量的变化](boosting-rounds.svg "图 2。同样 40 个点上的 boosting，每棵树深度 2，学习率 0.3。训练点上的误差一路降到接近零；同一条曲线上的新点的误差在 33 棵树时最低，之后模型开始拟合噪声，误差回升。")

从一个常数开始：目标值的均值。用一棵小树去拟合残差，也就是目标值和当前预测之差。把它缩小一些再加上去：

$$
F_m(x) = F_{m-1}(x) + \eta\, f_m(x)
$$

η（eta，学习率）是一个像 0.3 或 0.05 这样的数。然后重复。图 2 里，40 个训练点上的均方根误差（root-mean-square error，RMSE）从 1 棵树时的 1.24，到 5 棵树 0.62，30 棵树 0.22，300 棵树 0.01。在同一条曲线上另取的 400 个新点上，它从 1.32 降到 0.77，再到 0.47，然后又回升到 0.55。最好的是 33 棵树时的 0.47；生成这些点的那条真实曲线本身的得分是 0.36，也就是噪声水平。

所以树的数量不是“越多越好”。它要在树没见过的数据上选，通常用早停（early stopping）：训练时盯着一个验证集，留下验证误差最低的那一轮。

## 残差就是梯度

对平方误差 ½(y − F)²，残差 y − F 正是损失对预测求导再取负号。[Friedman](https://doi.org/10.1214/aos/1013203451) 把这一点变成了通用的配方，叫梯度提升机（gradient boosting machine，GBM）：对任何可导的损失，让每棵树去拟合负梯度。分类问题里，模型的输出 F 是对数几率（log-odds），概率是 \(p = 1/(1 + e^{-F})\)，用 log loss 时每一行的梯度就是 p − y。

XGBoost 还用到二阶导数，也就是 hessian。在当前预测处，对每一行 i：

| 损失 | 梯度 gᵢ | Hessian hᵢ |
|---|---|---|
| 平方误差 ½(yᵢ − F)² | F − yᵢ | 1 |
| Log loss，F 是对数几率 | pᵢ − yᵢ | pᵢ(1 − pᵢ) |

一棵树把每一行分到一个叶子，同一个叶子 j 里的行都得到同一个值 \(w_j\)。记 \(G_j\) 和 \(H_j\) 为这个叶子里所有行的 gᵢ 之和与 hᵢ 之和。把损失做二阶泰勒展开，再加上 XGBoost 的惩罚项，这个叶子的部分是

$$
G_j\, w_j + \tfrac{1}{2}\,(H_j + \lambda)\, w_j^2
$$

这是关于 \(w_j\) 的一条抛物线，最低点在

$$
w_j^\ast = -\frac{G_j}{H_j + \lambda}
$$

这就是一步牛顿法，被 λ（lambda，L2 惩罚，作用在叶子值的平方上）拉住了一点。完整的目标函数对每棵树 f 加惩罚 Ω(f) = γT + ½λ‖w‖²，其中 T 是叶子数，γ（gamma）是每个叶子的“价格”。把 w* 代回去，把一个叶子切成左右两个子节点的价值是

$$
\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma
$$

最主要的几个参数，直接从这两个公式里就能读出来：

- **λ 把叶子值往零拉，** 对 H 小的叶子拉得最狠，因为那里数据说得最少。
- **γ 是一个叶子的价格。** 损失下降不到这个数的切分会被剪掉。
- **`min_child_weight` 限的是 H 的下界，不是行数的下界。** 平方误差下每行 h = 1，两者一样。log loss 下预测为 p 的一行 h = p(1 − p)，模型越有把握，它越小。后面还会讲到。

## 手算一次切分

![12 笔交易排在数轴上，其中 4 笔是盗刷；下面是 11 个可能阈值各自的 gain 柱子](split-gain.svg "图 3。12 笔刷卡交易，按这张卡过去 24 小时的消费次数排序。每根柱子是在相邻两笔之间切一刀的 gain；灰色柱子被 min_child_weight 挡掉了。阈值 5.5 以 2.40 的 gain 胜出。")

12 笔刷卡交易，一个特征：这张卡过去 24 小时的消费次数，从 1 到 12。其中 4 笔是盗刷。XGBoost 3 让每一行都从标签均值出发，p = 1/3，对数几率是 ln(1/2) = −0.693。于是：

- 每笔盗刷 g = 1/3 − 1 = −2/3，每笔正常交易 g = +1/3；
- 每一行 h = (1/3)(2/3) = 2/9 ≈ 0.222；
- 12 行加起来，G = 0，H = 8/3。

试一下阈值 5.5。左边是消费次数 1 到 5 的那五笔正常交易：\(G_L = 5/3\)，\(H_L = 10/9\)。右边是另外七笔：\(G_R = -5/3\)，\(H_R = 14/9\)。λ = 1，父节点那一项是零（G = 0），方括号里是

$$
\frac{(5/3)^2}{10/9 + 1} + \frac{(5/3)^2}{14/9 + 1} = \frac{25}{19} + \frac{25}{23} = 2.40
$$

两个叶子的值是 \(w_L = -(5/3)/(19/9) = -0.789\) 和 \(w_R = +0.652\)。乘上 η = 0.3，树里存的是 −0.237 和 +0.196：左边的预测概率从 0.333 变成 0.283，右边变成 0.378。在同样 12 行上用 `tree_method="exact"` 跑 XGBoost，打印出来的正是这棵树：`[f0<5.5] gain=2.40274596`，叶子 `-0.236842126` 和 `0.195652172`，两个叶子的“cover”是 1.11 和 1.56，也就是 H。

图 3 画出了每个阈值的 gain。最高的柱子在 8.5（2.47），但它右边只有四行，H = 4 × 2/9 = 0.89，低于默认的 `min_child_weight` = 1。这里每行只“重” 0.222，所以一个子节点至少要五行；11 个阈值里有 8 个被挡掉。设 `min_child_weight=0`，XGBoost 就会改在 8.5 切。

这个例子还验证了库的两个细节：

- **它报告的 gain 就是方括号里那部分，没有乘 ½，** γ 也是跟这个数比的。这里 `gamma=2.41` 会剪掉这次切分，`gamma=2.40` 会留下。
- **导出的树里，叶子值已经乘过 η 了。**

在真实数据上，要盯的是 hessian。盗刷占 0.2% 时，一个有把握的模型对大多数交易预测 p ≈ 0.002，每笔贡献的 h ≈ 0.002，`min_child_weight=1` 就意味着每个叶子要大约 500 笔。同一个设置，在类别均衡的数据和稀有事件数据上，含义天差地别。

## 缺失值自己选边

![同样 12 笔交易加上 2 笔消费次数缺失的交易（都是盗刷），一边把它们放左边，一边放右边，各自的 gain](missing.svg "图 4。再加两笔消费次数缺失的交易，都是盗刷。在阈值 5.5 处，把它们放左边 gain 是 0.74，放右边是 3.50。XGBoost 对每个阈值都试两个方向，把更好的那个存进树里。")

XGBoost 不需要先填补缺失值。扫描一个特征时，它只对有值的行排序；对每个阈值，把没有值的那些行分别放到左边和右边各试一次（[Chen and Guestrin](https://doi.org/10.1145/2939672.2939785) 叫它 sparsity-aware split finding，稀疏感知的切分查找）。更好的方向作为这次切分的默认方向存进树里。

图 4 里，两笔没有消费次数的交易都是盗刷。在 5.5 处，把它们和五笔低次数的正常交易放一起，gain 是 0.74；放到右边是 3.50。所有阈值、两个方向一起比，胜出的是 8.5、缺失值走右边，gain 4.38。注意这两行还让这次切分变得合法了：右边现在有六行。

由此得出：

- **“缺失”本身可能就是信号。** 一张没有消费记录的卡，一项没开的化验。拿均值或零去填，这些信息就丢了。让它们保持 `NaN`（not a number）。
- **默认方向是悄悄生效的。** 如果上线后某个特征因为训练时从没出现过的原因而缺失（一个坏掉的 join，一个新的数据源），这些行都会走学到的默认分支，没有任何报警。要拿线上每个特征的缺失率和训练时对比。
- **稀疏输入里没存的就算缺失。** 对稀疏矩阵，没有存储的元素按缺失处理，不是按零处理。

## 快速找切分点

exact 方法对每个特征排序，扫描每个阈值，就像上面那样。从 2.0 版起默认是 `tree_method="hist"`：每个特征事先按分位数切成至多 `max_bin` = 256 个桶，找切分点时只用每个桶里 g 和 h 的和。论文里的 weighted quantile sketch（加权分位数草图）放切点时按每行的 hessian 加权；`approx` 方法仍然每一轮都这样重新做草图，而 `hist` 只在训练开始前切一次。在那 12 行的例子上，`hist` 切在“x < 6”，和“x < 5.5”分出来的两组完全一样。

XGBoost 快的其余原因都是系统层面的工作：数据按列压缩成块存储、对缓存友好的访问、多核、数据比内存还大时的外存训练（3.0 版重做过）、用 `device="cuda"` 跑在图形处理器（graphics processing unit，GPU）上，以及在 Dask、Spark 和 Ray 上做分布式训练。

## 要调的参数

| 参数 | 默认值 | 作用 | 我从哪里开始 |
|---|---|---|---|
| `n_estimators` / `num_boost_round` | 100（scikit-learn 接口）/ 10（`xgb.train`） | 树的数量 | 设大一些，配合验证集上的 `early_stopping_rounds` |
| `learning_rate`（`eta`） | 0.3 | 把每棵树缩小 | 0.03–0.1 |
| `max_depth` | 6 | 每棵树的深度；深度 d 的树最多组合 d 个特征 | 3–8 |
| `min_child_weight` | 1 | 子节点里 hessian 之和的下限 | 数据噪声大就调高；阳性很少时注意它的量级 |
| `gamma`（`min_split_loss`） | 0 | 保留一次切分所需的最小 gain（按库报告的数） | 0，再试小值 |
| `reg_lambda`（`lambda`） | 1 | 叶子值的 L2 惩罚（平方） | 1–10 |
| `reg_alpha`（`alpha`） | 0 | 叶子值的 L1 惩罚（绝对值） | 0 |
| `subsample` | 1 | 每棵树看到的行的比例 | 0.7–0.9 |
| `colsample_bytree` | 1 | 每棵树看到的列的比例 | 0.5–0.9 |
| `max_bin` | 256 | 每个特征的直方图桶数 | 不动 |
| `scale_pos_weight` | 1 | 把阳性行的权重乘上一个数 | 见下文 |
| `max_delta_step` | 0（不设上限） | 给每个叶子值设上限 | 阳性极少时设 1 |
| `monotone_constraints` | 无 | 强制某个特征的影响只能朝一个方向 | 会被审查的场景 |

行和列的抽样来自 [Friedman 的随机梯度提升](https://doi.org/10.1016/S0167-9473(01)00065-2)：每棵树只看数据的一个随机部分，树之间更不相像，加起来也就不那么容易追着噪声跑。XGBoost 的[调参说明](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html)也是这样分组的：`max_depth`、`min_child_weight` 和 `gamma` 直接控制复杂度；抽样，以及更小的 `eta` 配更多的树，用来增加稳健性。

## 阳性很少时的学习率

![四条曲线：学习率 0.3、0.1、0.03，以及 0.3 加叶子值上限 1，验证集 log loss 随树的数量变化](learning-rate.svg "图 5。信用卡盗刷数据，按时间切分。默认学习率下，最初几棵树一步迈过头，模型再也没缓过来；更小的学习率更晚、但到达更低的损失；把每个叶子的值限制在 1 以内，默认学习率也就没问题了。")

从这里开始用真实数据：ULB（Université libre de Bruxelles，布鲁塞尔自由大学）机器学习组的[信用卡盗刷数据集](https://api.openml.org/d/42175)，两天内的 284,807 笔刷卡交易，其中 492 笔是盗刷（0.17%）。列是 28 个匿名化的主成分（V1 到 V28）、金额，以及距第一笔交易的秒数。我按时间切分，就像上线的模型会遇到的那样：前 60% 的交易训练（360 笔盗刷，0.21%），接下来 20% 验证（57 笔，0.10%），最后 20% 测试（75 笔，0.13%）。

图 5 是验证集 log loss 随树的数量的变化。对所有交易都预测训练集盗刷率的模型得分是 0.0083。表的最后一列是在那个树数下，测试集上的 average precision（AP）。

| 学习率 | 验证集 log loss 最低值 | 在第几棵树 | 那时的测试集 AP |
|---|---|---|---|
| 0.3（默认） | 0.0062 | 27 | 0.58 |
| 0.1 | 0.0028 | 37 | 0.61 |
| 0.03 | 0.0022 | 187 | 0.79 |
| 0.3 加 `max_delta_step=1` | 0.0021 | 32 | 0.79 |

默认学习率只比常数模型好一点点，测试集 AP 是 0.58，别人是 0.79。原因在第一棵树。每一行都从 p = 0.0021 出发，所以每一行的 h ≈ 0.002。一个聚了些盗刷的叶子，G 是一个很大的负数，H 却很小，牛顿步 −G/(H + λ) 大致就是里面盗刷的笔数这个量级。第一棵树最大的叶子，乘过 0.3 以后仍然是对数几率 +35.7：这个叶子里每一笔正常交易的概率都从 0.002 变成了几乎是 1。一棵树之后，训练集 log loss 是 0.086，比起点的 0.015 还差。二阶近似只在当前预测附近才准，而这一步迈到了远远超出那个范围的地方。

两个办法在这里效果一样：调小学习率，代价是要更多的树；或者用 `max_delta_step` 给每个叶子值设上限（这里是 1，缩小后就是 0.3）。XGBoost 的参数文档建议，在类别极度不均衡的逻辑回归里设 1 到 10。加上上限后，默认学习率在 32 棵树时就达到了四种设置里最低的损失。

## 阳性很少：给类别加权，还是挪阈值？

![默认权重和 scale_pos_weight 两个模型的 PR 曲线几乎重合，两条曲线上各标出阈值 0.5 的位置](scale-pos-weight.svg "图 6。最后 20% 的交易，75 笔盗刷。AP：average precision；ROC-AUC：ROC 曲线（receiver operating characteristic，受试者工作特征曲线）下面积。把阳性的权重放大 474 倍，PR 曲线几乎没动；动的是阈值 0.5 落在曲线上的位置。默认模型在阈值 0.33 处，正好落在加权模型的 0.5 旁边。")

处理不均衡数据的常见建议是 `scale_pos_weight = 阴性数 / 阳性数`，这里是 474。它把每笔盗刷的梯度和 hessian 都乘上 474。我用 η = 0.05、以验证集 AP 早停，训练了两个版本，在测试集上用 AP 和 ROC-AUC（ROC 曲线下面积）来比（两者在[第二篇](/zh/notes/roc-auc-vs-pr-auc/)里拆开讲过）：

| | 默认权重 | `scale_pos_weight` = 474 |
|---|---|---|
| 树的数量 | 125 | 243 |
| AP | 0.80 | 0.78 |
| ROC-AUC | 0.980 | 0.976 |
| p ≥ 0.5 标出的交易 | 53 笔，其中 51 笔盗刷 | 60 笔，其中 53 笔盗刷 |
| 盗刷交易分数的中位数 | 0.85 | 1.00 |

AP 之差的 95% bootstrap 区间是 −0.01 到 +0.05：这个测试集分辨不出差别。排序基本一样，分数不一样。加权把盗刷的分数推向 1，所以 p ≥ 0.5 标出的交易更多。默认模型用一个更低的阈值就能到同一个地方：阈值 0.33 时它标出 62 笔、抓到 54 笔盗刷，加权模型在 0.5 时是 60 笔和 53 笔。

XGBoost 自己的[调参说明](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html)划的是同一条线：如果只关心排序（ROC-AUC），用 `scale_pos_weight` 平衡权重；如果需要正确的概率，就别重新平衡，设 `max_delta_step` 帮助收敛。写完前两篇之后，我会说得更重一些：加权模型的 0.5，是别人替你选的阈值，背后的代价比谁也没写下来。用真实的类别比例训练，保住有意义的概率，再按代价和人力去选阈值，就像[第一篇](/zh/notes/precision-vs-recall/#从代价到阈值)里那样。

## 树不会外推

![一条只在 x ≤ 10 处观测到的直线趋势：boosting 树在最后一个训练点之后变成水平线，直线则继续延伸](extrapolation.svg "图 7。一条直线趋势，训练时只见过 x 在 0 到 10 之间的点。boosting 树把训练点拟合得很紧，之后只给常数；直线一直延续下去。在 x = 15 处，真值是 24.5，直线给 25.1，树给 17.7。")

在某个特征上，一旦超过所有树用到的最大阈值，不管取值多大，每棵树给的都是同一个叶子，加起来也就是一条水平线。模型只会重复它在训练数据边缘说过的话。图 7 里，树对 40 个训练点的拟合比直线好（RMSE 0.27 对 0.94），对 x = 10 以外的 15 个点却差得多（3.78 对 1.26）。

盗刷数据里也有同样的问题，藏在一列里。Time 是距第一笔交易的秒数：训练集里是 0 到 120,396，测试集里是 145,248 到 172,792。训练好的模型在 Time 上切了 89 次，阈值全都小于 111,089 秒，所以每一笔测试交易在这些切分上都落在“晚”的那一边。Time 的 SHAP 值（SHapley Additive exPlanations，下一节解释）在测试集上平均是对数几率 −0.58，在训练集上是 −0.01：这是日历带来的一股向下的推力，和交易本身没有关系。去掉这一列，average precision 几乎不变（0.79 对 0.80）。

所以：随时间增长的特征（时间戳、从某个固定日期算起的账龄、累计计数、受通胀影响的价格）应该换成始终落在范围内的量：一天中的几点、距上一次事件的时间、比值、差分。如果目标本身有趋势，就单独给趋势建模，让树去拟合剩下的部分。

## 读懂一个训练好的模型

XGBoost 有好几种特征重要性，彼此并不一致。在上一节那个默认权重的盗刷模型上：

| 排序依据 | 前五名 | Amount | Time |
|---|---|---|---|
| weight（切分次数） | V4、V14、V10、V12、Amount | 第 5 | 第 16 |
| gain（每次切分的平均 gain） | V14、V10、V12、V27、V4 | 第 15 | 第 27 |
| cover（每次切分的平均 hessian） | V10、V14、V4、V20、V7 | 第 10 | 第 11 |
| 测试集上 \|SHAP\| 的均值 | V4、Time、V10、V12、V14 | 第 7 | 第 2 |

“weight”数的是切分次数，包括模型在训练后期为了微调而做的大量小切分：Amount 按次数排第 5，按 gain 排第 15。gain 和 cover 描述的是训练数据。它们都说明不了某一个预测。

SHAP 值可以。对每一行，它把模型的输出（对数几率）拆成每个特征一份贡献再加一个常数，用的是博弈论里的 Shapley 值；对树模型，[Lundberg 等人的 TreeSHAP](https://doi.org/10.1038/s42256-019-0138-9) 能在多项式时间内精确算出来，XGBoost 内置了它（`predict(..., pred_contribs=True)`）。在测试集上，这些贡献加起来和模型输出的差在 10⁻⁵ 以内。上面 Time 的问题也正是这样暴露出来的：按 |SHAP| 均值排第 2，按 gain 排第 27。

在会有人问“模型会不会在某处朝反方向走”的地方，比如信贷和医疗，`monotone_constraints` 能强制预测随某个特征只朝一个方向变，`interaction_constraints` 能限制哪些特征可以出现在同一棵树里。

## 用在哪里

| 领域 | 问题 | 目标函数 | 要留意什么 |
|---|---|---|---|
| 刷卡盗刷 | 这笔交易是盗刷吗？ | `binary:logistic`，用 AP 评估 | 阳性极少；标签要等几周后的拒付（chargeback）才来；盗刷手法会变 |
| 信贷风险 | 这个借款人会违约吗？ | `binary:logistic` | 单调约束；每次拒贷都要给出理由；人群随经济周期变化 |
| 临床风险 | 这个病人会再入院、恶化、存活吗？ | `binary:logistic`；`survival:cox`（Cox 比例风险）、`survival:aft`（accelerated failure time，加速失效时间） | 没开的化验本身就有信息；校准；医院之间的差异 |
| 广告与推荐 | 这个用户会点吗？ | `binary:logistic` | 校准直接影响竞价；[Facebook](https://doi.org/10.1145/2648584.2648589) 曾用 boosting 树给逻辑回归模型做特征变换 |
| 搜索排序 | 按什么顺序排？ | `rank:ndcg`（normalized discounted cumulative gain，归一化折损累计增益）、`rank:pairwise` | 以查询为组；树 boosting 排序器 LambdaMART 是经典做法 |
| 需求预测 | 能卖出多少？ | `reg:squarederror`、`reg:tweedie`、`reg:quantileerror` | 滞后项和滚动窗口；趋势要另外处理（见上）；[M5 竞赛](https://econpapers.repec.org/RePEc:eee:intfor:v:38:y:2022:i:4:p:1346-1364)的冠军用的是 LightGBM 模型 |
| 物理 | 这次碰撞是信号还是背景？ | `binary:logistic` | XGBoost 成名的地方（[Chen and He](https://proceedings.mlr.press/v42/chen14.html)） |
| 虚假账户 | 这个账户是假的吗？ | 在行为和图特征上用 `binary:logistic` | 对手会适应；标签有噪声 |

第一篇里的三个场景，癌症筛查、刷卡盗刷和虚假账户，都在这张表里。每一个场景里，模型都只是工作的一小部分；特征、标签和阈值决定了大部分结果。

## XGBoost 和它的邻居

| 模型 | 树 | 类别型列 | 长处 | 我什么时候选它 |
|---|---|---|---|---|
| XGBoost | 按层生长（或按最优叶子），带惩罚的牛顿步 | 原生支持，3.1 起不再是实验功能 | 缺失值、约束、GPU、成熟的工具链 | 表格数据的通用默认 |
| [LightGBM](https://mlanthology.org/neurips/2017/ke2017neurips-lightgbm/) | 按叶子生长；按梯度大小抽样（Gradient-based One-Side Sampling，GOSS），把稀疏列打包（Exclusive Feature Bundling，EFB） | 原生 | 大而宽的数据上速度快 | 上百万行、很多稀疏列 |
| [CatBoost](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html) | 对称树；用 ordered boosting 防止目标泄漏 | ordered target statistics | 类别型特征多的数据，默认值好用 | 很多高基数类别 |
| 随机森林 | 深、互相独立、取平均 | 需要编码 | 参数少，不容易用坏 | 又快又稳的基线 |
| 深度表格网络（TabM、RealMLP） | 无 | embedding | 调参加集成后在 TabArena 上打平；能和图像、文本一起端到端训练 | 表格只是多种输入之一 |
| TabPFN | 无：一个预训练的 transformer 在上下文里读训练集 | 支持 | 一万行以内最好，不用对每个数据集训练 | 小数据 |

## 什么时候我不会用它

- **原始图像、音频或文本。** 用预训练网络，或者把它的 embedding 喂给树。
- **目标必须延伸到训练范围以外。** 趋势、增长、远超观测范围的剂量。
- **只有几百行。** 用正则化线性模型或 TabPFN，并且老老实实看一眼评估本身的噪声（见下文）。
- **关系必须平滑或端到端可导时，** 比如放在一个更大的网络或优化器里面。

## 检查清单

1. **数据有时间就按时间切分。** 随机切分会让模型偷看到未来。
2. **先有基线：** 常数模型，然后是逻辑回归或线性模型。
3. **用早停选树的数量，** 指标要和用途一致：在乎概率就用 log loss，只在乎排序最前面就用 average precision。
4. **阳性很少时，看一眼最初几轮。** 调小 `eta`，或者设 `max_delta_step=1`。保持真实的类别比例，挪阈值而不是加权。
5. **缺失值保持 `NaN`，** 并对比训练和线上的缺失率。
6. **去掉或变换随时间增长的列。**
7. **用 SHAP，而不是切分次数，** 来看模型依赖什么。
8. **报不确定性：** 测试指标的 bootstrap 区间，或重复交叉验证。
9. **如果有人会把分数当概率读，就在留出数据上做校准。**

## 回头看我自己的项目

我的[心脏病课程项目](https://github.com/HenryVarro666/heart-disease-prediction)在 Cleveland 数据上比较了五个分类器。它的 notebook 导入了 `GradientBoostingClassifier`、`AdaBoostClassifier` 和 `ExtraTreesClassifier`，却一个都没用。我在同样的切分上（242 个病人训练，61 个测试）加了 XGBoost，又在全部 303 人上把 5 折交叉验证重复了 10 次，作为第二个参考：

| 模型 | 测试集准确率 | 测试集 ROC-AUC | 交叉验证 ROC-AUC（均值 ± 标准差） |
|---|---|---|---|
| 逻辑回归 | 0.918 | 0.963 | 0.905 ± 0.031 |
| 支持向量机 | 0.918 | 0.966 | 0.906 ± 0.031 |
| 随机森林 | 0.918 | 0.968 | 0.912 ± 0.029 |
| XGBoost，默认参数 | 0.885 | 0.957 | 0.887 ± 0.028 |
| XGBoost，小树 | 0.885 | 0.959 | 0.902 ± 0.028 |

“小树”是 300 棵深度为 2 的树，η = 0.05，行和列都抽样 0.8。有两点很明显。用默认参数时，100 棵深度 6、η = 0.3 的树对 242 个病人来说模型太大，XGBoost 垫底。换成小树后它追上来了，但没有超过别人。而且折与折之间的波动（标准差 0.03，范围从 0.81 到 0.97）比任何两个模型之间的差距都大。在 303 行上，“哪个模型最好”这个问题，这份数据给不出答案；基准测试在这个规模上指向的是 TabPFN 或正则化线性模型。

## 参考文献

1. Chen T, Guestrin C. [XGBoost: a scalable tree boosting system](https://doi.org/10.1145/2939672.2939785). KDD 2016.
2. Friedman JH. [Greedy function approximation: a gradient boosting machine](https://doi.org/10.1214/aos/1013203451). *Annals of Statistics* 2001;29(5):1189–1232.
3. Friedman JH. [Stochastic gradient boosting](https://doi.org/10.1016/S0167-9473(01)00065-2). *Computational Statistics & Data Analysis* 2002;38(4):367–378.
4. Chen T, He T. [Higgs boson discovery with boosted trees](https://proceedings.mlr.press/v42/chen14.html). NIPS 2014 Workshop on High-energy Physics and Machine Learning, PMLR 42:69–80.
5. XGBoost developers. [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html) 和 [notes on parameter tuning](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html)，官方文档。
6. Ke G, Meng Q, Finley T, et al. [LightGBM: a highly efficient gradient boosting decision tree](https://mlanthology.org/neurips/2017/ke2017neurips-lightgbm/). NeurIPS 2017.
7. Prokhorenkova L, Gusev G, Vorobev A, Dorogush AV, Gulin A. [CatBoost: unbiased boosting with categorical features](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html). NeurIPS 2018.
8. Lundberg SM, Erion G, Chen H, et al. [From local explanations to global understanding with explainable AI for trees](https://doi.org/10.1038/s42256-019-0138-9). *Nature Machine Intelligence* 2020;2:56–67.
9. Grinsztajn L, Oyallon E, Varoquaux G. [Why do tree-based models still outperform deep learning on typical tabular data?](https://arxiv.org/abs/2207.08815) NeurIPS 2022 Datasets and Benchmarks.
10. Erickson N, Purucker L, Tschalzev A, et al. [TabArena: a living benchmark for machine learning on tabular data](https://arxiv.org/abs/2506.16791). NeurIPS 2025 Datasets and Benchmarks.
11. Hollmann N, Müller S, Purucker L, et al. [Accurate predictions on small data with a tabular foundation model](https://doi.org/10.1038/s41586-024-08328-6). *Nature* 2025;637:319–326.
12. He X, Pan J, Jin O, et al. [Practical lessons from predicting clicks on ads at Facebook](https://doi.org/10.1145/2648584.2648589). ADKDD 2014.
13. Makridakis S, Spiliotis E, Assimakopoulos V. [M5 accuracy competition: results, findings, and conclusions](https://econpapers.repec.org/RePEc:eee:intfor:v:38:y:2022:i:4:p:1346-1364). *International Journal of Forecasting* 2022;38(4):1346–1364.
14. Machine Learning Group, ULB. [Credit card fraud detection dataset](https://api.openml.org/d/42175)，数据集说明。

玩具例子的图和数字来自 [`tools/notes/xgboost_figs.py`](https://github.com/HenryVarro666/henryvarro666.github.io/blob/master/tools/notes/xgboost_figs.py)，它用的是从零写的一个算法版本 [`tinyboost.py`](https://github.com/HenryVarro666/henryvarro666.github.io/blob/master/tools/notes/tinyboost.py)，大约一百行纯 Python。[`xgboost_runs.py`](https://github.com/HenryVarro666/henryvarro666.github.io/blob/master/tools/notes/xgboost_runs.py) 逐棵树检查它和 XGBoost 的结果一致，并算出信用卡和心脏病的数字。
