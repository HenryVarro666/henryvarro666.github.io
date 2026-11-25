---
title: "欺诈检测建模：练习题与参考答案"
date: 2026-11-23T20:00:00-06:00
lastmod: 2026-11-25T04:00:00-06:00
description: "欺诈建模方向的练习题和参考答案：类别不平衡、评估指标、按时间切分验证、延迟标签、特征选择、由成本定阈值，三道完整 case、一道实时系统设计、概率小题，以及运行验证过的 Python、SQL 和 PySpark 代码。"
topics: [data-science]
---
这套题是我准备欺诈与风控方向的应用机器学习（machine learning，ML）面试时整理的，分五部分：欺诈场景下的基本功、几道完整 case、一道系统设计、概率小题、带参考代码的编程练习。每个答案都按我开口回答的顺序写：先给结论，再讲原因和取舍，最后说怎么验证。

> **先说结论。** 欺诈建模的大多数问题都落回四个习惯：在工作点上、用更晚的数据评估；每个特征只用决策那一刻已知的信息；把标签当成迟到且有偏的；让成本决定阈值。

## 怎么用这套题 {#how-to}

先遮住答案，出声回答 60 到 90 秒，再对照，只记下漏掉的要点。除非题目另有说明，场景都是一家银行对支付做实时评分，大约每一千笔交易里有一笔欺诈。

时间紧的话，先练银行数据科学 panel 面试里最常被提到的几题：[类别不平衡](#imbalance)、[bagging 和 boosting](#trees)、[正则](#regularization)、[特征选择](#feature-selection)，以及[实时系统设计](#system-design)。

欺诈以外的基础题，比如假设检验、回归、经典算法和神经网络，见[机器学习与统计基础](/zh/notes/ml-stats-fundamentals/)。

有三篇笔记把单个主题讲得更细：[precision 和 recall](/zh/notes/precision-vs-recall/)、[ROC-AUC 和 PR-AUC](/zh/notes/roc-auc-vs-pr-auc/)、[A/B 测试](/zh/notes/ab-testing/)。[MLOps 学习笔记](/zh/notes/mlops-learning-guide/)里有可以运行的练习，把[泄漏](/zh/notes/mlops-learning-guide/#time-leakage)和[监控](/zh/notes/mlops-learning-guide/#monitoring)讲得更完整。

关于用词：precision、recall 等指标名沿用英文。false positive rate（FPR，假阳性率）指正常交易里被误报的比例；false positive ratio（误报比）是欺诈团队常用的口径，指每抓到一笔欺诈误伤几位正常客户。

## 欺诈场景下的基本功 {#fundamentals}

### 类别不平衡 {#imbalance}

**问题。** 欺诈只占交易的 0.1% 左右，类别不平衡怎么处理？

不平衡主要是评估和定阈值的问题，训练上的问题反而是次要的。我会做四件事：

1. 用不被海量负样本主导的指标：precision-recall 曲线下面积（PR-AUC）、固定 FPR 或固定报警量下的 recall、按金额加权的 recall。
2. 用全部欺诈样本加一部分抽样的正常样本训练，主要为了省算力；或者用类别权重，比如 `scale_pos_weight`（这个欺诈率下约为 999）。
3. 降采样之后，先把预测概率[校正](#drill-downsampling)回真实基率，再定阈值。
4. 评估始终在保持真实类别比例、没动过的数据上做。

我对 SMOTE（Synthetic Minority Over-sampling Technique，合成少数类过采样）比较谨慎。在高维、类型混杂的特征空间里给稀少的欺诈样本做插值，造出来的点现实中没有客户会产生；配合提升树时，它也很少比简单加权更好。神经网络可以考虑 focal loss。

### 评估指标 {#metrics}

**问题。** 你会报哪些指标？为什么不看 accuracy，或者只看 ROC-AUC？

Accuracy 在这里没有意义：全部预测成“正常”就有 99.9%。ROC 曲线（receiver operating characteristic，受试者工作特征曲线）下面积 ROC-AUC 适合判断排序好坏，但它对排序的每一段一视同仁，可能数字很好看，工作点上的 precision 却很差。[ROC-AUC 和 PR-AUC 那篇](/zh/notes/roc-auc-vs-pr-auc/)里有两个 ROC-AUC 相同、average precision 差四倍的模型。

我会报五项，全部在晚于训练数据的时间段上测：

- **PR-AUC**，旁边写上欺诈占比，那是它的随机基线。
- **固定 FPR 或固定报警量下的 recall**，因为审核团队一天能处理的案件是有限的。
- **金额检出率（value detection rate）**：抓到的欺诈金额占全部欺诈金额的比例。一笔 \$5,000 的欺诈比十笔 \$20 的更要紧。
- **误报比（false positive ratio）**：每抓到一笔欺诈，误伤几位正常客户。
- **校准**，如果分数要进入基于成本的决策。

### 验证集怎么切 {#validation}

**问题。** 验证时数据怎么切分？

按时间切，不随机切。用较早的时间窗训练，较晚的验证，再留一个最终的时间外（out-of-time）测试集，只用一次。

随机切分会把未来泄漏进训练集。同一个欺诈团伙、同一台设备、同一波攻击会同时落在两边，分数会比上线后好看。

还有两个细节。训练集和测试集之间要留出间隔，至少和标签成熟期一样长，因为最近的交易还来不及被报告。调参时用滚动向前（walk-forward）验证跑多个时间段，看的是稳定性，不是某个运气好的月份。

### 数据泄漏 {#leakage}

**问题。** 什么是数据泄漏？举欺诈场景的例子，并说明怎么发现。

泄漏指训练特征里含有决策时拿不到的信息。在欺诈场景里通常是下面几种：

- 滚动计数或求和把当前这笔交易也算了进去。
- 客户级聚合用了整份数据，而不是截止到该交易时间戳。
- 事后才写入的字段，比如案件调查代码、“因欺诈关户”标记。
- 在全量数据上拟合的 target encoding。

发现的办法：每个特征都按时点（point-in-time）构建，只用交易时间戳之前的信息；任何单个特征的重要性高得不合常理，都要怀疑；训练和线上用同一套特征代码，并且对比离线表现和影子模式（shadow mode）下的表现。

### 迟到且有偏的标签 {#labels}

**问题。** 欺诈标签来得晚，而且不完美，会带来什么问题？

三个问题。

**延迟。** 客户可能几周后才对交易提出争议，所以最近的数据标签不全。直接拿来训练，等于教模型把最近的欺诈当成正常交易。我只用超过成熟期的数据。

**选择偏差。** 只有被放行的交易才有结果。被拒绝的交易没有标签，模型学到的是被上一个模型筛过的人群。留一小部分随机放行的对照组（control group），在损失可以承受的范围内绕过模型，就能得到无偏的标签。信贷建模里把同一个问题叫拒绝推断（reject inference）。

**噪声。** 一部分被报告的欺诈其实是第一方欺诈或“友好欺诈”，即真实客户对真实消费提出争议；也有欺诈从未被报告。我会把不同来源的标签分开存，并尽量把第三方欺诈、第一方欺诈和诈骗（scam）当作不同的预测目标。

### Bagging 和 boosting {#trees}

**问题。** 以随机森林和梯度提升为例，比较 bagging 和 boosting。为什么在表格数据上提升树通常更好？

Bagging 在重采样的数据上独立训练很多个模型，再取平均。Boosting 一个接一个地训练，每个模型去纠正前面模型的错误。随机森林和梯度提升分别是这两种思路的树模型版本。

随机森林在 bootstrap 样本上独立地长出很深的树，每次分裂随机选一部分特征，最后取平均，主要降低方差。

梯度提升一棵接一棵地长浅树，每棵树拟合当前集成的损失梯度，主要降低偏差；方差靠学习率收缩、子采样和正则来控制。

| | Bagging：随机森林 | Boosting：梯度提升 |
|---|---|---|
| 树怎么建 | 相互独立，可以并行 | 依次建，每棵针对还没学对的部分 |
| 主要降低 | 方差 | 偏差 |
| 单棵树 | 深，单独看会过拟合 | 浅，单独看会欠拟合 |
| 继续加树 | 效果趋于平稳，没有坏处 | 最终会过拟合，所以要早停 |
| 标错的样本 | 被平均掉 | 不加正则的话，每一轮都追得更紧 |

最后一行在欺诈里很要紧，因为有些标签是错的。我会限制树深、用学习率收缩、早停，不让提升树去拟合训练集里最后那几个样本。

[XGBoost](https://arxiv.org/abs/1603.02754) 在此之上加了损失的二阶近似、叶子权重的 L1 和 L2 惩罚，以及为缺失值学习默认分裂方向。LightGBM 用直方图分箱和按叶生长（leaf-wise），在大数据上更快，但不限制叶子数时更容易过拟合。

提升树通常更好，是因为每棵新树都集中在模型还没学对的样本上，在欺诈里就是那些稀少而难分的案例。随机森林的好处是不怎么调参也不容易出错。

可能被追问的超参数：学习率、配合早停的树的数量、`max_depth` 或 `num_leaves`、`min_child_weight`、`subsample`、`colsample_bytree` 和 `scale_pos_weight`。

### 校准和阈值 {#threshold}

**问题。** 模型输出 0.8，这是概率吗？阈值怎么定？

不一定。用了类别权重或降采样训练出来的提升树，排序是好的，概率却没有校准。

如果训练时只保留了比例为 $\beta$ 的负样本，我先把分数[校正](#drill-downsampling)回真实基率，再在时间外数据上画可靠性曲线；必要时在留出集上拟合 Platt scaling 或保序回归（isotonic regression）。

阈值按期望成本定，不用 0.5：当概率乘以欺诈损失超过误报的期望成本时就干预。损失随金额增长，所以阈值也应该随金额变化。一笔 \$5,000 的转账，在比 \$50 的消费低得多的概率上就值得干预；[成本那道小题](#drill-cost)算了具体数字。

实际中阈值还受报警量预算和客户体验的约束，所以我会把取舍曲线拿给欺诈策略团队一起定，而不是自己选一个切点。

### 提升树的过拟合 {#overfitting}

**问题。** 提升树的过拟合怎么发现、怎么控制？

信号有两个：训练集和时间外数据上的表现差距越来越大；上线后的表现逐月快速衰减。

控制手段：在时间外验证集上早停，降低学习率并增加树的数量，把树做浅，增大叶子最小样本数，行采样和列采样，加 L1 或 L2 正则。

我还会剔除不稳定的特征，比如分布在不同时期变化很大的，或者重要性在不同时间折之间来回翻转的。在欺诈场景里，我宁愿牺牲一点样本内的提升，换一个三个月后还站得住的模型。

### 解释一次决策 {#explainability}

**问题。** 怎么向业务方或模型评审解释模型的决策？

全局层面，用 SHAP 值（[SHapley Additive exPlanations](https://arxiv.org/abs/1705.07874)）的汇总图和部分依赖图说明哪些特征在驱动模型，并检查方向是否符合业务直觉，比如风险应该随交易频次上升。

局部层面，对一笔被标记的交易，取贡献最大的几个 SHAP 值，翻译成直白的理由：新设备、第一次向这个收款人付款、金额是该客户平时的二十倍。

有两点要说明：特征相关时，SHAP 在它们之间分配贡献带有一定任意性；归因不等于因果。

治理方面，我会把数据、假设、局限、测试和监控方案写成文档，让独立验证方可以逐项质疑。美国的模型风险监管指引在 2026 年 4 月 17 日更新，[SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm) 取代了 SR 11-7。美联储负责监管的副主席在 [2026 年 5 月的讲话](https://www.federalreserve.gov/newsevents/speech/bowman20260501a.htm)里说明，新指引不适用于生成式和智能体（agentic）AI。

### 模型在变差 {#drift}

**问题。** 模型上线六个月，表现在下滑，你怎么办？

先确认是不是真的。最近这段时间的标签成熟了吗？交易量或产品结构变了吗？上游数据源有没有断？

然后定位。用群体稳定性指数（population stability index，PSI）跟踪分数分布和特征漂移，并按渠道、产品和欺诈类型拆开看表现。PSI 的常用读法是行业惯例，不是统计检验：0.1 以下算稳定，0.1 到 0.25 值得看一眼，0.25 以上是明显偏移。

常见原因有三类：数据管道改了；人群变了，比如上了新产品；欺诈者在适应，模型堵住一种手法，他们就换另一种。

应对按速度从快到慢：针对新手法加一条规则先顶上，调整阈值，用近期数据重训，再做新特征。长期要有定期重训、冠军–挑战者（champion-challenger）测试，以及和模型治理团队约定好的监控阈值。

### 序列模型和图模型 {#sequence-graph}

**问题。** 什么情况下用序列模型或图模型做欺诈检测？

当信号藏在人工聚合特征会丢掉的结构里时。

在客户事件历史上建的序列模型能捕捉顺序和时间间隔。先重置密码，再换新设备，再加新收款人，最后转走最大额度：这种模式用计数类的频次特征只能近似。

图模型捕捉关系：共用设备、地址、电话或收款人的账户。欺诈团伙和骡子账户网络表现为稠密的、或快速增长的子图。

代价是需要更多数据和算力，验证和解释更难，还有延迟。图特征通常要提前算好，从特征库（feature store）里取。

所以我会先把图上衍生的特征放进树模型，比如共享实体的数量、邻居中的欺诈率，计算时注意不泄漏标签。只有当学出来的 embedding 在工作点上确实增加 recall，才往那一步走。

### 异常检测 {#anomaly}

**问题。** 监督模型的边界在哪，异常检测从哪里接手？

只要有标签，监督模型就是主力，因为它学的是已知欺诈长什么样。它的盲区是还没有标签的新手法。

异常检测，比如孤立森林（isolation forest）或自编码器的重构误差，不需要标签，标出的是相对客户自己或整体人群不寻常的行为。单独使用时误报很多，因为不寻常不等于欺诈。

所以我把它当补充：异常分数作为输入特征；同时作为监控工具，把新出现的簇推给分析师审核和打标签。

### 离群值 {#outliers}

**问题。** 离群值怎么发现、怎么处理？

我先问这个离群值是什么：数据错误，罕见但真实的取值，还是我要找的东西本身。在欺诈里常常是第三种，所以我从不按规则直接删掉极端值。

**发现。** 单个变量用中位数和绝对中位差（median absolute deviation，MAD），或者四分位距，因为均值和标准差本身就会被离群值拉偏。金额严重右偏，所以放到对数尺度上看。多个变量一起看时，用孤立森林或到最近邻的距离。除了和整体人群比，还要和客户自己的历史比：\$3,000 对一个账户很平常，对另一个就是极端值。

**处理，按原因分。**

- **错误**，比如负金额、不可能的时间戳、重复的行：在源头修正或剔除，并且计数。错误数突然上升，本身就是数据质量报警。
- **真实的极端值：** 保留。树按顺序分裂，很大的数值不会扭曲它。线性模型和神经网络要对特征封顶或取对数，封顶值只在训练时间窗上学。
- **指标：** 少数几笔大额欺诈会主导所有按金额算的指标，所以结果按笔数和按金额各报一遍。

[稳健 z 分数那道题](#robust-z)写了按账户计算的版本。

### 强化学习和 bandit {#bandits}

**问题。** 强化学习在欺诈里能用在哪？

评分模型是监督学习，我会保持这一点。强化学习适合评分之后的那一步决策，因为这一步会改变我们接下来能观察到什么。

有三个位置。Contextual bandit 可以在几种干预方式（放行、加强验证、暂挂、拒绝）之间做选择，奖励是避免的欺诈损失减去打扰客户的成本。对照组本身就是一种探索策略，它留多大就是探索与利用的取舍。账户生命周期里的决策，比如什么时候提额，本来就是序列决策。

难点正是让强化学习在生产环境里有风险的那几条。奖励要几周后才到。探索要花真金白银。新策略在接触客户之前，必须用旧策略留下的日志做离线评估。所以我会从一小部分流量上带护栏的 bandit 做起，而不是一上来就做完整的序列策略。

### 特征族 {#features}

**问题。** 做交易欺诈，你会建哪些特征？

五类：

1. **交易本身：** 金额、渠道、商户类别、交易时间。
2. **频次（velocity）：** 按账户、卡、设备、收款人，在一小时、一天、七天窗口内的笔数和金额。
3. **偏离客户自己的基线：** 金额相对平时的倍数、新商户、新地点、不常见的时间。
4. **实体和关系：** 这台设备或这个 IP 地址关联了多少账户、和收款人的关系存在了多久、与已确认欺诈的关联。
5. **账户状态：** 开户时长、最近是否改过联系方式或密码、最近的登录失败次数。

全部按时点计算，并且都要能在决策时的延迟预算内取到。[第一道编程题](#velocity)做的就是频次这一类。

### 特征选择 {#feature-selection}

**问题。** 候选特征有几千个，怎么决定哪些进模型？

用提升树时，我把特征选择主要看成稳定性和成本的问题，其次才是精度，因为模型对无关特征的容忍度比较高。我分五轮做：

1. **去掉不能用的。** 决策时拿不到的、这类决策不允许使用的，以及取值不变或几乎全是缺失的字段。
2. **去掉撑不久的。** 分布逐月变化很大的（看 PSI），或者和欺诈的关系在不同时间折之间方向相反的。
3. **减少冗余。** 高度相关的一组里，留下线上计算最便宜、最容易解释的那个。
4. **在更晚的数据上按贡献排序。** 在时间外样本上算 permutation importance 或平均绝对 SHAP 值。训练时按分裂统计的重要性会偏向取值多的特征，[Strobl 等人](https://pmc.ncbi.nlm.nih.gov/articles/PMC1796903)在随机森林上展示过这一点。
5. **用消融确认。** 去掉一组特征重训，在工作点上比较。固定报警量下的 recall 不变，这一组就去掉。

教科书把方法分成三类：

| 类别 | 例子 | 我用在哪 |
|---|---|---|
| 过滤法（filter） | 缺失率、方差、单变量 AUC、相关性 | 便宜的第一轮筛选 |
| 嵌入法（embedded） | L1 惩罚、树的增益、SHAP | 主要的排序依据 |
| 包裹法（wrapper） | 递归消除、前向选择、消融 | 对短名单做最后确认，因为每一步都要重训 |

两点提醒。特征选择是训练的一部分，要放在每个训练折内部做；先在全量数据上选特征再做交叉验证，就是泄漏。单看很弱的特征可能在交互里有用，所以过滤这一轮不要下手太重。

回报主要在运维上：会出故障的管道更少，线上延迟更低，要监控、要写进模型评审文档的东西也更少。

### L1 和 L2 正则 {#regularization}

**问题。** L1 和 L2 正则有什么区别？

L2 惩罚权重的平方，把所有系数平滑地压向零，并在相关特征之间分摊权重。

L1 惩罚绝对值。它的约束区域在坐标轴上有尖角，最优点常常落在一部分系数恰好为零的位置，所以能做特征选择；但在相关特征之间选谁，带有一定任意性。

从贝叶斯角度看，L2 相当于给权重加高斯先验，L1 相当于拉普拉斯先验，带惩罚的拟合就是最大后验（maximum a posteriori，MAP）估计。两者都是用一点偏差换更低的方差。

Elastic net 把两种惩罚混在一起，既保留 L1 的稀疏性，又能更均匀地对待成组的相关特征。用之前要先把输入缩放到同一尺度，否则惩罚对各个特征不公平。

惩罚项只是正则的一种。早停、学习率收缩、行列子采样、叶子最小样本数、dropout 和数据增强，都是在限制模型把训练数据拟合得多紧。

### 经典机器学习快问 {#classic-ml}

panel 开场常问的几道教科书题，答案压到几句：

| 问题 | 回答 |
|---|---|
| 决策树怎么选分裂点？ | 对每个特征、每个候选切点，计算不纯度的下降（分类用 Gini 或熵，回归用方差），按子节点的样本数加权，取下降最大的。它是贪心的：只选当下最好的分裂，不往后看。靠限制树深、叶子最小样本数和剪枝，防止它把数据背下来。 |
| 欠拟合还是过拟合？ | 欠拟合：训练误差和验证误差都高（偏差）。过拟合：训练误差低，验证误差高出很多（方差）。前者要加模型容量或更好的特征。后者要更多数据、正则，或更简单的模型。 |
| 为什么风控里还在用逻辑回归？ | 系数可以读成 log-odds，单调关系容易保证也容易解释，数据少时稳定，分数通常接近校准。它是常用的基线，也常常是树模型必须打败的对手。 |
| k-means 的 k 怎么选？ | 没有标签可以优化，所以我综合几样：簇内方差的拐点、轮廓系数（silhouette）、重采样后簇是否稳定，以及分出来的群业务上能不能用。先做标准化，因为 k-means 用欧氏距离。 |
| 验证集和测试集有什么区别？ | 验证集用来做选择（超参数、特征、阈值），所以它上面的分数偏乐观。测试集在所有选择做完之后只用一次。在欺诈里，它取自更晚的时间段。 |
| 数据在 Python 里放不进内存，怎么办？ | 只读需要的列，用 Parquet 这类列式格式。用紧凑的数据类型和 category。分块处理，或者把聚合下推到 SQL 或 Spark。用向量化代替循环。开发时用保留全部欺诈样本的抽样数据。 |

### 深度学习快问 {#deep-learning}

面试官想考察树模型以外的基本功时，常会跟上这几题。答案压到两三句：

| 问题 | 回答 |
|---|---|
| 解释 self-attention。 | 每个 token 生成 query、key 和 value。它的输出是所有 value 的加权和，权重来自 query 与 key 点积除以维度平方根之后的 softmax。每个位置一步就能用到任意其他位置的信息。 |
| 局部极小还是鞍点？ | 高维下梯度为零的点大多是鞍点，不是差的极小值。随机梯度下降（stochastic gradient descent，SGD）的噪声和动量有助于逃离鞍点。Adam 这类自适应方法会放大梯度小的方向上的步长。 |
| Batch normalization 为什么有用？ | 它把每层的输入保持在稳定的范围内，可以用更大的学习率，优化更平滑。批内噪声还带来轻微的正则效果。 |
| Adam 还是 SGD？ | Adam 为每个参数自适应步长，不怎么调就收敛得快。带动量的 SGD 调好之后，在视觉任务上泛化常常略好。AdamW 把权重衰减从自适应步长里解耦出来。 |

## 完整 case {#cases}

### 七步框架 {#frame}

几道 case 用同一个框架。我会先把框架报出来，再逐步展开。

1. **澄清问题：** 针对哪类欺诈、分数出来后做什么动作、延迟预算和人工审核容量。
2. **目标和标签：** 预测单位、标签来源、标签多久成熟。
3. **数据和特征：** 特征族，按时点构建。
4. **模型：** 基线、主模型、后续增强。
5. **评估：** 时间外数据、指标、分群、和现有模型对比。
6. **决策和上线：** 阈值、动作、分阶段放量。
7. **监控：** 漂移、表现、重训触发条件。

### Case 1：实时交易欺诈 {#case-transactions}

**题目。** 为一家零售银行设计模型，实时识别卡交易或数字支付里的欺诈。

**澄清问题。** 我先问三个问题。针对哪类欺诈：盗刷、账户被接管这类未经授权的交易，还是客户自己授权付款的诈骗？分数出来之后做什么：拒绝、加强验证，还是暂挂等人工审核？延迟预算和审核容量是多少？得不到回答时，我会把假设说出来：针对未经授权的无卡（card-not-present）交易，一百毫秒左右内出决策，三种动作（放行、加强验证、拒绝）。

**目标和标签。** 预测单位是一次授权请求。标签是这笔交易后来是否通过客户申诉或争议被确认为未经授权。只用标签已经成熟的交易训练，友好欺诈单独标记。

**数据和特征。** 上面的[五类特征](#features)。每个特征都按交易发生的时点计算。变化快的聚合从流式存储里取，重的聚合用批处理提前算好。

**模型。** 先做一个基线：逻辑回归加现有规则，它给出下限，也是合理性检查。主模型是梯度提升树，用全部欺诈样本和降采样的正常样本训练，之后做概率校正。再往后，把序列或图的 embedding 当作额外特征来试，有提升才保留。

**评估。** 在时间外测试集上报 PR-AUC、业务可接受的误报比下的 recall 和金额检出率，以及校准情况。结果按渠道、金额段和客户开户时长拆开看，并在同一份样本上和现在的线上模型对比。

**决策和上线。** 分数映射到随金额变化的阈值和三种动作。最高一档拒绝。中间一档做加强验证，对正常客户来说，这比直接拒绝的代价小得多。其余放行。上线先走影子模式，只评分不动作；然后放一小部分流量；同时保留一个小的对照组，持续拿到无偏的标签。

**监控。** 每天看：分数分布、报警率、特征漂移和数据质量检查。标签成熟后看：分群的 recall、金额检出率和误报比。重训的触发条件事先约定好，并建立来自调查员的反馈渠道，他们最先看到新手法。

> **先说结论。** 要一直摆在业务面前的取舍是：欺诈损失、客户摩擦和运营成本。模型负责给风险排序，切在哪里是业务决策。

### Case 1 的追问 {#case-1-follow-ups}

**上线之后欺诈者会适应，你在设计上怎么应对？** 我假设模型从上线第一天就开始衰减。所以要安排：用近期数据频繁重训；用异常检测把模型没见过的簇找出来；重训期间用规则快速打补丁；除了交易字段，还要用更难伪造的信号，比如设备和行为。对照组在这里也很重要，它告诉我们不干预时欺诈是什么样子。

**离线结果很好，但影子模式下报警量远超预期，抓到的欺诈却没有增加。你查什么？** 按这个顺序查四件事：

1. **训练和线上不一致（training-serving skew）。** 每个特征在线上的算法是否一致？缺失值有没有被填成奇怪的默认值？我会逐个特征对比分布，并用线上记录下来的特征在离线重新给一批近期样本评分。
2. **离线泄漏。** 如果某个特征用了事后信息，就能解释为什么离线数字漂亮、上线后消失。
3. **校准。** 如果对负样本做了降采样，又在未校正的分数上定阈值，报警量会被放大。
4. **人群偏移。** 训练时间窗和今天之间，人群可能已经变了。

**怎么估算这个模型值多少钱？** 在相同误报比下，相对现有模型多拦下的欺诈金额，减去新增误报和加强验证的成本：流失的交易、客户来电、审核工时。干净的测法是按账户随机分流的冠军–挑战者测试，等标签成熟后再读数。实验设计见 [A/B 测试那篇](/zh/notes/ab-testing/)。

**客户自己授权付款的诈骗，有什么不同？** 用的是客户自己的设备和凭证，所以设备和身份验证信号看起来都是干净的。信号转移到收款方和行为上：第一次付款的收款人；收款账户从许多互不相关的人那里收钱并很快取走；金额远远超出客户的习惯。干预方式也要变：提醒或二次确认比直接拒绝更合适。标签更模糊，所以我会把诈骗单独作为一个预测目标。

### Case 2：新开户申请欺诈 {#case-applications}

**题目。** 线上来了一份新的支票账户申请。建一个模型，判断它是不是欺诈。

**和 Case 1 的不同。** 三点。这位客户在银行没有历史，是冷启动。标签要等几个月，而不是几周，因为它取决于账户后来的表现。被拒绝的申请，我们永远看不到结果。

**澄清问题。** 最担心哪类欺诈：冒用真实身份的身份盗用、合成身份，还是真实申请人本身就打算欺诈或充当骡子账户？可用的动作有哪些：通过、拒绝、人工审核，还是带限制通过？

**目标和标签。** 预测单位是一份申请。标签是账户在开户后的固定窗口内是否因欺诈被关闭或核销。数据允许的话，不同欺诈类型分开打标签。

**数据和特征。** 四类：

1. **身份一致性：** 姓名、出生日期、地址、电话和邮箱是否属于同一个人，它们一起出现了多久？
2. **设备和网络：** 这台设备或这个 IP 地址有没有出现在以前的申请里，最近从它发出了多少份申请？
3. **申请行为：** 填表用时、是否粘贴字段、申请时间。
4. **与已知欺诈的关联：** 和以前因欺诈关闭的账户共享哪些属性，这是一个图问题。

在银行获准使用的前提下，我还会用外部的身份数据和征信数据，并和合规团队确认哪些变量可以用于开户决策。

**模型和评估。** 仍然是梯度提升树，按时间外数据验证。基率低、审核容量固定，所以关键指标是在审核员处理得过来的量之内的 precision 和 recall，再加上避免的欺诈金额。

**决策。** 分四档：自动通过；带限制通过，比如更低的额度和开户初期监控；人工审核，量按容量来定；只有在 precision 非常高时才自动拒绝。针对被拒申请没有结果的问题，我会在临界档里随机放行一小部分并加上限制，或者把人工审核的结论当作标签。

**监控。** 按渠道看申请量和分数分布，因为攻击往往从某个渠道或某一簇设备集中爆发。再看已通过账户在开户初期的行为，作为最终标签成熟之前的先行指标。

### Case 2 的追问 {#case-2-follow-ups}

**身份盗用和合成身份的信号有什么不同？** 身份盗用里，身份是真的，历史很长，但联系方式和设备对不上这段历史。受害人迟早会投诉，所以标签会来。合成身份里，各个要素此前从未一起出现过，历史很薄或很新，而且可能有多个身份共用一个电话、地址或设备。没有受害人投诉，损失常被记成信用损失，欺诈标签就缺失了。

**欺诈占申请的 0.5%，团队能人工审核其中 2%。这个队列怎么用？** 按期望损失排序，而不只按概率。最顶端只有在 precision 足够高时才自动拒绝。接下来的一段，按 2% 的容量送去人工审核。我会跟踪队列内的 precision 和整体的 recall，把审核结论回流成标签，并且另外随机抽一小部分送审，避免标签偏向模型已经相信的那些样本。

### Case 3：交易量突然飙升 {#case-spike}

**题目。** 一小时内交易量远超预测。你怎么排查和应对？

我按顺序回答五个问题。

**是真的吗？** 先排除统计口径的问题：重试后被重放的事件、延迟入账的一批数据、时区或数据管道的改动。我会比较行数和去重后的交易 ID 数，再拿两个数据源互相核对。

**涨在哪里？** 把增量按渠道、商户、设备、IP 段、卡号段、地域、金额段、新老账户拆开。各处一起涨，通常有平常的原因，比如节假日、发薪日或促销。集中在少数商户、设备或某个金额段，就像是攻击。

| 分群 | 平时每小时 | 这一小时 | 变化 |
|---|---:|---:|---:|
| 全部交易 | 50,000 | 70,000 | +20,000 |
| 某一家商户、金额低于 \$2 | 300 | 19,300 | +19,000 |
| 其余全部 | 49,700 | 50,700 | +1,000 |

在这个虚构的例子里，总量涨了 40%，其中 95% 的增量来自一家商户、一个金额段，其余部分只涨了 2%。

**像欺诈吗？** 把集中的位置和已知手法对照。一家商户上、许多张卡的大量小额授权，拒绝率又高，是卡测试（card testing）。一台设备或一个 IP 段登录许多账户，是账户接管之前的撞库（credential stuffing）。突然出现大量首次收款人，是诈骗或骡子账户活动。标签要几周才到，所以我看早期信号：飙升部分的分数分布、授权拒绝率、客户来电。

**现在做什么？** 和受影响的范围相称的动作：对涉及的商户、设备簇或卡号段加频次规则、加强验证或限流，并通知欺诈运营团队。我会避免一刀切的拦截，那样伤到的主要是正常客户。如果飙升是正常的，我转而确认审核队列和服务扛得住，并留意频次特征有没有把正常的高峰变成误报。

**事后改什么？** 给这次事件打标签，衡量临时规则带来了多少误报，把这个手法加进特征或下一次重训，补上本该更早触发的报警，并给临时规则定一个下线日期。

> **先说结论。** 看集中度：正常的高峰会抬高每一个分群，攻击只抬高少数几个。

### 相邻的一题：贷款申请人的信用风险 {#case-credit}

**题目。** 怎么用数据评估贷款申请人的信用风险？

框架相同，目标不同。预测的是违约概率，比如借款人在放款后 12 到 24 个月内逾期达到 90 天。

- **数据。** 征信记录（还款记录、额度使用率、信用历史长度、近期查询次数）、收入和已有负债、申请表本身，以及申请人和银行已有的往来，比如存款和现金流。
- **模型。** 逻辑回归评分卡作为基线，往往也是线上模型。带单调约束的梯度提升树作为挑战者。
- **评估。** 用时间外数据。排序看 AUC 或 KS（Kolmogorov-Smirnov）统计量；还要分数段看校准，因为这个概率会进入定价和损失估计。
- **决策。** 批或不批，然后是额度和价格。期望损失等于违约概率乘以违约损失率，再乘以风险敞口。
- **缺失的结果。** 只有获批的申请人才看得到还款表现，和[欺诈标签](#labels)里的选择偏差是同一个问题。信贷建模把应对办法叫作拒绝推断。
- **解释。** 在美国，贷款机构必须告诉被拒的申请人主要原因，也不能使用受保护的特征。原因代码和公平借贷测试是硬性要求，所以可解释性是设计的一部分。

| | 欺诈 | 信用风险 |
|---|---|---|
| 预测什么 | 这笔支付或这位申请人不是真的 | 真实的客户还不上钱 |
| 基率 | 约 0.1% 的交易 | 常见是百分之几的账户 |
| 标签延迟 | 几天到几周 | 几个月到几年 |
| 对手 | 有，而且会适应 | 没有，但经济环境会变 |
| 决策时间 | 毫秒 | 几秒到几天 |
| 建模习惯 | 提升树，频繁更新 | 评分卡或带约束的树，更新慢，要给原因代码 |

## 系统设计：实时欺诈检测系统 {#system-design}

[Case 1](#case-transactions) 设计的是模型。系统设计这一轮问的是模型外面的整套服务，题目通常是“为银行的支付设计一个实时欺诈检测系统”。我分五步回答：需求、两条路径、延迟预算、故障处理、监控。

### 先问需求 {#sd-requirements}

| 我会问 | 没人回答时我的假设 |
|---|---|
| 决策放在哪一步？ | 内联，在支付授权之前 |
| 要多快？ | 端到端约 100 毫秒，按第 99 百分位算 |
| 流量多大？ | 高峰每秒几千笔请求，节假日有突发 |
| 模型不可用怎么办？ | 支付仍然要得到答复 |
| 有哪些动作？ | 放行、加强验证、拒绝，或暂挂等人工审核 |
| 要留存什么？ | 足以事后复现任何一次决策：输入、模型版本、规则版本 |

### 快路径和慢路径 {#sd-paths}

![一笔支付请求依次经过取特征、评分和决策，约 100 毫秒完成。下方的事件日志一边通过流式更新回到特征，一边在标签成熟后用于重训模型，监控同时覆盖两条路径。](fast-and-slow-path.zh.svg "图 1：上面一行回答一笔支付。下面一行为它供给：计数器几秒内更新，模型在标签成熟后重训。")

快路径回答一笔支付：

1. **决策服务。** 接收授权请求，守住时间预算，并且一定返回答复。
2. **取特征。** 请求里的字段，加上按键从低延迟在线存储里取的预计算特征。变化慢的聚合，比如 90 天均值或图特征，按批刷新。变化快的，比如最近一小时的笔数，由流处理在几秒内更新。
3. **评分。** 模型加载在内存里，版本固定。提升树评分只要几毫秒甚至更短。
4. **规则和策略。** 先过硬规则，比如黑名单和限额。再按金额和客群对分数设阈值，映射到动作。
5. **记录。** 线上实际用到的特征、分数、各个版本和所做动作写入事件日志，不占关键路径。

慢路径从已经发生的事情里学习：

1. **流式更新。** 流处理读取事件日志，更新下一笔请求要读的计数器。
2. **标签。** 争议、拒付（chargeback）和分析师的结论，在几周后按交易 ID 关联回日志里的事件。
3. **训练数据。** 用日志里记录的特征来建，天然是按时点的，也和线上看到的一致。
4. **发布。** 时间外验证、独立评审、影子模式、小流量、全量，上一个版本随时可以一步回滚。

### 延迟预算 {#sd-latency}

| 步骤 | 预算 |
|---|---:|
| 网络和请求解析 | 10 毫秒 |
| 特征查询（并行） | 30 毫秒 |
| 模型评分 | 10 毫秒 |
| 规则和决策 | 10 毫秒 |
| 留给重试和尾部延迟的余量 | 40 毫秒 |
| **合计** | **100 毫秒** |

数字只是示意。有两点适用于任何真实系统：最花时间的通常是取特征，不是模型；预算针对的是最慢的 1% 请求，不是平均值。

### 出故障时 {#sd-failure}

| 故障 | 应对 |
|---|---|
| 某次特征查询超时 | 用默认值，外加一个模型训练时见过的“缺失”标记，并计数 |
| 流式计数器落后 | 返回最近一次的值和它的时效，并对延迟报警 |
| 模型不可用或变慢 | 退回到只用规则，对大额交易用保守的限额 |
| 同一个请求到达两次 | 幂等键（idempotency key）返回第一次的决策，计数器不重复累加 |
| 模型发布出了问题 | 先影子模式，再小流量，可以一步回滚 |
| 流量突发 | 无状态的决策服务横向扩容；存储按账户或卡分区 |

### 上线后监控什么 {#sd-monitoring}

| 层 | 看什么 | 多久能看到 |
|---|---|---|
| 服务 | 中位数和第 99 百分位延迟、错误率和超时率、降级率 | 几分钟 |
| 数据 | 缺失率、特征新鲜度、特征漂移（PSI） | 几小时 |
| 分数 | 分数分布、报警率、分客群的拒绝率和加强验证率 | 几小时 |
| 结果 | recall、金额检出率、误报比、校准 | 几周，等标签成熟 |
| 业务 | 欺诈损失占交易额的基点数、客户来电和投诉、审核积压 | 几周 |

前三层是预警，因为结果这一层要等标签。每个报警都有负责人和事先约定好的动作。

### 系统设计的追问 {#sd-follow-ups}

**为什么不在线上直接调用大型神经网络或大语言模型？** 延迟、成本和可复现性。线上决策只有毫秒级的时间，而且重放时必须给出同样的答案。重模型可以在离线发挥作用：产出 embedding 或从文本提取的特征，由快模型从存储里读取；或者在决策之后辅助案件审核。

**怎么保证训练和线上一致？** 每个特征只有一份定义，批处理和流处理共用。训练数据取自线上评分时记录下来的特征。再加一个重放测试：离线重算一批样本，和日志里的值比较。

**人在哪个环节？** 分数中间那一档进入人工审核队列，队列大小按容量定。审核结论成为标签。审核员也最先看到新手法，所以系统要给他们留一个上报渠道。

**怎么安全地改阈值？** 分数都记在日志里，所以可以把上个月的流量按新阈值重放一遍，改动之前就能看到报警量和拒绝率。之后像发布模型一样分阶段放量。

## 概率与统计小题 {#drills}

下面每个数字都由[练习脚本](#coding)打印出来。

### 低基率下的 precision {#drill-bayes}

**问题。** 欺诈率 0.1%。模型能抓到 90% 的欺诈，同时会标记 1% 的正常交易。一笔交易被标记了，它是欺诈的概率有多大？

大约 8%。拿一百万笔交易来算：其中有 1,000 笔欺诈，模型抓到 900 笔。999,000 笔正常交易里被标记了 9,990 笔。所以 10,890 条报警里有 900 条是欺诈，即 8.3%，也就是每抓到一笔欺诈大约有 11 条误报。

$$
P(\text{fraud} \mid \text{alert}) = \frac{0.001 \times 0.9}{0.001 \times 0.9 + 0.999 \times 0.01} = \frac{0.0009}{0.01089} \approx 8.3\%
$$

如果 FPR 降到 0.1%，precision 升到 47%。基率很低时，precision 主要由 FPR 决定；整条曲线见 [precision 和 recall 那篇](/zh/notes/precision-vs-recall/)。

### recall 的置信区间 {#drill-interval}

**问题。** 测试集里有 400 笔已确认的欺诈，recall 是 80%。这个数字有多可靠？

标准误是 2 个百分点，所以 95% 置信区间大约是 76% 到 84%。

$$
SE = \sqrt{\frac{r(1-r)}{n_{\text{fraud}}}} = \sqrt{\frac{0.8 \times 0.2}{400}} = 0.02
$$

recall 估计的精度取决于欺诈样本的数量，而不是总行数。要把区间收窄到正负 2 个百分点，大约需要 1,540 笔欺诈。

### 冠军对挑战者 {#drill-ztest}

**问题。** 挑战者模型在 5,000 条报警上的 precision 是 12%，冠军模型在 5,000 条报警上是 10%。这个差异是真的吗？

两比例 z 检验给出 $z \approx 3.2$，双侧 p 值约 0.001，所以不太可能是偶然。

$$
z = \frac{0.12 - 0.10}{\sqrt{0.11 \times 0.89 \times \left(\frac{1}{5000} + \frac{1}{5000}\right)}} \approx 3.20
$$

相信它之前我会查四件事：账户是随机分到两组的；两组标签的成熟度相同；两组覆盖同一时间段；precision 是事先选定的指标，而不是从几个指标里挑出来最好的那个。

### 由成本定阈值 {#drill-cost}

**问题。** 一次误拒的成本是 $C$，漏掉一笔欺诈的损失是交易金额 $L$。概率到多少应该干预？

当期望欺诈损失超过误报的期望成本时干预：

$$
p \cdot L > (1-p) \cdot C \;\Rightarrow\; p^* = \frac{C}{L + C}
$$

金额越大，阈值越低。为了举例，把一次误拒的成本设为 \$5：

| 金额 $L$ | 误报成本 $C$ | 阈值 $p^*$ |
|---:|---:|---:|
| \$50 | \$5 | 9.1% |
| \$500 | \$5 | 1.0% |
| \$5,000 | \$5 | 0.1% |

这是 [Elkan 论文](https://mlanthology.org/ijcai/2001/elkan2001ijcai-foundations/)里代价敏感决策规则的二分类情形。

### 还原降采样 {#drill-downsampling}

**问题。** 你用全部欺诈样本和 1% 的正常样本训练。模型输出 0.5，真实概率是多少？

大约 1%。只保留比例为 $\beta$ 的负样本，相当于把 odds 乘以 $1/\beta$，所以再把 odds 乘回 $\beta$。odds 从 1 变成 0.01，对应的概率是 0.0099。

$$
p = \frac{\beta \, p_s}{\beta \, p_s + 1 - p_s}
$$

其中 $p_s$ 是在抽样数据上训练的模型给出的分数。用基率验一下：真实欺诈率 0.2%，保留 2% 的负样本，训练样本里的欺诈率是 9.1%；公式把 9.1% 映射回 0.2%。[Dal Pozzolo 等人](https://doi.org/10.1109/SSCI.2015.33)推导了这个校正，并说明降采样改变的是校准，不改变排序。

### 多少笔交易才大概率见到一笔欺诈 {#drill-geometric}

**问题。** 欺诈率 0.1% 时，需要多少笔交易，才有 95% 的把握至少见到一笔欺诈？

大约 3,000 笔。$n$ 笔交易里一笔欺诈都没有的概率是 $0.999^n$，到 $n = 2{,}995$ 时降到 5% 以下。只有 1,000 笔时，这个把握只有 63%。

$$
1 - 0.999^{\,n} \ge 0.95 \;\Rightarrow\; n \ge \frac{\ln 0.05}{\ln 0.999} \approx 2995
$$

### AUC 是什么意思 {#drill-auc}

**问题。** AUC 是什么意思？

ROC-AUC 是随机抽一笔欺诈、一笔正常交易，欺诈那笔得分更高的概率。随机模型是 0.5。precision-recall 曲线的随机基线是欺诈占比，这里是 0.001，所以正样本很少时 PR-AUC 的信息量更大。[第三道编程题](#auc)就是按这个定义来算 ROC-AUC。

## 编程练习 {#coding}

下载 [fraud-practice-lab.py](/notes/fraud-detection-practice/fraud-practice-lab.py)，用 Python 3.10 或更新版本运行，需要 NumPy 和 pandas。脚本自己生成数据，并打印这篇笔记里的每一个数字。如果装了 scikit-learn 或 DuckDB，还会交叉核对 AUC 和两个 SQL 查询。

```bash
python3 fraud-practice-lab.py
```

讲题时我按这个顺序：复述题意和边界情况，说思路，写代码，用小例子走一遍，最后说复杂度和放到生产环境里要改什么。

### 按时点计算的频次特征 {#velocity}

**题目。** 给定含 `account_id`、`ts` 和 `amount` 的交易表，为每笔交易添加：同一账户在此前 24 小时内的交易笔数和总金额、距上一笔交易的秒数、金额相对该账户此前平均金额的倍数。任何特征都不能用到当前行或之后的行。

```python
import pandas as pd

def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["account_id", "ts"]).reset_index(drop=True)
    g = df.set_index("ts").groupby("account_id")["amount"]
    # closed="left" -> the window is [t - 24h, t): the current row is excluded
    df["cnt_24h"] = g.rolling("24h", closed="left").count().fillna(0).to_numpy()
    df["sum_24h"] = g.rolling("24h", closed="left").sum().fillna(0).to_numpy()
    df["secs_since_prev"] = df.groupby("account_id")["ts"].diff().dt.total_seconds()
    prior_mean = df.groupby("account_id")["amount"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df["amt_to_prior_mean"] = df["amount"] / prior_mean
    return df
```

在六笔交易上的输出：

| account_id | ts | amount | cnt_24h | sum_24h | amt_to_prior_mean |
|---|---|---:|---:|---:|---:|
| A | 12-01 09:00 | 20 | 0 | 0 | NaN |
| A | 12-01 09:30 | 30 | 1 | 20 | 1.50 |
| A | 12-02 08:59 | 500 | 2 | 50 | 20.00 |
| A | 12-02 09:31 | 40 | 1 | 500 | 0.22 |
| B | 12-01 10:00 | 100 | 0 | 0 | NaN |
| B | 12-03 10:00 | 100 | 0 | 0 | 1.00 |

有三点要说出口。`closed="left"` 把当前行挡在窗口之外，防泄漏靠的就是它。第四行的窗口从 12-01 09:31 开始，所以只数到 \$500 那一笔。生产环境里这类特征来自特征库，训练和线上共用同一份定义。

### 同一特征的 SQL 和 PySpark 写法 {#velocity-sql}

```sql
SELECT account_id, ts, amount,
       COUNT(*) OVER w AS cnt_24h,
       COALESCE(SUM(amount) OVER w, 0) AS sum_24h
FROM transactions
WINDOW w AS (
  PARTITION BY account_id
  ORDER BY ts
  RANGE BETWEEN INTERVAL 24 HOUR PRECEDING
            AND INTERVAL 1 MICROSECOND PRECEDING
);
```

```python
from pyspark.sql import functions as F, Window

w = (Window.partitionBy("account_id")
           .orderBy(F.col("ts").cast("long"))   # seconds since the epoch
           .rangeBetween(-24 * 3600, -1))       # [t - 24h, t - 1s]: excludes the current row

out = (df.withColumn("cnt_24h", F.count("*").over(w))
         .withColumn("sum_24h", F.coalesce(F.sum("amount").over(w), F.lit(0.0))))
```

SQL 用的是 DuckDB 语法，其他引擎的时间区间写法略有不同。`rangeBetween` 按排序键的数值范围取窗，`rowsBetween` 按行数取窗，这个区别常被追问。[Spark 那篇笔记](/zh/notes/spark-pyspark/)里有完整的 PySpark 练习。

### 固定报警量下的指标 {#alert-budget}

**题目。** 给定标签、分数和金额，标记分数最高的 1% 交易，返回 precision、recall、金额检出率和误报比。

```python
import numpy as np

def fraud_metrics(y, score, amount, alert_rate=0.01):
    y, score, amount = map(np.asarray, (y, score, amount))
    k = max(1, int(np.ceil(alert_rate * len(y))))
    flagged = np.zeros(len(y), dtype=bool)
    flagged[np.argsort(-score, kind="stable")[:k]] = True
    tp = int((flagged & (y == 1)).sum())
    fp = k - tp
    return {
        "precision": tp / k,
        "recall": tp / int((y == 1).sum()),
        "value_detection_rate": float(amount[flagged & (y == 1)].sum() / amount[y == 1].sum()),
        "false_positive_ratio": fp / max(tp, 1),
    }
```

一个可以手算的小例子：十笔交易里有两笔欺诈，金额分别是 \$500 和 \$100。标记前 20%，抓到 \$500 那笔和一笔正常交易。precision 是 0.5，recall 是 0.5，金额检出率是 0.83，误报比是 1。按金额算的检出率高于按笔数算的，这正是两个都要报的原因。

### 用秩计算 ROC-AUC {#auc}

**题目。** 不用 scikit-learn，计算 ROC-AUC。

```python
import numpy as np
import pandas as pd

def roc_auc(y, score):
    y = np.asarray(y)
    ranks = pd.Series(score).rank(method="average").to_numpy()  # average rank handles ties
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
```

这是 Mann-Whitney 形式：正样本的秩和，减去它们可能取到的最小秩和，再除以正负样本对的数量。分数并列时取平均秩。排序决定了复杂度，是 $O(n \log n)$。在十笔交易的例子上它返回 0.9375，也就是 16 对里有 15 对顺序正确；在 10 万行模拟数据上和 scikit-learn 的结果一致。

### 群体稳定性指数 {#psi}

**题目。** 计算基准分数分布和近期分数分布之间的 PSI。

```python
import numpy as np

def psi(expected, actual, bins=10, eps=1e-6):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))
```

分箱边界取自基准分布的分位数，最外侧两个箱向两端开放，空箱给一个很小的下限，对数才不会发散。为了对量级有感觉，我把 5 万个标准正态样本的均值做了平移：

| 近期样本的均值偏移 | PSI |
|---|---:|
| 无 | 0.00 |
| 0.25 个标准差 | 0.06 |
| 0.5 个标准差 | 0.25 |
| 1 个标准差 | 0.92 |

所以惯用的 0.25 报警线，大约相当于均值偏移了半个标准差。

### 还原降采样的代码 {#downsampling-code}

```python
import numpy as np

def correct_for_downsampling(p_sampled, beta):
    """beta = share of negatives kept in training."""
    p_sampled = np.asarray(p_sampled, dtype=float)
    return beta * p_sampled / (beta * p_sampled + 1 - p_sampled)
```

`correct_for_downsampling(0.5, 0.01)` 返回 0.0099，`correct_for_downsampling(0.9, 0.01)` 返回 0.083。

### 用 SQL 找重复交易 {#duplicates}

**题目。** 写一个查询，找出重复的交易。

我先问“重复”指什么。同一个交易 ID 出现两次是数据管道的问题，`GROUP BY transaction_id HAVING COUNT(*) > 1` 就能找到。更有用的是重复扣款这一类：同一账户、同一商户、同样金额，和上一笔相隔很短。

```sql
SELECT account_id, merchant_id, amount, ts
FROM (
  SELECT t.*,
         LAG(ts) OVER (PARTITION BY account_id, merchant_id, amount ORDER BY ts) AS prev_ts
  FROM transactions AS t
) AS x
WHERE prev_ts IS NOT NULL
  AND ts - prev_ts <= INTERVAL 60 SECOND;
```

pandas 写法：

```python
def flag_near_duplicates(df, seconds=60):
    keys = ["account_id", "merchant_id", "amount"]
    df = df.sort_values(keys + ["ts"]).reset_index(drop=True)
    gap = df.groupby(keys)["ts"].diff().dt.total_seconds()
    df["near_duplicate"] = gap.le(seconds)   # the first row of each group has no gap -> False
    return df
```

在七行测试数据上，两种写法都标出两行：10:00:20 和 10:00:50 的两笔 \$25，它们跟在同一商户 10:00:00 的那一笔之后。五分钟后的第四笔没有被标记，商户、账户或金额不同的行也没有。

有三点要说出口。每一行只和它前面那一行比，所以一串重复里从第二行开始被标记，第一行保留。重复不一定是错误：一分钟内买两杯咖啡是真实发生的。数据管道本身应该按幂等键去重，而不是看字段是否相同。

### 对账户来说不寻常的金额 {#robust-z}

**题目。** 写一个函数，只用该账户此前的交易，标出金额对这个账户来说不寻常的交易。

```python
import numpy as np
import pandas as pd

def robust_z(s, min_history=5, mad_floor=0.05):
    med = s.expanding(min_history).median().shift()          # shift: the current row is excluded
    mad = s.expanding(min_history).apply(
        lambda w: np.median(np.abs(w - np.median(w))), raw=True).shift()
    return 0.6745 * (s - med) / mad.clip(lower=mad_floor)

def flag_unusual_amounts(df, threshold=3.5, min_history=5):
    df = df.sort_values(["account_id", "ts"]).reset_index(drop=True)
    log_amount = np.log1p(df["amount"])                      # amounts are heavily right-skewed
    df["robust_z"] = log_amount.groupby(df["account_id"]).transform(
        robust_z, min_history=min_history)
    df["unusual"] = df["robust_z"].abs() > threshold
    return df
```

这是[修正 z 分数（modified z-score）](https://itl.nist.gov/div898/handbook/eda/section3/eda35h.htm)：以 MAD 为单位，衡量一个值离中位数有多远；乘以 0.6745 之后，读起来和普通的 z 分数相当，常用的切点是 3.5。下面是一个账户在五笔 \$20 到 \$30 的平常交易之后的最后四笔：

| 金额 | 稳健 z | 是否标记 | 普通 z |
|---:|---:|---|---:|
| \$21 | −1.03 | 否 | −0.87 |
| \$900 | 29.28 | 是 | 25.68 |
| \$26 | 0.41 | 否 | −0.30 |
| \$850 | 18.90 | 是 | 2.42 |

最后一行说明了为什么用中位数。\$900 那一笔进入历史之后，均值和标准差都被抬高，普通 z 分数只给 \$850 打 2.4，低于 3 这个切点。中位数和 MAD 几乎没动，所以稳健分数仍然把它标了出来。两种分数都在对数金额上计算。

有四点要说出口。`shift()` 把当前行挡在它自己的基线之外。新账户没有基线，前几行得不到分数，需要用同类客群兜底。给 MAD 设下限，是为了在此前金额全都相同时不除以零。不寻常不等于欺诈：它是一个特征或一个审核队列，不是决策。

### 热身题 {#warmups}

有些数据科学面试流程里会有一两道简单的算法题。下面六道覆盖了最常见的几种思路：哈希表、排序、维护当前最优、滑动窗口。

```python
from collections import defaultdict

def two_sum(nums, target):                 # indices of two numbers that add up to target
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return seen[target - x], i
        seen[x] = i
    return None

def group_anagrams(words):                 # words that use the same letters
    groups = defaultdict(list)
    for w in words:
        groups["".join(sorted(w))].append(w)
    return list(groups.values())

def max_subarray(nums):                    # largest sum of a contiguous run
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best

def max_profit(prices):                    # buy and sell as often as you like
    return sum(max(b - a, 0) for a, b in zip(prices, prices[1:]))

def min_abs_difference(nums):              # smallest gap between any two values
    s = sorted(nums)
    return min(b - a for a, b in zip(s, s[1:]))

def longest_unique_substring(s):           # longest substring without a repeated character
    start, best, last = 0, 0, {}
    for i, ch in enumerate(s):
        if ch in last and last[ch] >= start:
            start = last[ch] + 1
        last[ch] = i
        best = max(best, i - start + 1)
    return best
```

| 函数 | 思路 | 时间 | 核对 |
|---|---|---|---|
| `two_sum` | 记下每个值的下标，查它的补数 | $O(n)$ | `two_sum([40, 15, 25, 60], 85)` 返回 `(2, 3)` |
| `group_anagrams` | 把字母排序后作为分组的键 | $O(n \cdot k \log k)$ | `listen`、`silent`、`enlist` 在同一组 |
| `max_subarray` | 接着当前这一段，或者重新开始 | $O(n)$ | `[3, -4, 5, -1, 2, -6, 4]` 得 6 |
| `max_profit` | 把每一段上涨都加起来 | $O(n)$ | `[10, 7, 12, 9, 15, 14]` 得 11 |
| `min_abs_difference` | 排序之后，最接近的一对一定相邻 | $O(n \log n)$ | `[31, 8, 20, 14]` 得 6 |
| `longest_unique_substring` | 把窗口起点移到上一次重复的位置之后 | $O(n)$ | `"fraudfree"` 得 6 |

动手之前先说出边界情况：没有满足条件的一对、全是负数、价格只跌不涨、空字符串。

## 一页复习 {#review}

| 项目 | 一句话 |
|---|---|
| 低基率下的 precision | 欺诈率 0.1%、recall 90%、FPR 1% 时，precision 是 8.3%：每抓到一笔欺诈约有 11 条误报 |
| accuracy 为什么没用 | 全部预测成“正常”就有 99.9% |
| 基线 | ROC-AUC 是 0.5；PR-AUC 等于欺诈占比 |
| 降采样校正 | $p = \beta p_s / (\beta p_s + 1 - p_s)$，$\beta$ 是保留的负样本比例 |
| 由成本定阈值 | $p^* = C / (L + C)$：金额越大，阈值越低 |
| PSI | $\sum (a - e)\ln(a / e)$；0.1 以下稳定，0.1 到 0.25 关注，0.25 以上明显偏移 |
| recall 的置信区间 | $\pm 1.96\sqrt{r(1-r)/n_{\text{fraud}}}$：取决于欺诈样本数 |
| `scale_pos_weight` | 负样本数除以正样本数：欺诈率 0.1% 时约 999 |
| 模型治理 | SR 26-2 于 2026 年 4 月 17 日取代 SR 11-7 |
| Bagging 和 boosting | Bagging 对独立的深树取平均，降方差；boosting 依次加浅树，降偏差 |
| 特征选择 | 能用、稳定、不冗余、在更晚的数据上排序、用消融确认 |
| 实时系统设计 | 快路径约 100 毫秒内做决策，慢路径负责学习；规则兜底 |
| 交易量飙升 | 先分清是真是假，再看涨在哪里：正常高峰抬高所有分群，攻击只抬高少数几个 |

值得随时说得出口的英文术语：point-in-time features、out-of-time validation、label maturity、control group、champion-challenger、shadow mode、value detection rate、false positive ratio、step-up authentication、training-serving skew、idempotency key、fallback、feature freshness。

## 参考资料 {#references}

1. Dal Pozzolo A, Caelen O, Johnson RA, Bontempi G. [Calibrating probability with undersampling for unbalanced classification](https://doi.org/10.1109/SSCI.2015.33). IEEE Symposium Series on Computational Intelligence 2015, 159–166.
2. Elkan C. [The foundations of cost-sensitive learning](https://mlanthology.org/ijcai/2001/elkan2001ijcai-foundations/). IJCAI 2001, 973–978.
3. Chen T, Guestrin C. [XGBoost: a scalable tree boosting system](https://arxiv.org/abs/1603.02754). KDD 2016.
4. Lundberg SM, Lee S-I. [A unified approach to interpreting model predictions](https://arxiv.org/abs/1705.07874). NeurIPS 2017.
5. Board of Governors of the Federal Reserve System. [SR 26-2: Revised guidance on model risk management](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm). 2026 年 4 月 17 日。
6. Bowman MW. [Artificial intelligence in the financial system](https://www.federalreserve.gov/newsevents/speech/bowman20260501a.htm). 讲话，2026 年 5 月 1 日。
7. Strobl C, Boulesteix A-L, Zeileis A, Hothorn T. [Bias in random forest variable importance measures: illustrations, sources and a solution](https://pmc.ncbi.nlm.nih.gov/articles/PMC1796903). BMC Bioinformatics 2007, 8:25.
8. NIST/SEMATECH. [e-Handbook of statistical methods, section 1.3.5.17: Detection of outliers](https://itl.nist.gov/div898/handbook/eda/section3/eda35h.htm). 其中给出 Iglewicz 和 Hoaglin（1993）的修正 z 分数。

这篇笔记里的每个数字都由 [`fraud-practice-lab.py`](/notes/fraud-detection-practice/fraud-practice-lab.py) 打印。频次特征的 SQL 和 PySpark 写法，分别用 DuckDB 2.0 和 PySpark 4.3 对照 pandas 的输出核对过；重复交易的查询用 DuckDB 2.0 核对过。
