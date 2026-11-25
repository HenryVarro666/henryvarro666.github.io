---
title: "机器学习与统计基础（二）：推导、评分卡、图和生存分析"
date: 2026-11-25T05:35:00-06:00
description: "基础题之后的第二层：要在纸上做的推导、WoE 和评分卡刻度这类风控常规、更多统计题、概率小题、图、生存分析、深一层的强化学习、SQL 套路，以及能运行的代码。"
topics: [data-science]
---
[基础题那篇](/zh/notes/ml-stats-fundamentals/)覆盖的是 panel 先问的问题。这一篇覆盖接下来会问的：“推导一下”“风控团队会怎么做”，以及反欺诈团队每天打交道的主题，比如图和迟到的标签。清单是这样来的：我拿一张主题表逐项核对之前的几篇笔记，把缺的写了出来。

> **先说结论。** 追问通常往三个方向走：往下，进入数学；往旁边，进入风控团队的实际做法；往前，进入上线之后会出什么问题。这里的每一节对应其中一个方向。

## 怎么用这篇笔记 {#how-to}

[推导](#derivations)要拿纸笔做，不要只是读。其余的出声回答。每个数字都由[练习脚本](#more-code)打印，SQL 在 DuckDB 里跑过。

面对反欺诈建模的 panel，我会先练[风控建模的常规](#staples)、[图](#graphs)和[生存分析](#survival)。

## 推导 {#derivations}

**推导偏差–方差分解。** 设 $y = f(x) + \varepsilon$，噪声方差为 $\sigma^2$；$\hat{f}$ 是在一个随机样本上训练出来的模型。在平方里加一项、减一项 $E[\hat{f}]$：

$$
\begin{aligned}
E\bigl[(y - \hat{f})^2\bigr] &= \bigl(f - E[\hat{f}]\bigr)^2 \\
&\quad + E\bigl[(\hat{f} - E[\hat{f}])^2\bigr] + \sigma^2
\end{aligned}
$$

三项分别是偏差的平方、方差和噪声。交叉项消失，是因为噪声和训练样本独立，而 $\hat{f} - E[\hat{f}]$ 的均值为零。

**推导逻辑回归的梯度和 Hessian。损失为什么是凸的？** 记 $p = \sigma(Xw)$，log loss 为 $L = -\sum_i \bigl[y_i \log p_i + (1 - y_i)\log(1 - p_i)\bigr]$，利用 $\sigma' = \sigma(1 - \sigma)$：

$$
\begin{aligned}
\nabla L &= X^\top (p - y) \\
\nabla^2 L &= X^\top W X, \quad W = \mathrm{diag}\bigl(p_i(1 - p_i)\bigr)
\end{aligned}
$$

对任意向量 $v$，$v^\top X^\top W X v = \sum_i p_i(1 - p_i)(x_i^\top v)^2 \ge 0$。Hessian 半正定，所以损失是凸的，任何局部最小值都是全局最小值。

**这里的牛顿法是什么？为什么叫迭代重加权最小二乘？** 更新式是 $w \leftarrow w - (X^\top W X)^{-1} X^\top (p - y)$。每一步解的是一个加权最小二乘问题，而权重 $p(1-p)$ 每次迭代都在变。脚本里它 7 次迭代就收敛，梯度下降要几千步。每次迭代要解一个规模等于特征数的线性方程组，所以统计软件在小模型上用它，大模型改用梯度类方法。

**写出岭回归的闭式解，并说明它做了什么。** 最小化 $\lVert y - Xw \rVert^2 + \lambda \lVert w \rVert^2$，得到 $\hat{w} = (X^\top X + \lambda I)^{-1} X^\top y$。加上 $\lambda I$ 之后，即使特征共线，矩阵也可逆。沿着 $X$ 的每个奇异方向（奇异值为 $d$），系数被乘以 $d^2/(d^2 + \lambda)$。$\lambda = 1$ 时，奇异值为 10 的方向保留 99% 的系数，奇异值为 1 的方向只保留 50%。数据说得最少的地方，岭回归收缩得最多。

**推导 XGBoost 的分裂增益。** 在当前预测值附近把损失展开到二阶，每一行贡献一个梯度 $g_i$ 和一个 Hessian $h_i$。一个叶子上两者的和为 $G$ 和 $H$、叶子权重为常数 $w$ 时，目标函数是 $Gw + \tfrac{1}{2}(H + \lambda)w^2$。它在 $w^* = -G/(H + \lambda)$ 处取最小值 $-\tfrac{1}{2}\,G^2/(H + \lambda)$。所以分裂一个叶子的增益是

$$
\begin{aligned}
\text{Gain} = \frac{1}{2}\Bigl[&\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} \\
&- \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\Bigr] - \gamma
\end{aligned}
$$

log loss 下，$g = p - y$，$h = p(1 - p)$。取六行，当前预测都是 $p = 0.5$：左子节点的标签是 1、1、0，右子节点是 0、0、0。于是 $G_L = -0.5$，$G_R = 1.5$，$H_L = H_R = 0.75$。$\lambda = 1$ 时，两个叶子的权重是 +0.286 和 −0.857，增益是 0.514。对应到库里的参数：`lambda` 收缩叶子权重，`gamma` 是分裂必须超过的增益，`min_child_weight` 是 $H$ 的下限。

**解释高斯混合模型的 EM 算法。** E 步为每个点计算各个成分对它的“责任”：$r_{ik} \propto \pi_k\,\mathcal{N}(x_i \mid \mu_k, \sigma_k^2)$。M 步用这些责任作权重，重新估计每个成分的权重、均值和方差。一次迭代不会让似然下降，因为 M 步最大化的是一个下界，而这个下界在当前参数处正好等于似然。它收敛到局部最优。k-means 是它的特例：硬分配，各成分方差相等且固定。在真实均值为 0 和 5 的混合数据上，脚本估出 0.01 和 4.93。

**softmax 配 cross-entropy，梯度为什么正好是 p − y？** 设 $p = \mathrm{softmax}(z)$，$L = -\sum_k y_k \log p_k$，求导得 $\partial L / \partial z_j = p_j - y_j$：对数抵消了指数，归一化项的导数正好是 $p$。再对线性层 $z = Wx + b$ 做反向传播，得到 $\partial L/\partial W = (p - y)\,x^\top$ 和 $\partial L/\partial x = W^\top (p - y)$。

**softmax 怎么算才数值稳定？** 取指数之前先减去最大的 logit，数学上结果不变。logit 为 1000、1001、1002 时，朴素写法溢出，返回 NaN；稳定写法返回 0.090、0.245 和 0.665。要算对数概率时，出于同样的原因用 log-sum-exp。

**写出 SVM 的优化问题和它的 hinge loss 形式，并说明核函数为什么可行。** 原问题是在 $y_i(w^\top x_i + b) \ge 1 - \xi_i$、$\xi_i \ge 0$ 的约束下最小化 $\tfrac{1}{2}\lVert w \rVert^2 + C\sum_i \xi_i$。它等价于最小化 hinge loss $\max(0,\, 1 - y f(x))$ 加 L2 惩罚。对偶问题只通过内积 $x_i^\top x_j$ 用到数据，所以可以用核函数 $k(x_i, x_j)$ 替换内积。解的形式是 $f(x) = \sum_i \alpha_i y_i\, k(x_i, x) + b$，只有支持向量的 $\alpha_i$ 不为零。

**PCA 和奇异值分解是什么关系？** 把数据中心化，写成 $X = U S V^\top$。$V$ 的各列是主方向，$S^2/(n - 1)$ 是各方向上的方差，投影后的数据是 $XV = US$。用奇异值分解不必先算协方差矩阵，数值上更稳定。脚本的例子里，前两个主成分分别解释了 91.8% 和 7.8% 的方差。

## 风控建模的常规 {#staples}

**证据权重和信息值是什么？** 对一个特征的某个分箱，证据权重（weight of evidence，WoE）是“全部好样本里落在这一箱的比例”除以“全部坏样本里落在这一箱的比例”，再取对数。为正表示比平均更安全。信息值（information value，IV）是对所有分箱，把这两个比例之差乘以 WoE 再求和。以“最近一小时的交易笔数”为例：

| 分箱 | 好样本占比 | 坏样本占比 | WoE |
|---|---:|---:|---:|
| 0 到 1 笔 | 60% | 45% | +0.288 |
| 2 到 3 笔 | 30% | 30% | 0.000 |
| 4 到 6 笔 | 8% | 17% | −0.754 |
| 7 笔以上 | 2% | 8% | −1.386 |

IV 是 0.194。信用评分里的经验读法：0.02 以下没用，0.02 到 0.1 弱，0.1 到 0.3 中等，0.3 以上强，0.5 以上就该查有没有泄漏。IV 和 PSI 是同一个公式，只是比较的对象换成了好样本和坏样本。做了 WoE 编码之后，逻辑回归也能处理非线性效应和缺失值（缺失值单独成一箱）。分箱和 WoE 值只在训练时间窗上学。

**评分卡的刻度怎么定？** 分数 = offset + factor × ln(好对坏的 odds)，其中 factor = PDO / ln 2，PDO 是让 odds 翻倍所需的分数。把 odds 为 50:1 定在 600 分、PDO 取 20，得到 factor 28.85、offset 487.1。于是 odds 为 100:1 对应 620 分，25:1 对应 580 分。模型有 $n$ 个特征时，一个分箱的分值是（WoE × 系数 + 截距 / $n$）× factor + offset / $n$。

**拒绝推断有哪些方法？你信哪个？** 只有获批的申请人才有结果。重新加权：每个获批样本按“获批概率的倒数”计数，前提是被拒的申请人和某些获批的申请人相像。Parceling：用只在获批样本上训练的模型给被拒样本打分，再按抬高后的坏账率给它们分配结果。Fuzzy augmentation：每个被拒样本加进去两次，一次算好、一次算坏，按预测概率加权。外部数据：看被拒的人在其他机构的表现。随机批准一小部分，是唯一不依赖不可检验假设的方法。不管用哪种，我都会把它依赖的假设讲出来。

**你在降采样后的负样本上训练了逻辑回归，系数会怎样？** 斜率在期望上不变。只有截距移动，移动量是抽样比例的对数：保留 1% 的负样本，截距会偏高 4.605。把 $\ln \beta$ 加回去就校正了，这是[odds 校正](/zh/notes/fraud-detection-practice/#drill-downsampling)的另一种写法。

**什么是完全分离？** 某个特征或特征组合把两类毫无差错地分开了。这时系数越大，似然越高，没有尽头，所以软件会报出巨大的系数和标准误，或者不收敛。在欺诈数据里，原因通常是泄漏。不是泄漏的话，加惩罚项就能解决。

**AIC 还是 BIC？** $\mathrm{AIC} = 2k - 2\ln L$，$\mathrm{BIC} = k \ln n - 2\ln L$，$k$ 是参数个数，越小越好。BIC 对参数的惩罚更重，选出的模型更小。AIC 以预测为目标，BIC 以找到真实模型为目标。预测模型我看的是时间外验证；这两个指标只用来快速比较嵌套的统计模型。

**什么是 swap-set 分析？** 在相同的报警量或批准量下比较两个模型，把样本分成三组：两个模型都标记的、只有新模型标记的（swap-in）、只有旧模型标记的（swap-out）。假设两个模型各报 1,000 条，其中 700 条相同；300 条 swap-in 里欺诈占 10%，300 条 swap-out 里占 3%。那么在同样的报警量下，新模型多抓 21 笔欺诈。这个分析还能准确指出哪些客户会受到不同的对待，这正是运营和合规会问的。

**测试集是抽样得到的，怎么给出生产环境里的 precision？** recall 和假阳性率不随基率 $\pi$ 变化，precision 会变：

$$
\text{precision} = \frac{\mathrm{TPR}\cdot\pi}{\mathrm{TPR}\cdot\pi + \mathrm{FPR}\cdot(1 - \pi)}
$$

recall 为 60%、假阳性率为 0.5% 时，欺诈占 1% 则 precision 是 54.8%，欺诈占 0.1% 则是 10.7%。

**模型验证方会问什么？** 设计在概念上是否站得住：为什么用这个方法，有哪些假设，用了什么数据。结果是否经得起检验：更晚时间段上的表现、校准、分群结果。和更简单的基准模型或挑战者相比如何。输入偏移或数据缺失时有多敏感。监控方案是什么，阈值和对应动作是什么。已知的局限有哪些，用什么来弥补。练习题里提到了[现行的监管指引](/zh/notes/fraud-detection-practice/#explainability)。

**什么是期望校准误差？** 把预测值分箱，对每一箱里“预测比率和实际比率之差”按箱的大小加权平均。三箱的预测值是 0.1、0.5、0.9，实际比率是 0.08、0.6、0.7，样本数是 800、150、50，结果是 0.041。这个值依赖分箱方式，所以我会把可靠性曲线和它一起给出。

**用一个回答讲清几个公平性指标。** Demographic parity 要求各组的批准率相等。Equalized odds 要求各组的真阳性率和假阳性率都相等。Predictive parity 要求各组的 precision 相等。各组基率不同时，三者无法同时成立。一个常用的筛查经验是五分之四规则：批准率之比低于 0.8 就要留意，45% 对 60% 是 0.75。受保护的属性不作为模型输入，但其他特征可能成为它们的替身，所以要分组检验结果。

## 更多统计 {#more-statistics}

**该知道哪几对共轭分布？**

| 先验 | 数据 | 后验 |
|---|---|---|
| Beta($a$, $b$) | $n$ 次试验里成功 $k$ 次 | Beta($a + k$, $b + n - k$) |
| Gamma（形状 $a$，速率 $b$） | $n$ 个时间段内泊松计数之和为 $s$ | Gamma($a + s$, $b + n$) |
| 正态 | 方差已知的正态 | 正态，均值按精度加权 |

先验 Gamma(2, 1)，10 天里观测到 30 次事件，后验是 Gamma(32, 11)，均值为每天 2.91 次。共轭先验相当于此前已有的观测：Beta($a$, $b$) 相当于 $a$ 次成功和 $b$ 次失败。

**写出全期望公式和全方差公式。** $E[X] = E\bigl[E[X \mid G]\bigr]$，$\mathrm{Var}(X) = E\bigl[\mathrm{Var}(X \mid G)\bigr] + \mathrm{Var}\bigl(E[X \mid G]\bigr)$：组内方差加组间方差。如果 90% 的账户平均消费 \$50、标准差 \$20，10% 的账户平均消费 \$300、标准差 \$100，那么总体均值是 \$75，组内部分是 1,360，组间部分是 5,625，标准差是 \$83.6。方差大部分来自分群之间，这就是分群或分层能去掉那么多方差的原因。

**马尔可夫不等式和切比雪夫不等式是什么？** 马尔可夫：对非负的随机变量，$P(X \ge a) \le E[X]/a$。切比雪夫：$P(\lvert X - \mu \rvert \ge k\sigma) \le 1/k^2$。两者都不假设分布。在三倍标准差处，切比雪夫给出的上界是 11.1%，而正态分布是 0.27%。只有数据接近正态时，“三倍标准差”这个阈值才算紧。

**什么是 delta 方法？** 给“估计量的函数”求标准误：$\mathrm{Var}\bigl(g(\hat\theta)\bigr) \approx g'(\theta)^2\,\mathrm{Var}(\hat\theta)$。比值类的量，我在对数尺度上做。12% 对 10% 的 lift，各 5,000 个样本，对数尺度上的标准误是 0.057，所以 lift 的 95% 区间是 1.07 到 1.34。

**Fisher 信息和 Cramér-Rao 下界是什么？** Fisher 信息是对数似然的期望曲率：峰越尖，数据对参数说得越多。Cramér-Rao 下界说，任何无偏估计量的方差都不会低于它的倒数。大样本下最大似然估计达到这个下界，所以标准误取自 Hessian 的逆。

**Wald 检验、似然比检验和 score 检验有什么区别？** 三者都建立在似然上。Wald 检验用估计值除以标准误。似然比检验比较两个嵌套模型的对数似然，差值的两倍服从卡方分布，自由度等于多出来的参数个数。Score 检验用对数似然在原假设处的斜率。大样本下三者一致，小样本下似然比检验最可靠。

**ANOVA 检验的是什么？** 三个或更多组的均值是否相等，方法是比较组间方差和组内方差。结果显著只说明均值不全相等，不说明是哪几组不同，所以之后还要做两两比较，并做多重检验校正。只有两组时，它等于 t 检验的平方。

**McNemar 检验是什么？** 用来比较在同一批样本上评分的两个分类器。只有两者结论不同的样本起作用：第一个对、第二个错的有 $b$ 个，反过来的有 $c$ 个。统计量是 $(b - c)^2/(b + c)$，服从自由度为 1 的卡方分布。$b = 40$、$c = 20$ 时是 6.67，p 值 0.010；做连续性校正后是 6.02 和 0.014。计数很小时，我对“$b + c$ 个里有 $b$ 个”做精确二项检验。

**CUPED 是什么？** 一种降低实验方差的方法：把实验前的协变量能预测的那部分去掉，$Y' = Y - \theta\,(X - \bar{X})$，其中 $\theta = \mathrm{Cov}(X, Y)/\mathrm{Var}(X)$。估计仍然无偏，因为实验处理不可能影响 $X$。方差下降的比例是相关系数的平方：相关系数 0.6 时下降 36%，所以达到同样的功效只需要 64% 的样本。[A/B 测试那篇](/zh/notes/ab-testing/)讲了更完整的设计。

**什么是广义线性模型？** 指数族里的一个分布、一个线性预测量，以及连接两者的 link 函数。线性回归是正态分布配恒等 link，逻辑回归是二项分布配 logit，泊松回归是计数配对数。对比率建模时我加一个 offset，也就是暴露量的对数，这样模型描述的是单位业务量上的事件数。如果方差大于均值，就用负二项模型代替泊松，否则标准误会偏小。

**回归诊断你看哪些？** 残差对拟合值的图：弯曲说明缺了非线性项，喇叭形说明方差在变。残差的分位数图，看尾部。杠杆值找出特征取值不寻常的点，Cook 距离找出既不寻常又拟合得差的点。方差膨胀因子看共线性。

**什么是分位数回归？** 它通过最小化 pinball loss 来对条件分位数建模：低估时每单位误差的代价是 $\tau$，高估时是 $1 - \tau$。尾部要紧的时候用它，比如延迟或损失的第 95 百分位；噪声有偏时也用。提升树可以直接用这个损失。

**建模的人对时间序列该知道些什么？** 平稳性：均值、方差和自相关不随时间变化，可以通过差分或去趋势得到。季节性和自相关。用“季节性朴素预测”作基线。ARIMA 是自回归加差分，再加过去误差的移动平均。验证要用滚动起点，绝不随机切分。自相关还会让独立数据的公式失效：一阶自相关系数为 0.5 时，均值的方差是通常公式的三倍，所以 300 个按小时的数据点只相当于 100 个独立的点。

**什么是回归均值？** 极端的观测值里有一部分是噪声，所以下一次观测往往更接近平均水平。脚本里，一个带噪声的风险指标排在前 10% 的账户，第一次平均是 2.48，第二次是 1.25，中间没有任何干预。所以“我们对最差的账户采取了措施，它们好转了”这句话，需要对照组才成立。

**什么是古德哈特定律？** 一个指标一旦变成目标，就不再是好的指标。按报警 precision 考核的团队，可以只对明显的案例报警来抬高它，recall 随之崩塌。我会把每个指标和它可以拿来交换的那个指标配成一对：precision 配 recall，损失配客户摩擦。

## 概率小题 {#puzzles}

| 题目 | 答案 | 理由 |
|---|---|---|
| 三门问题：换还是不换？ | 换，2/3 | 第一次选对的概率是 1/3，主持人开门不改变这一点 |
| 两个孩子，至少一个是男孩，两个都是男孩的概率？ | 1/3 | 男男、男女、女男三种情况等可能 |
| 抛硬币，直到第一次正面的期望次数 | 2 | 几何分布：$1/p$ |
| 直到连续两次正面的期望次数 | 6 | 对开头几次抛掷取条件：$E = 1 + \tfrac{1}{2}E_1 + \tfrac{1}{2}E$，$E_1 = 1 + \tfrac{1}{2}E$ |
| 集齐 6 种卡片要抽多少次 | 14.7 | $6\,(1 + \tfrac{1}{2} + \dots + \tfrac{1}{6})$ |
| 23 个人里有人生日相同的概率 | 50.7% | 1 减去所有人生日都不同的概率 |
| 两颗骰子较大点数的期望 | 4.47 | $\sum_k k\,(2k - 1)/36 = 161/36$ |
| 掷四次至少出现一次六点 | 51.8% | $1 - (5/6)^4$ |
| 随机排列里不动点个数的期望 | 1 | 期望的线性性：$n$ 个位置，每个不动的概率是 $1/n$ |

四招能解决其中大部分：写出样本空间，取补事件，用期望的线性性，对第一步取条件。卡住的时候，就说自己会怎么用模拟来验证；脚本对其中两题做了模拟。

## 图 {#graphs}

**图为什么对反欺诈有用？** 欺诈是有组织的。同一伙人控制的账户会共用设备、地址、电话和收款人；一个新账户如果和已确认的欺诈相连，它的风险比自身历史显示的要高。一张各行相互独立的表表达不了这一点。

**图怎么建？** 通常建成二部图：一边是账户或卡，另一边是设备、商户、地址或收款人，一起出现过就连一条边。把它投影到账户上，也就是给共用设备的两个账户连边，用起来更简单，但丢掉了“它们共用的是哪个实体”。每条边都带时间戳。

**哪些图特征可以放进树模型？** 度；我的每个实体被多少账户共用；一跳和两跳邻居里的欺诈率；所在连通分量的大小；到最近一个已确认欺诈的距离；personalized PageRank 分数。全部按交易发生的时刻计算，邻居的标签只有在当时已经确认才算数。

**怎么找出相互关联的账户群？** 连通分量，用并查集（union-find）：每条边合并两个集合，每条边的开销接近常数。在六个账户、三台设备的例子上，脚本找到一个含三个账户的分量和一个含两个账户的分量，第六个账户单独成组。连通分量只是起点：一个机场 Wi-Fi 的公共地址，就能把几千个毫不相干的账户并到一起。

**枢纽节点怎么处理？** 度极大的节点，比如公共 IP 地址或大商户，把什么都连在一起，本身不带信号。我会给度设上限，去掉这类实体或降低它们的权重，并对邻居做抽样。同时把度本身留作特征，因为“和 3 个账户共用设备”与“和 3,000 个账户共用设备”意思完全不同。

**什么是 personalized PageRank？** 一个随机游走的长期分布：每一步都有一定概率回到一组种子节点。用已确认的欺诈作种子，每个节点就得到一个“接近程度”的分数，它算进了所有路径，而不只是最短的那条。在脚本的小图上，以账户 a1 为种子，和 a1 共用一台设备的账户得 0.195，隔了两台设备的账户得 0.055，不相连的账户得零。

**什么是标签传播？** 每个节点反复取邻居标签的加权平均，已知的标签保持不动。它是一个很快的半监督基线，不需要特征，前提是相连的节点标签相同。

**解释一下图神经网络。** 每一层用节点自己的向量和邻居向量的聚合结果来更新这个节点的向量，所以 $k$ 层能看到 $k$ 跳。图卷积网络（GCN）用归一化的求和。GraphSAGE 抽取固定数量的邻居，并学习聚合函数，所以能扩展到大图，也能用在训练时没见过的节点上。图注意力网络（GAT）为每个邻居学一个权重。层数叠得太多，不同节点的向量会变得相似，这叫过平滑（over-smoothing）。

**transductive 和 inductive 有什么区别？** transductive 的模型为每个节点学一个 embedding，训练时不存在的节点它无法评分。inductive 的模型学的是“特征加邻域”的函数，所以能处理新账户，而反欺诈正需要这一点。

**图数据怎么切分来做验证？** 按时间切。训练用的图只包含截止时刻已知的边和标签，模型在之后的交易上测试。随机切分节点会通过共同邻居泄漏；用了事件发生之后才确认的邻居标签，就是泄漏未来。

**什么是社区发现？** 找出内部连接比外部更稠密的群。Louvain 方法贪心地最大化模块度，也就是群内边的占比减去随机情况下的期望占比。在反欺诈里，它给分析师提供候选的团伙；它不做任何决定。

## 生存分析 {#survival}

**为什么把标签看成“到事件发生的时间”？** 固定窗口的标签（“90 天内是否欺诈”）必须等满 90 天，更新的数据只能丢掉。时间到事件的模型，每个样本被观测了多久就用多久。

**生存函数和风险率是什么？** $S(t)$ 是到时刻 $t$ 事件还没有发生的概率。风险率（hazard）$h(t)$ 是在仍处于风险中的样本里，事件在 $t$ 时刻发生的速率。两者包含相同的信息：$S(t) = \exp\bigl(-\int_0^t h(u)\,du\bigr)$。

**Kaplan-Meier 估计怎么算？** 在每个事件发生的时刻，乘上“仍处于风险中的样本里熬过这一刻的比例”：$\hat{S}(t) = \prod_{t_i \le t}(1 - d_i/n_i)$。删失的样本退出风险集，但不算作事件。

| 天数 | 风险集 | 事件数 | 生存概率 |
|---:|---:|---:|---:|
| 5 | 10 | 1 | 0.900 |
| 12 | 8 | 2 | 0.675 |
| 25 | 5 | 1 | 0.540 |
| 40 | 2 | 1 | 0.270 |

这十个账户里有五个发生了事件，所以朴素的比率是 50%。Kaplan-Meier 估出到第 40 天发生事件的概率是 73%，因为有四个账户在第 40 天之前就退出了观测，期间没有发生事件。忽略删失会低估风险。

**两条生存曲线怎么比较？** 用 log-rank 检验：在每个事件发生的时刻，把各组实际发生的事件数和“风险率相同”时的预期数做比较。

**Cox 模型是什么？** $h(t \mid x) = h_0(t)\,\exp(\beta^\top x)$。基线风险率 $h_0$ 不指定形式，系数用偏似然来拟合，偏似然只用到事件发生的先后顺序。$e^{\beta}$ 是风险比（hazard ratio）。模型假设这个比值不随时间变化，这一点要检验。

**生存问题怎么用提升树来做？** 改写成离散时间。每个样本在它处于风险中的每个时间段各占一行，标签是这个时间段内事件是否发生。在这些行上训练的分类器估计的就是风险率；特征可以逐期变化；删失的处理很简单，行到此为止。

**近期的欺诈数怎么校正报告延迟？** 用已经成熟的批次估计“各个账龄之前已报告的欺诈占比”，再用已经看到的数量除以这个占比。如果第 7 天已报告 50%，第 30 天 90%，第 90 天 99%，那么在这三个账龄上分别看到 10、27、40 笔欺诈的批次，最终大约是 20、30、40 笔。同样的占比还可以在训练时给尚未有标签的新样本加权。

**什么是竞争风险？** 另一种事件可能先把样本带走：账户在发现任何欺诈之前就关闭了。把它当成普通的删失会高估欺诈概率，所以竞争事件也要一起建模。

**concordance index 是什么？** 在所有可比较的样本对里，预测风险更高的那个确实更早发生事件的比例。它是适用于删失数据的 AUC。

## 强化学习，再深一层 {#rl-deeper}

**写出 Bellman 方程。** 对策略 $\pi$，$V^{\pi}(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a)\,\bigl[r + \gamma V^{\pi}(s')\bigr]$。对最优策略，$Q^*(s, a) = E\bigl[r + \gamma \max_{a'} Q^*(s', a')\bigr]$。一个状态的价值，等于眼前的奖励加上后续价值的折现。$\gamma = 0.9$ 时，依次得到的奖励 1、0、2 价值 2.62。

**蒙特卡洛还是时序差分？** 蒙特卡洛等一个回合结束，用实际的回报：无偏，但噪声大。时序差分（temporal-difference）朝着“奖励加上对下一个状态的当前估计”更新：$V(s) \leftarrow V(s) + \alpha\,[r + \gamma V(s') - V(s)]$。它因为依赖自己的估计而有偏，但方差更小，而且回合没结束就能学。

**写出 Q-learning 的更新式。** $Q(s, a) \leftarrow Q(s, a) + \alpha\,[r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$。当前值为 2，奖励为 1，下一状态的最大值为 3，$\gamma = 0.9$，$\alpha = 0.5$，更新后是 2.85。它是 off-policy 的，因为目标里用的是下一步最好的动作，不管实际走了哪一步。DQN 在此基础上加了经验回放和缓慢更新的目标网络。对带噪声的估计取最大值会造成高估，double Q-learning 修正的就是这一点。

**写出策略梯度。** $\nabla J(\theta) = E\bigl[\nabla_\theta \log \pi_\theta(a \mid s)\; Q^{\pi}(s, a)\bigr]$：一个动作的结果越好，就把它的概率提得越多。减去一个只依赖状态的基线，期望不变，方差下降。用状态价值作基线时，权重就变成优势函数 $Q - V$。

**PPO 优化的是什么？** $E\bigl[\min\bigl(r_t A_t,\ \mathrm{clip}(r_t,\, 1 - \epsilon,\, 1 + \epsilon)\,A_t\bigr)\bigr]$，其中 $r_t$ 是新旧策略对该动作的概率之比，$A_t$ 是优势。裁剪去掉了把这个比值推离 1 很远的动力，所以同一批数据可以重复训练几轮。

**怎么用日志估计新策略的价值？** 逆倾向加权（inverse propensity scoring）：对新策略和日志动作一致的那些记录，把奖励按“日志策略选这个动作的概率的倒数”加权后取平均。它无偏，但噪声大。自归一化的版本改为除以权重之和，用一点点偏差换来小得多的方差。doubly robust 估计再加一个奖励模型，只要模型和倾向概率有一个是对的，它就是相合的。脚本里的八条日志，新策略的真实价值是 0.75；普通的逆倾向加权给出 1.48，自归一化的版本给出 0.86。

**contextual bandit 的算法你会提哪几个？** LinUCB 给每个动作拟合一个线性的奖励模型，再加一个奖励项；这个动作在相似的情境下试得越多，奖励项越小。Thompson sampling 维护模型的后验分布，从中抽样，再按抽到的模型行动。两者都要把所选动作的概率记进日志，留着以后做评估。

## 深度学习的机制 {#dl-mechanics}

**batch normalization 还是 layer normalization？** batch normalization 在一个 batch 内对每个特征做标准化。它依赖 batch 的统计量，所以训练和推理时行为不同，batch 小的时候效果变差。layer normalization 在单个样本的各个特征之间做标准化，不依赖 batch，所以 Transformer 和循环网络用的是它。

**写出 Adam 的更新式。** $m \leftarrow \beta_1 m + (1 - \beta_1)g$，$v \leftarrow \beta_2 v + (1 - \beta_2)g^2$。两者都除以 $1 - \beta^t$，校正从零开始带来的偏差，步长是 $\eta\,\hat{m}/(\sqrt{\hat{v}} + \epsilon)$。因为有这个校正，第一步的大小就等于学习率，和梯度的量级无关：梯度是 250、学习率是 0.001 时，权重移动 0.001。AdamW 把权重衰减直接作用在权重上，不经过这层缩放。

**残差连接为什么有用？** 一个模块计算的是 $x + F(x)$，所以梯度可以经由恒等映射直接传回去。每个模块只需要学一个修正量，很深的网络因此可以训练。

**注意力的开销是多少？为什么要编码位置？** self-attention 要比较每一对位置，所以时间和内存随序列长度的平方增长。它本身对顺序不敏感，所以要把位置信息加到输入上。

**LoRA 是什么？** 冻结权重矩阵 $W$，给它加一个可训练的低秩更新 $BA$，其中 $B$ 的大小是 $d \times r$，$A$ 的大小是 $r \times k$。对一个 4,096 × 4,096 的矩阵取秩 16，可训练参数是 131,072 个，而原矩阵有 1,680 万个，占 0.78%。训练完可以把更新合并进 $W$，推理没有额外开销。

**什么是对比学习？** 训练一个编码器，让同一个对象的两个视图靠近，不同对象的视图远离，通常是在相似度上做带温度的 softmax。它不需要标签就能得到有用的 embedding。

**什么是知识蒸馏？** 训练一个小的学生模型，去拟合大的教师模型软化之后的输出概率，这些概率比硬标签带有更多信息。它用一点精度换延迟，这对实时决策很要紧。

## 异常和解释 {#anomaly-interp}

**为什么用马氏距离而不用欧氏距离？** 它以数据自身的协方差为单位来量距离。两个特征的相关系数是 0.9 时，点 (2, 2) 和点 (2, −2) 到均值的欧氏距离相同，都是 2.83。它们的马氏距离却是 2.05 和 8.94，因为第二个点和相关的方向相反。协方差要用稳健的方法估计，否则离群值会把自己藏起来。

**局部离群因子还是 one-class SVM？** 局部离群因子（LOF）把一个点的密度和它邻居的密度相比，所以能找出“相对所在区域”显得异常的点。one-class SVM 学的是包住正常数据的一条边界，结果很依赖核函数和参数。两者在大数据上都会变慢，那时[孤立森林](/zh/notes/ml-stats-fundamentals/#algorithms)的扩展性更好。

**极值理论用在哪里？** 用来在尾部很远的地方定阈值。超过一个高阈值的那些值近似服从广义 Pareto 分布，所以可以估计很少出现、甚至从未出现过的水平的概率，经验分位数做不到这一点。

**部分依赖图、ICE 和 ALE 有什么区别？** 部分依赖图（partial dependence）展示一个特征变化时预测值的平均变化；特征相关时它会误导，因为它对现实中不会出现的特征组合也取了平均。ICE（individual conditional expectation）给每个样本画一条曲线，能看出被平均值掩盖的交互。ALE（accumulated local effects）只用这个特征真实取值附近的样本，所以能应对相关性。

**SHAP 值的哪些性质让它能用来给原因代码？** 可加：各特征的贡献之和，等于这次预测减去平均预测。一致：模型更依赖某个特征时，它的归因不会下降。对树模型可以精确而快速地计算。它们描述的是模型，不是真实世界。

**提升树里的单调约束是怎么实现的？** 在有约束的特征上，如果一次分裂会让两个子节点的取值顺序反了，这次分裂就被拒绝；同时把上下界向下传递，更深的分裂也无法破坏这个顺序。代价是一点精度，换来的是行为可以解释、可以辩护的模型，比如“交易频次上升时风险绝不下降”。

**什么是对抗验证？** 训练一个分类器来区分训练集的行和测试集的行。AUC 接近 0.5，说明两者看起来一样。AUC 很高，说明数据发生了偏移，而这个分类器最重要的特征指出了偏移发生在哪里。

## SQL 和数据处理的套路 {#sql}

除了普通的 join，最常被问到的大致就是下面四种。查询用的是 DuckDB 语法，在脚本里对九笔交易运行过。

**每组取前 N 个。** 每个账户金额最大的两笔交易：

```sql
SELECT account_id, ts, amount
FROM (
  SELECT t.*, ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY amount DESC, ts) AS rn
  FROM transactions AS t
) AS x
WHERE rn <= 2
ORDER BY account_id, amount DESC
```

排序键不完整时，`ROW_NUMBER` 对并列的行是任意取舍的；`RANK` 保留并列，可能返回多于 N 行。

**间隔与连续段（gaps and islands）。** 连续活跃的天数：

```sql
SELECT account_id, MIN(day) AS first_day, COUNT(*) AS days
FROM (
  SELECT account_id, day,
         day - CAST(ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY day) AS INTEGER) AS grp
  FROM (SELECT DISTINCT account_id, CAST(ts AS DATE) AS day FROM transactions) AS d
) AS x
GROUP BY account_id, grp
ORDER BY account_id, first_day
```

在一段连续的日期里，“日期减行号”是常数，所以可以拿它当分组键。账户 A 在 12 月 1、2、3、5 日活跃，得到一段三天和一段一天。

**会话划分。** 超过 30 分钟没有活动，就开始一个新会话：

```sql
SELECT account_id, ts,
       SUM(new_session) OVER (PARTITION BY account_id ORDER BY ts) AS session_id
FROM (
  SELECT t.*,
         CASE WHEN ts - LAG(ts) OVER (PARTITION BY account_id ORDER BY ts) <= INTERVAL 30 MINUTE
              THEN 0 ELSE 1 END AS new_session
  FROM transactions AS t
) AS x
ORDER BY account_id, ts
```

先标出每个会话的第一行，再对这个标记做累计求和。

**按时点关联（point-in-time join）。** 给每笔交易接上当时生效的风险等级：

```sql
SELECT t.account_id, t.ts, t.amount, r.risk_tier
FROM transactions AS t
ASOF LEFT JOIN risk_tiers AS r
  ON t.account_id = r.account_id AND t.ts >= r.valid_from
ORDER BY t.account_id, t.ts
```

pandas 里的写法是 `pd.merge_asof(transactions, tiers, left_on="ts", right_on="valid_from", by="account_id")`，两张表都要先按时间排序。如果只按账户做普通的 join，就会把最新的等级接到旧交易上，那是泄漏。没有 `ASOF` 的引擎，可以按账户和 `valid_from <= ts` 做 join，再用 `ROW_NUMBER` 给每笔交易留下最新的那个等级。

**关于 Spark 要能说出什么？** 转换操作是惰性的，要等某个 action 需要结果时才真正执行。join、group-by 这类宽依赖的转换要在机器之间 shuffle 数据，开销主要在这里。键分布倾斜时，大部分行会挤进少数几个分区；办法是广播小表、给热点键加盐（salting），或者用自适应执行。只缓存会被重复使用的结果；尽量用内置函数，少用逐行执行的 Python 函数。[Spark 那篇笔记](/zh/notes/spark-pyspark/)里有练习。

## 更多代码 {#more-code}

下载 [next-layer-lab.py](/notes/ml-stats-next-layer/next-layer-lab.py)，用 Python 3.10 或更新版本运行，需要 NumPy 和 pandas。它打印这篇笔记里的每一个数字。装了 scikit-learn 和 DuckDB 的话，它还会交叉核对模型的拟合结果，并运行上面的 SQL。

```bash
python3 next-layer-lab.py
```

### 用并查集求连通分量 {#code-union-find}

```python
def connected_components(edges):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]        # path halving
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    groups = {}
    for node in parent:
        groups.setdefault(find(node), set()).add(node)
    return list(groups.values())
```

路径减半（path halving）让树保持很浅。再加上按大小合并，就得到教科书里的结论：每次操作的开销接近常数。

### 蓄水池抽样 {#code-reservoir}

```python
import random

def reservoir_sample(stream, k, seed=0):
    rng = random.Random(seed)
    sample = []
    for i, item in enumerate(stream):
        if i < k:
            sample.append(item)
        else:
            j = rng.randint(0, i)                # item i is kept with probability k / (i + 1)
            if j < k:
                sample[j] = item
    return sample
```

它只扫一遍、用常数内存，就能从长度未知的数据流里抽出均匀的样本。从 10 个里抽 3 个，跑 20,000 次，每个元素被留下的比例在 29.3% 到 30.4% 之间。

### Average precision {#code-ap}

```python
import numpy as np

def average_precision(y, score):
    order = np.argsort(-np.asarray(score), kind="stable")
    y = np.asarray(y)[order]
    hits = np.cumsum(y)
    precision = hits / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())
```

Average precision 是沿着排序往下走时，在每个正样本处的 precision 的平均值。在没有并列分数的模拟数据上，它和 scikit-learn 的结果一致。

### Kaplan-Meier {#code-km}

```python
def kaplan_meier(durations, events):
    durations, events = np.asarray(durations), np.asarray(events)
    survival, rows = 1.0, []
    for t in np.unique(durations[events == 1]):
        at_risk = int((durations >= t).sum())
        d = int(((durations == t) & (events == 1)).sum())
        survival *= 1 - d / at_risk
        rows.append((int(t), at_risk, d, survival))
    return rows
```

### 证据权重和信息值 {#code-woe}

```python
def woe_iv(good, bad):
    good, bad = np.asarray(good, float), np.asarray(bad, float)
    pg, pb = good / good.sum(), bad / bad.sum()
    woe = np.log(pg / pb)
    return woe, float(((pg - pb) * woe).sum())
```

如果某一箱里没有好样本或没有坏样本，对数会变成无穷大。办法是和相邻的箱合并，或者给两边都加一个很小的计数。

### Off-policy 价值估计 {#code-ips}

```python
def ips_estimates(logs, policy):
    """logs: (context, action, propensity of that action under the logging policy, reward)."""
    weights = np.array([(policy(c) == a) / p for c, a, p, _ in logs])
    rewards = np.array([r for *_, r in logs])
    return float((weights * rewards).mean()), float((weights * rewards).sum() / weights.sum())
```

### 牛顿法和分裂增益 {#code-newton}

```python
def fit_logistic_newton(X, y, tol=1e-10, max_iter=50):
    """X must include a column of ones."""
    w = np.zeros(X.shape[1])
    for step in range(1, max_iter + 1):
        p = 1 / (1 + np.exp(-X @ w))
        gradient = X.T @ (p - y)
        hessian = X.T @ (X * (p * (1 - p))[:, None])
        delta = np.linalg.solve(hessian, gradient)
        w -= delta
        if np.abs(delta).max() < tol:
            break
    return w, step

def split_gain(G_L, H_L, G_R, H_R, lam=1.0, gamma=0.0):
    def score(G, H):
        return G * G / (H + lam)
    return 0.5 * (score(G_L, H_L) + score(G_R, H_R) - score(G_L + G_R, H_L + H_R)) - gamma
```

### Personalized PageRank {#code-ppr}

```python
def personalized_pagerank(edges, seeds, restart=0.15, n_iter=100):
    nodes = sorted({n for e in edges for n in e})
    index = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for a, b in edges:
        A[index[a], index[b]] = A[index[b], index[a]] = 1
    P = A / A.sum(axis=0, keepdims=True)         # column-stochastic transition matrix
    s = np.zeros(len(nodes))
    s[[index[n] for n in seeds]] = 1 / len(seeds)
    r = s.copy()
    for _ in range(n_iter):
        r = (1 - restart) * P @ r + restart * s
    return dict(zip(nodes, r))
```

这个稠密矩阵的版本只适合小图。规模大了之后矩阵是稀疏的，游走改用“从种子节点向外推送概率”的办法来近似。

## 参考资料 {#references}

1. Chen T, Guestrin C. [XGBoost: a scalable tree boosting system](https://arxiv.org/abs/1603.02754). KDD 2016.
2. Hastie T, Tibshirani R, Friedman J. The Elements of Statistical Learning, 2nd edition. Springer, 2009.
3. Sutton RS, Barto AG. Reinforcement Learning: An Introduction, 2nd edition. MIT Press, 2018.
4. Siddiqi N. Credit Risk Scorecards. Wiley, 2006. 信息值的通行读法出自这里。

这篇笔记里的每个数字都由 [`next-layer-lab.py`](/notes/ml-stats-next-layer/next-layer-lab.py) 打印，运行环境是 NumPy 2.5、pandas 3.0、scikit-learn 1.9 和 DuckDB 2.0。
