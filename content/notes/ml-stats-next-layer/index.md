---
title: "Machine learning and statistics, the next layer: derivations, scorecards, graphs and survival"
date: 2026-11-25T05:35:00-06:00
description: "A second pass after the fundamentals: derivations to do on paper, risk-modeling staples such as weight of evidence and scorecard scaling, more statistics, probability puzzles, graphs, survival analysis, reinforcement learning one level deeper, SQL patterns, and code that runs."
topics: [data-science]
---
The [fundamentals note](/notes/ml-stats-fundamentals/) covers what a panel asks first. This one covers what it asks next: "derive it", "what would a risk team do here", and the topics a fraud group works with every day, such as graphs and labels that arrive late. I built the list by checking my earlier notes against a list of topics and writing up what was missing.

> **Takeaway.** A follow-up question usually goes one of three ways: down into the mathematics, sideways into how a risk team actually does it, or forward into what breaks in production. Each section here takes one of those directions.

## How to use this note {#how-to}

Do the [derivations](#derivations) with pen and paper, not by reading. Say the rest aloud. Every number is printed by the [lab script](#more-code), and the SQL was run in DuckDB.

For a panel on fraud modeling, I would do [risk-modeling staples](#staples), [graphs](#graphs) and [survival analysis](#survival) first.

## Derivations {#derivations}

**Derive the bias-variance decomposition.** Let $y = f(x) + \varepsilon$ with noise variance $\sigma^2$, and let $\hat{f}$ be a model trained on a random sample. Add and subtract $E[\hat{f}]$ inside the square:

$$
\begin{aligned}
E\bigl[(y - \hat{f})^2\bigr] &= \bigl(f - E[\hat{f}]\bigr)^2 \\
&\quad + E\bigl[(\hat{f} - E[\hat{f}])^2\bigr] + \sigma^2
\end{aligned}
$$

The three terms are bias squared, variance and noise. The cross terms vanish because the noise is independent of the training sample and $\hat{f} - E[\hat{f}]$ has mean zero.

**Derive the gradient and the Hessian of logistic regression. Why is the loss convex?** With $p = \sigma(Xw)$ and log loss $L = -\sum_i \bigl[y_i \log p_i + (1 - y_i)\log(1 - p_i)\bigr]$, and using $\sigma' = \sigma(1 - \sigma)$:

$$
\begin{aligned}
\nabla L &= X^\top (p - y) \\
\nabla^2 L &= X^\top W X, \quad W = \mathrm{diag}\bigl(p_i(1 - p_i)\bigr)
\end{aligned}
$$

For any vector $v$, $v^\top X^\top W X v = \sum_i p_i(1 - p_i)(x_i^\top v)^2 \ge 0$. The Hessian is positive semidefinite, so the loss is convex and any local minimum is the global one.

**What is Newton's method here, and why is it called iteratively reweighted least squares?** The update is $w \leftarrow w - (X^\top W X)^{-1} X^\top (p - y)$. Each step solves a weighted least-squares problem, and the weights $p(1-p)$ change at every iteration. In the lab it converges in 7 iterations, against thousands of gradient steps. Each iteration solves a system the size of the feature count, so statistics packages use it for small models and gradient methods take over for large ones.

**Give the closed form of ridge regression and say what it does.** Minimizing $\lVert y - Xw \rVert^2 + \lambda \lVert w \rVert^2$ gives $\hat{w} = (X^\top X + \lambda I)^{-1} X^\top y$. Adding $\lambda I$ makes the matrix invertible even when features are collinear. Along each singular direction of $X$, with singular value $d$, the coefficient is multiplied by $d^2/(d^2 + \lambda)$. With $\lambda = 1$, a direction with singular value 10 keeps 99% of its coefficient and a direction with singular value 1 keeps 50%. Ridge shrinks most where the data says least.

**Derive the XGBoost split gain.** Expand the loss to second order around the current prediction, so each row contributes a gradient $g_i$ and a Hessian $h_i$. For a leaf with sums $G$ and $H$ and a constant weight $w$, the objective is $Gw + \tfrac{1}{2}(H + \lambda)w^2$. It is minimized at $w^* = -G/(H + \lambda)$, where it equals $-\tfrac{1}{2}\,G^2/(H + \lambda)$. Splitting a leaf therefore gains

$$
\begin{aligned}
\text{Gain} = \frac{1}{2}\Bigl[&\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} \\
&- \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\Bigr] - \gamma
\end{aligned}
$$

For log loss, $g = p - y$ and $h = p(1 - p)$. Take six rows that all sit at $p = 0.5$: the left child has labels 1, 1, 0 and the right child 0, 0, 0. Then $G_L = -0.5$, $G_R = 1.5$ and $H_L = H_R = 0.75$. With $\lambda = 1$ the leaf weights are +0.286 and −0.857 and the gain is 0.514. In the library's terms, `lambda` shrinks leaf weights, `gamma` is the gain a split must exceed, and `min_child_weight` is a floor on $H$.

**Explain expectation-maximization for a Gaussian mixture.** The E-step computes, for every point, how responsible each component is for it: $r_{ik} \propto \pi_k\,\mathcal{N}(x_i \mid \mu_k, \sigma_k^2)$. The M-step re-estimates each component's weight, mean and variance as averages weighted by those responsibilities. An iteration cannot lower the likelihood, because the M-step maximizes a lower bound that touches the likelihood at the current parameters. It converges to a local optimum. K-means is the special case with hard assignments and equal, fixed variances. On a mixture with true means 0 and 5, the lab recovers 0.01 and 4.93.

**Why is the gradient of softmax with cross-entropy simply p − y?** With $p = \mathrm{softmax}(z)$ and $L = -\sum_k y_k \log p_k$, differentiating gives $\partial L / \partial z_j = p_j - y_j$: the logarithm undoes the exponential, and the derivative of the normalizer is $p$. Backpropagation through a linear layer $z = Wx + b$ then gives $\partial L/\partial W = (p - y)\,x^\top$ and $\partial L/\partial x = W^\top (p - y)$.

**How do you compute a softmax safely?** Subtract the largest logit before exponentiating, which changes nothing mathematically. Logits of 1000, 1001 and 1002 overflow in the naive version and return NaN; the stable version returns 0.090, 0.245 and 0.665. For log-probabilities, use log-sum-exp for the same reason.

**Write the SVM problem, its hinge-loss form, and say why kernels work.** The primal problem minimizes $\tfrac{1}{2}\lVert w \rVert^2 + C\sum_i \xi_i$ subject to $y_i(w^\top x_i + b) \ge 1 - \xi_i$ and $\xi_i \ge 0$. That is the same as minimizing the hinge loss $\max(0,\, 1 - y f(x))$ plus an L2 penalty. The dual problem uses the data only through dot products $x_i^\top x_j$, so a kernel $k(x_i, x_j)$ can replace them. The solution is $f(x) = \sum_i \alpha_i y_i\, k(x_i, x) + b$, with $\alpha_i$ nonzero only for the support vectors.

**How is PCA related to the singular value decomposition?** Center the data and write $X = U S V^\top$. The columns of $V$ are the principal directions, $S^2/(n - 1)$ are their variances, and the projected data is $XV = US$. Using the decomposition avoids forming the covariance matrix and is numerically more stable. In the lab's example the first two components explain 91.8% and 7.8% of the variance.

## Risk-modeling staples {#staples}

**What are weight of evidence and information value?** For one bin of a feature, weight of evidence (WoE) is the log of the share of all good cases that fall in the bin over the share of all bad cases that do. Positive means safer than average. Information value (IV) sums, over the bins, the difference between the two shares times the WoE. For transactions in the last hour:

| Bin | Share of goods | Share of bads | WoE |
|---|---:|---:|---:|
| 0 to 1 | 60% | 45% | +0.288 |
| 2 to 3 | 30% | 30% | 0.000 |
| 4 to 6 | 8% | 17% | −0.754 |
| 7 or more | 2% | 8% | −1.386 |

The IV is 0.194. The rule of thumb from credit scoring reads below 0.02 as useless, 0.02 to 0.1 as weak, 0.1 to 0.3 as medium, above 0.3 as strong, and above 0.5 as a reason to look for leakage. IV has the same formula as PSI, applied to goods against bads. WoE encoding also lets a logistic regression handle nonlinear effects and missing values, which get their own bin. Bins and WoE values are learned on the training window only.

**How is a scorecard scaled?** Score = offset + factor × ln(odds of good), with factor = PDO / ln 2, where PDO is the number of points that doubles the odds. Anchoring 600 points at odds of 50 to 1 with a PDO of 20 gives a factor of 28.85 and an offset of 487.1. Odds of 100 to 1 then score 620 and odds of 25 to 1 score 580. For a model with $n$ features, a bin's points are (WoE × coefficient + intercept / $n$) × factor + offset / $n$.

**Which reject inference methods are there, and which do you trust?** Outcomes exist only for approved applicants. Reweighting counts each approved case by one over its probability of approval, which assumes rejected applicants resemble some approved ones. Parceling scores the rejects with the approved-only model and assigns outcomes at an inflated bad rate. Fuzzy augmentation adds each reject twice, as good and as bad, weighted by predicted probability. External data shows how rejects performed with other lenders. A small randomized set of approvals is the only method that rests on no untestable assumption. Whichever I use, I state the assumption.

**You trained a logistic regression on down-sampled negatives. What happens to the coefficients?** The slopes are unchanged in expectation. Only the intercept moves, by the log of the sampling ratio: keeping 1% of the negatives raises it by 4.605. Adding $\ln \beta$ back corrects it, which is the [odds correction](/notes/fraud-detection-practice/#drill-downsampling) in another form.

**What is perfect separation?** Some feature, or combination of features, splits the classes without error. The likelihood then keeps rising as the coefficient grows, so the software reports huge coefficients and standard errors, or fails to converge. In fraud data the usual cause is leakage. Otherwise a penalty fixes it.

**AIC or BIC?** $\mathrm{AIC} = 2k - 2\ln L$ and $\mathrm{BIC} = k \ln n - 2\ln L$, with $k$ parameters; lower is better. BIC penalizes parameters more and picks smaller models. AIC aims at prediction and BIC at finding the true model. For a predictive model I rely on out-of-time validation and use these only to compare nested statistical models quickly.

**What is a swap-set analysis?** Compare two models at the same alert or approval volume and split the cases into three groups: flagged by both, flagged by the new model only (swap-ins), and flagged by the old one only (swap-outs). With 1,000 alerts each and 700 in common, suppose the 300 swap-ins are 10% fraud and the 300 swap-outs 3%. The new model catches 21 more frauds at the same volume. It also shows exactly which customers are treated differently, which operations and compliance will ask about.

**Your test set was sampled. How do you state precision for production?** Recall and the false positive rate do not depend on prevalence $\pi$, and precision does:

$$
\text{precision} = \frac{\mathrm{TPR}\cdot\pi}{\mathrm{TPR}\cdot\pi + \mathrm{FPR}\cdot(1 - \pi)}
$$

At a recall of 60% and a false positive rate of 0.5%, precision is 54.8% when 1% of cases are fraud and 10.7% when 0.1% are.

**What does a model validator ask?** Whether the design is conceptually sound: why this approach, which assumptions, which data. How the outcomes hold up: performance on later periods, calibration, results by segment. How it compares with a simpler benchmark or a challenger. How sensitive it is to shifted inputs and missing data. What the monitoring plan is, with thresholds and actions. And which limitations are known and what compensates for them. The practice set notes [the current guidance](/notes/fraud-detection-practice/#explainability).

**What is expected calibration error?** Bin the predictions and average the gap between predicted and observed rates, weighted by bin size. Three bins predicting 0.1, 0.5 and 0.9 with observed rates of 0.08, 0.6 and 0.7 and sizes 800, 150 and 50 give 0.041. The value depends on the binning, so I show the reliability plot with it.

**Fairness metrics in one answer?** Demographic parity asks for equal approval rates across groups. Equalized odds asks for equal true and false positive rates. Predictive parity asks for equal precision. When base rates differ, the three cannot all hold. A common screening heuristic, the four-fifths rule, flags an approval-rate ratio below 0.8: 45% against 60% is 0.75. Protected attributes are not model inputs, but other features can stand in for them, so outcomes are tested by group.

## More statistics {#more-statistics}

**Which conjugate pairs should you know?**

| Prior | Data | Posterior |
|---|---|---|
| Beta($a$, $b$) | $k$ successes in $n$ trials | Beta($a + k$, $b + n - k$) |
| Gamma(shape $a$, rate $b$) | Poisson counts summing to $s$ over $n$ periods | Gamma($a + s$, $b + n$) |
| Normal | Normal with known variance | Normal, with a precision-weighted mean |

A Gamma(2, 1) prior and 30 events in 10 days give Gamma(32, 11), a mean of 2.91 per day. A conjugate prior acts like earlier observations: Beta($a$, $b$) is worth $a$ successes and $b$ failures.

**State the laws of total expectation and total variance.** $E[X] = E\bigl[E[X \mid G]\bigr]$ and $\mathrm{Var}(X) = E\bigl[\mathrm{Var}(X \mid G)\bigr] + \mathrm{Var}\bigl(E[X \mid G]\bigr)$: variance within groups plus variance between them. If 90% of accounts spend \$50 on average with a standard deviation of \$20, and 10% spend \$300 with a standard deviation of \$100, the overall mean is \$75, the within part is 1,360 and the between part is 5,625, for a standard deviation of \$83.6. Most of the variance sits between the segments, which is why segmenting or stratifying removes so much of it.

**Markov's and Chebyshev's inequalities?** Markov: for a nonnegative variable, $P(X \ge a) \le E[X]/a$. Chebyshev: $P(\lvert X - \mu \rvert \ge k\sigma) \le 1/k^2$. Neither assumes a distribution. At three standard deviations Chebyshev allows 11.1%, where a normal gives 0.27%. A three-sigma threshold is tight only if the data is close to normal.

**What is the delta method?** A standard error for a function of an estimate: $\mathrm{Var}\bigl(g(\hat\theta)\bigr) \approx g'(\theta)^2\,\mathrm{Var}(\hat\theta)$. For a ratio I work on the log scale. A lift of 12% over 10%, with 5,000 cases each, has a standard error of 0.057 on the log scale, so the 95% interval for the lift runs from 1.07 to 1.34.

**Fisher information and the Cramér-Rao bound?** Fisher information is the expected curvature of the log-likelihood: the sharper the peak, the more the data says about the parameter. The Cramér-Rao bound says no unbiased estimator has variance below its inverse. Maximum likelihood reaches the bound in large samples, which is why standard errors come from the inverse Hessian.

**Wald, likelihood ratio and score tests?** Three tests built on the likelihood. Wald divides the estimate by its standard error. The likelihood ratio test compares the log-likelihoods of nested models; twice the difference is chi-square with as many degrees of freedom as extra parameters. The score test uses the slope of the log-likelihood at the null value. They agree in large samples, and the likelihood ratio test is the most reliable in small ones.

**What does ANOVA test?** Whether three or more group means are equal, by comparing the variance between group means with the variance within groups. A significant result says the means are not all equal, not which ones differ, so pairwise comparisons follow with a multiple-testing correction. With two groups it is the t-test squared.

**What is McNemar's test?** A test for two classifiers scored on the same cases. Only the cases where they disagree count: $b$ where the first is right and the second wrong, $c$ the reverse. The statistic is $(b - c)^2/(b + c)$, chi-square with one degree of freedom. With 40 and 20 it is 6.67, a p-value of 0.010; with the continuity correction, 6.02 and 0.014. For small counts I use an exact binomial test on $b$ out of $b + c$.

**What is CUPED?** A way to cut variance in an experiment by removing what a pre-experiment covariate predicts: $Y' = Y - \theta\,(X - \bar{X})$ with $\theta = \mathrm{Cov}(X, Y)/\mathrm{Var}(X)$. The estimate stays unbiased because the treatment cannot affect $X$. Variance falls by the squared correlation: at a correlation of 0.6 it falls by 36%, so the same power needs 64% of the sample. The [A/B testing note](/notes/ab-testing/) has the wider design.

**What is a generalized linear model?** A distribution from the exponential family, a linear predictor, and a link between them. Linear regression uses a normal distribution with the identity link, logistic regression a binomial with the logit, Poisson regression counts with the log. For rates I add an offset, the log of the exposure, so the model describes events per unit of volume. If the variance exceeds the mean, a negative binomial model replaces Poisson, or the standard errors come out too small.

**Which regression diagnostics do you look at?** Residuals against fitted values: a curve means a missing nonlinear term and a funnel means changing variance. A quantile plot of the residuals for the tails. Leverage for unusual feature values, and Cook's distance for points that are both unusual and badly fitted. Variance inflation factors for collinearity.

**What is quantile regression?** It models a conditional quantile by minimizing the pinball loss, which charges $\tau$ per unit of under-prediction and $1 - \tau$ per unit of over-prediction. I use it when a tail matters, such as the 95th percentile of latency or loss, or when the noise is skewed. Boosted trees accept the same loss.

**What should a modeler know about time series?** Stationarity: a constant mean, variance and autocorrelation, reached by differencing or removing the trend. Seasonality and autocorrelation. A seasonal naive forecast as the baseline. ARIMA as autoregression plus differencing plus a moving average of past errors. And validation with a rolling origin, never a random split. Autocorrelation also breaks the formulas for independent data: with a lag-one coefficient of 0.5, the variance of a mean is three times the usual formula, so 300 hourly points carry the information of 100.

**What is regression to the mean?** An extreme observation is partly noise, so the next one tends to sit closer to the average. In the lab, the accounts in the top 10% of a noisy risk measure average 2.48 the first time and 1.25 the second, with no intervention. "We acted on the worst accounts and they improved" therefore needs a control group.

**What is Goodhart's law?** A measure that becomes a target stops measuring. A team rewarded on alert precision can raise it by alerting only on obvious cases, and recall collapses. I pair each metric with the one it can be traded against: precision with recall, loss with customer friction.

## Probability puzzles {#puzzles}

| Puzzle | Answer | Why |
|---|---|---|
| Monty Hall: switch or stay? | Switch, 2/3 | The first pick is right 1/3 of the time, and the host's reveal does not change that |
| Two children, at least one is a boy. Both boys? | 1/3 | Boy-boy, boy-girl and girl-boy are equally likely |
| Expected coin flips until the first head | 2 | Geometric: $1/p$ |
| Expected flips until two heads in a row | 6 | Condition on the first flips: $E = 1 + \tfrac{1}{2}E_1 + \tfrac{1}{2}E$ and $E_1 = 1 + \tfrac{1}{2}E$ |
| Draws to collect all 6 types of coupon | 14.7 | $6\,(1 + \tfrac{1}{2} + \dots + \tfrac{1}{6})$ |
| 23 people: a shared birthday? | 50.7% | One minus the chance that all differ |
| Expected maximum of two dice | 4.47 | $\sum_k k\,(2k - 1)/36 = 161/36$ |
| At least one six in four rolls | 51.8% | $1 - (5/6)^4$ |
| Expected fixed points of a random permutation | 1 | Linearity: $n$ positions, each fixed with probability $1/n$ |

Four moves solve most of these: write out the sample space, take the complement, use linearity of expectation, and condition on the first step. When stuck, say how I would simulate it; the lab does that for two of them.

## Graphs {#graphs}

**Why does a graph help in fraud?** Fraud is organized. Accounts controlled by one group share devices, addresses, phone numbers and payees, and a new account connected to confirmed fraud is riskier than its own history shows. A table of independent rows cannot express that.

**How do you build the graph?** Usually as a bipartite graph: accounts or cards on one side, and devices, merchants, addresses or payees on the other, with an edge when they appear together. Projecting it onto accounts, by linking two accounts that share a device, is simpler to use and loses which entity they shared. Every edge carries a timestamp.

**Which graph features go into a tree model?** Degree, the number of accounts sharing each of my entities, the fraud rate among neighbors at one and two hops, the size of the connected component, the distance to the nearest confirmed fraud, and a personalized PageRank score. All of them are computed as of the transaction time, with neighbor labels counted only if already confirmed.

**How do you find groups of linked accounts?** Connected components, with union-find: each edge merges two sets, in nearly constant time per edge. On six accounts and three devices, the lab finds one component of three accounts and one of two, and leaves the sixth account alone. Components are a starting point; one shared airport Wi-Fi address can merge thousands of unrelated accounts.

**What do you do about hubs?** A node with a huge degree, such as a public IP address or a large merchant, connects everything and carries no signal. I cap the degree, drop or down-weight such entities, and sample neighbors. I also keep the degree as a feature, because "shares a device with 3 accounts" and "with 3,000" mean different things.

**What is personalized PageRank?** The long-run distribution of a random walk that, at each step, returns to a set of seed nodes with some probability. Seeding it with confirmed fraud gives every node a proximity score that counts all the paths, not just the shortest. On the lab's small graph, seeded at account a1, the account sharing a device with a1 scores 0.195, the one two devices away scores 0.055, and unconnected accounts score zero.

**What is label propagation?** Each node repeatedly takes the weighted average of its neighbors' labels, while known labels stay fixed. It is a quick semi-supervised baseline that needs no features and assumes that connected nodes share a label.

**Explain a graph neural network.** Each layer updates a node's vector from its own vector and an aggregate of its neighbors' vectors, so $k$ layers see $k$ hops. A graph convolutional network uses a normalized sum. GraphSAGE samples a fixed number of neighbors and learns the aggregator, so it scales and works on nodes it never saw in training. A graph attention network learns a weight for each neighbor. Stack too many layers and the vectors of different nodes become alike, which is called over-smoothing.

**Transductive or inductive?** A transductive model learns an embedding per node and cannot score a node that was absent from training. An inductive model learns a function of features and neighborhood, so it handles new accounts, which fraud needs.

**How do you split graph data for validation?** By time. The training graph holds only the edges and labels known at the cutoff, and the model is tested on later transactions. A random split of nodes leaks through shared neighbors, and using a neighbor's label that was confirmed after the event leaks the future.

**What is community detection?** Finding groups that are denser inside than between. The Louvain method greedily maximizes modularity, the share of edges inside groups minus the share expected at random. In fraud it proposes candidate rings for analysts; it does not decide anything.

## Survival analysis {#survival}

**Why treat labels as a time-to-event problem?** A fixed-window label ("fraud within 90 days") must wait 90 days and throws away newer data. A time-to-event model uses every case for as long as it has been observed.

**Survival function and hazard?** $S(t)$ is the probability that the event has not happened by time $t$. The hazard $h(t)$ is the rate of the event at $t$ among those still at risk. They carry the same information: $S(t) = \exp\bigl(-\int_0^t h(u)\,du\bigr)$.

**How does the Kaplan-Meier estimator work?** At each event time, multiply by the share of those still at risk that survived it: $\hat{S}(t) = \prod_{t_i \le t}(1 - d_i/n_i)$. A censored case leaves the at-risk count without counting as an event.

| Day | At risk | Events | Survival |
|---:|---:|---:|---:|
| 5 | 10 | 1 | 0.900 |
| 12 | 8 | 2 | 0.675 |
| 25 | 5 | 1 | 0.540 |
| 40 | 2 | 1 | 0.270 |

Of these ten accounts, five had the event, so the naive rate is 50%. Kaplan-Meier puts the probability of an event by day 40 at 73%, because four accounts left observation before day 40 without an event. Ignoring censoring understates the risk.

**How do you compare two survival curves?** With the log-rank test, which at each event time compares the events in each group with what a common hazard would predict.

**What is the Cox model?** $h(t \mid x) = h_0(t)\,\exp(\beta^\top x)$. The baseline hazard $h_0$ is left unspecified, and the coefficients are fitted by a partial likelihood that uses only the order of the events. $e^{\beta}$ is a hazard ratio. The model assumes the ratio is constant over time, which should be checked.

**How do you use boosted trees for survival?** Recast it as discrete time. Each case becomes one row per period it was at risk, with a label saying whether the event happened in that period. A classifier on those rows estimates the hazard, features can change from period to period, and censoring is handled by simply stopping the rows.

**How do you correct recent fraud counts for reporting delay?** Estimate the share of fraud reported within each age from mature cohorts, then divide what has been seen by that share. If 50% is reported by day 7, 90% by day 30 and 99% by day 90, then cohorts showing 10, 27 and 40 frauds at those ages should end near 20, 30 and 40. The same shares can weight young, still-unlabeled cases in training.

**What are competing risks?** Another event can remove a case before the event of interest: an account closes before any fraud is found. Treating that as ordinary censoring overstates the fraud probability, so the competing event is modeled as well.

**What is the concordance index?** The share of comparable pairs in which the case with the higher predicted risk had the event earlier. It is AUC adapted to censored data.

## Reinforcement learning, one level down {#rl-deeper}

**Write the Bellman equations.** For a policy $\pi$, $V^{\pi}(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a)\,\bigl[r + \gamma V^{\pi}(s')\bigr]$. For the best policy, $Q^*(s, a) = E\bigl[r + \gamma \max_{a'} Q^*(s', a')\bigr]$. A value is the immediate reward plus the discounted value of what follows. Rewards of 1, 0 and 2 at $\gamma = 0.9$ are worth 2.62.

**Monte Carlo or temporal-difference learning?** Monte Carlo waits for the end of an episode and uses the actual return: unbiased and noisy. Temporal-difference learning updates toward the reward plus the current estimate of the next state: $V(s) \leftarrow V(s) + \alpha\,[r + \gamma V(s') - V(s)]$. It is biased by its own estimate, has less variance, and learns before the episode ends.

**Write the Q-learning update.** $Q(s, a) \leftarrow Q(s, a) + \alpha\,[r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$. With a current value of 2, a reward of 1, a best next value of 3, $\gamma = 0.9$ and $\alpha = 0.5$, the new value is 2.85. It is off-policy because the target uses the best next action whatever action is taken next. Deep Q-networks add a replay buffer and a slowly updated target network. Taking a maximum over noisy estimates overestimates, which double Q-learning corrects.

**State the policy gradient.** $\nabla J(\theta) = E\bigl[\nabla_\theta \log \pi_\theta(a \mid s)\; Q^{\pi}(s, a)\bigr]$: raise the probability of actions in proportion to how well they turned out. Subtracting a baseline that depends only on the state leaves the expectation unchanged and lowers the variance. With the state value as baseline, the weight becomes the advantage, $Q - V$.

**What does PPO optimize?** $E\bigl[\min\bigl(r_t A_t,\ \mathrm{clip}(r_t,\, 1 - \epsilon,\, 1 + \epsilon)\,A_t\bigr)\bigr]$, where $r_t$ is the ratio of the new policy's probability to the old one's and $A_t$ is the advantage. The clip removes any incentive to move the ratio far from one, so a batch can be reused for several epochs.

**How do you estimate a new policy's value from logs?** Inverse propensity scoring averages the logged rewards where the new policy agrees with the logged action, each weighted by one over the logging probability. It is unbiased and noisy. The self-normalized version divides by the sum of the weights, accepting a small bias for much less variance. A doubly robust estimator adds a reward model and stays consistent if either the model or the propensities are right. In the lab's eight logged decisions the true value of the new policy is 0.75; plain scoring says 1.48 and the self-normalized version 0.86.

**Which contextual bandit algorithms would you name?** LinUCB fits a linear reward model per action and adds a bonus that shrinks as that action is tried in similar contexts. Thompson sampling keeps a posterior over the model, draws from it, and acts on the draw. Both need the probability of the chosen action logged, for later evaluation.

## Deep learning mechanics {#dl-mechanics}

**Batch normalization or layer normalization?** Batch normalization standardizes each feature across the batch. It depends on batch statistics, so it behaves differently in training and inference and degrades with small batches. Layer normalization standardizes across the features of one example and does not depend on the batch, which is why Transformers and recurrent networks use it.

**Write the Adam update.** $m \leftarrow \beta_1 m + (1 - \beta_1)g$ and $v \leftarrow \beta_2 v + (1 - \beta_2)g^2$. Both are divided by $1 - \beta^t$ to correct their start at zero, and the step is $\eta\,\hat{m}/(\sqrt{\hat{v}} + \epsilon)$. Because of that correction, the first step has the size of the learning rate whatever the gradient's scale: a gradient of 250 at a learning rate of 0.001 moves the weight by 0.001. AdamW applies weight decay to the weights directly, outside this rescaling.

**Why do residual connections help?** A block computes $x + F(x)$, so the gradient has a direct path back through the identity. Each block only has to learn a correction, which makes very deep networks trainable.

**What does attention cost, and why are positions encoded?** Self-attention compares every pair of positions, so time and memory grow with the square of the sequence length. It is also indifferent to order, so position information is added to the inputs.

**What is LoRA?** A frozen weight matrix $W$ gets a trainable low-rank update $BA$, with $B$ of size $d \times r$ and $A$ of size $r \times k$. For a 4,096 by 4,096 matrix at rank 16, that is 131,072 trainable parameters out of 16.8 million, or 0.78%. The update can be merged into $W$ afterwards, so inference costs nothing extra.

**What is contrastive learning?** Training an encoder so that two views of the same item are close and views of different items are far apart, usually with a softmax over similarities and a temperature. It yields useful embeddings without labels.

**What is knowledge distillation?** A small student model is trained to match a large teacher's softened output probabilities, which carry more information than hard labels. It trades a little accuracy for latency, which matters in a real-time decision.

## Anomalies and explanations {#anomaly-interp}

**Why Mahalanobis distance and not Euclidean?** It measures distance in units of the data's own covariance. With two features correlated at 0.9, the points (2, 2) and (2, −2) are equally far from the mean in Euclidean terms, 2.83. Their Mahalanobis distances are 2.05 and 8.94, because the second point goes against the correlation. The covariance should be estimated robustly, or the outliers hide themselves.

**Local outlier factor or one-class SVM?** The local outlier factor compares a point's density with that of its neighbors, so it finds points that are odd for their region. A one-class SVM learns a boundary around the normal data and depends heavily on the kernel and its settings. Both slow down on large data, where an [isolation forest](/notes/ml-stats-fundamentals/#algorithms) scales better.

**Where does extreme value theory fit?** In setting thresholds far in the tail. Values above a high threshold follow a generalized Pareto distribution approximately, so I can estimate the probability of levels rarely or never seen, which empirical quantiles cannot do.

**Partial dependence, ICE or ALE?** Partial dependence shows the average prediction as one feature varies, and misleads when features are correlated, because it averages over combinations that never occur. Individual conditional expectation draws one curve per case, which reveals interactions the average hides. Accumulated local effects use only realistic neighborhoods of the feature, so they cope with correlation.

**Which properties make SHAP values usable for reason codes?** They add up: the contributions sum to the prediction minus the average prediction. And they are consistent: if a model relies more on a feature, its attribution does not fall. For trees they are computed exactly and quickly. They describe the model, not the world.

**How does a monotonic constraint work in boosted trees?** A split on a constrained feature is rejected if the child values would be in the wrong order, and bounds are passed down so deeper splits cannot break the order either. It costs a little accuracy, and buys a model whose behavior can be explained and defended, such as risk never falling as velocity rises.

**What is adversarial validation?** Train a classifier to tell training rows from test rows. An AUC near 0.5 means the two look alike. A high AUC means the data shifted, and the classifier's top features say where.

## SQL and data patterns {#sql}

These four patterns cover most of what gets asked beyond a plain join. The queries are DuckDB syntax and run in the lab on nine transactions.

**Top N per group.** The two largest transactions of each account:

```sql
SELECT account_id, ts, amount
FROM (
  SELECT t.*, ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY amount DESC, ts) AS rn
  FROM transactions AS t
) AS x
WHERE rn <= 2
ORDER BY account_id, amount DESC
```

`ROW_NUMBER` breaks ties arbitrarily unless the ordering is complete; `RANK` keeps ties and can return more than N rows.

**Gaps and islands.** Streaks of consecutive active days:

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

Within a streak, the date minus the row number is constant, so it works as a group key. Account A, active on December 1, 2, 3 and 5, gets a streak of three days and a streak of one.

**Sessions.** A new session starts after 30 minutes without activity:

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

Flag the rows that start a session, then take a running sum of the flags.

**Point-in-time join.** Attach to each transaction the risk tier that was in force at that moment:

```sql
SELECT t.account_id, t.ts, t.amount, r.risk_tier
FROM transactions AS t
ASOF LEFT JOIN risk_tiers AS r
  ON t.account_id = r.account_id AND t.ts >= r.valid_from
ORDER BY t.account_id, t.ts
```

In pandas it is `pd.merge_asof(transactions, tiers, left_on="ts", right_on="valid_from", by="account_id")`, with both frames sorted by time. An ordinary join on the account alone would attach the latest tier to old transactions, which is leakage. Engines without `ASOF` join on the account with `valid_from <= ts` and keep the latest tier for each transaction with `ROW_NUMBER`.

**What should you be able to say about Spark?** Transformations are lazy, and nothing runs until an action asks for a result. A wide transformation such as a join or a group-by shuffles data between machines and dominates the cost. Skewed keys put most rows in a few partitions; the fixes are broadcasting the small table, salting the hot keys, or adaptive execution. Cache only what is reused, and prefer built-in functions to Python functions that run row by row. The [Spark note](/notes/spark-pyspark/) has a lab.

## More code {#more-code}

Download [next-layer-lab.py](/notes/ml-stats-next-layer/next-layer-lab.py) and run it with Python 3.10 or later, NumPy and pandas. It prints every number in this note. With scikit-learn and DuckDB installed it also cross-checks the model fits and runs the SQL.

```bash
python3 next-layer-lab.py
```

### Connected components with union-find {#code-union-find}

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

Path halving keeps the trees shallow. Adding union by size gives the textbook bound, which is nearly constant time per operation.

### Reservoir sampling {#code-reservoir}

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

It draws a uniform sample from a stream of unknown length in one pass and constant memory. Over 20,000 runs of 3 out of 10, each item is kept between 29.3% and 30.4% of the time.

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

Average precision is the mean of the precision at each positive, going down the ranking. It matches scikit-learn on simulated data without tied scores.

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

### Weight of evidence and information value {#code-woe}

```python
def woe_iv(good, bad):
    good, bad = np.asarray(good, float), np.asarray(bad, float)
    pg, pb = good / good.sum(), bad / bad.sum()
    woe = np.log(pg / pb)
    return woe, float(((pg - pb) * woe).sum())
```

A bin with no goods or no bads makes the logarithm infinite. Merge it with a neighbor or add a small count to both.

### Off-policy value estimates {#code-ips}

```python
def ips_estimates(logs, policy):
    """logs: (context, action, propensity of that action under the logging policy, reward)."""
    weights = np.array([(policy(c) == a) / p for c, a, p, _ in logs])
    rewards = np.array([r for *_, r in logs])
    return float((weights * rewards).mean()), float((weights * rewards).sum() / weights.sum())
```

### Newton's method and the split gain {#code-newton}

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

This dense version is for small graphs. At scale the matrix is sparse, and the walk is approximated by pushing probability outward from the seeds.

## References {#references}

1. Chen T, Guestrin C. [XGBoost: a scalable tree boosting system](https://arxiv.org/abs/1603.02754). KDD 2016.
2. Hastie T, Tibshirani R, Friedman J. The Elements of Statistical Learning, 2nd edition. Springer, 2009.
3. Sutton RS, Barto AG. Reinforcement Learning: An Introduction, 2nd edition. MIT Press, 2018.
4. Siddiqi N. Credit Risk Scorecards. Wiley, 2006. The source of the usual reading of information value.

Every number in this note is printed by [`next-layer-lab.py`](/notes/ml-stats-next-layer/next-layer-lab.py). It was run with NumPy 2.5, pandas 3.0, scikit-learn 1.9 and DuckDB 2.0.
