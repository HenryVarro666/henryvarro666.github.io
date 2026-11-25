---
title: "Machine learning and statistics fundamentals: interview questions with short answers"
date: 2026-11-25T05:00:00-06:00
description: "Short answers to about 100 questions on probability, estimation, hypothesis testing, regression, sampling bias, model evaluation, classic algorithms, neural networks and reinforcement learning, with worked numbers and from-scratch code that runs."
topics: [data-science]
---
This is the wider net behind my [fraud modeling practice set](/notes/fraud-detection-practice/). That note goes deep on one domain. This one covers the fundamentals a modeling panel can reach for from any direction: probability, statistics, evaluation, the classic algorithms, neural networks and reinforcement learning. Each answer is short enough to say in under a minute.

> **Takeaway.** A fundamentals question usually tests three things: whether I can state the definition exactly, whether I know when it breaks, and whether I can tie it to a decision. The answers here are written in that order.

## How to use this note {#how-to}

Cover the answer, say it aloud, then compare. Where an answer quotes a number, the [lab script](#scratch) prints it. Where a topic has a longer treatment on this site, the answer links to it instead of repeating it.

If time is short, do [hypothesis testing](#testing), [model evaluation](#evaluation) and [classic algorithms](#algorithms) first. They produce the most follow-up questions.

A second note, [the next layer](/notes/ml-stats-next-layer/), continues with derivations, scorecards, graphs, survival analysis and SQL patterns.

## Probability {#probability}

**State Bayes' rule. When does the prior matter most?** $P(A \mid B) = P(B \mid A)\,P(A) / P(B)$. The prior matters most when the event is rare. At a 0.1% base rate, a detector with 90% recall and a 1% false positive rate still gives a posterior of only about 8%. The practice set [works that number](/notes/fraud-detection-practice/#drill-bayes).

**Independence or conditional independence?** $A$ and $B$ are independent when $P(A \cap B) = P(A)\,P(B)$. They are conditionally independent given $C$ when that holds within each value of $C$. Neither implies the other. Two symptoms with a common cause are dependent overall and independent given the cause. Two independent causes become dependent once I condition on their shared effect.

**Which rules for expectation and variance do you use most?** Expectation is linear, with or without independence: $E[aX + bY] = aE[X] + bE[Y]$. Variance is not: $\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathrm{Cov}(X, Y)$, and $\mathrm{Var}(aX) = a^2\,\mathrm{Var}(X)$. The covariance term is why averaging correlated models helps less than averaging independent ones.

**Which distributions should you recognize on sight?**

| Distribution | What it models | Mean and variance |
|---|---|---|
| Binomial | Frauds among $n$ transactions | $np$ and $np(1-p)$ |
| Poisson | Events in a fixed interval, such as alerts per hour | $\lambda$ and $\lambda$ |
| Geometric | Transactions until the first fraud | $1/p$ and $(1-p)/p^2$ |
| Exponential | Time between events | $1/\lambda$ and $1/\lambda^2$ |
| Normal | Sums and averages of many small effects | $\mu$ and $\sigma^2$ |
| Log-normal | Amounts and other positive, right-skewed values | Its logarithm is normal |

**Law of large numbers or central limit theorem?** The law of large numbers says the sample mean converges to the true mean. The central limit theorem says more: the distribution of the sample mean approaches a normal with standard deviation $\sigma/\sqrt{n}$, whatever the shape of the data, as long as the variance is finite. That is what justifies z-intervals for means and proportions. With a standard deviation of \$80 and 400 transactions, the mean is known to about \$4. Convergence is slow for heavy tails and rare events.

**When is the Poisson approximation useful?** A binomial with large $n$ and small $p$ is close to a Poisson with mean $np$. For 10,000 transactions at a 0.1% fraud rate, the mean is 10 and the standard deviation is 3.2. Seeing no fraud at all has probability 0.00005, and seeing 20 or more has probability 0.35%. So a doubled count is a signal, not noise.

**You saw zero frauds in 1,000 transactions. What can you say about the rate?** Not that it is zero. Solving $(1 - p)^n = 0.05$ gives a 95% upper bound of about $3/n$, here 0.3%. This is the rule of three.

**How do you use expected value in a decision?** For each action I sum probability times payoff over the outcomes and take the action with the best total. In fraud that turns a probability and two costs into a threshold; the practice set [derives it](/notes/fraud-detection-practice/#drill-cost).

## Estimation and intervals {#estimation}

**What is maximum likelihood estimation?** Choose the parameter values under which the observed data are most probable. For a rate it is $k/n$. Under regularity conditions the estimate is consistent and asymptotically normal, which is where standard errors come from. Logistic regression and most neural networks are fitted this way: minimizing log loss is maximizing the likelihood.

**Maximum likelihood or maximum a posteriori?** Maximum a posteriori (MAP) maximizes the likelihood times a prior. A Gaussian prior on the weights gives an L2 penalty and a Laplace prior gives L1, as in the [regularization answer](/notes/fraud-detection-practice/#regularization). With enough data the two estimates agree.

**Bias and variance of an estimator?** Bias is the expected estimate minus the truth. Variance is its spread across samples. Mean squared error is bias squared plus variance. The sample variance divides by $n - 1$ to remove a bias. A biased estimator can still have the lower error, which is the case for shrinkage.

**Why shrink a rate estimated from few observations?** A merchant with 2 frauds in 50 transactions has a raw rate of 4%. With a Beta(1, 999) prior, whose mean is the portfolio rate of 0.1%, the posterior mean is 0.29%. A merchant with 200 frauds in 5,000, also 4%, moves only to 3.35%. Small samples are pulled toward the prior and large ones are not. Target encoding with smoothing is the same idea.

**Standard deviation or standard error?** The standard deviation describes the spread of the data. The standard error describes the spread of an estimate, and for a mean it is the standard deviation divided by $\sqrt{n}$. Four times the data halves it.

**What does a 95% confidence interval mean?** The procedure that built it covers the true value in 95% of repeated samples. It does not say there is a 95% probability that the parameter lies in this particular interval. That statement belongs to a Bayesian credible interval, which needs a prior.

**Which interval for a small proportion?** The Wald interval, $\hat{p} \pm 1.96\,SE$, fails when counts are small. For 3 frauds in 2,000 it runs from −0.02% to 0.32%, with a negative lower bound. The Wilson interval gives 0.05% to 0.44%. For rare events I use Wilson, which [Brown, Cai and DasGupta](https://doi.org/10.1214/ss/1009213286) recommend for small samples.

**When do you bootstrap?** When a metric has no simple formula for its standard error: PR-AUC, recall at a fixed alert rate, a ratio of medians. I resample rows with replacement, recompute the metric, and take percentiles. The unit I resample must be the independent one, accounts and not transactions, or the interval comes out too narrow. For a recall of 320 out of 400, the bootstrap gives 0.757 to 0.838 and the formula 0.761 to 0.839.

**Frequentist or Bayesian?** A frequentist treats the parameter as fixed and the data as random, and makes statements about how a procedure behaves over repeated samples. A Bayesian puts a distribution on the parameter and updates it with data, which allows direct probability statements at the price of choosing a prior. In practice I use shrinkage for small-sample rates and frequentist tests for experiments.

## Hypothesis testing {#testing}

**What is a p-value?** The probability, if the null hypothesis is true, of a result at least as extreme as the one observed. It is not the probability that the null hypothesis is true, and it says nothing about the size of the effect.

**Type I error, type II error and power?** A type I error rejects a true null hypothesis; its rate is $\alpha$. A type II error misses a real effect; its rate is $\beta$. Power is $1 - \beta$. At a fixed sample size, lowering one error rate raises the other, exactly like moving a fraud threshold.

**What determines the sample size of a test?** The effect size, the variance, $\alpha$ and the power. For two proportions, the size of each arm is:

$$
n \approx \frac{(z_{\alpha/2} + z_{\beta})^2\,\bigl[p_1(1-p_1) + p_2(1-p_2)\bigr]}{(p_1 - p_2)^2}
$$

Detecting a move from 10% to 12% at $\alpha = 5\%$ and 80% power needs 3,839 per arm. Halving the effect quadruples the sample. The [A/B testing note](/notes/ab-testing/#power) goes further.

**A result is significant. How likely is it to be real?** That depends on how often tested ideas are real, the same way precision depends on the base rate. If 10% of the ideas I test are real, with 80% power and $\alpha = 5\%$, then 64% of my significant results are real.

**z-test, t-test or Welch's test?** A z-test when the variance is known or the sample is large, as with proportions. A t-test for a mean with an estimated variance and a small sample; its tails are heavier. Welch's version does not assume equal variances and is the safe default for two groups. A paired t-test when the same units are measured twice: it tests the differences and removes the variation between units.

**What is a chi-square test for?** It compares observed counts with the counts expected under a hypothesis. The usual use is independence in a contingency table, such as whether the fraud rate is the same across channels. It needs expected counts of about five or more per cell. Below that, Fisher's exact test.

**Which tests assume no particular distribution?** Mann-Whitney asks whether one group tends to have larger values, and its statistic is [AUC under another name](/notes/fraud-detection-practice/#drill-auc). Kolmogorov-Smirnov measures the largest gap between two cumulative distributions, so it reacts to any difference in shape. A permutation test shuffles the group labels and recomputes any statistic I choose; it needs only that the labels be exchangeable under the null hypothesis.

**You run 20 tests. What changes?** At 5% each, the chance of at least one false positive is 64%. Bonferroni controls the chance of any false positive by testing each at $\alpha/m$, here 0.0025, and is conservative. [Benjamini-Hochberg](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) controls the expected share of false positives among the rejections: sort the p-values, find the largest rank $k$ with $p_{(k)} \le (k/m)\,q$, and reject the first $k$. On ten p-values in the lab, Bonferroni rejects one and Benjamini-Hochberg three. I use the first when any false positive is costly and the second when screening many features.

**Why is peeking at a running experiment a problem?** Every look is another chance to cross the threshold, so stopping at the first significant result pushes the false positive rate far above $\alpha$. Either fix the sample size in advance or use a sequential design that spends $\alpha$ across the looks. The A/B testing note has a [section on stopping](/notes/ab-testing/#stopping).

**Statistically significant or practically significant?** With millions of rows, a trivial difference is significant. I report the effect size with its interval and compare it with the smallest effect that would change a decision.

**One-sided or two-sided?** Two-sided by default. One-sided only when a result in the other direction would lead to the same action as no effect, and only if that was decided before seeing the data.

## Regression {#regression}

**What does ordinary least squares assume?** It minimizes squared residuals, with the solution $\hat{\beta} = (X^\top X)^{-1} X^\top y$. For unbiased coefficients it needs a model that is linear in the parameters, errors with mean zero given the features, and no perfect collinearity. For valid standard errors it also needs independent errors with constant variance. Normal errors matter only for inference in small samples.

**How do you read a coefficient?** As the expected change in the outcome for a one-unit change in that feature with the others held fixed. With a logged outcome it is roughly a percentage change. With correlated features, "the others held fixed" may describe a case that never occurs.

**What does multicollinearity do?** Predictions stay fine. Individual coefficients become unstable, with large standard errors and signs that can flip. The variance inflation factor, $1/(1 - R_j^2)$, measures it, and values above 5 to 10 deserve attention. I drop or combine features, or use ridge.

**What is heteroscedasticity, and does it matter?** The error variance changes with the features: larger amounts, larger errors. The coefficients remain unbiased and the standard errors are wrong. I use robust standard errors, model the log of the outcome, or weight the observations.

**What is wrong with R-squared?** It is the share of variance explained, and it never falls when a feature is added. So I use adjusted R-squared or, better, error on held-out data. A high value does not show the model is right, and a low one can still be useful.

**Explain logistic regression.** It models the log-odds as linear in the features: $\log\frac{p}{1-p} = \beta_0 + \beta^\top x$. It is fitted by maximum likelihood, which is the same as minimizing log loss. The problem is convex with no closed form, so it is solved iteratively. The decision boundary is linear. $e^{\beta}$ is an odds ratio: a coefficient of 0.7 doubles the odds, which moves a 1% probability to 1.99%.

**Why not linear regression for a binary target?** Its predictions leave the range from 0 to 1, its errors do not have constant variance, and squared loss penalizes a confident correct prediction as if it were an error. Log loss matches the likelihood of a yes-or-no outcome.

**What is omitted variable bias?** A variable that affects both a feature and the outcome is left out, and its effect is absorbed into that feature's coefficient. That is harmless for prediction while the relationship holds, and fatal for a causal reading. The [causal inference note](/notes/causal-inference/#dag) draws the cases.

**What is Simpson's paradox?** A pattern that holds in every subgroup reverses in the total, because the subgroups have different sizes. Rule B is more precise than rule A on web alerts (95% against 90%) and on mobile alerts (40% against 30%). Overall, A scores 84.5% and B 45.0%, because A's alerts are mostly web and B's are mostly mobile. I compare within segments, or reweight to a common mix. The causal inference note has [another example](/notes/causal-inference/#simpson).

**Pearson or Spearman correlation?** Pearson measures linear association and is sensitive to outliers. Spearman is Pearson computed on ranks, so it measures any monotonic association and is robust. A correlation of zero does not mean independence.

## Sampling, bias and shift {#sampling}

**Which sampling schemes matter?** Simple random sampling. Stratified sampling draws within groups, which guarantees rare groups and lowers variance. Cluster sampling draws whole groups, which is cheaper and noisier. Weighted sampling over-samples a rare class and weights it back; the practice set shows [the correction](/notes/fraud-detection-practice/#drill-downsampling).

**Selection bias and survivorship bias?** Selection bias: the sample was chosen by a process related to the outcome, as when labels exist only for approved transactions. Survivorship bias: only cases that passed a filter remain, as with accounts that were never closed. Either way the sample no longer represents the population the model will score. Fraud labels have [both problems](/notes/fraud-detection-practice/#labels).

**Why split by customer and not by row?** Rows from one customer are correlated. If they sit on both sides of the split, the model is tested on something it has partly seen. I split by entity, and by time as well when the data has an order.

**Confounder, mediator, collider?** A confounder causes both the feature and the outcome: adjust for it. A mediator lies on the path from feature to outcome: adjusting for it removes part of the effect. A collider is caused by both: adjusting for it creates an association that is not there. Selection bias is collider bias.

**Covariate shift, label shift and concept drift?** Covariate shift: the inputs change and the relationship stays, so I reweight or retrain. Label shift: the base rate changes and each class looks the same, so I adjust the predicted probabilities for the new rate, with the same odds correction as for down-sampling. Concept drift: the relationship itself changes, as when fraudsters adapt, and only new labels fix it. The practice set covers [the response](/notes/fraud-detection-practice/#drift).

**Missing completely at random, at random, or not at random?** Completely at random: missingness is unrelated to anything, so dropping rows is unbiased but wasteful. At random: it depends on observed variables, so I impute from them. Not at random: it depends on the missing value itself, as when low incomes go unreported, and the data alone cannot fix it. For prediction, an indicator that the value was missing is often a useful feature, and trees handle missing values natively. Imputation is fitted on training data only.

## Model evaluation {#evaluation}

**Define precision, recall, specificity and F1.** Precision is $TP/(TP+FP)$. Recall, also called sensitivity, is $TP/(TP+FN)$. Specificity is $TN/(TN+FP)$, one minus the false positive rate. F1 is the harmonic mean of precision and recall, and $F_\beta$ weights recall $\beta$ times as much. The [precision and recall note](/notes/precision-vs-recall/) draws the trade-off.

**ROC-AUC or PR-AUC?** ROC-AUC does not change with the class balance, which is its strength for comparing rankings and its weakness when positives are rare. The precision-recall curve shows what an alert queue will look like. [A separate note](/notes/roc-auc-vs-pr-auc/) compares them.

**What are KS and Gini in credit scoring?** KS is the largest gap between the cumulative score distributions of good and bad accounts, which equals the largest value of true positive rate minus false positive rate. Gini is $2 \times \mathrm{AUC} - 1$. For scores distributed N(0, 1) and N(1, 1), AUC is 0.760, Gini is 0.520 and KS is 0.383. KS describes one threshold; AUC and Gini describe the whole ranking.

**What are lift and capture rate?** The capture rate is the share of all positives found in the top $x\%$ of scores. Lift is that share divided by $x$: how many times better than random. If the top 1% of scores holds 50% of the fraud, the lift is 50.

**Log loss or Brier score?** Both are proper scoring rules: the true probabilities minimize them, so they reward calibration and ranking together. Log loss punishes a confident mistake without limit. The Brier score is the mean squared error of the probability and is bounded. With one positive among four cases, a sharp model that says 0.7, 0.2, 0.1, 0.1 scores 0.0375 on Brier and 0.198 on log loss. Saying 0.25 for all four scores 0.1875 and 0.562.

**Calibration or discrimination?** Discrimination is ranking positives above negatives; AUC measures it. Calibration is whether a predicted 10% happens 10% of the time. Any monotonic transformation of the score changes calibration and leaves AUC alone. Calibration matters when the number feeds a cost calculation, as in [setting a threshold](/notes/fraud-detection-practice/#threshold).

**Which cross-validation, and when?** K-fold for independent rows. Stratified when classes are imbalanced. Grouped when rows share an entity. Forward-chaining when time matters. Nested when the same data both tunes the model and estimates its performance. In all of them, scaling, encoding and feature selection are fitted inside each fold.

**MAE, RMSE or MAPE?** Root mean squared error weights large errors more and is minimized by the mean. Mean absolute error is robust and is minimized by the median. Mean absolute percentage error is relative, undefined at zero, and treats over- and under-prediction unequally. I choose by what a large error costs.

**How do you compare two models on one test set?** As a paired comparison on the same rows. I bootstrap the difference in the metric, resampling accounts, and check whether the interval excludes zero. For AUC there is also DeLong's test. An offline win still has to be confirmed by a [champion-challenger test](/notes/fraud-detection-practice/#drill-ztest).

## How models learn {#learning}

**What is the bias-variance trade-off?** Expected squared error splits into bias squared, variance and irreducible noise. Simple models have high bias; flexible ones have high variance. Regularization, more data and ensembling move a model along the trade-off: bagging cuts variance and boosting cuts bias.

**How do you recognize overfitting, and what do you do?** Training error is low and validation error is much higher. The practice set lists [the controls](/notes/fraud-detection-practice/#overfitting) for boosted trees and the [textbook answer](/notes/fraud-detection-practice/#classic-ml).

**More data or a better model?** I read the learning curves. If validation error is still falling as data is added, more data helps. If training and validation error are both high, the model is biased and needs better features or more capacity. If the gap between them is wide, it is variance: regularize or add data.

**What is the curse of dimensionality?** As dimensions grow, data becomes sparse, the nearest and farthest neighbors end up almost equally far away, and the sample needed to cover the space grows exponentially. Nearest neighbors, kernels and clustering suffer most. Feature selection, dimensionality reduction and models with built-in structure are the remedies.

**Parametric or non-parametric?** A parametric model has a fixed number of parameters, as linear and logistic regression do. It is efficient, and biased if the form is wrong. A non-parametric model grows with the data, as nearest neighbors, trees and kernel methods do. It is flexible and needs more data.

**Generative or discriminative?** A generative model describes how each class produces features, $P(x \mid y)\,P(y)$: naive Bayes, Gaussian mixtures. A discriminative model describes $P(y \mid x)$ directly: logistic regression, trees, neural networks. With enough data the discriminative model is usually more accurate. The generative one copes better with little data, missing features and a changed base rate.

**Which models need feature scaling?** Anything built on distances, gradient descent or a penalty on coefficients: nearest neighbors, k-means, support vector machines, principal components, regularized linear models and neural networks. Trees do not, because they split on order.

**How do you encode categorical features?** One-hot for a few categories. Ordinal when there is an order, or for trees. Target encoding for many categories, computed out of fold and smoothed toward the overall rate, because otherwise it leaks the label. Hashing when there are too many to store. Learned embeddings in neural networks.

**How do you tune hyperparameters?** Random search beats a grid when only a few hyperparameters matter. Bayesian optimization pays off when each fit is expensive. Early stopping sets the number of trees or epochs. I tune on validation data, from a later period if time matters, and report on test data once. Most of the gain comes from the learning rate, the tree size and the regularization strength.

## Classic algorithms {#algorithms}

**How does a decision tree choose a split?** By the largest drop in impurity. A node with 4 positives among 10 rows has Gini impurity 0.48. Splitting it into a group with 3 positives of 4 and a group with 1 of 6 gives 0.375 and 0.278, a weighted 0.317, so the gain is 0.163. The tree tries every feature and threshold and keeps the best. The [code below](#scratch) does it for one feature.

**Why does a random forest sample features at each split?** To make the trees less alike. The variance of an average of $B$ trees with pairwise correlation $\rho$ is $\rho\sigma^2 + (1-\rho)\sigma^2/B$. More trees shrink only the second term, so lowering $\rho$ is what helps. The rows left out of each bootstrap sample give a validation estimate for free.

**What does "gradient" mean in gradient boosting?** Each new tree is fitted to the negative gradient of the loss with respect to the current predictions. For squared error that is the residual; for log loss it is $y - p$. The learning rate scales each step. The practice set [compares it with bagging](/notes/fraud-detection-practice/#trees).

**XGBoost, LightGBM or CatBoost?** XGBoost uses second-order information and explicit regularization, and grows trees level by level by default. LightGBM bins features into histograms and grows leaf by leaf; it is the fastest on large data and needs a limit on leaves. CatBoost handles categorical features with ordered target statistics, uses symmetric trees, and has strong defaults.

**Explain a support vector machine.** It finds the separating hyperplane with the widest margin, and only the points on or inside the margin, the support vectors, determine it. The parameter C trades margin width against violations. A kernel replaces dot products, which gives a nonlinear boundary without computing the expanded features. With the radial basis function kernel, gamma sets how local the influence of a point is. Features must be scaled. Training cost grows faster than linearly in the rows, and the outputs are not probabilities until calibrated.

**K-nearest neighbors?** Predict from the $k$ closest training points. There is no training and prediction is slow. A small $k$ gives variance and a large $k$ gives bias. It needs scaling and degrades in high dimensions.

**Naive Bayes?** Bayes' rule with the features assumed independent given the class. It is fast and works with little data and with text. Its probabilities are too extreme, because correlated evidence is counted more than once, while its ranking is often fine.

**K-means, Gaussian mixtures or DBSCAN?** K-means assigns points to the nearest center and recomputes the centers. It suits round clusters of similar size and depends on the starting points and the scale. A Gaussian mixture gives soft assignments and elliptical clusters, fitted by expectation-maximization; k-means is its limiting case. DBSCAN grows clusters from dense regions, finds arbitrary shapes and marks noise, and struggles when density varies. Choosing $k$ is in the [quick checks](/notes/fraud-detection-practice/#classic-ml).

**What does principal component analysis do?** It finds orthogonal directions of greatest variance, the eigenvectors of the covariance matrix. I keep the first few; the eigenvalues give the variance each explains. Features are standardized first. It is linear, its components are hard to name, and high variance is not the same as predictive value.

**How does an isolation forest find anomalies?** It splits the data at random. An unusual point is separated after few splits, so a short average path means an anomaly. It computes no distances and scales well. The practice set says [where it fits](/notes/fraud-detection-practice/#anomaly).

**What is stacking?** A second model is trained on the predictions of several first models. Those predictions must come from folds the first models did not see, or the second model learns from overfitted inputs. The gain is usually small, and it costs complexity in serving and explaining.

**Linear model or trees?** A linear model when rows are few, effects are mostly additive, or I need to extrapolate or explain coefficients. Trees when there are interactions, thresholds, mixed types and missing values. Trees cannot predict beyond the range they saw.

## Neural networks {#neural-networks}

**Backpropagation in one paragraph.** The forward pass computes the loss. The backward pass applies the chain rule layer by layer from the output to the input, reusing the intermediate results, so all the gradients cost about as much as one more forward pass. An optimizer then updates the weights.

**Why ReLU and not sigmoid in hidden layers?** Sigmoid and tanh flatten out at both ends, so their gradients approach zero and vanish across many layers. ReLU has a gradient of one for positive inputs and is cheap. A unit can get stuck at zero, which leaky ReLU and GELU avoid.

**Vanishing and exploding gradients?** A gradient through many layers or time steps is a product of many terms, so it shrinks or grows exponentially. The remedies are ReLU-type activations, careful initialization, normalization and residual connections. Clipping handles the exploding case, and gated units handle recurrent networks.

**How do you regularize a neural network?** Weight decay, dropout, early stopping, data augmentation, a smaller network, label smoothing. Dropout zeroes random activations during training, which approximates averaging many thinned networks.

**What is a convolutional layer good for?** It slides small filters across the input. Sharing weights and connecting locally cut the number of parameters and make the layer respond the same way wherever a pattern appears. Pooling and stride reduce resolution and widen the receptive field. It suits images and, in one dimension, sequences with local patterns.

**RNN, LSTM or GRU?** A recurrent network carries a hidden state along a sequence with the same weights at every step, and its long-range gradients vanish. An LSTM adds a cell state with input, forget and output gates, so information and gradient survive many steps. A GRU merges these into two gates with fewer parameters. Transformers have replaced them where data is plentiful; recurrent models remain fine for short or streaming sequences. Self-attention is in the practice set's [quick checks](/notes/fraud-detection-practice/#deep-learning).

**What are embeddings for?** Dense vectors learned for discrete things such as words, merchants or devices, so that similar ones lie close together. They replace one-hot encoding when there are many categories, and they can be trained with the task or trained first and handed to a tree model.

**How does an autoencoder detect anomalies?** It is trained to reconstruct normal data through a narrow layer. A point unlike the training data reconstructs badly, and the reconstruction error is the score. If anomalies are in the training data, or the network is too large, it learns to reconstruct them too.

**Why do trees still beat neural networks on tabular data?** A [benchmark by Grinsztajn and colleagues](https://arxiv.org/abs/2207.08815) found trees ahead on medium-sized tabular datasets and offered three reasons. Tabular targets are irregular and networks prefer smooth functions. Uninformative features hurt networks more. And tabular features each have their own meaning, which a rotation-invariant learner ignores. Networks win where the input has raw structure: sequences, text, images, graphs.

**Batch size and learning rate?** A larger batch gives a less noisy gradient and tolerates a higher learning rate; the noise of a small batch acts as a regularizer. A common schedule warms up and then decays. The learning rate is the hyperparameter that matters most.

**Cross-entropy, class weights or focal loss?** Cross-entropy is log loss for classification. Class weights scale the loss of the rare class. Focal loss multiplies each term by $(1 - p_t)^\gamma$, which shrinks the contribution of easy examples. Weights and focal loss both distort the probabilities, so I recalibrate afterwards.

**Fine-tune or train from scratch?** A pretrained model brings general features, so it needs less data and time. I can freeze the backbone and train a head, fine-tune everything at a small learning rate, or train small adapters. The risks are forgetting what was learned and a gap between the two domains.

## Reinforcement learning {#reinforcement-learning}

**What is a Markov decision process?** States, actions, transition probabilities, rewards and a discount factor. A policy maps states to actions, and the goal is the highest expected discounted return. "Markov" means the next state depends only on the current state and action.

**Value-based or policy-based?** Value-based methods such as Q-learning learn the value of each action in each state and act on the best. They reuse old data well and suit discrete actions. Policy-based methods such as PPO adjust the policy's parameters along the gradient of expected return. They handle continuous actions and stochastic policies, with higher variance. Actor-critic methods combine them: a learned value estimate reduces the variance of the policy gradient.

**Exploration or exploitation?** Acting on the current best estimate earns reward now, and trying something else earns information. Epsilon-greedy explores at random. An upper confidence bound adds a bonus for uncertainty. Thompson sampling draws from the posterior and acts on the draw. The practice set discusses [where this fits in fraud](/notes/fraud-detection-practice/#bandits).

**On-policy or off-policy, and why does off-policy evaluation matter?** On-policy methods learn from data produced by the current policy. Off-policy methods learn from data produced by any policy. In production I cannot deploy a new policy just to see what happens, so I estimate its value from logs of the old one. Importance sampling reweights each logged reward by the ratio of the two policies' probabilities for the action taken. It needs logged probabilities and overlap between the policies, and its variance is high when they differ a lot.

**Bandit or full reinforcement learning?** In a bandit, an action affects only the immediate reward. In full reinforcement learning, it also changes the next state. I start with a bandit when the long-run effect is small or cannot be measured.

**What makes reward design hard?** Rewards that arrive late or rarely make it hard to credit the right action. A proxy reward gets gamed: optimizing clicks is not optimizing value. Before training I ask what behavior would maximize the reward as written.

**PPO in two sentences.** It is a policy-gradient method that clips the ratio between the new and the old policy's probabilities, so one batch can be reused for several updates without the policy moving too far. It is simple and stable, which made it the default for many problems.

## Partly labeled data {#weak-labels}

**Semi-supervised learning and pseudo-labels?** With few labels and much unlabeled data, I train on the labels, predict on the rest, add the confident predictions as labels, and retrain. The risk is that the model's own mistakes are fed back and reinforced.

**What is positive-unlabeled learning, and why does fraud fit?** Only some positives are labeled, and everything else is a mix of positives and negatives. Confirmed fraud is positive; "not reported" is unlabeled, not negative. If the labeled positives are a random sample of all positives, a model trained on labeled against unlabeled gives the true probability times a constant. The ranking survives and the probabilities need rescaling.

**What is active learning?** Choosing which examples to label, usually the ones the model is least sure of. In fraud the review queue is the labeling budget, so some of it should go to uncertain cases and not only to the highest scores.

**What is censoring, and why does it matter for recent labels?** An outcome that has not happened by the cutoff is right-censored: all I know is that it has not happened yet. Counting it as a negative understates the rate. If half of all fraud is reported within seven days, a rate of 0.05% observed at day seven points to about 0.10% in the end. Survival analysis, with Kaplan-Meier curves and hazard models, uses the partial information properly.

## Statistics in monitoring {#monitoring-stats}

**KL divergence, Jensen-Shannon and PSI?** KL divergence is $\sum p \log(p/q)$. It is asymmetric, and infinite where $q$ is zero and $p$ is not. Jensen-Shannon is a symmetric, bounded version. PSI on binned data is the sum of the two KL divergences, one in each direction. The practice set [codes it](/notes/fraud-detection-practice/#psi).

**How do you set an alert threshold on a count?** I model the count. If alerts per hour are Poisson with mean 100, the standard deviation is 10, and 130 or more has probability 0.23%. Real counts vary more than Poisson and follow the clock, so I compare with the same hour last week and widen the limits.

**Why do many monitors produce false alarms?** It is multiple testing. With 200 metrics each using a three-sigma rule, the chance that at least one fires on a normal day is 42%. I keep fewer alerts, set them higher, and require them to persist.

**How do you detect a small, sustained shift?** A three-sigma rule misses it. A cumulative sum (CUSUM) or an exponentially weighted moving average adds up small deviations over time and triggers earlier.

## From scratch in code {#scratch}

Download [ml-stats-lab.py](/notes/ml-stats-fundamentals/ml-stats-lab.py) and run it with Python 3.10 or later and NumPy. It prints every number in this note. If SciPy or scikit-learn is installed, it also checks the KS statistic, the logistic regression and k-means against them.

```bash
python3 ml-stats-lab.py
```

### Logistic regression by gradient descent {#code-logistic}

```python
import numpy as np

def fit_logistic(X, y, lr=0.5, steps=5000, l2=0.0):
    X, y = np.asarray(X, float), np.asarray(y, float)
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w)   # gradient of the mean log loss
        b -= lr * np.mean(p - y)
    return w, b
```

The gradient of the mean log loss is $X^\top(p - y)/n$, which is the whole derivation. On 4,000 simulated rows with true weights 1.5 and −1.0 and intercept −0.5, it returns 1.477, −1.033 and −0.516, and scikit-learn agrees to three decimals. Things to say: standardize the features so one learning rate suits all of them, the L2 term does not apply to the intercept, and a real implementation stops on convergence.

### The best split by Gini impurity {#code-gini}

```python
def gini(y):
    p = np.mean(y)
    return 2 * p * (1 - p)             # 1 - p^2 - (1 - p)^2 for two classes

def best_split(x, y):
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    parent, best = gini(y), (None, 0.0)
    for i in range(1, len(x)):
        if x[i] == x[i - 1]:
            continue                    # no threshold separates equal values
        child = (i * gini(y[:i]) + (len(x) - i) * gini(y[i:])) / len(x)
        if parent - child > best[1]:
            best = ((x[i] + x[i - 1]) / 2, parent - child)
    return best
```

On ten amounts with four frauds, the best threshold is 75, with a gain of 0.309: every row above it is fraud. This version recomputes the impurity at each position, so it is quadratic. Running counts of positives on each side make it linear after the sort.

### K-means {#code-kmeans}

```python
def kmeans(X, k, n_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)]
    for _ in range(n_iter):
        labels = np.argmin(((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
        new = np.array([X[labels == j].mean(axis=0) if (labels == j).any() else centers[j]
                        for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    inertia = float(((X - centers[labels]) ** 2).sum())
    return centers, labels, inertia
```

On three simulated blobs it recovers the centers and reaches the same within-cluster sum of squares, the value it returns as `inertia`, as scikit-learn. Each step can only lower that sum, so the loop converges, to a local optimum that depends on the starting points. Real use runs several starts or k-means++.

### Intervals and tests {#code-stats}

```python
import math

def wilson_interval(k, n, z=1.959964):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half

def bootstrap_interval(values, metric=np.mean, n_boot=2000, level=0.95, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    stats = [metric(values[rng.integers(0, len(values), len(values))]) for _ in range(n_boot)]
    lo, hi = np.quantile(stats, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)

def permutation_test(a, b, n_perm=10_000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        hits += abs(pooled[:len(a)].mean() - pooled[len(a):].mean()) >= observed
    return (hits + 1) / (n_perm + 1)

def benjamini_hochberg(p_values, q=0.05):
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, len(p) + 1) / len(p)
    k = passed.nonzero()[0].max() + 1 if passed.any() else 0   # largest rank that passes
    reject = np.zeros(len(p), dtype=bool)
    reject[order[:k]] = True
    return reject

def ks_statistic(a, b):
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.abs(cdf_a - cdf_b).max())
```

Two details that get asked. The permutation p-value adds one to the numerator and the denominator, so it can never be zero. And Benjamini-Hochberg rejects everything up to the largest rank that passes, including smaller ranks that failed their own threshold.

## Formula sheet {#formulas}

| Item | Formula |
|---|---|
| Bayes' rule | $P(A \mid B) = P(B \mid A)\,P(A)/P(B)$ |
| Variance of a sum | $\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathrm{Cov}(X,Y)$ |
| Standard error of a mean | $\sigma/\sqrt{n}$ |
| Standard error of a proportion | $\sqrt{p(1-p)/n}$ |
| Rule of three | Zero events in $n$ trials: 95% upper bound $3/n$ |
| Sample size, two proportions | $(z_{\alpha/2}+z_\beta)^2\,[p_1(1-p_1)+p_2(1-p_2)]/(p_1-p_2)^2$ |
| Bonferroni | Test each of $m$ hypotheses at $\alpha/m$ |
| Benjamini-Hochberg | Reject up to the largest $k$ with $p_{(k)} \le (k/m)\,q$ |
| Least squares | $\hat{\beta} = (X^\top X)^{-1}X^\top y$ |
| Logistic regression | $\log\frac{p}{1-p} = \beta_0 + \beta^\top x$; gradient $X^\top(p-y)/n$ |
| Variance inflation factor | $1/(1-R_j^2)$ |
| Gini impurity | $1 - \sum_c p_c^2$ |
| Gini coefficient and KS | $2\,\mathrm{AUC} - 1$; $\max(\mathrm{TPR} - \mathrm{FPR})$ |
| Bias and variance | Error = bias$^2$ + variance + noise |
| Average of correlated trees | $\rho\sigma^2 + (1-\rho)\sigma^2/B$ |
| Log loss and Brier score | $-\frac{1}{n}\sum[y\log p + (1-y)\log(1-p)]$; $\frac{1}{n}\sum(p-y)^2$ |
| KL divergence | $\sum p\log(p/q)$ |

## References {#references}

1. Hastie T, Tibshirani R, Friedman J. The Elements of Statistical Learning, 2nd edition. Springer, 2009.
2. Brown LD, Cai TT, DasGupta A. [Interval estimation for a binomial proportion](https://doi.org/10.1214/ss/1009213286). Statistical Science 2001, 16(2):101–117.
3. Benjamini Y, Hochberg Y. [Controlling the false discovery rate: a practical and powerful approach to multiple testing](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x). Journal of the Royal Statistical Society, Series B 1995, 57(1):289–300.
4. Grinsztajn L, Oyallon E, Varoquaux G. [Why do tree-based models still outperform deep learning on typical tabular data?](https://arxiv.org/abs/2207.08815) NeurIPS 2022, Datasets and Benchmarks.

Every number in this note is printed by [`ml-stats-lab.py`](/notes/ml-stats-fundamentals/ml-stats-lab.py). It was run with NumPy 2.5, SciPy 1.18 and scikit-learn 1.9. For where to practice further, see the [ML and AI study route](/notes/ml-ai-interview-prep/).
