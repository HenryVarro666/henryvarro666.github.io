---
title: "Fraud detection modeling: practice questions with worked answers"
date: 2026-11-23T20:00:00-06:00
lastmod: 2026-11-25T04:00:00-06:00
description: "Practice questions on fraud modeling with worked answers: class imbalance, metrics, time-based validation, late labels, feature selection, thresholds from costs, three case walk-throughs, a real-time system design, probability drills, and tested Python, SQL and PySpark code."
topics: [data-science]
---
I put this set together while preparing for applied machine learning (ML) interviews on fraud and risk modeling. It has five parts: fundamentals as they show up in fraud, case walk-throughs, a system design walk-through, probability drills, and coding exercises with tested code. Each answer is written the way I would say it: the conclusion first, then the reason, the trade-off, and how I would check it.

> **Takeaway.** Most fraud modeling questions come back to four habits: measure at the operating point on later data, build every feature as of decision time, treat labels as late and biased, and let costs set the threshold.

## How to use this set {#how-to}

Cover the answer, speak for 60 to 90 seconds, then compare and note only what you missed. Unless a question says otherwise, the setting is a bank that scores payments in real time, with about one fraud per thousand transactions.

If time is short, start with the topics candidates most often report from data science panels at banks: [class imbalance](#imbalance), [bagging and boosting](#trees), [regularization](#regularization), [feature selection](#feature-selection) and [the real-time system design](#system-design).

For fundamentals outside fraud, such as hypothesis tests, regression, the classic algorithms and neural networks, see [Machine learning and statistics fundamentals](/notes/ml-stats-fundamentals/).

Three earlier notes go deeper on single topics: [precision and recall](/notes/precision-vs-recall/), [ROC-AUC and PR-AUC](/notes/roc-auc-vs-pr-auc/), and [A/B testing](/notes/ab-testing/). The [MLOps guide](/notes/mlops-learning-guide/) covers [leakage](/notes/mlops-learning-guide/#time-leakage) and [monitoring](/notes/mlops-learning-guide/#monitoring) with a runnable lab.

## Fundamentals in a fraud setting {#fundamentals}

### Class imbalance {#imbalance}

**Question.** Fraud is about 0.1% of transactions. How do you handle the class imbalance?

Imbalance is mostly an evaluation and thresholding problem, not a training problem. I would do four things:

1. Use metrics that ignore the huge negative class: the area under the precision-recall curve (PR-AUC), recall at a fixed false positive rate (FPR) or alert budget, and recall weighted by dollars.
2. Train on all the fraud and a sample of the non-fraud, mainly to save compute, or use class weights such as `scale_pos_weight` (about 999 at this fraud rate).
3. After down-sampling, [correct the predicted probabilities](#drill-downsampling) back to the true base rate before setting a threshold.
4. Evaluate on data with the real, untouched class distribution.

I would be cautious with the Synthetic Minority Over-sampling Technique (SMOTE). Interpolating between rare fraud cases in a high-dimensional space of mixed types creates points that no customer would produce, and it rarely beats simple weighting with boosted trees. For neural networks, focal loss is an option.

### Metrics {#metrics}

**Question.** Which metrics would you report, and why not accuracy or only ROC-AUC?

Accuracy is useless here: predicting "legitimate" for everything scores 99.9%. The area under the receiver operating characteristic curve (ROC-AUC) is fine for judging the ranking, but it gives every part of the ranking the same weight. It can look excellent while precision at the operating point is poor. The [ROC-AUC and PR-AUC note](/notes/roc-auc-vs-pr-auc/) has two models with the same ROC-AUC and a fourfold gap in average precision.

I would report five things, all measured on a later period than the training data:

- **PR-AUC**, next to the prevalence, which is its random baseline.
- **Recall at a fixed FPR or alert volume**, because a review team can only work so many cases.
- **Value detection rate**: the share of fraud dollars caught. One \$5,000 fraud matters more than ten \$20 ones.
- **False positive ratio**: good customers flagged per fraud caught.
- **Calibration**, if the score feeds a cost-based decision.

### Validation split {#validation}

**Question.** How do you split the data for validation?

By time, not at random. I train on an earlier window, validate on a later one, and keep a final out-of-time test set that I touch once.

A random split leaks the future. The same fraud ring, device or campaign lands on both sides, and the score looks better than it will in production.

Two details matter. I leave a gap between training and test at least as long as the label maturity window, because recent transactions have not had time to be reported. And for tuning I use walk-forward validation over several periods, which shows stability instead of one lucky month.

### Leakage {#leakage}

**Question.** What is data leakage? Give fraud examples and say how you would catch it.

Leakage is any information in the training features that would not be available at decision time. In fraud it usually looks like one of these:

- A rolling count or sum that includes the current transaction.
- A customer aggregate computed over the whole dataset instead of up to the timestamp.
- A field written after the event, such as a case-investigation code or an "account closed for fraud" flag.
- Target encoding fitted on the full data.

To catch it, I build every feature point-in-time, as of the transaction timestamp. I am suspicious of any single feature with implausibly high importance. And I compare offline performance with shadow-mode performance, using the same feature code in training and serving.

### Late and biased labels {#labels}

**Question.** Fraud labels arrive late and are imperfect. What problems does that cause?

Three problems.

**Delay.** A customer may dispute a transaction weeks later, so the most recent data is under-labeled. Training on it naively teaches the model that recent fraud is legitimate. I use only data older than the maturity window.

**Selection bias.** We observe outcomes only for transactions we approved. Declined ones have no label, so the model learns from a population filtered by the previous model. A small random control group that bypasses the model, where the loss is tolerable, gives unbiased labels. Credit modeling calls the same problem reject inference.

**Noise.** Some reported fraud is first-party or friendly fraud, where a real customer disputes a real purchase, and some fraud is never reported. I keep label sources separate and, where possible, model third-party fraud, first-party fraud and scams as different targets.

### Bagging and boosting {#trees}

**Question.** Compare bagging with boosting, using random forest and gradient boosting as the examples. Why does boosting usually win on tabular data?

Bagging trains many models independently on resampled data and averages them. Boosting trains them one after another, each correcting the errors of those before it. Random forest and gradient boosting are the tree versions of the two ideas.

A random forest grows deep trees independently on bootstrap samples with random feature subsets and averages them, which mainly reduces variance.

Gradient boosting grows shallow trees one after another, each fitted to the gradient of the loss of the current ensemble. That mainly reduces bias, and shrinkage, subsampling and regularization keep the variance in check.

| | Bagging: random forest | Boosting: gradient boosting |
|---|---|---|
| How the trees are built | Independently, in parallel | In sequence, each on what is still wrong |
| What it mainly reduces | Variance | Bias |
| The individual tree | Deep; overfits on its own | Shallow; underfits on its own |
| Adding more trees | Levels off and does no harm | Overfits in the end, so stop early |
| Mislabeled examples | Averaged out | Chased harder each round unless regularized |

The last row matters in fraud, where some labels are wrong. I limit depth, use shrinkage and stop early, and I do not push a boosted model to fit the last few training cases.

[XGBoost](https://arxiv.org/abs/1603.02754) adds a second-order approximation of the loss, L1 and L2 penalties on leaf weights, and a learned default direction for missing values. LightGBM uses histogram binning and leaf-wise growth, so it is faster on large data and overfits more easily without limits on leaves.

Boosting usually wins because each new tree concentrates on what the model still gets wrong, which in fraud means the rare, hard cases. A random forest is more forgiving with little tuning.

The hyperparameters I expect to be asked about: learning rate, number of trees with early stopping, `max_depth` or `num_leaves`, `min_child_weight`, `subsample`, `colsample_bytree` and `scale_pos_weight`.

### Calibration and the threshold {#threshold}

**Question.** Your model outputs 0.8. Is that a probability? How do you set the threshold?

Not necessarily. Boosted trees trained with class weights or on down-sampled data rank well and are still miscalibrated.

If I kept a share $\beta$ of the negatives, I first [correct the score](#drill-downsampling) back to the true base rate. Then I check a reliability curve on out-of-time data and, if needed, fit Platt scaling or isotonic regression on a held-out set.

For the threshold I use expected cost, not 0.5: intervene when the probability times the fraud loss exceeds the expected cost of a false positive. Because the loss scales with the amount, the threshold should too. A \$5,000 transfer deserves intervention at a much lower probability than a \$50 purchase; the [cost drill](#drill-cost) works the numbers.

In practice the threshold is also bounded by the alert budget and by limits on customer friction, so I would bring the trade-off curve to the fraud strategy team instead of picking the cut alone.

### Overfitting in boosted trees {#overfitting}

**Question.** How do you detect and control overfitting in boosted trees?

I detect it by a widening gap between training and out-of-time performance, and by performance that decays quickly month over month.

To control it: early stopping on an out-of-time validation set, a lower learning rate with more trees, shallower trees, a larger minimum leaf size, row and column subsampling, and L1 or L2 regularization.

I also remove unstable features, for example ones whose distribution shifts a lot between periods or whose importance flips across time folds. In fraud I would give up a little in-sample lift for a model that still holds three months later.

### Explaining a decision {#explainability}

**Question.** How do you explain a model's decision to a business partner or a model reviewer?

At the global level, I show which features drive the model, with a summary of SHAP values ([SHapley Additive exPlanations](https://arxiv.org/abs/1705.07874)) and partial-dependence plots, and check that the directions make business sense. Risk should rise with velocity, for example.

At the local level, for one flagged transaction, I take the largest SHAP contributions and translate them into plain reasons: new device, first payment to this payee, amount twenty times the customer's usual.

Two caveats: SHAP splits credit among correlated features somewhat arbitrarily, and attribution is not causation.

For governance, I document the data, assumptions, limitations, tests and monitoring plan so that an independent validator can challenge them. US supervisory guidance on model risk was revised on April 17, 2026, when [SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm) replaced SR 11-7. A [May 2026 speech](https://www.federalreserve.gov/newsevents/speech/bowman20260501a.htm) by the Federal Reserve's Vice Chair for Supervision says the revised guidance does not apply to generative or agentic AI.

### A model that is getting worse {#drift}

**Question.** The model has been live for six months and performance is dropping. What do you do?

First I confirm it is real. Are the labels for the recent period mature? Did volumes or the product mix change? Did an upstream data feed break?

Then I localize it. I track the score distribution and feature drift with the population stability index (PSI), and I break performance down by channel, product and fraud type. The usual reading of PSI is a convention, not a statistical test: below 0.1 is stable, 0.1 to 0.25 deserves a look, and above 0.25 is a large shift.

The usual causes are a data pipeline change, a population shift such as a new product, or fraudsters adapting: the model blocks one pattern and they move to another.

My responses, fastest first: a targeted rule for the new pattern as a stopgap, a threshold adjustment, retraining on recent data, then new features. Longer term I want scheduled retraining, champion-challenger tests, and monitoring thresholds agreed with model governance.

### Sequence and graph models {#sequence-graph}

**Question.** When would you use sequence or graph models for fraud?

When the signal sits in structure that hand-built aggregates lose.

Sequence models over a customer's event history capture order and timing. Password reset, then new device, then new payee, then a maximum transfer is a pattern that count-based velocity features only approximate.

Graph models capture relations: accounts sharing a device, address, phone or beneficiary. Fraud rings and mule networks show up as dense or fast-growing subgraphs.

The costs are more data and compute, harder validation and explanation, and latency. Graph features usually have to be precomputed and served from a feature store.

So I would start with graph-derived features in the tree model, such as shared-entity counts and the fraud rate among neighbors, computed without label leakage. I would move to learned embeddings only if they add recall at the operating point.

### Anomaly detection {#anomaly}

**Question.** Where do supervised models stop and anomaly detection start?

Supervised models are the workhorse wherever there are labels, because they learn what known fraud looks like. Their blind spot is a new pattern with no labels yet.

Anomaly detection, such as an isolation forest or the reconstruction error of an autoencoder, flags behavior that is unusual for the customer or the population, without labels. On its own it raises many false positives, because unusual is not the same as fraudulent.

So I use it as a complement: anomaly scores as input features, and as a monitoring tool that surfaces emerging clusters for analysts to review and label.

### Outliers {#outliers}

**Question.** How do you detect and handle outliers?

First I ask what the outlier is: a data error, a rare but real value, or the thing I am looking for. In fraud it is often the third, so I never delete extremes by rule.

**Detecting.** For one variable I use the median and the median absolute deviation (MAD), or the interquartile range, because the mean and the standard deviation are themselves pulled by the outlier. Amounts are heavily right-skewed, so I look on a log scale. For several variables together, I use an isolation forest or the distance to the nearest neighbors. And I compare a value with the customer's own history as well as with the population: \$3,000 is ordinary for one account and extreme for another.

**Handling, by cause.**

- **Errors**, such as a negative amount, an impossible timestamp or a duplicated row: fix or remove them at the source, and count them. A jump in the error count is a data-quality alert.
- **Real extremes:** keep them. Trees split on order, so a huge value does not distort them. For linear models and neural networks I cap or log-transform the feature, with the caps learned on the training window only.
- **Metrics:** a few very large frauds can dominate anything measured in dollars, so I report results by count and by value.

The [robust z-score exercise](#robust-z) codes the per-account version.

### Reinforcement learning and bandits {#bandits}

**Question.** Where could reinforcement learning fit in fraud?

The scoring model is supervised, and I would keep it that way. Reinforcement learning fits the decision that follows the score, because that decision changes what we observe next.

Three places stand out. A contextual bandit can choose among interventions (approve, step-up authentication, hold, decline), with fraud loss avoided minus the cost of friction as the reward. The control group is already an exploration policy, and its size is an exploration-exploitation choice. And decisions over an account's lifetime, such as when to raise limits, are sequential.

The hard parts are the ones that make reinforcement learning risky in production. The reward arrives weeks late. Exploration costs real money. And a new policy has to be evaluated offline, from logs collected under the old policy, before it touches customers. So I would start with a bandit on a small slice of traffic with guardrails, not a full sequential policy.

### Feature families {#features}

**Question.** What features would you build for transaction fraud?

Five families:

1. **Transaction:** amount, channel, merchant category, time of day.
2. **Velocity:** counts and sums per account, card, device and payee over one hour, one day and seven days.
3. **Deviation from the customer's own baseline:** amount against their usual, new merchant, new geography, unusual hour.
4. **Entity and relationship:** how many accounts share this device or IP address, how old the payee relationship is, links to confirmed fraud.
5. **Account state:** tenure, recent changes to contact details or password, recent failed logins.

All of them are computed point-in-time, and all must be available inside the latency budget at decision time. The [first coding exercise](#velocity) builds the velocity family.

### Feature selection {#feature-selection}

**Question.** You have a few thousand candidate features. How do you decide which ones go into the model?

With boosted trees I treat selection as a question of stability and cost more than of accuracy, because the model tolerates irrelevant features fairly well. I work in five passes:

1. **Remove what cannot be used.** Anything not available at decision time, anything not permitted for this decision, and fields that are constant or almost always missing.
2. **Remove what will not last.** Features whose distribution moves a lot from month to month (PSI), or whose relationship with fraud changes sign across time folds.
3. **Thin out redundancy.** Among highly correlated features I keep the one that is cheapest to serve and easiest to explain.
4. **Rank by contribution on later data.** Permutation importance or mean absolute SHAP on an out-of-time sample. Importance counted from training splits is biased toward features with many distinct values, as [Strobl and colleagues](https://pmc.ncbi.nlm.nih.gov/articles/PMC1796903) showed for random forests.
5. **Confirm by ablation.** Drop a group, retrain, and compare at the operating point. If recall at the alert budget does not move, the group goes.

The textbook answer sorts the methods into three families:

| Family | Examples | Where I use it |
|---|---|---|
| Filter | Missing rate, variance, univariate AUC, correlation | A cheap first cut |
| Embedded | L1 penalty, tree gain, SHAP | The main ranking |
| Wrapper | Recursive elimination, forward selection, ablation | The final check on a short list, because each step retrains |

Two cautions. Selection is part of training, so it happens inside each training fold; choosing features on all the data and then cross-validating leaks. And a feature that is weak on its own can matter in an interaction, so the filter pass should be gentle.

The payoff is mostly operational: fewer pipelines to break, lower serving latency, and less to monitor and to document for model review.

### L1 and L2 regularization {#regularization}

**Question.** What is the difference between L1 and L2 regularization?

L2 penalizes squared weights. It shrinks all coefficients smoothly toward zero and shares weight among correlated features.

L1 penalizes absolute values. Its constraint region has corners on the axes, so the optimum often lands with some coefficients exactly zero. It selects features, and it picks somewhat arbitrarily among correlated ones.

In Bayesian terms, L2 is a Gaussian prior on the weights and L1 is a Laplace prior, and the penalized fit is the maximum a posteriori (MAP) estimate. Both trade a little bias for lower variance.

Elastic net mixes the two penalties, which keeps the sparsity of L1 and treats groups of correlated features more evenly. The inputs have to be on the same scale first, or the penalty treats features unequally.

A penalty is only one kind of regularization. Early stopping, shrinkage, row and column subsampling, a minimum leaf size, dropout and data augmentation all limit how closely a model can fit its training data.

### Classic ML quick checks {#classic-ml}

Short answers to the textbook questions that tend to open a panel:

| Question | Answer |
|---|---|
| How does a decision tree choose a split? | For each feature and candidate cut it computes the drop in impurity (Gini or entropy for classification, variance for regression), weighted by the size of the children, and takes the largest. It is greedy: the best split now, with no look-ahead. Depth limits, a minimum leaf size and pruning keep it from memorizing. |
| Underfitting or overfitting? | Underfitting: training and validation error are both high (bias). Overfitting: training error is low and validation error is much higher (variance). The first needs more capacity or better features. The second needs more data, regularization or a simpler model. |
| Why is logistic regression still used in risk? | Its coefficients read as log-odds, monotonic effects are easy to enforce and to explain, it is stable on little data, and its scores are usually close to calibrated. It is the usual baseline, and often the model a tree ensemble has to beat. |
| How do you choose k in k-means? | There is no label to optimize, so I combine the elbow in within-cluster variance, the silhouette score, whether the clusters stay put across resamples, and whether the business can use the segments. Standardize first, because k-means uses Euclidean distance. |
| Validation data or test data? | Validation data is used to make choices (hyperparameters, features, thresholds), so its score is optimistic. Test data is used once, after every choice is made. In fraud it comes from a later period. |
| A dataset does not fit in memory in Python. What do you do? | Read only the columns I need, from a columnar format such as Parquet. Use compact dtypes and categoricals. Process in chunks, or push the aggregation down to SQL or Spark. Vectorize instead of looping. Develop on a sample that keeps all the fraud. |

### Deep learning quick checks {#deep-learning}

Short answers for the questions that follow when the interviewer wants to test fundamentals outside trees:

| Question | Answer |
|---|---|
| Explain self-attention. | Each token builds a query, a key and a value. Its output is a weighted sum of all values, with weights from the softmax of query-key dot products scaled by the square root of the dimension. Every position can use any other position in one step. |
| Local minima or saddle points? | In high dimensions most zero-gradient points are saddles, not bad minima. The noise in stochastic gradient descent (SGD) and momentum help escape them. Adaptive methods such as Adam rescale directions with small gradients. |
| Why does batch normalization help? | It keeps layer inputs in a stable range, which allows higher learning rates and smoother optimization. Batch noise adds mild regularization. |
| Adam or SGD? | Adam adapts the step size per parameter and converges fast with little tuning. SGD with momentum often generalizes a little better in vision when tuned. AdamW decouples weight decay from the adaptive step. |

## Case walk-throughs {#cases}

### A seven-step frame {#frame}

The cases use the same frame. I say the frame first, then fill it in.

1. **Clarify:** which fraud, what action follows the score, the latency budget and the review capacity.
2. **Target and label:** the unit of prediction, where the label comes from, how long it takes to mature.
3. **Data and features:** feature families, built point-in-time.
4. **Model:** a baseline, the main model, later additions.
5. **Evaluation:** out-of-time data, metrics, segments, comparison with the current model.
6. **Decision and deployment:** thresholds, actions, staged rollout.
7. **Monitoring:** drift, performance, retraining triggers.

### Case 1: real-time transaction fraud {#case-transactions}

**Prompt.** Design a model that detects fraudulent card or digital payment transactions in real time for a consumer bank.

**Clarify.** I ask three questions. Which fraud are we targeting: unauthorized use, such as a stolen card or an account takeover, or scams in which the customer authorizes the payment? What happens after the score: a decline, step-up authentication, or a hold for review? And what are the latency budget and the review capacity? Without answers, I state my assumptions: unauthorized card-not-present transactions, a decision within about a hundred milliseconds, and three possible actions (approve, step up, decline).

**Target and label.** The unit is one authorization request. The label is whether that transaction was later confirmed as unauthorized through a customer claim or dispute. I train only on transactions old enough for labels to mature, and I tag friendly fraud separately.

**Data and features.** The [five families](#features) above. Every feature is computed as of the transaction time. Fast-moving aggregates come from a streaming store; heavy ones are precomputed in batch.

**Model.** I start with a baseline of logistic regression plus the existing rules, which gives a floor and a sanity check. The main model is gradient-boosted trees on all the fraud and down-sampled non-fraud, with probability correction afterwards. Later I test sequence or graph embeddings as extra features and keep them only if they add lift.

**Evaluation.** On an out-of-time test set I report PR-AUC, recall and value detection rate at the false positive ratio the business accepts, and calibration. I break the results down by channel, amount band and customer tenure, and I compare against the current production model on the same sample.

**Decision and deployment.** The score maps to amount-dependent thresholds and three actions. The top band is declined. The middle band gets step-up authentication, which costs a good customer far less than a decline. The rest is approved. I roll out in shadow mode first, scoring without acting, then on a small share of traffic, and I keep a small control group to keep collecting unbiased labels.

**Monitoring.** Daily: score distribution, alert rate, feature drift and data-quality checks. Once labels mature: recall, value detection and false positive ratio by segment. I agree retraining triggers in advance and build a feedback channel from investigators, who see new patterns first.

> **Takeaway.** The trade-off to keep in front of the business is fraud loss against customer friction against operations cost. The model ranks risk; where to cut is a business decision.

### Follow-ups to case 1 {#case-1-follow-ups}

**Fraudsters adapt after you deploy. How do you design for that?** I assume the model starts decaying on day one. So I plan frequent retraining on recent data, anomaly detection to surface clusters the model has not seen, rules as a fast patch while a retrain is in progress, and signals that are harder to fake, such as device and behavior, alongside transaction fields. The control group matters here too, because it shows what fraud looks like when we do not intervene.

**Offline results are strong, but in shadow mode the model raises far more alerts than forecast and catches no more fraud. What do you check?** Four things, in this order:

1. **Training-serving skew.** Is each feature computed the same way online, and are missing values defaulting to something odd? I compare distributions feature by feature and re-score a recent sample offline using the logged online features.
2. **Leakage offline.** A feature that used information from after the event would explain strong offline numbers that vanish live.
3. **Calibration.** If I down-sampled the negatives and set the threshold on uncorrected scores, the alert volume is inflated.
4. **Population shift** between the training window and today.

**How would you estimate the dollar value of the model?** Incremental fraud dollars prevented against the current model at the same false positive ratio, minus the cost of any extra false positives and step-ups: lost transactions, customer contacts, review time. The clean way to measure it is a champion-challenger test randomized by account and read out after the labels mature. The [A/B testing note](/notes/ab-testing/) covers the design.

**How is a scam different, when the customer authorizes the payment?** The customer's own device and credentials are used, so device and authentication signals look clean. The signal moves to the payee side and to behavior: a first-time payee, a recipient account that receives from many unrelated senders and cashes out quickly, an amount far outside the customer's pattern. The intervention changes too: a warning or a confirmation step fits better than a hard decline. Labels are murkier, so I would model scams as their own target.

### Case 2: new-account application fraud {#case-applications}

**Prompt.** A new checking-account application arrives online. Build a model that decides whether it is fraudulent.

**What changes from case 1.** Three things. There is no history for this customer at the bank, so it is a cold start. The label takes months, not weeks, because it comes from what the account does later. And we never see outcomes for applications we reject.

**Clarify.** Which fraud are we most worried about: identity theft using a real person's details, a synthetic identity, or a real applicant who intends to defraud or to act as a mule? And which actions are available: approve, reject, manual review, or approve with restrictions?

**Target and label.** The unit is one application. The label is whether the account was closed or charged off for fraud within a fixed window after opening. I keep the fraud types as separate labels where the data allows.

**Data and features.** Four families:

1. **Identity consistency:** do the name, date of birth, address, phone and email belong together, and how long have they been seen together?
2. **Device and network:** has this device or IP address appeared on earlier applications, and how many applications came from it recently?
3. **Application behavior:** time to complete the form, pasted fields, time of day.
4. **Links to known fraud:** attributes shared with accounts previously closed for fraud, which is a graph problem.

I would also use external identity and bureau data where the bank is permitted to, and work with compliance on which variables are allowed in an account-opening decision.

**Model and evaluation.** Gradient-boosted trees again, validated out of time. Because the base rate is low and review capacity is fixed, the key metrics are precision and recall within the volume reviewers can handle, plus fraud dollars avoided.

**Decision.** Four bands: approve automatically, approve with restrictions such as lower limits and early-life monitoring, manual review sized to capacity, and reject automatically only at very high precision. To handle the missing outcomes for rejected applications, I approve a small random sample of the borderline band under restrictions, or use manual-review outcomes as labels.

**Monitoring.** Application volume and score distribution by channel, since attacks arrive in bursts from one channel or device cluster. Then the early-life behavior of approved accounts, as a leading indicator before the final label matures.

### Follow-ups to case 2 {#case-2-follow-ups}

**How do the signals differ between identity theft and a synthetic identity?** With identity theft, the identity is real and has a long history, but the contact details and device do not match that history. The victim eventually complains, so a label arrives. With a synthetic identity, the elements have never been seen together, the history is thin or recent, and several identities may share a phone, address or device. No victim complains, so the loss is often booked as a credit loss and the fraud label is missing.

**Fraud is 0.5% of applications and the team can manually review 2% of them. How do you use the queue?** I rank by expected loss, not by probability alone. The very top goes to automatic rejection only if precision there is high enough. The next slice, sized to the 2% capacity, goes to manual review. I track precision in the queue and recall overall, feed reviewer decisions back as labels, and send a small random sample to review as well, so the labels are not biased toward what the model already believes.

### Case 3: a sudden spike in transactions {#case-spike}

**Prompt.** Transaction volume jumps far above forecast within an hour. How do you investigate and respond?

I take five questions in order.

**Is it real?** I rule out a measurement problem first: events replayed after a retry, a batch that posted late, a time-zone or pipeline change. I compare the number of rows with the number of distinct transaction IDs, and one data source with another.

**Where is it?** I break the increase down by channel, merchant, device, IP range, card range, geography, amount band, and new against existing accounts. A rise spread across everything usually has an ordinary cause, such as a holiday, payday or a promotion. A rise that sits in a few merchants, devices or one amount band looks like an attack.

| Segment | Usual per hour | This hour | Change |
|---|---:|---:|---:|
| All transactions | 50,000 | 70,000 | +20,000 |
| One merchant, amounts under \$2 | 300 | 19,300 | +19,000 |
| Everything else | 49,700 | 50,700 | +1,000 |

In this made-up example, volume is up 40%, and 95% of the increase is one merchant and one amount band. Everything else is up 2%.

**Does it look like fraud?** I match the concentration to known patterns. Many small authorizations across many cards at one merchant, with a high decline rate, is card testing. One device or IP range logging in to many accounts is credential stuffing ahead of account takeover. A burst of first-time payees is a scam or mule campaign. Labels will take weeks, so I read the early signals: the score distribution inside the spike, authorization declines, and customer contacts.

**What do I do now?** Something proportionate to the segment: a velocity rule, step-up authentication or a rate limit on the affected merchant, device cluster or card range, and a call to fraud operations. I avoid a blanket block, which mostly hits good customers. If the spike is legitimate, I check instead that the review queue and the service can absorb it, and that velocity features are not turning a normal peak into false alerts.

**What changes afterwards?** I label the incident, measure what the stopgap rule cost in false positives, add the pattern to the features or the next retrain, add the alert that would have fired earlier, and retire the rule on a date.

> **Takeaway.** Concentration is the tell: an ordinary peak lifts every segment, and an attack lifts a few.

### A neighboring case: credit risk for loan applicants {#case-credit}

**Prompt.** How would you assess credit risk for loan applicants using data?

The frame is the same and the target is different. I predict the probability of default: for example, that the borrower becomes 90 days past due within the first 12 to 24 months.

- **Data.** Credit bureau history (payment record, utilization, length of history, recent inquiries), income and existing debt, the application itself, and the applicant's relationship with the bank, such as deposits and cash flow.
- **Model.** A logistic regression scorecard as the baseline, and often as the production model. Gradient-boosted trees with monotonic constraints as the challenger.
- **Evaluation.** Out of time. AUC or the Kolmogorov-Smirnov statistic for ranking, and calibration by score band, because the probability feeds pricing and loss estimates.
- **Decision.** Approve or decline, then the limit and the price. Expected loss is the probability of default times the loss given default times the exposure.
- **Missing outcomes.** Repayment is observed only for approved applicants, the same selection problem as in [fraud labels](#labels). Credit modeling calls the remedies reject inference.
- **Explanations.** In the United States a lender must tell a declined applicant the main reasons, and may not use protected characteristics. Reason codes and fair-lending tests are requirements, so explainability is part of the design.

| | Fraud | Credit risk |
|---|---|---|
| What is predicted | The payment or the applicant is not genuine | A genuine customer will not repay |
| Base rate | About 0.1% of transactions | Often a few percent of accounts |
| Label delay | Days to weeks | Months to years |
| Adversary | Yes, and it adapts | No, but the economy shifts |
| Time to decide | Milliseconds | Seconds to days |
| Modeling habits | Boosted trees, frequent refresh | Scorecards or constrained trees, slow refresh, reason codes |

## System design: a real-time fraud detection system {#system-design}

[Case 1](#case-transactions) designs the model. A system design round asks for the service around it, usually as "design a real-time fraud detection system for a bank's payments". I answer in five steps: requirements, the two paths, the latency budget, failure handling and monitoring.

### Requirements first {#sd-requirements}

| What I ask | What I assume if nobody answers |
|---|---|
| Where does the decision sit? | Inline, before the payment is authorized |
| How fast? | About 100 ms end to end at the 99th percentile |
| How much traffic? | Thousands of requests per second at peak, with holiday bursts |
| What if the model is unavailable? | The payment still gets an answer |
| Which actions? | Approve, step up, decline, or hold for review |
| What must be kept? | Enough to reproduce any decision later: inputs, model version, rules version |

### A fast path and a slow path {#sd-paths}

![A payment request passes through feature lookup, scoring and a decision in about 100 milliseconds. Below, an event log feeds stream updates to the features and, after labels mature, retraining of the model, while monitoring watches both.](fast-and-slow-path.svg "Figure 1: The top row answers one payment. The bottom row feeds it: counters within seconds, a retrained model after labels mature.")

The fast path answers one payment:

1. **Decision service.** Receives the authorization request, enforces the time budget, and always returns an answer.
2. **Feature retrieval.** Request fields, plus precomputed features fetched by key from a low-latency online store. Slow-moving aggregates, such as a 90-day average or graph features, are refreshed in batch. Fast-moving ones, such as the count in the last hour, are updated by a stream processor within seconds.
3. **Scoring.** The model is loaded in memory with a pinned version. Boosted trees score in a few milliseconds or less.
4. **Rules and policy.** Hard rules first, such as blocked lists and limits. Then thresholds on the score, by amount and segment, map to an action.
5. **Logging.** The features as served, the score, the versions and the action go to an event log, off the critical path.

The slow path learns from what happened:

1. **Stream updates.** A stream processor reads the event log and updates the counters that the next request will read.
2. **Labels.** Disputes, chargebacks and analyst outcomes are joined to the logged events by transaction ID, weeks later.
3. **Training data.** Built from the logged features, so it is point-in-time by construction and matches what serving saw.
4. **Release.** Out-of-time validation, independent review, shadow mode, a small share of traffic, then full traffic, with the previous version one step away.

### A latency budget {#sd-latency}

| Step | Budget |
|---|---:|
| Network and request parsing | 10 ms |
| Feature lookups, in parallel | 30 ms |
| Model scoring | 10 ms |
| Rules and decision | 10 ms |
| Headroom for retries and the tail | 40 ms |
| **Total** | **100 ms** |

The numbers are illustrative. Two points carry over to any real system: feature retrieval, not the model, usually takes the most time, and the budget is for the slowest 1% of requests, not the average.

### When something fails {#sd-failure}

| Failure | Response |
|---|---|
| A feature lookup times out | Use a default plus a "missing" flag that the model saw in training, and count it |
| The streaming counters fall behind | Serve the last value with its age, and alert on the lag |
| The model is down or slow | Fall back to rules alone, with conservative limits on large amounts |
| The same request arrives twice | An idempotency key returns the first decision, and the counters are not updated twice |
| A bad model release | Shadow mode first, then a small share of traffic, with a one-step rollback |
| A traffic burst | Stateless decision services scaled out; the store partitioned by account or card |

### What to monitor after launch {#sd-monitoring}

| Layer | What I watch | Visible within |
|---|---|---|
| Service | Latency at the median and the 99th percentile, error and timeout rate, fallback rate | Minutes |
| Data | Missing rate, feature freshness, feature drift (PSI) | Hours |
| Score | Score distribution, alert rate, decline and step-up rate by segment | Hours |
| Outcome | Recall, value detection rate, false positive ratio, calibration | Weeks, once labels mature |
| Business | Fraud loss in basis points of volume, customer contacts and complaints, review backlog | Weeks |

The first three layers are the early warning, because the outcome layer waits for labels. Each alert has an owner and an action agreed in advance.

### Follow-ups to the design {#sd-follow-ups}

**Why not call a large neural network or a language model inline?** Latency, cost and reproducibility. The inline decision has milliseconds, and it must give the same answer when replayed. Heavy models can still contribute offline. They produce embeddings or text-derived features that the fast model reads from the store, or they support case review after the decision.

**How do you keep training and serving consistent?** One definition per feature, shared by the batch and the streaming computation. Training data comes from the features logged at serving time. And a replay test recomputes a sample offline and compares it with the logged values.

**Where do people fit?** The middle band of scores goes to a review queue sized to capacity. Reviewer outcomes become labels. Reviewers are also the first to see a new pattern, so the system needs a channel for them to report it.

**How do you change a threshold safely?** Scores are logged, so I can replay last month's traffic through the new threshold and see the alert volume and the decline rate before anything changes. Then the change is staged like a model release.

## Probability and statistics drills {#drills}

Every number below is printed by the [lab script](#coding).

### Precision at a low base rate {#drill-bayes}

**Question.** The fraud rate is 0.1%. The model catches 90% of fraud and flags 1% of legitimate transactions. A transaction is flagged. How likely is it to be fraud?

About 8%. Take a million transactions: there are 1,000 frauds and the model catches 900. Of the 999,000 legitimate ones it flags 9,990. So 900 of 10,890 alerts are fraud, which is 8.3%, or about 11 false alerts per fraud caught.

$$
P(\text{fraud} \mid \text{alert}) = \frac{0.001 \times 0.9}{0.001 \times 0.9 + 0.999 \times 0.01} = \frac{0.0009}{0.01089} \approx 8.3\%
$$

If the FPR falls to 0.1%, precision rises to 47%. At a low base rate, precision is driven by the FPR; the [precision and recall note](/notes/precision-vs-recall/) draws the whole curve.

### An interval for recall {#drill-interval}

**Question.** On a test set with 400 confirmed frauds, recall is 80%. How sure are you?

The standard error is 2 points, so the 95% interval runs from about 76% to 84%.

$$
SE = \sqrt{\frac{r(1-r)}{n_{\text{fraud}}}} = \sqrt{\frac{0.8 \times 0.2}{400}} = 0.02
$$

The precision of a recall estimate depends on the number of frauds, not the number of rows. Narrowing the interval to plus or minus 2 points takes about 1,540 frauds.

### Champion against challenger {#drill-ztest}

**Question.** The challenger's alert precision is 12% on 5,000 alerts. The champion's is 10% on 5,000 alerts. Is the difference real?

A two-proportion z-test gives $z \approx 3.2$ and a two-sided p-value of about 0.001, so chance is an unlikely explanation.

$$
z = \frac{0.12 - 0.10}{\sqrt{0.11 \times 0.89 \times \left(\frac{1}{5000} + \frac{1}{5000}\right)}} \approx 3.20
$$

Before trusting it I check four things: accounts were randomized to each arm, the labels are equally mature, both arms cover the same period, and precision was the metric chosen in advance, not the best of several.

### A threshold from costs {#drill-cost}

**Question.** A false decline costs $C$. A missed fraud costs the transaction amount $L$. At what probability should you intervene?

Intervene when the expected fraud loss exceeds the expected cost of a false positive:

$$
p \cdot L > (1-p) \cdot C \;\Rightarrow\; p^* = \frac{C}{L + C}
$$

The larger the amount, the lower the threshold. With a false decline costed at \$5 for illustration:

| Amount $L$ | False-positive cost $C$ | Threshold $p^*$ |
|---:|---:|---:|
| \$50 | \$5 | 9.1% |
| \$500 | \$5 | 1.0% |
| \$5,000 | \$5 | 0.1% |

This is the two-class case of the cost-sensitive decision rule in [Elkan's paper](https://mlanthology.org/ijcai/2001/elkan2001ijcai-foundations/).

### Undoing down-sampling {#drill-downsampling}

**Question.** You trained on all the fraud and 1% of the non-fraud. The model says 0.5. What is the real probability?

About 1%. Keeping only a share $\beta$ of the negatives multiplies the odds by $1/\beta$, so I multiply the odds back by $\beta$. Odds of 1 become odds of 0.01, which is a probability of 0.0099.

$$
p = \frac{\beta \, p_s}{\beta \, p_s + 1 - p_s}
$$

Here $p_s$ is the score from the model trained on the sample. A check on base rates: with a true fraud rate of 0.2% and 2% of the negatives kept, the fraud rate in the training sample is 9.1%, and the formula maps 9.1% back to 0.2%. [Dal Pozzolo and colleagues](https://doi.org/10.1109/SSCI.2015.33) derive the correction and show that down-sampling changes calibration but not the ranking.

### Transactions until a fraud is likely {#drill-geometric}

**Question.** At a 0.1% fraud rate, how many transactions do you need for a 95% chance of seeing at least one fraud?

About 3,000. The chance of no fraud in $n$ transactions is $0.999^n$, and that drops below 5% at $n = 2{,}995$. With only 1,000 transactions the chance is 63%.

$$
1 - 0.999^{\,n} \ge 0.95 \;\Rightarrow\; n \ge \frac{\ln 0.05}{\ln 0.999} \approx 2995
$$

### What AUC means {#drill-auc}

**Question.** What does AUC mean?

ROC-AUC is the probability that a randomly chosen fraud gets a higher score than a randomly chosen legitimate transaction. A random model scores 0.5. For the precision-recall curve, the random baseline is the prevalence, here 0.001, which is why PR-AUC says more when positives are rare. The [third coding exercise](#auc) computes ROC-AUC from that definition.

## Coding exercises {#coding}

Download [fraud-practice-lab.py](/notes/fraud-detection-practice/fraud-practice-lab.py) and run it with Python 3.10 or later, NumPy and pandas. It generates its own data and prints every number in this note. If scikit-learn or DuckDB is installed, it also cross-checks the AUC and the SQL queries.

```bash
python3 fraud-practice-lab.py
```

When I talk through one of these, I restate the problem and its edge cases, say the approach, write the code, walk a small example, and finish with the complexity and what changes in production.

### Point-in-time velocity features {#velocity}

**Prompt.** Given transactions with `account_id`, `ts` and `amount`, add for each transaction: the count and total amount of the same account's transactions in the previous 24 hours, the seconds since the previous transaction, and the amount relative to the account's prior average. No feature may use the current row or any later row.

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

The output on six transactions:

| account_id | ts | amount | cnt_24h | sum_24h | amt_to_prior_mean |
|---|---|---:|---:|---:|---:|
| A | 12-01 09:00 | 20 | 0 | 0 | NaN |
| A | 12-01 09:30 | 30 | 1 | 20 | 1.50 |
| A | 12-02 08:59 | 500 | 2 | 50 | 20.00 |
| A | 12-02 09:31 | 40 | 1 | 500 | 0.22 |
| B | 12-01 10:00 | 100 | 0 | 0 | NaN |
| B | 12-03 10:00 | 100 | 0 | 0 | 1.00 |

Three things to say out loud. `closed="left"` keeps the current row out of the window, which is what prevents leakage. The fourth row's window starts at 12-01 09:31, so it counts only the \$500 transaction. And in production these features come from a feature store, with one definition shared by training and serving.

### The same feature in SQL and PySpark {#velocity-sql}

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

The SQL is DuckDB syntax; other engines write the interval differently. `rangeBetween` selects by the value of the ordering key and `rowsBetween` by a number of rows, a difference that often comes up as a follow-up. The [Spark note](/notes/spark-pyspark/) has a full PySpark lab.

### Metrics at a fixed alert budget {#alert-budget}

**Prompt.** Given labels, scores and amounts, flag the top 1% of transactions by score and return precision, recall, value detection rate and the false positive ratio.

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

A small example to check by hand: ten transactions hold two frauds, of \$500 and \$100. Flagging the top 20% catches the \$500 fraud and one legitimate transaction. Precision is 0.5, recall is 0.5, the value detection rate is 0.83 and the false positive ratio is 1. Detection by value is higher than detection by count, which is the reason to report both.

### ROC-AUC from ranks {#auc}

**Prompt.** Compute ROC-AUC without scikit-learn.

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

This is the Mann-Whitney form: the rank sum of the positives, minus the smallest rank sum they could have, divided by the number of positive-negative pairs. Ties get the average rank. Sorting sets the cost at $O(n \log n)$. On the ten-transaction example it returns 0.9375, which is 15 of 16 pairs in the right order, and on 100,000 simulated rows it matches scikit-learn.

### Population stability index {#psi}

**Prompt.** Compute the PSI between a baseline score distribution and a recent one.

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

The bin edges come from the quantiles of the baseline, the two outer bins are open-ended, and empty bins get a small floor so the logarithm stays finite. To get a feel for the scale, I shifted the mean of 50,000 standard normal draws:

| Mean shift of the recent sample | PSI |
|---|---:|
| none | 0.00 |
| 0.25 standard deviations | 0.06 |
| 0.5 standard deviations | 0.25 |
| 1 standard deviation | 0.92 |

So the conventional alarm level of 0.25 corresponds to about half a standard deviation of mean shift.

### Undoing down-sampling in code {#downsampling-code}

```python
import numpy as np

def correct_for_downsampling(p_sampled, beta):
    """beta = share of negatives kept in training."""
    p_sampled = np.asarray(p_sampled, dtype=float)
    return beta * p_sampled / (beta * p_sampled + 1 - p_sampled)
```

`correct_for_downsampling(0.5, 0.01)` returns 0.0099 and `correct_for_downsampling(0.9, 0.01)` returns 0.083.

### Duplicate transactions in SQL {#duplicates}

**Prompt.** Write a query that finds duplicate transactions.

I ask what "duplicate" means. The same transaction ID twice is a pipeline problem, and `GROUP BY transaction_id HAVING COUNT(*) > 1` finds it. The more useful case is a repeat: the same account, merchant and amount within a short time of the previous one.

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

The same in pandas:

```python
def flag_near_duplicates(df, seconds=60):
    keys = ["account_id", "merchant_id", "amount"]
    df = df.sort_values(keys + ["ts"]).reset_index(drop=True)
    gap = df.groupby(keys)["ts"].diff().dt.total_seconds()
    df["near_duplicate"] = gap.le(seconds)   # the first row of each group has no gap -> False
    return df
```

On seven test rows, both versions flag two: the \$25 payments at 10:00:20 and 10:00:50, which follow one at 10:00:00 at the same merchant. A fourth payment five minutes later is not flagged, and neither are rows that differ in merchant, account or amount.

Three things to say out loud. Each row is compared with the one before it, so a run of repeats is flagged from the second row on and the first is kept. A repeat is not always an error: two coffees in a minute are real. And the pipeline itself should deduplicate on an idempotency key, not on matching fields.

### Amounts that are unusual for the account {#robust-z}

**Prompt.** Write a function that flags transactions whose amount is unusual for the account, using only that account's earlier transactions.

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

This is the [modified z-score](https://itl.nist.gov/div898/handbook/eda/section3/eda35h.htm): the distance from the median in units of the MAD, scaled by 0.6745 so that it reads like an ordinary z-score, with 3.5 as the usual cutoff. Here are one account's last four transactions, after five ordinary ones between \$20 and \$30:

| Amount | Robust z | Flagged | Ordinary z |
|---:|---:|---|---:|
| \$21 | −1.03 | no | −0.87 |
| \$900 | 29.28 | yes | 25.68 |
| \$26 | 0.41 | no | −0.30 |
| \$850 | 18.90 | yes | 2.42 |

The last row is the reason to use the median. Once the \$900 payment is in the history, the mean and the standard deviation are inflated, and an ordinary z-score rates \$850 at 2.4, under a cutoff of 3. The median and the MAD barely move, so the robust score still flags it. Both scores are computed on log amounts.

Four things to say out loud. `shift()` keeps the current row out of its own baseline. A new account has no baseline, so its first rows get no score and need a peer-group fallback. The floor on the MAD avoids dividing by zero when every earlier amount is the same. And unusual is not fraudulent: this is a feature or a review queue, not a decision.

### Warm-ups {#warmups}

Some data science interview loops include one or two easy algorithm questions. These six cover the patterns that come up most: a hash map, sorting, a running best, and a sliding window.

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

| Function | Idea | Time | Check |
|---|---|---|---|
| `two_sum` | Remember each value's index and look up the complement | $O(n)$ | `two_sum([40, 15, 25, 60], 85)` returns `(2, 3)` |
| `group_anagrams` | Sorted letters are the group key | $O(n \cdot k \log k)$ | `listen`, `silent` and `enlist` share a group |
| `max_subarray` | Extend the current run or start a new one | $O(n)$ | `[3, -4, 5, -1, 2, -6, 4]` gives 6 |
| `max_profit` | Add up every rise | $O(n)$ | `[10, 7, 12, 9, 15, 14]` gives 11 |
| `min_abs_difference` | After sorting, the closest pair is adjacent | $O(n \log n)$ | `[31, 8, 20, 14]` gives 6 |
| `longest_unique_substring` | Move the window start past the last repeat | $O(n)$ | `"fraudfree"` gives 6 |

The edge cases to name before coding: no valid pair, all-negative numbers, prices that only fall, and an empty string.

## One-page review {#review}

| Item | In one line |
|---|---|
| Precision at a low base rate | At a 0.1% fraud rate, 90% recall and 1% FPR, precision is 8.3%: about 11 false alerts per fraud caught |
| Why accuracy fails | Predicting "legitimate" for everything scores 99.9% |
| Baselines | ROC-AUC 0.5; PR-AUC equals the prevalence |
| Down-sampling correction | $p = \beta p_s / (\beta p_s + 1 - p_s)$, with $\beta$ the share of negatives kept |
| Threshold from costs | $p^* = C / (L + C)$: the larger the amount, the lower the threshold |
| PSI | $\sum (a - e)\ln(a / e)$; below 0.1 stable, 0.1 to 0.25 watch, above 0.25 large |
| Interval for recall | $\pm 1.96\sqrt{r(1-r)/n_{\text{fraud}}}$: it depends on the number of frauds |
| `scale_pos_weight` | Negatives divided by positives: about 999 at a 0.1% fraud rate |
| Model governance | SR 26-2 replaced SR 11-7 on April 17, 2026 |
| Bagging and boosting | Bagging averages independent deep trees and cuts variance; boosting adds shallow trees in sequence and cuts bias |
| Feature selection | Usable, stable, not redundant, ranked on later data, confirmed by ablation |
| Real-time design | A fast path that decides in about 100 ms and a slow path that learns; rules as the fallback |
| A spike in volume | First real or artifact, then where it sits: an ordinary peak lifts every segment, an attack lifts a few |

Phrases worth having ready: point-in-time features, out-of-time validation, label maturity, control group, champion-challenger, shadow mode, value detection rate, false positive ratio, step-up authentication, training-serving skew, idempotency key, fallback, feature freshness.

## References {#references}

1. Dal Pozzolo A, Caelen O, Johnson RA, Bontempi G. [Calibrating probability with undersampling for unbalanced classification](https://doi.org/10.1109/SSCI.2015.33). IEEE Symposium Series on Computational Intelligence 2015, 159–166.
2. Elkan C. [The foundations of cost-sensitive learning](https://mlanthology.org/ijcai/2001/elkan2001ijcai-foundations/). IJCAI 2001, 973–978.
3. Chen T, Guestrin C. [XGBoost: a scalable tree boosting system](https://arxiv.org/abs/1603.02754). KDD 2016.
4. Lundberg SM, Lee S-I. [A unified approach to interpreting model predictions](https://arxiv.org/abs/1705.07874). NeurIPS 2017.
5. Board of Governors of the Federal Reserve System. [SR 26-2: Revised guidance on model risk management](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm). April 17, 2026.
6. Bowman MW. [Artificial intelligence in the financial system](https://www.federalreserve.gov/newsevents/speech/bowman20260501a.htm). Speech, May 1, 2026.
7. Strobl C, Boulesteix A-L, Zeileis A, Hothorn T. [Bias in random forest variable importance measures: illustrations, sources and a solution](https://pmc.ncbi.nlm.nih.gov/articles/PMC1796903). BMC Bioinformatics 2007, 8:25.
8. NIST/SEMATECH. [e-Handbook of statistical methods, section 1.3.5.17: Detection of outliers](https://itl.nist.gov/div898/handbook/eda/section3/eda35h.htm). It gives the modified z-score of Iglewicz and Hoaglin (1993).

Every number in this note is printed by [`fraud-practice-lab.py`](/notes/fraud-detection-practice/fraud-practice-lab.py). The SQL and PySpark versions of the velocity feature were checked against the pandas output with DuckDB 2.0 and PySpark 4.3, and the duplicate-transaction query with DuckDB 2.0.
