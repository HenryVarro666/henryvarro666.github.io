---
title: "Fraud detection modeling: practice questions with worked answers"
date: 2026-11-23T20:00:00-06:00
lastmod: 2026-11-23T20:00:00-06:00
description: "Practice questions on fraud modeling with worked answers: class imbalance, metrics, time-based validation, late labels, thresholds from costs, two case walk-throughs, probability drills, and tested Python, SQL and PySpark code."
topics: [data-science]
---
I put this set together while preparing for applied machine learning (ML) interviews on fraud and risk modeling. It has four parts: fundamentals as they show up in fraud, two case walk-throughs, probability drills, and coding exercises with tested code. Each answer is written the way I would say it: the conclusion first, then the reason, the trade-off, and how I would check it.

> **Takeaway.** Most fraud modeling questions come back to four habits: measure at the operating point on later data, build every feature as of decision time, treat labels as late and biased, and let costs set the threshold.

## How to use this set {#how-to}

Cover the answer, speak for 60 to 90 seconds, then compare and note only what you missed. Unless a question says otherwise, the setting is a bank that scores payments in real time, with about one fraud per thousand transactions.

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

### Random forest and gradient boosting {#trees}

**Question.** Compare random forest with gradient boosting. Why does boosting usually win on tabular data?

Both are tree ensembles. A random forest grows deep trees independently on bootstrap samples with random feature subsets and averages them, which mainly reduces variance.

Gradient boosting grows shallow trees one after another, each fitted to the gradient of the loss of the current ensemble. That mainly reduces bias, and shrinkage, subsampling and regularization keep the variance in check.

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

### L1 and L2 regularization {#regularization}

**Question.** What is the difference between L1 and L2 regularization?

L2 penalizes squared weights. It shrinks all coefficients smoothly toward zero and shares weight among correlated features.

L1 penalizes absolute values. Its constraint region has corners on the axes, so the optimum often lands with some coefficients exactly zero. It selects features, and it picks somewhat arbitrarily among correlated ones.

In Bayesian terms, L2 is a Gaussian prior on the weights and L1 is a Laplace prior, and the penalized fit is the maximum a posteriori (MAP) estimate. Both trade a little bias for lower variance.

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

Both cases use the same frame. I say the frame first, then fill it in.

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

Download [fraud-practice-lab.py](/notes/fraud-detection-practice/fraud-practice-lab.py) and run it with Python 3.10 or later, NumPy and pandas. It generates its own data and prints every number in this note. If scikit-learn or DuckDB is installed, it also cross-checks the AUC and the SQL query.

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

Phrases worth having ready: point-in-time features, out-of-time validation, label maturity, control group, champion-challenger, shadow mode, value detection rate, false positive ratio, step-up authentication, training-serving skew.

## References {#references}

1. Dal Pozzolo A, Caelen O, Johnson RA, Bontempi G. [Calibrating probability with undersampling for unbalanced classification](https://doi.org/10.1109/SSCI.2015.33). IEEE Symposium Series on Computational Intelligence 2015, 159–166.
2. Elkan C. [The foundations of cost-sensitive learning](https://mlanthology.org/ijcai/2001/elkan2001ijcai-foundations/). IJCAI 2001, 973–978.
3. Chen T, Guestrin C. [XGBoost: a scalable tree boosting system](https://arxiv.org/abs/1603.02754). KDD 2016.
4. Lundberg SM, Lee S-I. [A unified approach to interpreting model predictions](https://arxiv.org/abs/1705.07874). NeurIPS 2017.
5. Board of Governors of the Federal Reserve System. [SR 26-2: Revised guidance on model risk management](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm). April 17, 2026.
6. Bowman MW. [Artificial intelligence in the financial system](https://www.federalreserve.gov/newsevents/speech/bowman20260501a.htm). Speech, May 1, 2026.

Every number in this note is printed by [`fraud-practice-lab.py`](/notes/fraud-detection-practice/fraud-practice-lab.py). The SQL and PySpark versions of the velocity feature were checked against the pandas output with DuckDB 2.0 and PySpark 4.3.
