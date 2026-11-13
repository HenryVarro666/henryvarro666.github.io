---
title: "ROC-AUC and PR-AUC: two areas, two questions"
date: 2026-11-12T20:00:00-06:00
lastmod: 2026-11-12T20:00:00-06:00
description: "ROC-AUC is the chance that a random positive outranks a random negative; PR-AUC is the average precision met on the way down the list. Two models with the same ROC-AUC can differ fourfold in PR-AUC, and which one is better depends on where the list will be cut."
topics: [data-science]
---
The [previous note](/notes/precision-vs-recall/) looked at a classifier one threshold at a time. Most model comparisons skip the threshold and report an area instead: ROC-AUC, PR-AUC, or both. This note works out what each area measures, why the two can disagree about the same pair of models, and what I report.

> **Takeaway.** ROC-AUC asks how often the model puts a random positive above a random negative. It ignores prevalence and gives every part of the ranking the same weight. PR-AUC, computed as average precision, asks how clean the list is as you walk down it. It moves with prevalence and is dominated by the top of the list. Neither picks a threshold. Report the one whose question matches the deployment, with its baseline and an interval, and put the operating-point numbers next to it.

## From one threshold to all of them

Every threshold on the score gives one confusion matrix. Sweep the threshold from the highest score to the lowest and you get a sequence of matrices, one per cut. The two curves plot that same sequence on different axes:

| | ROC curve | Precision-recall curve |
|---|---|---|
| x-axis | false positive rate, FP / (FP + TN) | recall, TP / (TP + FN) |
| y-axis | recall, TP / (TP + FN) | precision, TP / (TP + FP) |
| Random ranking | the diagonal | a flat line at the prevalence |
| Perfect ranking | the top-left corner | the top-right corner |
| Sees prevalence | no: both axes are row rates | yes: precision is a column rate |

The last row is the whole story of the previous note in one line. The area under each curve folds all thresholds into a single number. That is convenient, and it is also where the trouble starts, because the number no longer says which thresholds it was good at.

## ROC-AUC: the chance of ordering a pair correctly

![A ranked list of fifteen cases beside a grid of all fifty positive–negative pairs, with the ROC staircase drawn over the grid](pairs.svg "Figure 1. Fifteen cases ranked by score. Each cell of the grid is one positive–negative pair; it is blue when the positive is ranked above the negative. The staircase is the ROC curve, and the blue cells are exactly the area under it.")

Take every pair made of one positive and one negative. ROC-AUC is the share of pairs in which the positive gets the higher score, with ties counting half:

$$
\text{ROC-AUC} = P(s^+ > s^-) + \tfrac{1}{2}\,P(s^+ = s^-)
$$

In Figure 1 there are 5 positives and 10 negatives, so 50 pairs. The positive is on top in 39 of them: ROC-AUC = 0.78. Walk down the list and draw the ROC curve as you go: each positive is a step up, each negative a step right. The cells below the staircase are exactly the pairs in which the positive came first, which is why the area and the pair count agree. The same number is the Mann–Whitney U statistic divided by the number of pairs; [Hanley and McNeil](https://doi.org/10.1148/radiology.143.1.7063747) made this three-way equivalence (area, correct ranking of a random pair, Wilcoxon statistic) the standard reading in radiology.

The same quantity goes by AUROC, AUC, or, in medicine, the c-statistic (concordance). Credit scoring reports the Gini coefficient, 2 × AUC − 1, which is 0.80 for an AUC of 0.90.

What follows from the pair reading:

- **0.5 is a coin flip.** Below 0.5 the scores are upside down, and flipping them gives 1 − AUC.
- **Only the order counts.** Any monotone transformation of the scores leaves it unchanged, so it says nothing about calibration. A model that overstates every probability has the same ROC-AUC as a calibrated one, as long as the order is the same.
- **Prevalence does not enter.** Copy every negative ten times and no pair changes order.
- **The classes are interchangeable.** Call the other class positive, flip the scores, and the pairs are the same pairs.
- **The uncertainty is set mostly by the smaller class.** With the Hanley–McNeil formula, an AUC of 0.90 measured on 50 positives and 50,000 negatives has a 95% interval of 0.84 to 0.96. On 250 positives and 360 negatives it is 0.87 to 0.93. Over a hundred times as many negatives do not make up for a fifth as many positives.

## PR-AUC: the precision you meet on the way down

![The same ranked list with the precision noted at each positive, and the step-shaped precision-recall curve whose shaded area is the average precision](average-precision.svg "Figure 2. The same fifteen cases. At each positive, the precision so far; average precision is the mean of those five numbers, and equals the shaded area under the step-shaped precision-recall curve. The dashed line is a random ranking.")

Walk down the same list. Each time you reach a positive, write down the precision so far. Average precision (AP) is the mean of those numbers:

$$
\text{AP} = \sum_n (R_n - R_{n-1})\,P_n = \frac{1}{n^+}\sum_{k=1}^{n^+} \big(\text{precision at the } k\text{th positive}\big)
$$

In Figure 2 the five precisions are 1.00, 0.67, 0.75, 0.57 and 0.45, so AP = 0.69. It is also the area under the precision-recall curve drawn as a staircase. The list is the same one that scored 0.78 on ROC-AUC; the two numbers answer different questions about it.

What follows from this reading:

- **The baseline is the prevalence, not 0.5.** A random ranking meets precision ≈ π all the way down. Here that is 0.33. An AP of 0.30 is excellent when 1% of cases are positive and worse than random when half are.
- **It moves with prevalence.** More on this below.
- **The classes are not interchangeable.** AP for "disease" and AP for "no disease" are different numbers on the same predictions.
- **How it is computed matters.** scikit-learn's `average_precision_score` uses the step sum above and [warns](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) that the trapezoidal rule, which uses linear interpolation, can be too optimistic. On the toy list, trapezoids through the same points give 0.74 instead of 0.69. [Davis and Goadrich](https://doi.org/10.1145/1143844.1143874) show why: between two points in PR space, a straight line overstates what any threshold in between can achieve.
- **"PR-AUC" is not one estimator.** Papers use the step sum, trapezoids, or Davis–Goadrich interpolation under the same name. [Boyd and colleagues](https://doi.org/10.1007/978-3-642-40994-3_29) compare the estimators and their confidence intervals and recommend two simple intervals. I write the function I used next to the number.
- **The scale is arithmetic.** [Flach and Kull](https://papers.nips.cc/paper/2015/file/33e8075e9970de0cfea955afd4644bb2-Paper.pdf) point out that the area averages precision arithmetically while F-scores use a harmonic mean, and propose precision-recall-gain curves to fix the scale and the baseline.

## Same ROC-AUC, different models

![Two ROC curves with the same area, on ordinary axes and with the false positive rate on a log scale](same-auc.svg "Figure 3. Two models with ROC-AUC 0.90 each. On ordinary axes they look alike; on a log scale for the false positive rate, B is far ahead where false positives are rare, and A is ahead once a fifth of the negatives are flagged.")

Here are two models with exactly the same ROC-AUC, 0.90. Both are binormal: model A's positive and negative scores have equal spread; model B's positives are spread twice as wide, so some of them score far above every negative and others sink among them.

On ordinary axes the ROC curves are hard to tell apart. Put the false positive rate on a log scale and they are not:

| False positive rate | Recall, model A | Recall, model B |
|---:|---:|---:|
| 0.1% | 10% | 46% |
| 1% | 30% | 61% |
| 10% | 70% | 79% |
| 30% | 90% | 88% |

The curves cross at a false positive rate of 22%, where both catch 85% of positives. Past that point A is better: to catch 95% of positives, A flags 43% of negatives and B flags 66%.

![Precision-recall curves of the same two models at 0.5% prevalence; B stays near precision 1 up to recall 0.3, A starts below 0.9 and falls fast](same-auc-pr.svg "Figure 4. The same two models when 0.5% of cases are positive. Average precision is 0.52 for B and 0.13 for A, with identical ROC-AUC.")

At 0.5% prevalence the precision-recall curves show the gap plainly. At 30% recall, B's alerts are 97% positive and A's are 13%. Average precision is 0.52 for B and 0.13 for A: a fourfold difference between two models that ROC-AUC calls equal.

Which model is better depends on where the list will be cut:

- **If only the top of the list gets acted on** (an alert queue with fixed capacity, the accounts a platform removes outright), B is much better.
- **If nearly every positive must be found** (a first-pass screen that sends everything suspicious to a second test), A is better.

ROC-AUC calls it a tie because it averages recall over every false positive rate from 0 to 1, including the region past 22% where A wins and which few deployments ever use. AP favours B because it is dominated by the top of the list. [McDermott and colleagues](https://arxiv.org/abs/2401.06091) make this precise: ROC-AUC rewards fixing a mistake by the same amount wherever it sits in the ranking, while AP rewards fixes near the top of the list far more than fixes further down.

[Hand](https://doi.org/10.1007/s10994-009-5119-5) goes further: when ROC curves cross, the model with the larger AUC can be the worse one over almost the whole range of thresholds anyone would use. He traces this to an implicit weighting of misclassification costs that differs from one classifier to the next, and proposes the H-measure as a fixed alternative.

A simpler fix keeps the ROC curve and integrates only the part you would operate in. [McClish](https://doi.org/10.1177/0272989X8900900307) proposed this partial AUC with a standardization that maps random performance to 0.5 and perfect to 1. Over false positive rates up to 1%, A scores 0.60 and B 0.77; up to 10%, A scores 0.75 and B 0.85. The partial area picks the same winner as AP, for a stated reason.

## Prevalence moves PR-AUC, not ROC-AUC

![Average precision against prevalence for three models whose ROC-AUC stays fixed](ap-vs-prevalence.svg "Figure 5. Average precision against prevalence for the screening model of the previous note and for models A and B. Their ROC-AUC values (0.95, 0.90, 0.90) do not move; average precision falls with prevalence.")

Hold each model fixed and change only the share of positives:

| Prevalence | Screening model (ROC-AUC 0.95) | Model A (0.90) | Model B (0.90) |
|---:|---:|---:|---:|
| 50% | 0.95 | 0.90 | 0.93 |
| 5% | 0.67 | 0.46 | 0.71 |
| 1% | 0.41 | 0.20 | 0.58 |
| 0.58% | 0.33 | 0.14 | 0.54 |
| 0.1% | 0.14 | 0.04 | 0.42 |

The screening model is the binormal model fitted through the mammography operating point in the previous note. Its ROC-AUC is 0.95 and its average precision at screening prevalence is 0.33. Both are true: the ranking is good, and most of the list is still healthy women.

Three practical consequences:

- **AP from datasets with different prevalence cannot be compared.** A model does not get worse because it moved to a population with fewer positives; its AP does.
- **A test set with undersampled negatives inflates AP.** ROC-AUC is unaffected by the undersampling; AP has to be recomputed at deployment prevalence.
- **Report the prevalence next to the AP.** Dividing by it helps reading: model A's 0.13 at 0.5% prevalence is 26 times better than random.

## Is PR-AUC the metric for imbalanced data?

That is the common advice. The description of the widely used [ULB credit-card fraud dataset](https://api.openml.org/d/42175), in which 0.17% of transactions are fraud, recommends AUPRC because of the class imbalance.

McDermott and colleagues dispute the general claim. In their analysis the deciding factor is not imbalance at all but how the model will be used: whether the mistakes that matter sit near the top of the ranking or anywhere in it. They also show that AP can favour improvements in subgroups where positives are more common, which widens gaps between groups.

The example above agrees with them. Prevalence decides how *large* the gap between A and B looks in AP; it does not decide which model is right. That is decided by where the list will be cut. My rule of thumb:

| How the scores will be used | Summary I report |
|---|---|
| Only the top of the list is acted on: an alert queue with fixed capacity, retrieval, outright removal | AP with the prevalence, precision at the review capacity, recall at a required precision |
| Nearly every positive must be found: a first-pass screen, a rule-out test | ROC-AUC, partial AUC over the usable false positive range, specificity at the required sensitivity |
| Groups with different prevalence are compared | ROC-AUC per group; AP only alongside each group's prevalence |
| Datasets are compared | ROC-AUC; AP only at equal prevalence |
| A threshold is being chosen | neither: the confusion matrix at that threshold, with the costs |

## What neither area tells you

- **Where to cut.** Both average over thresholds. The deployment uses one, or a few.
- **Whether the probabilities mean anything.** Both depend only on the order of the scores. Calibration needs its own check: a reliability plot, or the Brier score.
- **What a mistake costs.** That was the subject of the previous note; Hand's critique is that ROC-AUC smuggles in a cost assumption that changes from model to model.
- **Whether a difference is real.** Two models scored on the same cases are correlated; [Hanley and McNeil's 1983 method](https://doi.org/10.1148/radiology.148.3.6878708) takes that pairing into account and narrows the interval on the difference. A bootstrap over cases does the same for AP.
- **Which average was taken.** With more than two classes, or with one-hot labels, "the AUC" can be per-class, micro-averaged or macro-averaged, and the three differ (an example follows).

## From my own work

Re-running my [heart-disease course project](https://github.com/HenryVarro666/heart-disease-prediction) with this in mind turned up three of the points above in one figure.

**One plot, four AUCs.** The ROC panel for the support vector machine prints four numbers: 0.97 for class "False", 0.97 for class "True", 0.96 for the micro-average and 0.97 for the macro-average. The two per-class values agree because ROC-AUC does not care which class is called positive. The micro-average pools the two columns of one-hot labels into a single ranking, which is a different quantity; it happens to be the lowest.

**The PR curve was for the healthy class.** The PR panel reports average precision 0.98. It is computed for class 1, and in this data `target = 1` means *no disease*: 36 of the 61 test patients, a baseline of 0.59. Re-running the notebook reproduces the plotted values (ROC-AUC 0.97, AP 0.98). For the disease class, 25 of 61 and a baseline of 0.41, the same predictions give AP 0.95, and the random forest 0.96. ROC-AUC is 0.97 either way.

**The differences are inside the noise.** With 25 patients with disease, an AUC of 0.97 has a Hanley–McNeil interval of about 0.92 to 1.00. The support vector machine (0.966), random forest (0.968) and logistic regression (0.96) cannot be told apart on this test set, which is what the project README already concludes.

## What I report

1. **Both curves, with their baselines:** the diagonal on the ROC plot, the prevalence line on the PR plot. When the ROC curves cross, a log-scale false positive rate axis as well.
2. **ROC-AUC with an interval,** and a paired comparison when models share a test set.
3. **AP with the prevalence it was computed at, an interval, and the function used** (`average_precision_score`, not trapezoids over the PR points).
4. **The numbers at the operating region:** precision at the review capacity, recall at a required precision, or a partial AUC over the false positive rates that will be used.
5. **Calibration, separately.**
6. **AP across datasets or subgroups only at equal prevalence,** or with each prevalence stated.
7. **The class that is positive.** It changes AP, and it changes what "recall" means.

## References

1. Hanley JA, McNeil BJ. [The meaning and use of the area under a receiver operating characteristic (ROC) curve](https://doi.org/10.1148/radiology.143.1.7063747). *Radiology* 1982;143(1):29–36.
2. Hanley JA, McNeil BJ. [A method of comparing the areas under receiver operating characteristic curves derived from the same cases](https://doi.org/10.1148/radiology.148.3.6878708). *Radiology* 1983;148(3):839–843.
3. McClish DK. [Analyzing a portion of the ROC curve](https://doi.org/10.1177/0272989X8900900307). *Medical Decision Making* 1989;9(3):190–195.
4. Davis J, Goadrich M. [The relationship between precision-recall and ROC curves](https://doi.org/10.1145/1143844.1143874). ICML 2006.
5. Hand DJ. [Measuring classifier performance: a coherent alternative to the area under the ROC curve](https://doi.org/10.1007/s10994-009-5119-5). *Machine Learning* 2009;77:103–123.
6. Boyd K, Eng KH, Page CD. [Area under the precision-recall curve: point estimates and confidence intervals](https://doi.org/10.1007/978-3-642-40994-3_29). ECML PKDD 2013.
7. Saito T, Rehmsmeier M. [The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets](https://doi.org/10.1371/journal.pone.0118432). *PLOS ONE* 2015;10(3):e0118432.
8. Flach P, Kull M. [Precision-recall-gain curves: PR analysis done right](https://papers.nips.cc/paper/2015/file/33e8075e9970de0cfea955afd4644bb2-Paper.pdf). NeurIPS 2015.
9. McDermott MBA, Zhang H, Hansen LH, Angelotti G, Gallifant J. [A closer look at AUROC and AUPRC under class imbalance](https://arxiv.org/abs/2401.06091). NeurIPS 2024.
10. scikit-learn. [`average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), API documentation.
11. Machine Learning Group, ULB. [Credit card fraud detection dataset](https://api.openml.org/d/42175), dataset description.

The figures and every number computed in this note come from one script, [`tools/notes/roc_auc_vs_pr_auc_figs.py`](https://github.com/HenryVarro666/henryvarro666.github.io/blob/master/tools/notes/roc_auc_vs_pr_auc_figs.py). The heart-disease numbers come from re-running that project's notebook.
