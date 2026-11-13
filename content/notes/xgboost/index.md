---
title: "XGBoost: from one tree to a few hundred"
date: 2026-11-12T21:30:00-06:00
lastmod: 2026-11-12T21:30:00-06:00
description: "XGBoost (eXtreme Gradient Boosting) adds small decision trees one at a time, each one fitted to the slope and curvature of the loss and kept modest by penalties. A walk through the algorithm with a split worked by hand, the knobs that matter, what goes wrong with rare positives and with anything outside the training range, and where it is used."
topics: [data-science]
---
The [first](/notes/precision-vs-recall/) and [second](/notes/roc-auc-vs-pr-auc/) notes were about judging a model's scores. This one is about a model that produces many of those scores in practice. XGBoost, short for eXtreme Gradient Boosting, is a library for gradient-boosted decision trees: it builds a prediction out of hundreds of small trees, added one at a time, each correcting what the previous ones got wrong. Tianqi Chen started it in 2014 as a research project at the University of Washington. It became known through the Higgs Boson Machine Learning Challenge that year, where he and Tong He received the special "High Energy Physics meets Machine Learning" award, and through the competitions that followed: of the 29 winning solutions published on Kaggle's blog in 2015, 17 used XGBoost ([Chen and Guestrin, 2016](https://doi.org/10.1145/2939672.2939785)). The numbers below come from version 3.4.1, released in August 2026.

> **Takeaway.** Each new tree is fitted to the gradient (slope) and hessian (curvature) of the loss at the current predictions. Every leaf gets the Newton step −G/(H + λ), and every split must pay for itself in loss reduction. That is the whole algorithm; the rest is fast split finding and good defaults. In practice: pick the number of trees on held-out data, be careful with the learning rate when positives are rare, prefer moving the threshold to reweighting the classes, leave missing values missing, and never ask trees to predict outside the range they were trained on.

## The problem: tables

Most data that businesses and hospitals hold is a table: one row per transaction, patient or account; columns of mixed kinds (an amount, a count, a category, a lab value), on different scales, with gaps. The signal is often in thresholds and interactions: a purchase is suspicious when it is large *and* the card was used in another country an hour ago.

Trees suit this. A split asks "is this column below this value?", so the scale of a column does not matter and any monotone transformation of it gives the same tree. Nesting splits builds interactions. Missing values can be routed instead of imputed. And trees are cheap to train and to run.

The benchmarks agree, with a qualification that has grown over time. On 45 medium-sized datasets (around 10,000 rows), [Grinsztajn and colleagues](https://arxiv.org/abs/2207.08815) found tree ensembles ahead of deep networks, and traced the gap to irregular target functions and uninformative columns. The newer [TabArena](https://arxiv.org/abs/2506.16791) benchmark, run with careful tuning and ensembling, finds the best deep tabular models now equal to or better than boosted trees, and the tabular foundation model [TabPFN](https://doi.org/10.1038/s41586-024-08328-6) (Tabular Prior-data Fitted Network) ahead on datasets of up to 10,000 rows. Boosted trees remain the strong default; they are no longer the automatic winner.

## One tree

![A depth-two regression tree next to the staircase function it computes over forty points](one-tree.svg "Figure 1. A regression tree of depth two. Three questions about x send each point to one of four leaves; each leaf predicts the mean of its points. Drawn as a function, the tree is a staircase.")

A decision tree is a set of yes-or-no questions that sends every row to a leaf, and a number stored in each leaf. Figure 1 shows one of depth two fitted to forty points: three questions, four leaves, a staircase. A deep tree can fit any staircase, including the noise; a shallow tree is too coarse. There are two ways to combine many trees:

- **Average many deep trees** grown independently on resampled data. That is a random forest. Averaging lowers the variance of trees that each overfit.
- **Add many shallow trees** one after another, each fitted to what the sum so far still gets wrong. That is boosting. Adding lowers the bias of trees that each underfit.

## Boosting: each tree fixes what is left

![Four panels showing the boosted model after 1, 5, 30 and 300 trees, and below them training and test error against the number of trees](boosting-rounds.svg "Figure 2. Boosting on the same forty points, with trees of depth two and learning rate 0.3. The error on the training points falls towards zero; the error on new points from the same curve is lowest at 33 trees and then climbs as the model starts fitting noise.")

Start from a constant, the mean of the targets. Fit a small tree to the residuals, the differences between the targets and the current prediction. Add a shrunken copy of it:

$$
F_m(x) = F_{m-1}(x) + \eta\, f_m(x)
$$

where η (eta, the learning rate) is a number such as 0.3 or 0.05. Repeat. In Figure 2 the root-mean-square error (RMSE) on the forty training points goes from 1.24 after one tree to 0.62 after five, 0.22 after thirty and 0.01 after three hundred. On 400 new points from the same curve it goes from 1.32 to 0.77, then 0.47, then back up to 0.55. The best is 0.47 at 33 trees; the curve the points were drawn from would score 0.36, the noise level.

So the number of trees is not "as many as possible". It is chosen on data the trees have not seen, usually by early stopping: train while watching a validation set and keep the round where its error was lowest.

## Residuals are gradients

For squared error, ½(y − F)², the residual y − F is minus the derivative of the loss with respect to the prediction. [Friedman](https://doi.org/10.1214/aos/1013203451) turned that into a general recipe, the gradient boosting machine (GBM): for any differentiable loss, fit each tree to the negative gradient. For classification the model's output F is a log-odds, the probability is \(p = 1/(1 + e^{-F})\), and with log loss the gradient for a row is simply p − y.

XGBoost also uses the second derivative, the hessian. For each row i at the current prediction:

| Loss | Gradient gᵢ | Hessian hᵢ |
|---|---|---|
| Squared error, ½(yᵢ − F)² | F − yᵢ | 1 |
| Log loss, F a log-odds | pᵢ − yᵢ | pᵢ(1 − pᵢ) |

A tree assigns every row to a leaf, and every row in leaf j gets the same value \(w_j\). Write \(G_j\) and \(H_j\) for the sums of gᵢ and hᵢ over the rows in that leaf. A second-order Taylor expansion of the loss, plus the penalty XGBoost adds, gives for that leaf

$$
G_j\, w_j + \tfrac{1}{2}\,(H_j + \lambda)\, w_j^2
$$

a parabola in \(w_j\). Its minimum is at

$$
w_j^\ast = -\frac{G_j}{H_j + \lambda}
$$

That is a Newton step, damped by λ (lambda, an L2 penalty, on the squares of the leaf values). The full objective penalizes each tree f with Ω(f) = γT + ½λ‖w‖², where T is the number of leaves and γ (gamma) a price per leaf. Plugging w* back in, a split of one leaf into a left and a right child is worth

$$
\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma
$$

Three of the main knobs fall straight out of these two formulas:

- **λ shrinks leaf values towards zero,** most of all in leaves with a small H, where the data say least.
- **γ is the price of a leaf.** A split that does not reduce the loss by at least that much is removed.
- **`min_child_weight` is a lower bound on H, not on the number of rows.** For squared error every row has h = 1 and the two coincide. For log loss a row predicted at p has h = p(1 − p), which is small when the model is confident. This comes back below.

## A split worked by hand

![Twelve transactions on a number line, four of them fraud, and a bar for the gain of each of the eleven possible thresholds](split-gain.svg "Figure 3. Twelve card transactions sorted by the number of purchases on the card in the past 24 hours. Each bar is the gain of a split between neighbours; grey bars are ruled out by min_child_weight. The threshold 5.5 wins with a gain of 2.40.")

Twelve card transactions, one feature: the number of purchases on the card in the past 24 hours, from 1 to 12. Four are fraud. XGBoost 3 starts every row at the mean label, p = 1/3, a log-odds of ln(1/2) = −0.693. Then:

- every fraud has g = 1/3 − 1 = −2/3, every legitimate transaction g = +1/3;
- every row has h = (1/3)(2/3) = 2/9 ≈ 0.222;
- over all twelve, G = 0 and H = 8/3.

Try the threshold 5.5. The left side holds the five legitimate transactions with counts 1 to 5: \(G_L = 5/3\), \(H_L = 10/9\). The right side holds the other seven: \(G_R = -5/3\), \(H_R = 14/9\). With λ = 1 and the parent term zero (G = 0), the bracket is

$$
\frac{(5/3)^2}{10/9 + 1} + \frac{(5/3)^2}{14/9 + 1} = \frac{25}{19} + \frac{25}{23} = 2.40
$$

The leaves get \(w_L = -(5/3)/(19/9) = -0.789\) and \(w_R = +0.652\). With η = 0.3 the tree stores −0.237 and +0.196, which moves the predicted probability from 0.333 to 0.283 on the left and 0.378 on the right. XGBoost, run on the same twelve rows with `tree_method="exact"`, prints exactly this tree: `[f0<5.5] gain=2.40274596`, leaves `-0.236842126` and `0.195652172`, and a "cover" of 1.11 and 1.56 for the two leaves, which is H.

Figure 3 shows the gain at every threshold. The highest bar is at 8.5 (2.47), but its right side would hold four rows with H = 4 × 2/9 = 0.89, below the default `min_child_weight` of 1. Each row here weighs 0.222, so a child needs at least five rows; eight of the eleven thresholds are ruled out. With `min_child_weight=0`, XGBoost splits at 8.5 instead.

Two details of the library, both checked on this example:

- **The gain it reports is the bracket, without the ½,** and γ is compared with that number. Here `gamma=2.41` removes the split and `gamma=2.40` keeps it.
- **Leaf values in a dump already include η.**

On real data the hessian is the thing to watch. With 0.2% fraud, a confident model predicts p ≈ 0.002 for most transactions, so each contributes h ≈ 0.002, and `min_child_weight=1` asks for about 500 of them per leaf. The same setting means very different things on balanced and on rare-event data.

## Missing values pick a side

![The twelve transactions plus two with a missing count, both fraud, sent left in one panel and right in the other, with the gain of each choice](missing.svg "Figure 4. Two more transactions whose count is missing, both fraud. At the threshold 5.5, sending them left gives a gain of 0.74, sending them right 3.50. XGBoost tries both directions for every threshold and stores the better one in the tree.")

XGBoost does not need missing values filled in. When it scans a feature it sorts only the rows that have a value, and for every threshold it tries the rows without one on the left and on the right ([Chen and Guestrin](https://doi.org/10.1145/2939672.2939785) call this sparsity-aware split finding). The better direction is stored in the tree as the default for that split.

In Figure 4 the two transactions with no count are both fraud. At 5.5, putting them with the five low-count legitimate transactions gives a gain of 0.74; putting them with the right side gives 3.50. Over all thresholds and both directions the winner is 8.5 with missing values sent right, gain 4.38. Note that the two extra rows also make that split legal: the right side now has six rows.

What follows:

- **Missingness can carry signal.** A card with no purchase history, a lab that was not ordered. Filling the gaps with the mean or with zero throws that away. Leave them as `NaN` (not a number).
- **The default direction is applied silently.** If a feature is missing at serving time for a reason that never happened in training (a broken join, a new data source), every such row goes down the learned default branch and nothing warns you. Check the missing rate of each feature in production against training.
- **Sparse inputs count as missing.** For a sparse matrix, entries that are not stored are treated as missing, not as zero.

## Finding splits fast

The exact method sorts every feature and scans every threshold, as above. Since version 2.0 the default is `tree_method="hist"`: each feature is cut once into at most `max_bin` = 256 buckets at quantiles, and split finding works on per-bucket sums of g and h. The paper's weighted quantile sketch places the cuts with each row weighted by its hessian; the `approx` method still re-sketches that way every round, while `hist` places the cuts once, before training. On the twelve-row example, `hist` splits at "x < 6", the same partition as "x < 5.5".

The rest of what made XGBoost fast is systems work: data stored in compressed column blocks, cache-aware access, many cores, out-of-core training on data larger than memory (reworked in version 3.0), graphics processing units (GPUs) with `device="cuda"`, and distributed training on Dask, Spark and Ray.

## The knobs

| Parameter | Default | What it does | Where I start |
|---|---|---|---|
| `n_estimators` / `num_boost_round` | 100 (scikit-learn API) / 10 (`xgb.train`) | number of trees | large, with `early_stopping_rounds` on a validation set |
| `learning_rate` (`eta`) | 0.3 | shrinks every tree | 0.03–0.1 |
| `max_depth` | 6 | depth of each tree; a tree of depth d can combine up to d features | 3–8 |
| `min_child_weight` | 1 | minimum sum of hessians in a child | raise for noisy data; mind the scale with rare positives |
| `gamma` (`min_split_loss`) | 0 | minimum reported gain to keep a split | 0, then small values |
| `reg_lambda` (`lambda`) | 1 | L2 penalty (squares) on leaf values | 1–10 |
| `reg_alpha` (`alpha`) | 0 | L1 penalty (absolute values) on leaf values | 0 |
| `subsample` | 1 | share of rows each tree sees | 0.7–0.9 |
| `colsample_bytree` | 1 | share of columns each tree sees | 0.5–0.9 |
| `max_bin` | 256 | histogram buckets per feature | leave |
| `scale_pos_weight` | 1 | multiplies the weight of positive rows | see below |
| `max_delta_step` | 0 (no cap) | caps every leaf value | 1 when positives are very rare |
| `monotone_constraints` | none | forces a feature's effect to go one way | where a reviewer will ask |

Row and column subsampling come from [Friedman's stochastic gradient boosting](https://doi.org/10.1016/S0167-9473(01)00065-2): each tree sees a random part of the data, which makes the trees less alike and the sum less likely to chase noise. XGBoost's [tuning notes](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html) group the rest the same way: `max_depth`, `min_child_weight` and `gamma` control complexity directly; subsampling and a smaller `eta` with more trees add robustness.

## The learning rate on rare positives

![Validation log loss against number of trees for learning rates 0.3, 0.1 and 0.03, and for 0.3 with leaf values capped at 1](learning-rate.svg "Figure 5. Credit-card fraud, split by time. With the default learning rate the first trees overshoot and the model never recovers; a smaller rate reaches a lower loss later; capping each leaf at 1 fixes the default rate.")

From here on the data are real: the [credit-card fraud dataset](https://api.openml.org/d/42175) from the Machine Learning Group at ULB (Université libre de Bruxelles), 284,807 card transactions over two days, 492 of them fraud (0.17%). The columns are 28 anonymized principal components (V1 to V28), the amount, and the time in seconds since the first transaction. I split by time, as a deployed model would meet it: the first 60% of transactions to train (360 frauds, 0.21%), the next 20% to validate (57, 0.10%) and the last 20% to test (75, 0.13%).

Figure 5 shows validation log loss as trees are added. A model that predicts the training fraud rate for every transaction scores 0.0083. The last column of the table is the average precision (AP) on the test set at that number of trees.

| Learning rate | Lowest validation log loss | at trees | Test AP there |
|---|---|---|---|
| 0.3 (default) | 0.0062 | 27 | 0.58 |
| 0.1 | 0.0028 | 37 | 0.61 |
| 0.03 | 0.0022 | 187 | 0.79 |
| 0.3 with `max_delta_step=1` | 0.0021 | 32 | 0.79 |

The default rate barely beats the constant, and its test AP is 0.58 against 0.79. The cause is in the first tree. Every row starts at p = 0.0021, so every row has h ≈ 0.002. A leaf that collects some frauds has a large negative G and a small H, and the Newton step −G/(H + λ) is of the order of the number of frauds in it. The largest leaf of the first tree is +35.7 in log-odds even after shrinking by 0.3: it turns a probability of 0.002 into almost 1 for every legitimate transaction in that leaf. Training log loss after one tree is 0.086, worse than the 0.015 it started from. The quadratic approximation is only good near the current prediction, and the step lands far outside that range.

Two fixes, with the same effect here: a smaller learning rate, which needs more trees, or `max_delta_step`, which caps every leaf value (here at 1, so 0.3 after shrinkage). XGBoost's parameter documentation suggests 1 to 10 for logistic regression with extremely imbalanced classes. With the cap, the default rate reaches the lowest loss of the four, in 32 trees.

## Rare positives: reweight, or move the threshold?

![Precision-recall curves for XGBoost with default weights and with scale_pos_weight, nearly on top of each other, with the threshold 0.5 marked on each](scale-pos-weight.svg "Figure 6. The last 20% of transactions, 75 frauds. AP: average precision; ROC-AUC: area under the receiver operating characteristic curve. Reweighting the positives 474-fold barely moves the precision-recall curve; it moves where the threshold 0.5 lands on it. The default model cut at 0.33 lands next to the reweighted model's 0.5.")

The usual advice for imbalanced data is `scale_pos_weight = negatives / positives`, here 474. It multiplies the gradient and hessian of every fraud by 474. I trained both versions with η = 0.05 and early stopping on validation AP, and scored them on the test set with AP and ROC-AUC, the area under the receiver operating characteristic curve (both are taken apart in the [second note](/notes/roc-auc-vs-pr-auc/)):

| | Default weights | `scale_pos_weight` = 474 |
|---|---|---|
| Trees | 125 | 243 |
| AP | 0.80 | 0.78 |
| ROC-AUC | 0.980 | 0.976 |
| Flagged at p ≥ 0.5 | 53, of which 51 fraud | 60, of which 53 fraud |
| Median score of a fraud | 0.85 | 1.00 |

The 95% bootstrap interval for the difference in AP is −0.01 to +0.05: no difference this test set can detect. The ranking is essentially the same; the scores are not. Reweighting pushes the frauds' scores towards 1, so p ≥ 0.5 flags more transactions. The default model reaches the same place with a lower threshold: at 0.33 it flags 62 transactions and catches 54 frauds, against 60 and 53 for the reweighted model at 0.5.

XGBoost's own [tuning notes](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html) draw the same line: if only the ranking (ROC-AUC) matters, balance the weights with `scale_pos_weight`; if you need the right probability, do not rebalance, and set `max_delta_step` to help convergence. After the first two notes I would put it more strongly. A reweighted model's 0.5 is a threshold someone else chose, at a cost ratio no one wrote down. Train on the real class balance, keep probabilities that mean something, and pick the threshold from costs and capacity, as in the [first note](/notes/precision-vs-recall/#from-costs-to-a-threshold).

## Trees do not extrapolate

![A straight-line trend observed only up to x = 10: boosted trees go flat beyond the last training point while a straight line continues](extrapolation.svg "Figure 7. A straight-line trend seen only for x between 0 and 10. Boosted trees fit the training points closely, then predict a constant; the straight line keeps going. At x = 15 the truth is 24.5, the line says 25.1, the trees 17.7.")

Past the largest threshold any tree uses on a feature, every tree gives the same leaf whatever the value, so the sum is flat too. The model can only repeat what it said at the edge of the training data. In Figure 7 the trees fit the forty training points better than a straight line does (RMSE 0.27 against 0.94), and do much worse on the fifteen points beyond x = 10 (3.78 against 1.26).

The fraud data have the same problem, hidden in a column. Time is seconds since the first transaction: 0 to 120,396 in training, 145,248 to 172,792 in test. The trained model splits on Time 89 times, always below 111,089 seconds, so every test transaction falls on the "late" side of every one of those splits. Its SHAP value (SHapley Additive exPlanations, explained below) for Time averages −0.58 in log-odds on the test set, against −0.01 on training rows: a downward push that comes from the calendar, not from anything about the transaction. Dropping the column leaves average precision where it was (0.79 against 0.80).

So: features that grow with time (timestamps, account ages measured from a fixed date, cumulative counts, prices under inflation) should be replaced by things that stay in range: hour of day, time since the previous event, ratios, differences. If the target itself trends, model the trend separately and let the trees fit what is left.

## Reading a trained model

XGBoost reports feature importance in several ways, and they disagree. On the fraud model with default weights from the previous section:

| Ranking by | Top five | Amount | Time |
|---|---|---|---|
| weight (number of splits) | V4, V14, V10, V12, Amount | 5th | 16th |
| gain (average gain per split) | V14, V10, V12, V27, V4 | 15th | 27th |
| cover (average hessian per split) | V10, V14, V4, V20, V7 | 10th | 11th |
| mean \|SHAP\| on the test set | V4, Time, V10, V12, V14 | 7th | 2nd |

"Weight" counts splits, including the many small ones a model makes late in training to fine-tune its answers: Amount is fifth by count and fifteenth by gain. Gain and cover describe the training data. None of them is a statement about a single prediction.

SHAP values are. For each row they split the model's output, in log-odds, into one contribution per feature plus a constant, using Shapley values from game theory; for trees, [Lundberg and colleagues' TreeSHAP](https://doi.org/10.1038/s42256-019-0138-9) computes them exactly in polynomial time, and XGBoost has it built in (`predict(..., pred_contribs=True)`). On the test set the contributions add up to the model's output to within 10⁻⁵. They are also how the Time problem above showed up: second place by mean |SHAP|, twenty-seventh by gain.

Where a reviewer will ask "does the model ever go the wrong way?", say in credit or medicine, `monotone_constraints` forces the prediction to move in one direction with a feature, and `interaction_constraints` limits which features may appear together in a tree.

## Where it is used

| Field | Question | Objective | What to watch |
|---|---|---|---|
| Card fraud | is this transaction fraud? | `binary:logistic`, evaluated by AP | rare positives; labels arrive weeks later as chargebacks; fraud patterns drift |
| Credit risk | will this borrower default? | `binary:logistic` | monotone constraints; reasons for each decline; populations shift with the economy |
| Clinical risk | will this patient be readmitted, deteriorate, survive? | `binary:logistic`; `survival:cox` (Cox proportional hazards), `survival:aft` (accelerated failure time) | missing labs carry information; calibration; shift between hospitals |
| Ads and recommendations | will this user click? | `binary:logistic` | calibration feeds the auction; [Facebook](https://doi.org/10.1145/2648584.2648589) used boosted trees as feature transforms for a logistic model |
| Search ranking | in what order? | `rank:ndcg` (normalized discounted cumulative gain), `rank:pairwise` | queries as groups; LambdaMART, the tree-boosting ranker, is the classic |
| Demand forecasting | how many will sell? | `reg:squarederror`, `reg:tweedie`, `reg:quantileerror` | lags and rolling windows; trends need handling (above); the [M5 competition](https://econpapers.repec.org/RePEc:eee:intfor:v:38:y:2022:i:4:p:1346-1364) was won by LightGBM models |
| Physics | is this collision signal or background? | `binary:logistic` | where XGBoost made its name ([Chen and He](https://proceedings.mlr.press/v42/chen14.html)) |
| Fake accounts | is this account fake? | `binary:logistic` on behaviour and graph features | adversaries adapt; labels are noisy |

The three settings of the first note, cancer screening, card fraud and fake accounts, all appear in this table. In each, the model is the smaller part of the work; the features, the labels and the threshold decide most of the outcome.

## XGBoost and its neighbours

| Model | Trees | Categorical columns | Strength | I pick it when |
|---|---|---|---|---|
| XGBoost | grown level by level (or by best leaf), Newton steps with penalties | native support, no longer experimental since 3.1 | missing values, constraints, GPU, mature tooling | general default for tables |
| [LightGBM](https://mlanthology.org/neurips/2017/ke2017neurips-lightgbm/) | grown leaf by leaf; samples rows by gradient size (Gradient-based One-Side Sampling, GOSS) and bundles sparse columns (Exclusive Feature Bundling, EFB) | native | speed on large, wide data | millions of rows, many sparse columns |
| [CatBoost](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html) | symmetric trees; ordered boosting against target leakage | ordered target statistics | categorical-heavy data, good defaults | many high-cardinality categories |
| Random forest | deep, independent, averaged | via encoding | few knobs, hard to break | a quick, robust baseline |
| Deep tabular networks (TabM, RealMLP) | none | embeddings | with tuning and ensembling, on par in TabArena; train jointly with images or text | tables are one input among several |
| TabPFN | none: a pretrained transformer reads the training set in context | supported | best on up to 10,000 rows, no training per dataset | small data |

## When I would not reach for it

- **Raw images, audio or text.** Use a pretrained network, or feed its embeddings to the trees.
- **Targets that must extend past the training range.** Trends, growth, dose far beyond what was observed.
- **A few hundred rows.** A regularized linear model or TabPFN, and an honest look at the evaluation noise (below).
- **When the relation must be smooth or differentiable end to end,** for instance inside a larger network or an optimizer.

## Checklist

1. **Split by time** if the data have time. A random split lets the model peek at the future.
2. **Start with a baseline:** the constant, then a logistic or linear model.
3. **Pick the number of trees by early stopping,** with the metric that matches the use: log loss when the probabilities matter, average precision when only the top of the ranking does.
4. **With rare positives, look at the first rounds.** Lower `eta` or set `max_delta_step=1`. Keep the real class balance and move the threshold rather than reweighting.
5. **Leave missing values as `NaN`,** and compare missing rates between training and serving.
6. **Remove or transform columns that grow with time.**
7. **Use SHAP, not split counts,** to see what the model relies on.
8. **Report uncertainty:** a bootstrap interval on the test metric, or repeated cross-validation.
9. **Calibrate on held-out data** if anyone will read the scores as probabilities.

## From my own work

My [heart-disease course project](https://github.com/HenryVarro666/heart-disease-prediction) compares five classifiers on the Cleveland data. Its notebook imports `GradientBoostingClassifier`, `AdaBoostClassifier` and `ExtraTreesClassifier` and never uses them. I added XGBoost on the same split, 242 patients to train and 61 to test, and repeated 5-fold cross-validation ten times on all 303 for a second opinion:

| Model | Test accuracy | Test ROC-AUC | Cross-validated ROC-AUC (mean ± standard deviation) |
|---|---|---|---|
| Logistic regression | 0.918 | 0.963 | 0.905 ± 0.031 |
| Support vector machine | 0.918 | 0.966 | 0.906 ± 0.031 |
| Random forest | 0.918 | 0.968 | 0.912 ± 0.029 |
| XGBoost, defaults | 0.885 | 0.957 | 0.887 ± 0.028 |
| XGBoost, small trees | 0.885 | 0.959 | 0.902 ± 0.028 |

"Small trees" is 300 trees of depth 2 at η = 0.05, with row and column subsampling at 0.8. Two things stand out. With defaults, 100 trees of depth 6 at η = 0.3 are too much model for 242 patients, and XGBoost comes last. With small trees it catches up but does not pass the others. And the spread between folds, a standard deviation of 0.03 and a range from 0.81 to 0.97, is larger than any gap between models. On 303 rows the question "which model is best?" has no answer this data can give; that is the regime where the benchmarks point to TabPFN or a regularized linear model.

## References

1. Chen T, Guestrin C. [XGBoost: a scalable tree boosting system](https://doi.org/10.1145/2939672.2939785). KDD 2016.
2. Friedman JH. [Greedy function approximation: a gradient boosting machine](https://doi.org/10.1214/aos/1013203451). *Annals of Statistics* 2001;29(5):1189–1232.
3. Friedman JH. [Stochastic gradient boosting](https://doi.org/10.1016/S0167-9473(01)00065-2). *Computational Statistics & Data Analysis* 2002;38(4):367–378.
4. Chen T, He T. [Higgs boson discovery with boosted trees](https://proceedings.mlr.press/v42/chen14.html). NIPS 2014 Workshop on High-energy Physics and Machine Learning, PMLR 42:69–80.
5. XGBoost developers. [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html) and [notes on parameter tuning](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html), documentation.
6. Ke G, Meng Q, Finley T, et al. [LightGBM: a highly efficient gradient boosting decision tree](https://mlanthology.org/neurips/2017/ke2017neurips-lightgbm/). NeurIPS 2017.
7. Prokhorenkova L, Gusev G, Vorobev A, Dorogush AV, Gulin A. [CatBoost: unbiased boosting with categorical features](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html). NeurIPS 2018.
8. Lundberg SM, Erion G, Chen H, et al. [From local explanations to global understanding with explainable AI for trees](https://doi.org/10.1038/s42256-019-0138-9). *Nature Machine Intelligence* 2020;2:56–67.
9. Grinsztajn L, Oyallon E, Varoquaux G. [Why do tree-based models still outperform deep learning on typical tabular data?](https://arxiv.org/abs/2207.08815) NeurIPS 2022 Datasets and Benchmarks.
10. Erickson N, Purucker L, Tschalzev A, et al. [TabArena: a living benchmark for machine learning on tabular data](https://arxiv.org/abs/2506.16791). NeurIPS 2025 Datasets and Benchmarks.
11. Hollmann N, Müller S, Purucker L, et al. [Accurate predictions on small data with a tabular foundation model](https://doi.org/10.1038/s41586-024-08328-6). *Nature* 2025;637:319–326.
12. He X, Pan J, Jin O, et al. [Practical lessons from predicting clicks on ads at Facebook](https://doi.org/10.1145/2648584.2648589). ADKDD 2014.
13. Makridakis S, Spiliotis E, Assimakopoulos V. [M5 accuracy competition: results, findings, and conclusions](https://econpapers.repec.org/RePEc:eee:intfor:v:38:y:2022:i:4:p:1346-1364). *International Journal of Forecasting* 2022;38(4):1346–1364.
14. Machine Learning Group, ULB. [Credit card fraud detection dataset](https://api.openml.org/d/42175), dataset description.

The toy figures and their numbers come from [`tools/notes/xgboost_figs.py`](https://github.com/HenryVarro666/henryvarro666.github.io/blob/master/tools/notes/xgboost_figs.py), which uses a from-scratch version of the algorithm, [`tinyboost.py`](https://github.com/HenryVarro666/henryvarro666.github.io/blob/master/tools/notes/tinyboost.py), about a hundred lines of plain Python. [`xgboost_runs.py`](https://github.com/HenryVarro666/henryvarro666.github.io/blob/master/tools/notes/xgboost_runs.py) checks that it matches XGBoost tree for tree and produces the credit-card and heart-disease numbers.
