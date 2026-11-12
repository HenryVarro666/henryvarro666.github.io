---
title: "The confusion matrix: precision, recall and who pays for each mistake"
date: 2026-11-12
lastmod: 2026-11-12
description: "Precision and recall are two readings of one 2×2 table. Worked through cancer screening, card fraud and fake accounts: each ends up with several thresholds and graded actions rather than one trade-off."
topics: [data-science, healthcare]
---
Sooner or later every binary classifier meets the same question: should it favour precision or recall? The stock answer is a slogan (recall for medicine, precision for spam), and it stops helping as soon as real numbers arrive. This note works through what the confusion matrix says, why precision collapses when positives are rare, how costs become a threshold, and how three settings resolve the trade-off: cancer screening, card fraud, and fake accounts on a short-video platform.

> **Takeaway.** Recall is a property of the model and its threshold. Precision is a property of those *and* of the population the model runs on. Which one to favour is a property of the action that follows a positive. All three settings below end up with several thresholds and graded actions, not one trade-off.

## The problem

A model produces a score. A product needs a decision, so someone puts a threshold on the score, and from then on the model makes two kinds of mistake. It flags things that are fine, and it lets through things that are not. The two mistakes are rarely paid for by the same person, and rarely cost the same.

"Precision or recall?" compresses three separate questions:

1. **How rare is the positive class** where the model will run?
2. **What does each mistake cost, and who pays?**
3. **What happens after the model says "positive"?** A second test, a human review, or an irreversible action?

The confusion matrix is the place where all three become visible, so I start there.

## One table, read three ways

![A two-by-two confusion matrix with recall and specificity read along the rows and precision and NPV read down the columns](confusion-matrix.svg "Figure 1. The four cells and the two directions of reading. Recall and specificity divide along a row; precision and NPV divide down a column.")

### The four cells

Rows are the truth; columns are the model's call. Before reading anything else, pin down which class is "positive". By convention it is the rare one that triggers an action: the cancer, the fraud, the fake account. Every name below flips if that choice flips.

Here is a real matrix: 1,682,504 screening mammograms read in U.S. community practice from 2007 to 2013, from the [Breast Cancer Surveillance Consortium](https://www.bcsc-research.org/statistics/screening-performance-benchmarks-archive/screening-sens-spec-false-negative-archive) (BCSC). "Positive" means cancer diagnosed within a year of the exam.

| | Called back | Read as normal | Total |
|---|---:|---:|---:|
| Cancer | 8,529 (TP) | 1,283 (FN) | 9,812 |
| No cancer | 186,139 (FP) | 1,486,553 (TN) | 1,672,692 |
| Total | 194,668 | 1,487,836 | 1,682,504 |

### Across a row: what the model does to each class

Divide along a row and the denominator is the number of real positives or real negatives.

- **Recall** (sensitivity): $8{,}529 / 9{,}812 = 86.9\%$ of cancers were called back.
- **Specificity**: $1{,}486{,}553 / 1{,}672{,}692 = 88.9\%$ of healthy exams were read as normal. Its complement, the **false positive rate**, is 11.1%.

These two numbers describe the test. They do not depend on how many of the people screened have cancer.

### Down a column: what a result means

Divide down a column and the denominator is the number of exams that got a given answer.

- **Precision** (positive predictive value): $8{,}529 / 194{,}668 = 4.4\%$ of call-backs were cancer.
- **Negative predictive value**: $1{,}486{,}553 / 1{,}487{,}836 = 99.91\%$ of normal readings were right.

These are what the person holding the result needs, and they depend on prevalence as much as on the test.

### The whole table: accuracy

Accuracy adds the diagonal: 88.9%. A reader who called every exam normal would score 99.4% and find no cancers. Accuracy weights each cell by its count, so with 0.58% prevalence it is almost entirely a statement about healthy people.

The same six quantities carry different names in medicine, signal detection and machine learning:

| Quantity | Formula | The question it answers |
|---|---|---|
| **Recall**: sensitivity, true positive rate, hit rate | TP&nbsp;/&nbsp;(TP&nbsp;+&nbsp;FN) | Of the real positives, how many did I catch? |
| **Miss rate**: false negative rate, type II error rate | FN&nbsp;/&nbsp;(TP&nbsp;+&nbsp;FN) | How many did I let through? |
| **Specificity**: true negative rate | TN&nbsp;/&nbsp;(TN&nbsp;+&nbsp;FP) | Of the real negatives, how many did I leave alone? |
| **False positive rate**: fall-out, type I error rate | FP&nbsp;/&nbsp;(TN&nbsp;+&nbsp;FP) | How many innocents did I flag? |
| **Precision**: positive predictive value (PPV) | TP&nbsp;/&nbsp;(TP&nbsp;+&nbsp;FP) | Of my alarms, how many were real? |
| **NPV**: negative predictive value | TN&nbsp;/&nbsp;(TN&nbsp;+&nbsp;FN) | Of my all-clears, how many were right? |

## Why precision collapses when positives are rare

![One thousand dots: 116 called back, of which 5 are cancers; 884 cleared, of which 1 is a missed cancer](thousand-screens.svg "Figure 2. The BCSC rates applied to 1,000 screening mammograms. Almost everyone called back is healthy, because almost everyone screened is healthy.")

The screening test above catches 87% of cancers and clears 89% of healthy women, and still 96 of every 100 call-backs are false alarms. Nothing is wrong with the test. There are about 170 healthy women for every woman with cancer, so an 11% false positive rate applied to the large group swamps an 87% recall applied to the small one.

Bayes' rule in odds form makes this a single multiplication:

$$
\underbrace{\frac{P(\text{positive} \mid \text{flag})}{P(\text{negative} \mid \text{flag})}}_{\text{odds after the flag}}
= \underbrace{\frac{\pi}{1-\pi}}_{\text{odds before}} \times \underbrace{\frac{\text{recall}}{\text{FPR}}}_{\text{likelihood ratio}}
$$

For screening, the odds before are 1 to 170 and the likelihood ratio is $0.869 / 0.111 = 7.8$. The flag makes cancer 7.8 times more likely, which is a strong test, and the odds after are still only 1 to 22. Written as a probability:

$$
\text{precision} = \frac{\text{recall} \cdot \pi}{\text{recall} \cdot \pi + \text{FPR} \cdot (1 - \pi)}
$$

![Precision plotted against prevalence on a log axis for false positive rates of 11%, 1% and 0.1%](precision-vs-prevalence.svg "Figure 3. Precision against prevalence at 87% recall, for three false positive rates. The lightest curve is the screening operating point. The vertical rules are rough prevalences for the three settings in this note: card fraud by value, screening from BCSC, fake accounts from Meta's estimate for Facebook.")

Three things follow.

**The false positive rate has to be about as small as the prevalence.** Precision reaches 50% only when $\text{FPR} \le \text{recall} \cdot \pi / (1 - \pi)$. With positives at 0.1% and recall at 80%, that means flagging fewer than 0.08% of negatives. A false positive rate of 1% sounds excellent and gives 7% precision there.

**Precision measured on a rebalanced test set does not transfer.** A model with 90% recall and a 5% false positive rate shows 94.7% precision on a 50/50 test set and 15.4% where positives are 1%. Recall and specificity carry over, as long as the kinds of cases stay the same; precision has to be recomputed at the prevalence of deployment with the formula above.

**The same model is a different product in each population.** Move the screening test to a group where 4% are positive and its precision is 25%; move it to card-fraud prevalence and it is 0.5%.

## The matrix is a snapshot of one threshold

![Two bell curves of scores, for actual negatives and actual positives, cut by a threshold; the tails on the wrong side are misses and false alarms](threshold.svg "Figure 4. One threshold on two score distributions, drawn at the screening operating point. Moving it trades misses for false alarms; nothing else about the model changes.")

A confusion matrix belongs to a model *at a threshold*. Sliding the threshold moves cases within a row: true positives become false negatives, or false positives become true negatives. So recall and the false positive rate rise and fall together, and precision is a by-product of where both land and of the prevalence.

That is why two curves that describe the same model can tell such different stories.

![Left, one ROC curve with area 0.95. Right, three precision-recall curves for the same model at 50%, 5% and 0.58% positives](roc-vs-pr.svg "Figure 5. One model (two unit-variance Gaussians fitted through the screening operating point), plotted two ways. The ROC curve cannot show prevalence; the precision-recall curves show little else.")

The ROC curve plots the two row rates against each other, so it is blind to prevalence. Its area here is 0.95 whether positives are half the data or one in 170. The precision-recall curve plots a row rate against a column rate. At the same operating point precision is 89%, 29% or 4.4% as positives go from 50% to 5% to 0.58%.

[Saito and Rehmsmeier](https://doi.org/10.1371/journal.pone.0118432) argue that this makes ROC plots easy to misread on imbalanced data: a large pool of negatives makes specificity look reassuring. [Davis and Goadrich](https://doi.org/10.1145/1143844.1143874) prove the two spaces agree about dominance (a curve that wins everywhere in one wins everywhere in the other), and also that optimising ROC area does not optimise precision-recall area. My reading: use ROC to compare models, and the precision-recall curve at deployment prevalence to decide whether a model is usable.

A note on single numbers. $F_1$ for the screening matrix is 0.08, which says nothing about whether screening is worth doing. $F_\beta$ lets recall count $\beta$ times as much as precision, without saying where $\beta$ comes from, and neither uses the true negatives. I would rather write the costs down.

## From costs to a threshold

Give each error a cost and the expected cost per case is

$$
\mathbb{E}[\text{cost}] = c_{FN} \cdot \pi \, (1 - \text{recall}) \; + \; c_{FP} \cdot (1 - \pi) \, \text{FPR}
$$

If the model outputs a calibrated probability $p$ that a case is positive, flagging costs $(1-p)\,c_{FP}$ on average and not flagging costs $p\,c_{FN}$. Flag when the first is smaller, which gives the threshold from [Elkan's cost-sensitive learning paper](https://cseweb.ucsd.edu/~elkan/rescale.pdf):

$$
p^* = \frac{c_{FP}}{c_{FP} + c_{FN}}
$$

| A miss costs … as much as a false alarm | Flag when the probability exceeds |
|---|---:|
| 1,000 times | 0.1% |
| 100 times | 1% |
| 10 times | 9% |
| exactly | 50% |
| a tenth | 91% |
| a hundredth | 99% |

This reframes the slogan. "Favour recall" means misses are expensive relative to false alarms, so the threshold is low. "Favour precision" means the opposite. And the costs attach to the **action**, not to the field: a call-back for more imaging and a biopsy are both "medicine", and they sit at very different thresholds.

Four practical complications:

- **The threshold assumes calibrated probabilities.** Training on resampled data breaks calibration. If only a fraction $\beta$ of negatives is kept, the model's odds are inflated by $1/\beta$; the corrected probability is $p = \beta p_s / (\beta p_s + 1 - p_s)$.
- **0.5 is a special case.** It is right only when the two errors cost the same and the probabilities are calibrated.
- **Costs can differ per case.** A missed \$2,000 fraud is not a missed \$20 fraud. Then every case has its own threshold.
- **Capacity can set the threshold instead.** If people review every alert and can handle 500 a day, the operating point is "the 500 highest scores", and the useful metrics are precision within that budget and the recall it buys.

## Three settings

| | Screening mammography | Card fraud | Fake accounts |
|---|---|---|---|
| Unit scored | one exam | one payment | one account, again and again |
| Positives | 0.58% of exams | under 0.1% of card volume | a few percent of active accounts, not known exactly |
| A miss costs | a cancer found later; paid by the patient | the amount; paid by the bank or merchant | spam and fake engagement; paid thinly by everyone |
| A false alarm costs | more imaging, sometimes a biopsy, worry; paid by a healthy person | a declined purchase, a support call, perhaps a lost customer | a real person locked out, an appeal |
| The truth arrives | within a year, from biopsy and cancer registries | days to weeks later, and never for declined payments | rarely; from human review of samples |
| After a positive | a second, more specific test | a step-up check, a review, or a decline | a challenge, reduced reach, a review, or removal |
| Numbers people report | sensitivity, specificity, PPV at each stage | fraud dollars stopped, precision of alerts, false declines | sampled precision of each action, appeals overturned, estimated prevalence left |

### Medical screening: recall first, then a second test

In the [BCSC benchmarks](https://doi.org/10.1148/radiol.2016161174), 11.6% of screening exams were called back and 5.1 cancers were found per 1,000 exams. Precision at that step was 4.4%. Among women who went on to a biopsy recommendation, it was 25.6%.

A precision of 4.4% is acceptable because of what a positive triggers: more imaging, not treatment. The false alarm costs an appointment and some anxious days. The miss costs a cancer found a year or more later. With a cost ratio that lopsided, the threshold belongs low.

The more important point is that screening is not one decision. Each stage raises the prevalence for the next: 0.58% in the screened population, 4.4% among those called back, about 26% among those sent to biopsy. A sensitive first test followed by a specific second test gets recall *and* precision out of a pipeline, which no single threshold could.

Recall is still not pushed to 100%. False alarms repeat at every screening round, follow-up has its own harms, and readers' time is finite. In the same BCSC data only about six in ten radiologists met the recommended range for the call-back rate; the profession audits the false-alarm side as well.

### Card fraud: both errors are priced in money

Fraud is rarer still. [The Nilson Report](https://nilsonreport.com/articles/card-fraud-losses-worldwide-in-2023/) puts 2023 card fraud losses at \$33.83 billion worldwide, which its press release expresses as 6.58 cents per \$100 of volume: under 0.1% by value.

Take 1,000,000 payments of which 1,000 are fraudulent, and a model with 80% recall. At a 0.5% false positive rate it raises 4,995 false alarms for 800 frauds caught: 13.8% precision, six interrupted customers per fraud. Cutting the false positive rate to 0.1% lifts precision to 44.5%. That is the prevalence rule from earlier, in a setting where each false alarm is a customer standing at a checkout.

Here both errors are in dollars, so the cost formula applies almost literally, one payment at a time. As an illustration with made-up but plausible numbers, let a miss cost the amount $A$ and a false decline cost 2% of $A$ in lost margin plus \$3 of friction. Then $p^*$ is 14.5% for a \$20 payment, 3.4% for \$200 and 2.1% for \$2,000. Larger payments deserve lower thresholds, which is also why fraud teams report recall weighted by value (the share of fraud *dollars* stopped) and not only the share of fraud cases.

Three things make the fraud matrix harder to measure than the medical one:

- **Labels are late.** A fraud is confirmed when the customer reports it or a chargeback arrives. Last week's matrix is still filling in.
- **Labels are selective.** A declined payment never reveals whether it was fraud. Only alerted payments are checked quickly, so the fastest feedback comes from the cases the model already suspected. [Dal Pozzolo and colleagues](https://doi.org/10.1109/TNNLS.2017.2736643) call these verification latency and the alert-feedback interaction, and note that for investigators, who can check only a limited number of alerts, the precision of those alerts is the measure that matters.
- **The positives fight back.** Fraud patterns change in response to the model, so an operating point chosen in January drifts.

And as in screening, the answer is more than one threshold: approve, ask for a one-time code, hold for review, decline. The one-time code plays the role of the second mammogram. It is a cheap extra test that a real customer passes in seconds, so it can sit at a low, recall-friendly threshold.

### Fake accounts: scale turns small rates into crowds

On a platform, the positives are less rare and much harder to count. Meta estimates the share of fake accounts on Facebook by [sampling monthly active users and labelling them](https://transparency.meta.com/policies/improving/prevalence-metric/), and has [recently put it at about 4%](https://www.socialmediatoday.com/news/meta-outlines-latest-data-on-content-removals-and-fake-accounts/807727/). TikTok publishes removals rather than a prevalence estimate: [146.8 million fake accounts removed worldwide in the second quarter of 2026](https://www.dhakatribune.com/feature/tech/423261/tiktok-releases-q2-2026-community-guidelines).

Suppose a platform has a billion accounts, 4% of them fake, and a detector with 90% recall and a 0.1% false positive rate. Precision is 97.4%, a number any classifier would be proud of. It is also 960,000 real people actioned by mistake. Precision as a percentage hides the crowd. The authors of Meta's [Deep Entity Classification](https://www.usenix.org/conference/usenixsecurity21/presentation/xu-teng) system make the same point about why abuse classifiers need very high precision.

The two errors also land differently. A missed fake account spreads its cost thinly: slightly distorted rankings, inflated follower counts, less trustworthy ad metrics. A false positive lands on one person, who may have lost an audience or an income. So the action ladder matters more here than anywhere:

- **Quiet, reversible actions run at high recall.** Not counting a suspicious account's likes, or asking for a CAPTCHA, costs a real user almost nothing.
- **Removal runs at very high precision,** usually with an appeal path, because it is the one action the user cannot undo.

Measurement is the hardest of the three settings, because nobody hands over the truth. Precision is estimated by having people review a sample of actioned accounts; overturned appeals give a floor on false positives. Recall cannot be observed at all. It is inferred from the prevalence that sampling finds among the accounts still standing.

There is also a third axis the matrix has no cell for: **time**. An account removed after it has posted its spam is a true positive in the table and a failure in practice. Blocking at sign-up and removing after the fact are different products even at identical precision and recall.

### What the three have in common

![Three score bars, for a screening mammogram, a card payment and a platform account, each cut by several thresholds into graded actions](graded-actions.svg "Figure 6. One score, several thresholds. Low thresholds guard cheap, reversible actions; high thresholds guard costly ones. Segment widths are schematic.")

None of the three picks precision over recall once. Each has rare positives and lopsided costs, and each answers with the same design: several thresholds on one score, with actions that get more expensive and less reversible as the score rises. The early, cheap stage is tuned for recall and tolerates terrible precision. The late, costly stage is tuned for precision and tolerates missing cases, because the earlier stage already caught the attention of a human or a second test. The trade-off is real at each threshold; the system as a whole gets both.

## What I check before reporting a classifier

1. **Name the positive class and the unit.** Per exam or per patient, per payment or per card, per account or per action.
2. **Use the prevalence where the model will run,** not the one in the dataset.
3. **Say what follows a positive and who pays for each error.** That fixes the cost ratio, and the cost ratio fixes the threshold.
4. **Report the matrix at the operating point, as counts.** Rates without counts hide small samples: 27 hits out of 30 positives is 90% recall with a 95% interval from 74% to 97%.
5. **Compare with doing nothing.** "Always negative" scored 99.4% accuracy above.
6. **Show the precision-recall curve at deployment prevalence,** and check calibration before thresholding a probability.
7. **Ask where the labels came from and how late they are.**
8. **Re-measure after launch.** Prevalence moves, and in fraud and abuse the positives adapt.

## From my own work

Three earlier projects of mine read differently with this in hand.

**RadQNLI: accuracy fell, recall did not.** In [RadQNLI](https://github.com/HenryVarro666/RadQNLI) I tested ChatGPT on four radiology test sets whose negatives shared more and more words with the question. Accuracy fell from 0.81 to 0.62, which looks like a difficulty effect. But recall on the positive class stayed between 0.52 and 0.53 on every set, while the share of positives rose from 21% to 70% and precision rose from 0.56 to 0.89. A control with the same number of *random* negatives fell almost as far, from 0.83 to 0.64. Accuracy is $\pi \cdot \text{recall} + (1-\pi) \cdot \text{specificity}$, so with recall near 0.53 and specificity near 0.85, raising $\pi$ drags accuracy down toward recall. Most of the "harder test set" effect was class mix.

**Heart disease: which class was positive?** In my [heart-disease course project](https://github.com/HenryVarro666/heart-disease-prediction) the data file codes `target = 1` as *no disease*. Every per-class metric for class 1 therefore describes healthy patients, and "recall" for that class is what a clinician would call specificity. The test set had 61 patients in total, so recall for either class rests on roughly thirty cases and carries an interval about twenty points wide. And the headline number was ROC AUC, which says how well a model ranks but not where to cut.

**De-identification: a recall task with a misleading floor.** In [open-LLM de-identification](https://github.com/HenryVarro666/open-llm-deid) the costly error is a leaked name, so the task is recall-first; over-redaction only costs readability. The score I used counted the share of protected strings removed, and because of how matches were counted, a note that had not been touched at all already scored 0.597. Every result has to be read against that floor. It is the same lesson as "always negative gets 99.4%": check what doing nothing scores before trusting a metric.

## References

1. Lehman CD, Arao RF, Sprague BL, et al. [National Performance Benchmarks for Modern Screening Digital Mammography: Update from the Breast Cancer Surveillance Consortium](https://doi.org/10.1148/radiol.2016161174). *Radiology* 2017;283(1):49–58. Counts from the [BCSC benchmark tables](https://www.bcsc-research.org/statistics/screening-performance-benchmarks-archive/screening-sens-spec-false-negative-archive); the false positive count is non-cancers minus true negatives.
2. Saito T, Rehmsmeier M. [The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets](https://doi.org/10.1371/journal.pone.0118432). *PLOS ONE* 2015;10(3):e0118432.
3. Davis J, Goadrich M. [The Relationship Between Precision-Recall and ROC Curves](https://doi.org/10.1145/1143844.1143874). ICML 2006.
4. Elkan C. [The Foundations of Cost-Sensitive Learning](https://cseweb.ucsd.edu/~elkan/rescale.pdf). IJCAI 2001.
5. Dal Pozzolo A, Boracchi G, Caelen O, Alippi C, Bontempi G. [Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy](https://doi.org/10.1109/TNNLS.2017.2736643). *IEEE Transactions on Neural Networks and Learning Systems* 2018;29(8):3784–3797.
6. The Nilson Report. [Card Fraud Losses Worldwide in 2023](https://nilsonreport.com/articles/card-fraud-losses-worldwide-in-2023/). December 2024; the per-\$100 figure is from its press release of 6 January 2025.
7. Meta Transparency Center. [Prevalence](https://transparency.meta.com/policies/improving/prevalence-metric/): how the share of fake accounts is estimated. The 4% figure as reported by [Social Media Today](https://www.socialmediatoday.com/news/meta-outlines-latest-data-on-content-removals-and-fake-accounts/807727/).
8. TikTok Community Guidelines Enforcement Report, Q2 2026, as reported by the [Dhaka Tribune](https://www.dhakatribune.com/feature/tech/423261/tiktok-releases-q2-2026-community-guidelines).
9. Xu T, Goossen G, Cevahir HK, et al. [Deep Entity Classification: Abusive Account Detection for Online Social Networks](https://www.usenix.org/conference/usenixsecurity21/presentation/xu-teng). USENIX Security 2021.

The figures and every number computed in this note come from one script, [`tools/notes/precision_vs_recall_figs.py`](https://github.com/HenryVarro666/henryvarro666.github.io/blob/master/tools/notes/precision_vs_recall_figs.py).
