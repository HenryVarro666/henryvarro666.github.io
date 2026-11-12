---
title: "Notes style check"
date: 2026-11-12
description: "Unlisted page for checking how notes render: contents box, math, tables, callouts and code."
robots: "noindex"
sitemap:
  disable: true
build:
  list: never
  render: always
---
This page is not listed anywhere. It shows every element a note can use, so changes to the
styles can be checked in one place. Delete it once real notes cover the same elements.

## Math

Inline math sits in the text: precision is $P = \frac{TP}{TP + FP}$ and recall is
\(R = \frac{TP}{TP + FN}\). A dollar amount needs a backslash: a \$500 transfer and a
\$20 refund stay plain text.

Display math goes on its own lines:

$$
F_\beta = (1 + \beta^2)\,\frac{P \cdot R}{\beta^2 P + R}
$$

### Expected cost of a threshold

$$
\mathbb{E}[\text{cost}](t) = c_{FN}\,\pi\,\bigl(1 - R(t)\bigr) + c_{FP}\,(1 - \pi)\,\mathrm{FPR}(t)
$$

## Tables

| | Medical screening | Card fraud | Fake accounts |
|---|---|---|---|
| Cost of a miss (FN) | 漏诊，后果可能很严重 | Direct loss, grows with the amount | Spam and fake engagement spread |
| Cost of a false alarm (FP) | 多做检查，病人焦虑 | Declined purchase, customer churn | A real user is banned |
| Usual metric | Sensitivity at fixed specificity | PR-AUC, recall weighted by amount | Precision of automatic actions |

| Threshold | Precision | Recall |
|---:|---:|---:|
| 0.3 | 0.41 | 0.92 |
| 0.5 | 0.63 | 0.81 |
| 0.7 | 0.84 | 0.58 |

## Callouts and code

> **Takeaway.** None of these settings picks precision over recall once and for all: unequal
> error costs and rare positives push all three toward several thresholds with graded actions.

```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_true, scores)
```

- A list item with `inline code`
- A second item
