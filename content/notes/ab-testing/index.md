---
title: "A/B Testing: From Checkout Reminder to Rollout Decision"
date: 2026-11-13T19:49:02-06:00
description: "Design a checkout experiment, compute sample size and conversion lift, check the logs, and decide whether the result clears a rollout guardrail."
topics: [data-science]
---

Suppose a shopping site wants to offer an optional reminder at checkout. Would offering it lead more users to buy? If the current policy targets people who are less likely to buy in the first place, the reminder group can convert at a lower rate in historical logs even when the reminder helps. Randomizing eligible users and comparing their 24-hour purchase rates gives an estimate of the effect of **offering the reminder**.

This note follows one hypothetical experiment from the design document to the rollout decision. If you can read proportions and averages, you can start here. Run the [calculation lab](#lab) first if you prefer to see the numbers before reading the [design](#design).

## Define the comparison before assigning users {#design}

An A/B test is a randomized controlled experiment. Group A keeps the current checkout, while group B gets the optional reminder. Random assignment makes pre-existing purchase intent comparable between arms on average. Chance imbalance can still occur in the realized sample. With the assignment implemented, outcomes measured, and no interference between users, the difference in means estimates the average effect of assignment for the experiment population. See the [Causal Inference note]({{< relref "/notes/causal-inference" >}}) for the assumptions and the target effect.

![Eligible accounts are split with equal probability between the current checkout and the optional reminder. Both arms use the same 24-hour user purchase outcome.](allocation.svg "Figure 1. A returning account keeps its first assignment. Both arms follow the same rules for recording outcomes.")

| Design choice | Plan for this experiment |
|---|---|
| Population and entry | Signed-in accounts that can receive reminders and meet the checkout criteria for the first time. Apply the same entry rule to both arms before showing anything different. |
| Assignment and analysis unit | The account, identified by `user_id`. Keep its first assignment across devices and repeat visits. Reduce its behavior to one user-level outcome. |
| Intervention | A keeps the current checkout and B offers the optional reminder. Log delivery and clicks separately to explain how the offer works. |
| Primary metric | Users who complete at least one purchase in `[t0, t0 + 24h)`, divided by enrolled users whose outcome windows are complete. |
| Rollout guardrail | Use site-wide notification opt-outs in the same 24-hour window. Require the one-sided 95% upper confidence bound for B minus A to be below +0.20 percentage points. Either group can opt out through the existing settings. |
| Business hurdle | Plan around a minimum detectable effect of +1.00 percentage point. For rollout, require the lower endpoint of the conversion 95% confidence interval to exceed +0.30 percentage points, then check data quality and the guardrail. |
| Readout time | Set the enrollment window and fixed endpoint before launch. Cover a full week in this example, allow the final user's 24-hour window to finish, and verify log completeness. Decide what to count during ramp-up in advance. |

**Analyze by intention to treat (ITT).** Include eligible users in their assigned arms. A B user stays in B if the reminder fails to send, goes unopened, or gets no click. The resulting estimate answers the rollout question: what happens on average when eligible users are offered the reminder? Delivery and clicks can explain how the treatment performed. Clicks can depend on both assignment and purchase intent, so selecting only clickers can undo comparability.

If you need an estimate among users who had a chance to see the feature, record the same trigger in **both** arms before the treatment can affect it. For example, use shared rules to detect the reminder opportunity before either experience is shown. This changes the population being studied. Validate the corresponding counterfactual trigger logs in the control arm and check the complement of the triggered group. [Microsoft's post-experiment guidance on triggered analysis](https://www.microsoft.com/en-us/research/articles/patterns-of-trustworthy-experimentation-post-experiment-stage/)

## Turn event logs into one row per user {#metric}

An account is assigned at noon. A purchase at 11:58 occurred before assignment. Purchases at 12:05 and 16:10 give that user a single 1. A purchase at noon the next day is outside the window. Compare occurrence timestamps in Coordinated Universal Time (UTC), and keep ingestion timestamps to tell when the inputs are complete.

![Purchases before assignment and exactly 24 hours later fall outside the window. Two purchases inside the window yield an outcome of 1 for this user.](outcome-window.svg "Figure 2. The 24-hour outcome window starts at the user's own assignment time. The final boundary is exclusive. The timeline spacing is schematic.")

The assignment ledger records `experiment_id, user_id, variant, assigned_at`. Use the experiment and account as the key for the first assignment and retain its configuration or version. One way to obtain a stable bucket is to hash the experiment identifier, a fixed salt, and the account ID. This teaching example uses the 256-bit Secure Hash Algorithm (SHA-256). In production, specify the encoding, bucket algorithm, variant boundaries, and versions. Use test vectors to make sure services agree, and read the original assignment when an account returns after a change in the bucketing rules.

```python
from hashlib import sha256
from json import dumps

key = dumps([experiment_id, salt, user_id], ensure_ascii=False, separators=(",", ":"))
bucket = int.from_bytes(sha256(key.encode("utf-8")).digest()[:8], "big")
variant = "A" if bucket < 2**63 else "B"
```

Purchase records need at least `event_id, user_id, purchased_at`. Deduplicate by a trusted event key and apply the business rules for corrections. Aggregate eligible events to a binary outcome, or use `EXISTS` to check whether any purchase qualifies. Start with all mature assignments and left join the outcome so the denominator retains users with zero purchases. Fill a missing purchase with zero after checking that the source is complete and there is no qualifying record for the user. Otherwise, a payment log outage in one arm can appear to be a treatment effect. The engineering notes use this same metric definition. [Spark / PySpark]({{< relref "/notes/spark-pyspark" >}}) · [Airflow]({{< relref "/notes/airflow" >}})

## Plan the sample size and calendar {#power}

Suppose the baseline conversion rate is 10%. We want an 80% chance of rejecting equal conversion rates if the treatment lifts conversion to 11%. The minimum detectable effect (MDE) is **+1.0 percentage point**, or **+10%** relative to baseline. The Type I error rate $\alpha=0.05$ is the long-run chance of a false positive when the null holds. Power $1-\beta=0.80$ is the long-run probability of detecting the specified effect when it is real. With equal allocation, independent users, and one two-sided large-sample test at a fixed endpoint, approximate the sample size per arm by:

$$
n \approx
\left(
\frac{
z_{1-\alpha/2}\sqrt{2\bar p(1-\bar p)}
+
z_{1-\beta}\sqrt{p_A(1-p_A)+p_B(1-p_B)}
}{
p_B-p_A
}
\right)^2,
\qquad \bar p=(p_A+p_B)/2.
$$

Here $z_q$ is the $q$ quantile of the standard normal distribution. The first term sets an approximate threshold under the null. The second makes room for sampling variation under the alternative. Plug in $p_A=0.10$ and $p_B=0.11$ to get about 14,750.79. Round up to **14,751 users per arm, 29,502 total**. For very rare outcomes, unequal allocation, or team-level assignment, choose a calculation that matches the design or simulate power. [statsmodels two-proportion sample size documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.samplesize_proportions_2indep_onetail.html)

![At a 10 percent baseline, detecting lifts of 0.5, 1.0, and 2.0 percentage points requires 57,763, 14,751, and 3,841 users per arm, respectively.](sample-size.svg "Figure 3. Halving the target lift requires about four times as many users. The squared effect is in the denominator, and the rates also affect the variance.")

At 5,000 newly enrolled and measurable accounts per day, the primary metric reaches 29,502 users in about six days. This experiment would cover a full week of enrollment and allow the last batch another 24 hours plus the expected logging delay. Estimate the sample required for the guardrail before launch too. Rare opt-outs can require more data, and 80% power on the primary metric does not give an 80% chance of passing the complete rollout rule. The MDE is a planning input. The business hurdle sets the decision standard. If the goal is an 80% chance to show a lift above +0.30 points, plan power against that threshold.

## Monitor data quality and follow the stopping rule {#quality}

### Check assignment counts and data maturity first {#srm}

A **sample ratio mismatch (SRM)** occurs when observed arm counts depart from the planned proportions more than random variation would plausibly explain. Under a fixed 50/50 design with 40,000 independent assignment units, expect 20,000 per arm. With 21,000 in A and 19,000 in B:

$$
\chi^2=\sum_g \frac{(O_g-E_g)^2}{E_g}=100.
$$

There is one degree of freedom here, giving $p\approx 1.52\times 10^{-23}$. Investigate the counts by time and platform across the **assignment ledger → entry logs → mature cohort → metric table**. Verify first assignments, check whether B loses more logs, find filters that depend on post-assignment behavior, and look for unintended inner joins. If the planned ratio changes during ramp-up, use the recorded allocation by time or stratum and the prespecified inclusion rules. A 50/50 split can miss an outage that affects both arms, so check event coverage, duplicate keys, log freshness, and cross-arm exposure too. For sparse counts, use an exact test matched to the assignment rule. [Microsoft's SRM diagnostic guide and case studies](https://www.microsoft.com/en-us/research/articles/diagnosing-sample-ratio-mismatch-in-a-b-testing/)

Run an **A/A test** with the same experience in both arms to check assignment, logging, and the analysis pipeline. A single A/A test can be significant by chance. Frequent extreme $p$ values across controlled replays or variants disappearing on the same platform are reasons to inspect randomization and variance estimates.

### Monitor for safety and agree in advance on when to call a winner {#stopping}

In a fixed-horizon design, refreshing an ordinary $p$ value each day and declaring a win at the first reading below 0.05 raises the cumulative false-positive rate. Watch assignment, logs, and safety metrics throughout the test, and pause on failures according to the plan. If the business needs an early efficacy or futility decision, choose a group-sequential design that allocates the error budget across planned looks, or use an anytime-valid procedure such as a confidence sequence under its required assumptions. If a test stops unexpectedly, record the reason and distinguish a safety stop or data failure from a confirmatory result. [Microsoft's during-experiment guidance](https://www.microsoft.com/en-us/research/articles/patterns-of-trustworthy-experimentation-during-experiment-stage/) · [Johari et al. on sequential inference](https://arxiv.org/abs/1512.04922)

## Calculate the result and make a rollout decision {#readout}

These are **hypothetical counts**. Assume the test reached its planned endpoint, both arms completed follow-up, and data checks passed.

| Metric | A · 20,000 users | B · 20,000 users | B minus A |
|---|---:|---:|---:|
| Purchase within 24 hours | 2,000 / 20,000 = 10.00% | 2,200 / 20,000 = 11.00% | +1.00 percentage point |
| Notification opt-out within 24 hours | 100 / 20,000 = 0.50% | 120 / 20,000 = 0.60% | +0.10 percentage points |

The absolute conversion lift of +1.00 percentage point corresponds to about 100 additional purchasers per 10,000 similar users. The relative lift is $0.01/0.10=10\%$. Use each arm's own observed rate to estimate the standard error of the difference:

$$
\widehat{SE}(\hat\Delta)
=
\sqrt{\frac{\hat p_A(1-\hat p_A)}{n_A}
+\frac{\hat p_B(1-\hat p_B)}{n_B}}
\approx 0.00307.
$$

Adding and subtracting $1.96\,\widehat{SE}$ gives the large-sample 95% Wald confidence interval **[+0.40, +1.60] percentage points**. Across repetitions of the same procedure, approximately 95% of these intervals would cover the true difference. For the null test, the arms share a rate under the null, so use the pooled estimate $\hat p=(2000+2200)/40000=0.105$ in the null standard error. This gives $z\approx3.26$ and a **two-sided $p=0.001106$**. Under the null and the test assumptions, the probability of a statistic at least this extreme is about 0.11%. When counts are small or events are rare, consider a score-based method such as the Newcombe difference interval and plan the sample using the intended inference method. [Newcombe's original paper](https://pubmed.ncbi.nlm.nih.gov/9595617/)

![Conversion's confidence interval clears the +0.30-point business threshold. The one-sided upper bound on opt-outs is +0.222 points, above the +0.20-point guardrail.](readout.svg "Figure 4. Conversion uses a lower endpoint and the opt-out guardrail uses an upper endpoint. Both horizontal axes are in percentage points and use different scales.")

The +0.40 conversion lower endpoint clears the +0.30 rollout hurdle. The estimated opt-out increase is +0.10 percentage points, with a one-sided 95% upper bound of $0.001+1.645\times0.000740\approx0.002216$, or **+0.222 percentage points**. An increase of +0.20 points remains compatible with the data, so **hold the wider rollout**. The team can test a revised reminder frequency or message, or collect more data under a prespecified follow-up design. The guardrail measures net impact on the same user population and window as conversion. Both criteria have to pass.

When a confidence interval crosses zero, report the point estimate and the plausible range. For instance, an interval of [−0.40, +0.70] percentage points allows both a small harm and a meaningful benefit. An equivalence claim requires prespecified bounds and a test designed for them. In a different channel or with a seven-day retention goal, define the population and outcome again, then check whether the original treatment, coverage, and observation window support that decision.

### Run the calculation lab {#lab}

Download [checkout-ab-lab.py](/notes/ab-testing/checkout-ab-lab.py) and run it with Python 3.10 or later. It uses the standard library.

```bash
python3 checkout-ab-lab.py
```

```text
MDE 1.0 pp: 14,751
Conversion: A 10.00%, B 11.00%
Absolute lift: +1.00 pp; relative lift: +10.0%
95% CI: [+0.40, +1.60] pp
Two-sided p-value: 0.001106
One-sided 95% upper bound: +0.222 pp
Guardrail upper bound < +0.20 pp: False
```

The script also reproduces the 21,000-to-19,000 SRM calculation and assigns the same account to the same arm across two simulated devices. Change `k_b` to `2_040` and keep 20,000 per arm, then read the new interval. The change gives +0.20 points with a 95% interval of approximately [−0.39, +0.79] points. The six-user synthetic data in the Spark and Airflow notes is for checking time windows and duplicate events. The larger hypothetical counts here are for practicing statistical inference.

## Adapt the design to more complex metrics or populations {#extensions}

| Scenario | Design and analysis |
|---|---|
| One user creates multiple visits or orders | For the primary metric, aggregate to the user. For order or session outcomes, estimate variance from user-level totals or use cluster-robust inference or a bootstrap that resamples users. Treating all event rows as independent can understate uncertainty. |
| Treatment affects coworkers, teammates, or others in a market | Consider randomizing teams, stores, or markets within which users interact. Plan power and inference around the independent clusters, and specify whether the target weights users or clusters equally. Check for spillovers across clusters. |
| Revenue per session or click-through rate | Distinguish the ratio of arm totals $\sum_i X_i/\sum_i D_i$ from the mean user ratio $\mathrm{mean}_i(X_i/D_i)$. With clicks/impressions of 1/1 and 0/9 for two users, the first rate is 10% and the second is 50%. For the first metric, apply the delta method to the joint user-level numerator and denominator or resample whole users. |
| Use prior behavior to improve precision | Adjust outcome $Y$ using a pre-assignment outcome predictor $X$. Controlled Experiments Using Pre-Experiment Data (CUPED) can improve precision when $X$ is predictive. Prespecify covariates, missing data handling, and fitting so that treatment cannot change the inputs to the adjustment. |
| Multiple variants, metrics, or subgroups | Name a primary hypothesis and define families of comparisons. For confirmatory claims, consider Holm correction or a prespecified testing hierarchy. To claim a different effect on mobile and desktop, compare effects or test the interaction. One significant result and one nonsignificant result are insufficient. Confirm exploratory findings in a new experiment. |

One CUPED adjustment is $Y^{adj}=Y-\theta(X-\bar X)$. Fix a common $\theta$ in suitable historical data or estimate it with an analysis procedure that accounts for uncertainty. Randomization balances the pre-assignment covariate in expectation, so the adjusted and unadjusted mean differences have the same expectation under a fixed common coefficient. With one covariate and the optimal linear coefficient, the outcome variance is multiplied by $1-\rho^2$, where $\rho$ is the correlation between $X$ and $Y$. Whether a user opened this experiment's reminder is a post-assignment variable and adjusting for it would change the question. [Original CUPED paper](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf) · [Delta method for ratio metrics](https://arxiv.org/abs/1803.06336)

## Reuse a design and a readout template {#templates}

**Save a one-page design before launch.**

1. Hypothesis and audience. Record the reason for testing the reminder, the entry event, exclusions, the unit of randomization, the variants, and the allocation configuration.
2. Metrics and data. Write down the numerator, denominator, observation window, source, deduplication key, and maturity rules for the primary, guardrail, and diagnostic metrics.
3. Power and stopping. Record the baseline, MDE, business thresholds, sample size, traffic budget, enrollment window, monitoring rules, and planned readout date.
4. Decision and ownership. Specify the conditions for continuing, pausing, rolling out, or retesting, along with who handles alerts, who can disable the feature, and where to rebuild the data.

**Fill in a one-page readout after the experiment.**

1. Population. Report enrollment dates, mature user counts, allocation and event coverage by arm, SRM checks, and deviations from the plan.
2. Results and scope. Give raw counts, effect units, intervals, tests, guardrail bounds, and planned thresholds. Label exploratory findings.
3. Decision. Record whether to roll out, retest, or stop. Save the query, code version, input snapshot, and figures so the analysis can be reproduced.

## Exercises and worked answers {#exercises}

1. **One assignment, three purchases.** A user is assigned to B at noon. A purchase occurs at 11:59, a second at noon is delivered to the logs twice, and a third occurs at noon the next day. What do you count?\
   **Answer.** Deduplicate the second purchase. Count it at the inclusive start and exclude the first and third. The user contributes one row to B with a primary outcome of 1.
2. **Change the MDE to +0.5 points.** Can you keep 14,751 users per arm with all other inputs fixed?\
   **Answer.** Recalculate to about 57,763 per arm. Check the script and explain why the result is close to four times the original sample.
3. **A significant short-term conversion gain.** Use the example's counts, including the opt-out upper bound of +0.222 points. An analyst proposes recalculating only among people who saw the reminder. How do you respond?\
   **Answer.** Keep the ITT denominator. Evidence falls short of ruling out the +0.20-point opt-out limit. Use exposure logs to examine delivery and copy, then design a follow-up to test a revised version.
4. **Change the business metric.** Compare arms on total revenue divided by total sessions. Accounts were randomized, and the reminder can cause a B user to make an additional visit. Why is a session-level independent standard error inappropriate?\
   **Answer.** Visits from the same user are correlated, and treatment can change their number. Aggregate revenue and sessions by account and use the delta method or a user-level bootstrap for the ratio of totals. Explain how the ratio answers a different question from revenue per user.

## External reading path {#resources}

- **Learn the vocabulary and run the first case.** Kohavi, Tang, and Xu work through an experiment in Chapter 2 of *Trustworthy Online Controlled Experiments*. Use the book to study the full workflow, including A/A checks and ramp-up. {{< new-tab "Cambridge University Press · Book and chapters" "https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/D97B26382EB0EB2DC2019A7A7B518F59" >}}
- **Design a first experiment.** Read the sections on eligibility, metrics, and staged ramp-up. Draft the one-page plan above before comparing it with Microsoft ExP's process. {{< new-tab "Microsoft ExP · Pre-experiment stage" "https://www.microsoft.com/en-us/research/articles/patterns-of-trustworthy-experimentation-pre-experiment-stage/" >}}
- **Diagnose and interpret a readout.** Use the four stages in the SRM guide to trace users lost from the metrics table. Then study triggers, counterfactual logging, and metric interpretation. {{< new-tab "Microsoft ExP · SRM diagnosis" "https://www.microsoft.com/en-us/research/articles/diagnosing-sample-ratio-mismatch-in-a-b-testing/" >}} · {{< new-tab "Microsoft ExP · Post-experiment stage" "https://www.microsoft.com/en-us/research/articles/patterns-of-trustworthy-experimentation-post-experiment-stage/" >}}
- **Check power independently.** Reproduce 14,751 per arm with the two-proportion sample-size function, then vary the baseline and effect separately. {{< new-tab "statsmodels · Two-proportion sample size" "https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.samplesize_proportions_2indep_onetail.html" >}}
- **Analyze harder metrics.** Read the adjustment formula and covariate selection in the CUPED paper, then start with the ratio-metric example in the delta method paper. {{< new-tab "CUPED paper (PDF)" "https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf" >}} · {{< new-tab "Delta method paper" "https://arxiv.org/abs/1803.06336" >}}
- **Make a decision before the end.** Start with the continuous monitoring problem in the introduction, then read the definition and stopping rule of an anytime-valid $p$ value. {{< new-tab "Always Valid Inference" "https://arxiv.org/abs/1512.04922" >}}
