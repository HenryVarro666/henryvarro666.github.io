---
title: "Causal Inference: From Counterfactuals to a Checkout Reminder Study"
date: 2026-11-13T19:49:03-06:00
description: "Learn potential outcomes and causal graphs, then work through a checkout example with standardization, weighting, and doubly robust estimation."
topics: [data-science]
---

A checkout system used to offer reminders more often to users with low purchase intent. In the logs, 29% of users assigned an offer bought something, compared with 50% of users assigned no offer. Split users by purchase intent measured before the decision and the comparison reverses in both groups. The decision the team faces is more specific. **For the same set of eligible users, how would 24-hour purchase rates change if everyone were assigned the reminder offer versus if nobody were?**

You need proportions and weighted averages to follow the first half. Start with the [four cell counts](#simpson) or [run the Python lab](#lab). If you already have historical logs and a decision to make, start with the [study protocol](#target-trial).

## Specify the intervention, outcome, and population {#question}

The unit is a user. At their first eligible checkout decision, record an assignment and its time $t_0$. $A=1$ assigns the optional reminder offer. $A=0$ assigns no offer, with the rest of checkout held fixed. Delivery can fail and a user can ignore the reminder. The analysis uses the assigned group in either case. Set $Y=1$ when the user completes at least one purchase in $[t_0,t_0+24\text{h})$, and $Y=0$ when complete records show no purchase during that window. The target population is all eligible users in the study period. All numbers in this note are synthetic.

Three similar questions require different evidence.

| Question | Quantity | Use |
|---|---|---|
| Will this user buy tomorrow? | Probability of purchase given available information | Predict a purchase |
| Were users assigned an offer more likely to buy in the old logs? | Difference between observed purchase rates | Describe the historical association |
| How many more users would buy if we assigned the offer to the same target population? | Difference in purchase rates under two strategies | Decide whether to offer the reminder |

A purchase model can rank users who were already going to buy at the top. To evaluate the reminder, ask how much the assignment changes buying. Write $P(Y=1\mid A=1)$ for the observed purchase rate among users assigned an offer. Write $P(Y=1\mid\operatorname{do}(A=1))$ for the rate under an intervention that sets the assignment to an offer. How the old system selected users affects the first quantity. The second describes the strategy we want to evaluate. [Pearl's overview of causal inference, section 2](https://escholarship.org/uc/item/36w8n7pg)

## Write down two potential outcomes per user {#potential-outcomes}

$Y_i(1)$ is whether user $i$ would buy if assigned an offer. $Y_i(0)$ is whether they would buy if assigned no offer. For someone who would buy only under the offer, those values are 1 and 0. An assignment allows us to observe just one of them. The other is the **counterfactual**. A single log entry cannot recover $Y_i(1)-Y_i(0)$, but a study can identify an average effect for a well-defined population under suitable assumptions. [Hernán and Robins, *Causal Inference: What If*, chapters 1–3](https://miguelhernan.org/whatifbook)

| Estimand | Definition | Checkout question |
|---|---|---|
| Average Treatment Effect (ATE) | $\mathbb E[Y(1)-Y(0)]$ | What is the average effect of offering a reminder to every eligible user? |
| Average Treatment Effect on the Treated (ATT) | $\mathbb E[Y(1)-Y(0)\mid A=1]$ | What was the average effect for the users whom the old system selected? |
| Conditional Average Treatment Effect (CATE) | $\mathbb E[Y(1)-Y(0)\mid X=x]$ | What is the average effect among low-intent users or among high-intent users? |

**Specify units for absolute and relative changes.** An increase from 35% to 42.5% is **7.5 percentage points**, or about **21.4% relative to 35%**. If the effect carries over to 10,000 eligible users, it would correspond to roughly 750 additional buyers. Each user contributes at most one to the outcome, so you would need other metrics to estimate incremental orders or revenue.

In a randomized controlled trial (RCT), random assignment makes the groups comparable in expectation. An **intention-to-treat (ITT)** comparison keeps each user in the assigned group, including delivery failures and unopened reminders. It estimates the effect of assigning the strategy. To see what changes when you filter to people who saw the reminder, consider that browser state and purchase intent could affect both who sees it and who buys. The filtered sample can put unlike users on either side of the comparison. [Hernán and Robins, chapters 2 and 8](https://miguelhernan.org/whatifbook)

## Work through Simpson's paradox {#simpson}

Let $X$ record purchase intent before assignment. The old policy assigns the offer to 80% of low-intent users and 20% of high-intent users. Each cell uses the number of users in that intent stratum and assignment group as its denominator.

| Prior purchase intent | Offer assigned | No offer assigned | Gap within stratum |
|---|---:|---:|---:|
| Low (200 users) | 32 / 160 = 20% | 4 / 40 = 10% | +10 percentage points |
| High (200 users) | 26 / 40 = 65% | 96 / 160 = 60% | +5 percentage points |
| Crude overall (400 users) | 58 / 200 = 29% | 100 / 200 = 50% | −21 percentage points |

![The offer group has a higher purchase rate within both intent strata. The crude overall rates reverse direction, and the standardized rates favor the offer again.](simpson-rates.svg "Figure 1. Four in five users in the offer arm have low intent. Four in five in the no-offer arm have high intent. Combining the groups as observed compares two different mixes.")

Our target population is half low intent and half high intent. Use that same mix for both strategies. This is **standardization**. The estimated purchase rate for offering to all is $0.5\times20\%+0.5\times65\%=42.5\%$. For offering to none it is $0.5\times10\%+0.5\times60\%=35.0\%$. The difference is **+7.5 percentage points**. To estimate the ATT for the old policy, use the mix in the offer arm instead. With 80% low intent and 20% high intent, the effect is $0.8\times10+0.2\times5=9$ percentage points. A causal interpretation depends on the assumptions below. [Hernán and Robins, chapter 3 on standardization](https://miguelhernan.org/whatifbook)

## Use a causal graph to choose what to adjust for {#dag}

A **directed acyclic graph (DAG)** represents assumptions about how the data came to be. An arrow specifies an assumed causal direction. Before drawing, ask how the old assignment rule worked and when each field was recorded. In this example $X$ affects both assignment and purchase, creating the backdoor path $A\leftarrow X\rightarrow Y$. Compare within $X$ first, then average over the target population.

![Prior intent points to both assignment and purchase. A mediator sits along a path from offer to purchase. A collider is caused by both the offer and a separate cause of purchase.](intent-dag.svg "Figure 2. Confounders, mediators, and colliders play different roles in the graph. The effect you want and the paths in the graph determine what to adjust for.")

| Role in the graph | Checkout example | When estimating the total effect of assignment |
|---|---|---|
| Confounder | Prior intent $X$ influences the assignment and whether users buy. | Adjust for $X$ to close the backdoor path in this graph. |
| Mediator | The reminder brings some users back to checkout $M$, which can lead to purchase. | Leave the path $A\rightarrow M\rightarrow Y$ open so that it contributes to the total effect. |
| Collider | Reopening the app $S$ is affected by both assignment and an independent promotion $U$, which also affects purchase. | Filtering to app reopeners induces an association between $A$ and $U$ in the selected sample, opening a path from $A$ to $Y$. |

Roles depend on the causal question. A field recorded before assignment is a candidate for adjustment. Check the graph for paths that conditioning on it would open. A regression coefficient also needs a specific interpretation. With a treatment-by-intent interaction, for example, compute each stratum's effect and average over the target population. [Pearl's backdoor criterion](https://escholarship.org/uc/item/36w8n7pg) · [DAGitty](https://dagitty.net/)

### Identify the effect before estimating it {#identification}

**Identification** asks whether the observed data distribution and the assumptions determine the causal quantity. **Estimation** asks how to calculate it from the available sample and describe its uncertainty. Three assumptions let us use standardization here.

1. **Consistency.** Define each strategy precisely. For a user assigned $A=a$, we observe $Y=Y(a)$. Different reminder messages, timing, and concurrent discounts can amount to different versions of the intervention. This setup also assumes no interference. Another user's assignment cannot change the first user's outcome. A shared coupon budget could require a different intervention or randomization unit.
2. **Conditional exchangeability.** Given $X$, potential outcomes are independent of assignment, written $Y(a)\perp A\mid X$. This requires adjusting adequately for common causes of assignment and outcome. If the old policy also read an unlogged marketing list and membership affects purchases, intent alone is insufficient.
3. **Positivity.** Both assignments must be possible at every $X=x$ present in the target population, or $0<P(A=1\mid X=x)<1$. A rule that always offers a reminder in one stratum leaves no observed comparison arm there. A tiny arm in finite data is also a practical overlap problem.

Expand the population mean of $Y(a)$ over $X$, then use exchangeability and consistency to replace each conditional potential outcome mean with an observed mean.

$$
\mathbb E[Y(a)]
=\sum_x\mathbb E[Y(a)\mid X=x]P(X=x)
=\sum_x\mathbb E[Y\mid A=a,X=x]P(X=x)
$$

Review the policy and data collection before choosing a set of covariates. More data reduces sampling error, and an unlogged common cause or a missing comparison arm can still leave the effect unidentified. [Hernán and Robins, chapters 3 and 7](https://miguelhernan.org/whatifbook)

## Estimate the effect three ways in Python {#lab}

Download [causal-lab.py](/notes/causal-inference/causal-lab.py) and run it with Python 3. It uses only the standard library and contains the four cell counts.

~~~bash
python3 causal-lab.py
~~~

| Output | Value | Calculation |
|---|---:|---|
| Crude difference | −21.0 percentage points | Use each arm's own mix |
| Standardized difference | +7.5 percentage points | Use 50% low intent and 50% high intent in both arms |
| Inverse probability weighting | +7.5 percentage points | Reweight the observations to represent the target population |
| Augmented inverse probability weighting | +7.5 percentage points | Add weighted residual corrections to predicted outcomes |

### 1. Outcome models and standardization {#outcome-adjustment}

Estimate purchase risk $m_a(x)=\mathbb E[Y\mid A=a,X=x]$ under each assignment. The four observed proportions suffice here. For many or continuous covariates, fit an outcome model and predict each target user's purchase probability twice, once with assignment set to 1 and once with it set to 0. Average the predicted differences. This is regression standardization, an application of the g-formula. If the outcome model gets the risk wrong in an important subgroup, the error carries over to the average. [Hernán and Robins, chapter 13](https://miguelhernan.org/whatifbook)

You can compute the first estimate with two multiplication steps. Change the target weights to `(0.7, 0.3)` for the first exercise.

~~~python
target_weights = (0.5, 0.5)  # low and high intent
offer_risks = (32 / 160, 26 / 40)
no_offer_risks = (4 / 40, 96 / 160)
ate = sum(
    weight * (offer - no_offer)
    for weight, offer, no_offer in zip(target_weights, offer_risks, no_offer_risks)
)
print(f"{100 * ate:+.1f} percentage points")
~~~

### 2. Inverse probability weighting {#ipw}

The **propensity score** is $e(x)=P(A=1\mid X=x)$, the probability of assignment given baseline features. **Inverse Probability Weighting (IPW)** weights users assigned the offer by $1/e(X)$ and users assigned no offer by $1/(1-e(X))$. A low-intent user rarely went without an offer in the old policy, so give each of those users weight $1/0.2=5$. Each low-intent user with an offer gets weight $1/0.8=1.25$. Sum the weighted purchases within each arm and divide each sum by the total number of users, 400.

$$
\hat{\tau}_{\mathrm{IPW}}=\frac{1}{n}\sum_{i=1}^n
\left[\frac{A_iY_i}{\hat e(X_i)}-\frac{(1-A_i)Y_i}{1-\hat e(X_i)}\right]
$$

For instance, the resulting purchase risk for offering to all is $(32/0.8+26/0.2)/400=42.5\%$. If someone's observed assignment has probability 0.01, their weight is 100. Report balance before and after weighting, the propensity distribution by arm, and each arm's effective sample size $\mathrm{ESS}=(\sum_i w_i)^2/\sum_i w_i^2$. Inspect the observations with the largest weights. If you trim to a region with common support, name the new target population. If you cap weights, report the cap and its bias and variance trade-off. [Hernán and Robins, chapter 12](https://miguelhernan.org/whatifbook)

### 3. Augmented weighting and machine learning {#aipw}

**Augmented Inverse Probability Weighting (AIPW)** starts with the difference in predicted outcomes. It corrects each arm using its weighted observed-minus-predicted outcomes.

$$
\begin{aligned}
\hat\tau_{\mathrm{AIPW}}=\frac{1}{n}\sum_i\Bigg[
 &\hat m_1(X_i)-\hat m_0(X_i) \\
 &+\frac{A_i[Y_i-\hat m_1(X_i)]}{\hat e(X_i)}
 -\frac{(1-A_i)[Y_i-\hat m_0(X_i)]}{1-\hat e(X_i)}
\Bigg]
\end{aligned}
$$

Given the identification assumptions and suitable estimation conditions, AIPW can be consistent when the assignment model is correct or when the two outcome risks are correctly modeled. This is **double robustness**. The script replaces each model in turn with incorrect constants and still gets +7.5 percentage points. Replacing both models gives −21 points in this example. These exact equalities follow from fitting the cell counts used for the illustration. They generally will not hold in finite samples. [Bang and Robins on doubly robust estimation](https://pubmed.ncbi.nlm.nih.gov/16401269/)

With many features, use flexible predictors such as trees to estimate $e$ and $m_a$. **Double/Debiased Machine Learning (DML)** uses an orthogonal score and **cross-fitting**. Train the nuisance models on other folds so each observation gets predictions from models that did not train on it. Then aggregate the estimated effect scores. Valid inference still depends on overlap, sample structure, and how accurately the nuisance functions are learned. To find which users benefit more, try causal forests after checking overlap and look for evidence of practical value on independent data. [Chernozhukov et al. on DML](https://arxiv.org/abs/1608.00060) · [Athey and Wager's causal forest application](https://arxiv.org/abs/1902.07409)

The script also gives an illustrative 95% interval of **[−2.5, +17.5] percentage points**, using fixed target weights and independent binomial cells. It includes zero, so the +7.5 point estimate comes with substantial sampling uncertainty in this example. Real studies may need to account for dependence within stores or regions when they share a rollout decision. Investigate hidden confounding separately through the study design and sensitivity analysis.

### 4. Matching {#matching}

To inspect comparisons user by user, match users on important baseline features. Matching each offered user to a comparable no-offer user can estimate an ATT when the design supports it. Check covariate balance after matching and report how many offered users had no suitable match. If you drop those users, state that the estimate applies to the matched portion of the offered group. Matching, standardization, and weighting all depend on what you measured and where both arms are observed. [Scott Cunningham on unconfoundedness and matching](https://mixtape.scunning.com/05-causal_cross_sectional_designs)

## Choose a design based on how treatment is assigned {#design-choice}

First ask whether you can randomize. If you are working with historical data, look for a mechanism that supplies a credible comparison.

| Available evidence | Design and checkout example | What the design can identify | What to check first |
|---|---|---|---|
| Control over assignment | **Randomized experiment.** Assign each user at their first eligible checkout. | The assignment effect for the trial population | Confirm stable assignment, comparable outcome collection, and whether missing users differ by arm. |
| Historical logs that capture the assignment rules | **Backdoor adjustment.** Adjust for prior intent, eligibility rules, and other common causes. | ATE or ATT for a specified population given conditional exchangeability and overlap | Ask whether the old rules read unlogged information and look for strata with sparse or empty arms. |
| A rollout with before-and-after data and credible comparison markets | **Difference-in-Differences (DiD).** Pilot markets go from 30% to 36% and comparison markets from 28% to 31%. The difference is $(36-30)-(31-28)=3$ percentage points. | The effect for the pilot markets under parallel untreated trends and other design assumptions | Check for spillovers, anticipation, and concurrent market specials. Inspect several pre-periods. With staggered adoption, use a method that distinguishes adoption cohorts and periods. [DiD chapter](https://mixtape.scunning.com/09-difference_in_differences) · [Callaway and Sant'Anna](https://arxiv.org/abs/1803.09015) |
| Assignment changes at a known cutoff | **Regression Discontinuity (RD).** The rule offers reminders for baseline scores below 50. Fit locally on either side and estimate the purchase-rate jump at 50. | An effect near the cutoff under continuity of potential outcomes and a credible threshold rule | Inspect score bunching, other policies at 50, baseline covariates, and sensitivity to bandwidth. [Cattaneo et al. on RD](https://arxiv.org/abs/1906.04242) |
| Randomized encouragement with incomplete receipt | **Instrumental Variables (IV).** A randomized assignment $Z$ increases actual delivery by 40 percentage points and purchasing by 2 points. The Wald ratio is $0.02/0.40=0.05$. | The **Local Average Treatment Effect (LATE)** for users whose receipt changes with the encouragement, under relevance, independence, exclusion, and monotonicity | Report the strength of the first stage. If assignment changes the page or applies a discount directly, exclusion is doubtful. The 2-point purchase gap itself is the assignment effect. [Angrist, Imbens, and Rubin](https://www.nber.org/papers/t0136) |

For a staged rollout or threshold rule, say which markets, time periods, or score ranges the effect represents. When reusing a past experiment, check whether its checkout flow and buying cycle still describe the new launch.

## Turn historical logs into a target trial protocol {#target-trial}

Write out the trial you would run with random assignment and compare its protocol with what the historical data can reproduce. **Target trial emulation** makes it possible to evaluate the eligibility rules, assignment, start of follow-up, and treatment of missing outcomes before estimating an effect. [Hernán and Robins on target trial emulation](https://pubmed.ncbi.nlm.nih.gov/26994063/)

![The eligibility check and assignment anchor the start of the clock. Both groups count purchases during the same 24-hour window.](target-trial.svg "Figure 3. At t₀, check eligibility, fix baseline variables, and record assignment. Use the same outcome definition and follow-up window in both groups.")

| Protocol item | Ideal randomized trial | Emulation with historical logs |
|---|---|---|
| Eligibility and baseline | Each user's first checkout that meets the study's criteria, regardless of future behavior | Find the first eligible user event in the eligibility log. Freeze baseline information as of that decision. |
| Strategies | Randomly assign the specified offer or no-offer strategy at that decision | Record the policy decision time, rule version, and original group. Identify eligible users in the no-offer group too. |
| Outcome and follow-up | Use the same purchase source and 24-hour window for both arms | Join purchases by user, filter to the window, and collapse to one row per user. Include a window after follow-up closes and its events are complete. |
| Analysis | Report the absolute purchase-rate difference by assignment, an interval, and guardrails | Choose covariates from the policy and graph, diagnose overlap, then report standardization and a weighting or AIPW comparison. |
| Assumptions and limitations | Document randomization, interference, missing outcomes, and execution failures | Explain whether the logs capture all of the policy inputs, whether missing outcomes vary by arm, and how these issues affect interpretation. |

A common timing error labels users by whether they open a reminder in the next 24 hours but starts tracking purchases at checkout. This groups users according to future behavior. Group them at the policy decision and use clicks and opens to diagnose implementation. Incomplete windows should wait for a common data watermark. Missing logs need investigation and a reported missingness rate. Use a left join to preserve everyone who was assigned and encode no matched purchase as zero once you establish that the outcome was observable.

### Check the analysis before making a decision {#diagnostics}

- **Denominator.** Build a roster with one eligible assignment per user. Check for joins that multiply records. Compare missing outcome rates and ingestion lags by arm.
- **Comparability.** Plot important baseline variables and propensity scores. If an entire region has only one arm, report the gap and define any effect for the population with overlap explicitly.
- **Hidden confounding.** Suppose an unlogged coupon was issued before assignment and raises the chance of purchase by 10 percentage points. Within each intent stratum, the share already holding a coupon is 20 percentage points higher in the offer group. A simple additive bias model gives $0.10\times0.20=0.02$, or 2 percentage points of upward bias. Try a range of scenario values and show where the decision changes. [Cinelli and Hazlett on sensitivity analysis](https://carloscinelli.com/files/Cinelli%20and%20Hazlett%20-%20Making%20Sense%20of%20Sensitivity.pdf)
- **Falsification.** Check whether assignment predicts a purchase from before assignment as a negative control outcome. If it does, inspect policy rules, omitted confounders, and data extraction. Treat the check as one diagnostic.
- **Decision.** Report reminder costs, opt-outs, complaints, and repeat purchases alongside the estimated difference and its interval. If a key input to the historical assignment policy is missing, plan a randomized trial to fill the evidence gap. [A/B Testing study note]({{< relref "/notes/ab-testing" >}})

### Apply the same protocol to brain MRI review {#mri-example}

Suppose a brain magnetic resonance imaging (MRI) lab wants to evaluate whether draft segmentation reduces manual review time. The intervention is making the draft available at the start of review. The outcome is minutes of reviewer time per scan, with the final quality check as a guardrail. Use all scans assigned to each workflow, including those whose draft a reviewer chooses to discard.

If reviewers can be assigned each scan under either workflow, consider randomizing within reader and site, accounting for the same reader evaluating multiple scans. If seeing drafts on earlier scans might change later work, plan for carryover or randomize at reader or session level and account for that design in the analysis. In historical data, check whether scanner, protocol, reader experience, and baseline scan quality drove both workflow choice and time. If only one center ever used segmentation assistance, the observed difference may reflect center workflows. To estimate the effect, find comparable scans and readers in both workflows or plan a trial that varies the workflow within sites.

## Exercises and worked answers {#exercises}

1. **What if the target population is 70% low intent?** If within-stratum effects carry over, use $0.7\times10+0.3\times5=8.5$ percentage points. Reweighting the two potential outcome means separately gives the same difference.
2. **What if every low-intent user received an offer?** That stratum's no-offer risk is missing, so standardization cannot estimate the original population's effect from these cells. The script raises an error. You could randomize a holdout in that stratum, or define a separate estimand for the population with overlap.
3. **Should we analyze only people who returned to checkout?** Their purchase rate can help diagnose the checkout funnel. For the total assignment effect, use the complete roster. Restricting to returners can remove part of the mediated effect and introduce selection bias.
4. **DiD gives +3 percentage points, but the pilot markets started offering free shipping on the same day. Can we attribute the increase to the reminder?** Free shipping creates a concurrent shock only in the pilot markets. Look for an adoption period or comparison group without the shock, or assign the offer randomly.
5. **How would you report the interval that includes zero?** “Standardization estimates a 7.5 percentage point increase for the specified population. The illustrative 95% interval extends from −2.5 to +17.5 points, so sampling uncertainty is substantial under the stated assumptions.” Then state the causal assumptions you used and how you would check them.

## A reading plan with practice {#resources}

| Stage | Read or watch | Practice |
|---|---|---|
| Start here (60–90 min) | Chapters 1–3 of {{< new-tab "Causal Inference: What If, Hernán and Robins" "https://miguelhernan.org/whatifbook" >}}. Pair with the Potential Outcomes and Backdoor Adjustment lectures from {{< new-tab "Brady Neal's free course" "https://www.bradyneal.com/causal-inference-course" >}}. | Compute 29%, 50%, 42.5%, and 35% by hand. Explain the denominator and weights in each. |
| Draw graphs (30 min) | Read the backdoor criterion in {{< new-tab "Causal inference in statistics, Judea Pearl" "https://escholarship.org/uc/item/36w8n7pg" >}}. Check a graph in {{< new-tab "DAGitty" "https://dagitty.net/" >}}. | Add a concurrent promotion to the checkout graph. Compare the available adjustment sets with and without promotion records. |
| Estimate effects (60 min) | Read *What If*, chapters 12–13. Run the {{< new-tab "PyWhy DoWhy introductory notebook" "https://www.pywhy.org/dowhy/v0.13/example_notebooks/dowhy_simple_example.html" >}} to specify a graph, identify an effect, estimate it, and try a refutation. | Change the target population weights in the formula or remove an arm in the Python example. Predict what happens before running it. |
| Study quasi-experiments (about 60 min each) | Read Scott Cunningham's chapters on {{< new-tab "DiD" "https://mixtape.scunning.com/09-difference_in_differences" >}}, {{< new-tab "RD" "https://mixtape.scunning.com/06-regression_discontinuity" >}}, and {{< new-tab "IV" "https://mixtape.scunning.com/07-instrumental_variables" >}}. | For each method, write down one identification assumption, one practical check, and the population the effect represents. |
| Work through an application (2–3 h) | Start with the {{< new-tab "target trial paper by Hernán and Robins" "https://pubmed.ncbi.nlm.nih.gov/26994063/" >}}. For high-dimensional estimation, consult the {{< new-tab "original DML paper" "https://arxiv.org/abs/1608.00060" >}}. | Write a one-page protocol. Review it for how treatment was assigned, completeness of outcomes, overlap, and sensitivity to confounding. |

Continue with the same checkout example in [A/B Testing]({{< relref "/notes/ab-testing" >}}) to design a randomized comparison, [Spark and PySpark]({{< relref "/notes/spark-pyspark" >}}) to compute user-level outcomes, and [Airflow]({{< relref "/notes/airflow" >}}) to schedule work by data interval.
