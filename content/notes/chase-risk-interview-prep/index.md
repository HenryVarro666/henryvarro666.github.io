---
title: "Interview practice for a Chase fraud modeling role"
date: 2026-11-23T00:00:00-06:00
description: "A 60-minute mock panel for a consumer fraud modeling role, with worked answers on review capacity, delayed labels, model selection, and customer impact."
topics: [data-science, ml-systems]
resourceGuide: true
---

Suppose a new model finds ten more fraudulent payments a week, but the old model catches more dollars. Which one would you recommend? This question gets to the heart of Chase's **Risk Program Senior Associate** posting in Plano. A good answer explains what happens after a payment is flagged, who bears the cost of a mistake, and which evidence would change the recommendation.

[Read the full bilingual questions and answers](/zh/notes/chase-risk-interview-prep/) · [Download the practice workbook (PDF)](chase-risk-interview-prep.pdf) · [Download the editable workbook (Word)](chase-risk-interview-prep.docx)

## What the role and candidate reports establish

Requisition **210788388** covers fraud ranking for transactions and applications for new products. The job description asks for two or more years of Python data analysis and commercial modeling experience using at least three of the listed techniques. The list spans CNNs and RNNs, reinforcement learning, random forests and gradient boosting, transformers, and graph methods. For each technique you have actually used, prepare one decision you made, the alternative you tested, and how you evaluated it. [Job description with requisition number](https://www.olive.jobs/job/pst_a1d582d70a429280)

A Plano candidate for **Risk Strategic Analytics, Associate** reported that three colleagues covered a business case, technical questions, and behavioral questions in the second round in 2023. A separate, family-posted account of a 2022 **Risk Modeling Program** interview describes a research presentation and a conversation about explaining work to people outside statistics. These are observations from adjacent roles. The mock panel, interview prompts and worked scenarios below are original practice examples with invented details. [Plano candidate report](https://www.glassdoor.com/Interview/JPMorganChase-Interview-Questions-E5224839.htm?filter.jobTitleExact=+Risk+Strategic+Analytics%2C+Associate) · [Risk Modeling Program account](https://www.1point3acres.com/bbs/thread-1085929-1-1.html)

## Run a 60-minute mock panel

| Minutes | Interviewer prompt | What the listener should hear |
|---|---|---|
| 0–10 | “Pick one project. Which decision was yours?” | A two-minute account of the user, decision, baseline, personal contribution and result. |
| 10–25 | “Explain the logit. Now calculate AUC for five payments.” | An interpretation in plain English, followed by correct arithmetic and denominators. |
| 25–43 | “With 1,000 reviews available, should we adopt this model?” | An operating point, a business objective, and a reason to revisit the decision. |
| 43–54 | “Tell us about a disagreement over customer friction.” | Both parties' reasoning, the evidence you brought and what changed. |
| 54–60 | “What would you like to ask us?” | One question about model ownership and another about how the team measures a launch. |

Ask one listener to challenge the data, another to challenge the method, and the third to ask for a decision. Record the first attempt. Replay the point where you switched from describing the model to recommending an action. If the listener still has to ask what you would do, try delivering that recommendation in two sentences.

## Case 1. A full review queue

In one week, 200 of 100,000 payments were eventually confirmed as fraudulent. Operations can review 1,000. The current model puts 120 frauds worth &#36;36,000 in that queue. A challenger puts 130 worth &#36;32,500 there. All figures in this exercise are hypothetical.

**Which model wins?** “At the same workload, the challenger raises precision from 12% to 13% and recall from 60% to 65%. The current queue contains &#36;3,500 more fraud exposure. If our objective is preventable loss, I would keep the current model while we check what reviewers can recover in each queue.”

**What does that difference buy?** Suppose review prevents half the flagged fraud dollars and costs &#36;3 per alert. The current queue yields an estimated &#36;18,000 in prevented losses less &#36;3,000 in costs. The challenger yields &#36;16,250 less &#36;3,000. The estimated weekly advantage is &#36;1,750. A different intervention success rate could reverse the decision. Check whether a few high-value cases dominate, repeat on later weeks, and measure the burden on genuine customers. A review request and a declined payment create different costs.

## Case 2. A new applicant with no history

**How would you decide whether to ask for extra verification?** “First I would define the decision time, the fraud outcome, and how long it takes to confirm that outcome. I would start with application information available at submission. Next I would test whether links to previously seen devices improve fraud capture at a fixed referral rate. I would track how often legitimate applicants sharing a device get referred and how many complete verification.”

**What would make a backtest misleading?** A graph feature can leak the future if it counts later applications sharing the same device. Build the graph as it was known at each historical decision. A fraud confirmed on Friday could not have informed Tuesday's score. Declined applicants may never produce an observable downstream fraud outcome. Describe which applicants each metric represents and what independent verification could help assess the others.

## Technical questions you can check by hand

**What does doubling the odds mean?** In logistic regression, a coefficient of log(2) doubles the odds for a one-unit increase, holding the other predictors fixed, in a model without an interaction involving that variable. Starting from a fraud probability of 20%, the odds are 1 to 4. Doubling them gives 1 to 2, or a probability of one third.

**Can you calculate AUC with a tie?** Two frauds score 0.9 and 0.6. Three genuine payments score 0.8, 0.6 and 0.1. Across six fraud–genuine pairs, the first fraud wins three comparisons. The second wins one and ties one. Counting the tie as half gives an AUC of 4.5/6, or 0.75. At a threshold of 0.6 inclusive, four payments are flagged, two are fraud, precision is 50%, and recall is 100%. The false-positive rate is two out of three genuine payments. [ROC AUC and PR AUC](/notes/roc-auc-vs-pr-auc/)

**Why test a boosted tree before a sequence model?** “I can train a tabular baseline using prior transaction counts, amounts and merchant history. I would test whether event order adds information beyond those summaries, then compare both models on a later period at the same queue size. A small gain may disappear if sequence features arrive too late for authorization.”

**How would you check a Python feature?** For each payment, count the customer's transactions in the previous 24 hours. Exclude the current transaction. Test an empty history, two records with identical timestamps, and an older payment whose record arrived after scoring. Give the offline calculation and the service the same availability rules.

## Put your own experience into the answer

Pick a project where a real result changed your next step. Say who used it, what you personally chose, how the baseline performed on the same data, and the outcome you could verify. If the result was measured offline, give the evaluation window and describe what you would test in production. Research experience can supply a strong example when you can explain why a promising first experiment failed and how you found the problem.

For a disagreement, practice this sequence aloud. “My colleague preferred [action] because [reason]. I was concerned about [consequence]. We compared [options] on [evidence]. That changed my view about [point]. We agreed to [decision], then observed [actual outcome].” Before finishing, ask the team which decision a new associate would own and what evidence would convince them that the decision had improved customer outcomes.
