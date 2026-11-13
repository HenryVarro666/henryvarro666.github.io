---
title: "ML foundations and AI interview prep"
date: 2026-11-13T17:29:06-06:00
description: "A study route from classic machine learning questions to coding from scratch, system design, and language models, with links to practice problems and GitHub repositories."
topics: [data-science, llm]
---

This is a study index for machine learning (ML) and artificial intelligence (AI) interviews. Start with the fundamentals, then write one implementation from a blank file. After that, follow the work the interview will cover. For data science, spend more time on data splits, evaluation, and A/B tests. For applied ML engineering, add a design case. For research engineering or large language model (LLM) roles, spend more time writing and debugging Transformer code.

## A 45-minute first pass

1. **Explain a model for 15 minutes.** Open [MLQuestions](https://github.com/andrewekhalel/MLQuestions) and pick a question about overfitting or evaluation. Close the answer and spend one minute explaining how you would spot the problem and what you would check next.
2. **Write code for 20 minutes.** Try [the Deep-ML softmax problem](https://www.deep-ml.com/problems/23). Use large inputs to check numerical stability. Then add the same constant to every score and check that the output stays the same.
3. **Spend 10 minutes on a follow-up.** For classic ML, read [Precision vs. recall](/notes/precision-vs-recall/) and explain who is affected when you raise a decision threshold. For language models, open *Let's build GPT* in [Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) and sketch which tokens each position can see through a causal mask.

## ML foundations: explain the choice

- **Probability and models.** Look up conditional probability, expectation, or variance in the [Probability Cheatsheet](https://github.com/wzchen/probability_cheatsheet). Use [ML-NLP](https://github.com/NLP-LOVE/ML-NLP) for Chinese explanations of logistic regression, decision trees, and ensembles, then check for gaps with the [AIMLInterviews fundamentals chapter](https://github.com/alirezadir/AIMLInterviews/blob/main/src/ml-fundamental.md).
- **Validation and evaluation.** Pick classic questions on overfitting, leakage, and class imbalance from the [Educatum AI/ML Interview Playbook](https://www.educatum.com/the-ai-ml-interview-playbook). Compare the [areas under the receiver operating characteristic (ROC) and precision–recall (PR) curves](/notes/roc-auc-vs-pr-auc/). Now suppose positives are rare and you can review only 100 cases a day. Which metric would you report, and how would you select a threshold on the validation set?
- **Statistics and experiments.** Pick a statistics question from [Data Science Interview Questions and Answers](https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers). Then try an A/B test scenario: if the same user visits multiple times a day, what can go wrong when you assign a variant per visit? State the unit of randomization and the primary metric before discussing significance.

You have done enough for this pass when you can pick a baseline for a tabular dataset, split the examples the way the task calls for, and explain what result would make you try a different model.

## ML coding: from NumPy to a Transformer

- **Short problems.** Pick one problem each on matrix operations, gradient descent, and clustering from [Deep-ML](https://www.deep-ml.com/problems). The [AIMLInterviews coding chapter](https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLC/ml-coding.md) has more prompts. Write down the input and output shapes, check an example by hand, then test an edge case.
- **Two-layer network.** Start with *micrograd* in [Zero to Hero](https://github.com/karpathy/nn-zero-to-hero), then look at backpropagation through a multilayer perceptron. Write the forward and backward passes for a small two-layer network and use finite differences to check the gradient for one weight.
- **Small language model.** Try the [multi-head attention exercise](https://www.deep-ml.com/problems/94) for tensor shapes, then add a causal mask in your own version. [Stanford CS336 Assignment 1](https://github.com/stanford-cs336/assignment1-basics) brings tokenization, Transformer components, and a training loop together. If time is short, implement one component and pass its tests.

## ML system design: put the model in a product

Start with the [AIMLInterviews system design guide](https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/ml-system-design.md). Use [Chip Huyen's Machine Learning Systems Design](https://huyenchip.com/machine-learning-systems-design/toc.html) to revisit gaps in data, deployment, or monitoring.

For a practice case, look at retrieval and ranking in [Datawhale fun-rec](https://github.com/datawhalechina/fun-rec) (in Chinese). Sketch how a recommendation moves from candidate retrieval through ranking and display, then back to training as feedback. Say when your labels will be available. If offline ranking improves but the live click rate stays flat, start with one check: experiment assignment, retrieval coverage, or serving latency. Explain what the result would rule out.

## Language models and AI engineering: pick a specialty

- **Model internals.** Use [Dive into LLMs](https://github.com/Lordog/dive-into-llms) (in Chinese) to review Transformers and training. Then pick a question to answer aloud from [Educatum's advanced language model and generative AI questions](https://www.educatum.com/llm-and-genai-advanced-interview-questions). Start by explaining how causal attention keeps a model from reading future tokens.
- **Retrieval and agents.** Run the semantic search and retrieval-augmented generation (RAG) examples in [Hands-On Large Language Models, Chapter 8](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main/chapter08). Pick one failure and inspect the retrieved evidence separately from the generated answer. For agent interviews, try the tool calling and evaluation exercises in [learn-harness-engineering](https://github.com/walkinglabs/learn-harness-engineering) and pick a prompt from [Educatum's AI engineering questions](https://www.educatum.com/ai-engineering-interview-questions) to answer aloud.
- **Inference and post-training.** For inference, write a key–value (KV) cache and check that token-by-token decoding matches a full forward pass within floating-point tolerance. Then read [vLLM](https://github.com/vllm-project/vllm) and reason through the effect of a longer context on memory and latency. For post-training, trace a reinforcement learning loop in [verl](https://github.com/verl-project/verl), from generation to reward calculation and the policy update.

For more questions, pick a category in [Cheng Zhang's ML engineering question bank](https://chengz.art/en/mle-prep), or use [Silvia Sapora's interview guide](https://silviasapora.github.io/blog/ml-interviews.html) to choose another coding or debugging exercise. Write down the first follow-up you could not answer and try it again the next day.
