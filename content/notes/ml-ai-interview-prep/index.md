---
title: "ML foundations and AI interview prep"
date: 2026-11-13T17:29:06-06:00
description: "A study route from classic machine learning questions to coding from scratch, system design, and language models, with links to practice problems and GitHub repositories."
topics: [data-science, llm]
resourceGuide: true
---

This study guide covers machine learning (ML) and artificial intelligence (AI) interviews. Try the first practice session, then pick the topic your interview calls for. Spend more time on data splits and evaluation for data science, system design for applied ML engineering, and implementation and debugging for research engineering or large language model (LLM) roles.

## A 45-minute first pass

1. **Explain a model for 15 minutes.** Pick an overfitting or evaluation question. Close the answer and spend one minute explaining how you would spot the problem and what you would check next. [MLQuestions fundamentals](https://github.com/andrewekhalel/MLQuestions#machine-learning-fundamentals)
2. **Write code for 20 minutes.** Implement softmax from scratch. Test it on large inputs, then add the same constant to every score and check that the output stays the same. [Deep-ML softmax exercise](https://www.deep-ml.com/problems/23)
3. **Spend 10 minutes on a follow-up.** For classic ML, explain how raising a threshold changes false positives and false negatives. For language models, sketch which tokens each position can see through a causal mask. [Precision vs. recall](/notes/precision-vs-recall/) · [Zero to Hero GPT lesson](https://karpathy.ai/zero-to-hero.html)

## ML foundations: explain the choice

- **[Probability Cheatsheet](https://github.com/wzchen/probability_cheatsheet)**\
  A compact reference for probability formulas and common distributions.\
  **Study** Conditional probability, total expectation, and variance. Pick a two-stage sampling example and use conditioning to compute the overall expectation.

- **[ML-NLP](https://github.com/NLP-LOVE/ML-NLP)**\
  Chinese-language notes on algorithms with code and interview topics.\
  **Study** Compare logistic regression, decision trees, and ensembles. Explain what result would make you replace a logistic regression baseline.

- **[AIMLInterviews · ML Fundamentals](https://github.com/alirezadir/AIMLInterviews/blob/main/src/ml-fundamental.md)**\
  The fundamentals chapter of a broader technical interview guide.\
  **Study** Use validation performance to distinguish underfitting from overfitting. Explain how you would split the data.

- **[Educatum · AI/ML Interview Playbook](https://www.educatum.com/the-ai-ml-interview-playbook)**\
  Study paths and classic interview questions organized by topic.\
  **Study** Pick a question on data leakage or class imbalance. With only 100 reviews available per day, choose a metric and explain how you would set the threshold on the validation set. [Areas under the receiver operating characteristic and precision–recall curves](/notes/roc-auc-vs-pr-auc/)

- **[Data Science Interview Questions and Answers](https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers)**\
  A question bank covering statistics, probability, and modeling.\
  **Study** Review sampling and uncertainty, then design an A/B test. If each user can visit several times, explain the unit of randomization and the primary metric.

- **[MLQuestions](https://github.com/andrewekhalel/MLQuestions)**\
  Technical interview questions on machine learning and computer vision.\
  **Study** Pick an evaluation question and revisit your first answer. Describe the pattern of training and validation performance that would make you suspect overfitting. [Model evaluation and validation questions](https://github.com/andrewekhalel/MLQuestions#model-evaluation-and-validation)

## ML coding: from NumPy to a Transformer

- **[Deep-ML](https://www.deep-ml.com/problems)**\
  Python exercises you can run and test in the browser.\
  **Study** Work from matrix operations and gradient descent to multi-head attention. Write down the tensor shapes, then add a causal mask to your own implementation. [Multi-head attention exercise](https://www.deep-ml.com/problems/94)

- **[AIMLInterviews · ML Coding](https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLC/ml-coding.md)**\
  Practice prompts for implementing algorithms and model components in interviews.\
  **Study** Pick a familiar problem and implement it under a time limit. Test one example you can check by hand and one edge case.

- **[Karpathy · Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero)**\
  A hands-on course that starts with automatic differentiation and builds up to a small language model.\
  **Study** Use micrograd to review computation graphs. Implement the forward and backward passes of a two-layer perceptron, then check one weight with finite differences.

- **[Stanford CS336 · Assignment 1](https://github.com/stanford-cs336/assignment1-basics)**\
  A course assignment that brings tokenization, Transformer components, and the training loop together.\
  **Study** Implement one attention component and pass its tests before putting it back in the model. [Assignment handout](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_assignment1_basics.pdf)

## ML system design: put the model in a product

- **[AIMLInterviews · ML System Design](https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/ml-system-design.md)**\
  A framework for answering system design questions, with worked cases.\
  **Study** Turn the product goal into something you can measure. Define the label and offline metric, then say what you would check after launch.

- **[Chip Huyen · Machine Learning Systems Design](https://huyenchip.com/machine-learning-systems-design/toc.html)**\
  A guide to data pipelines, training, and serving.\
  **Study** Read the data pipeline and serving sections. Check which training features are available at prediction time and how you would measure serving latency. [System design chapter](https://huyenchip.com/machine-learning-systems-design/design-a-machine-learning-system.html)

- **[Datawhale fun-rec](https://github.com/datawhalechina/fun-rec)**\
  A Chinese-language recommendation systems textbook and exercises, from retrieval through reranking.\
  **Study** Start with the two-tower model. Explain how many candidates each stage handles, then sketch how impressions become training feedback. [Two-tower model chapter](https://datawhalechina.github.io/fun-rec/chapter_1_retrieval/3.two_tower/index.html)

> **Interview follow-up** Offline ranking improves and the live click rate stays flat. Which check would you start with among experiment assignment, retrieval coverage, and serving latency? Explain what it could rule out.

## Language models and AI engineering: pick a specialty

### Model internals and training

- **[Dive into LLMs](https://github.com/Lordog/dive-into-llms)**\
  Chinese-language labs from Shanghai Jiao Tong University coursework, including fine-tuning and deployment.\
  **Study** Use the fine-tuning lab to identify the pretrained weights, the task data, and the parameters that get updated. Say how you would compare results before and after training. [Fine-tuning and deployment lab](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter1)

- **[Educatum · Advanced language model and generative AI questions](https://www.educatum.com/llm-and-genai-advanced-interview-questions)**\
  A question bank for oral practice on language models and generative AI.\
  **Study** Sketch a causal mask, then explain why training can process positions in parallel and generation decodes one token at a time.

### Retrieval and agents

- **[Hands-On Large Language Models](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main/chapter08)**\
  Code from the book of the same name. Chapter 8 covers semantic search and retrieval-augmented generation (RAG).\
  **Study** Run a retrieval example and pick a question it gets wrong. Inspect the retrieved evidence and generated answer separately.

- **[learn-harness-engineering](https://github.com/walkinglabs/learn-harness-engineering)**\
  Projects on agent workspaces, execution feedback, and verification.\
  **Study** Try P05. Give the system an unsupported answer and see whether the verification step catches it. [P05 project brief](https://github.com/walkinglabs/learn-harness-engineering/blob/main/docs/en/projects/project-05-grounded-qa-verification/index.md)

- **[Educatum · AI engineering questions](https://www.educatum.com/ai-engineering-interview-questions)**\
  Questions and answers for artificial intelligence engineering interviews.\
  **Study** Pick a question to answer aloud, then practice a follow-up. What would you log to reproduce and diagnose a failed agent task?

### Inference and post-training

- **[vLLM](https://github.com/vllm-project/vllm)**\
  An inference and serving engine for language models.\
  **Study** Read how PagedAttention stores the key–value (KV) cache in blocks. Write a small cache, compare token-by-token decoding with a full forward pass, then reason about memory and latency as context grows. [PagedAttention documentation](https://docs.vllm.ai/en/latest/design/paged_attention/)

- **[verl](https://github.com/verl-project/verl)**\
  A framework for post-training language models with reinforcement learning.\
  **Study** Trace group relative policy optimization (GRPO). Follow several responses to the same prompt through reward scoring, advantage estimation, and the policy update. [GRPO documentation](https://verl.readthedocs.io/en/latest/algo/grpo.html)

## More questions and practice

- **[Cheng Zhang's ML engineering question bank](https://chengz.art/en/mle-prep)**\
  Interview questions organized by topic and difficulty.\
  **Study** Pick a weak topic, answer three questions aloud, and write down the first follow-up that stops you.

- **[Silvia Sapora's ML interview guide](https://silviasapora.github.io/blog/ml-interviews.html)**\
  A preparation guide with programming and debugging exercises.\
  **Study** Time yourself on one implementation or debugging exercise, then redo the part you missed the next day.
