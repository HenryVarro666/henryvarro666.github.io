---
title: "ML foundations and AI interview prep"
date: 2026-11-13T17:29:06-06:00
description: "A study route from classic machine learning questions to coding from scratch, system design, and language models, with links to practice problems and GitHub repositories."
topics: [data-science, llm]
resourceGuide: true
---

This study guide covers machine learning (ML) and artificial intelligence (AI) interviews. Try the first practice session, then pick the topic your interview calls for. Spend more time on data splits and evaluation for data science, system design for applied ML engineering, and implementation and debugging for research engineering or large language model (LLM) roles.

## A 45-minute first pass

1. **Explain a model for 15 minutes.** Spend one minute explaining how you would spot overfitting and what you would check next. Read the answer after you finish. [MLQuestions on overfitting and underfitting](https://github.com/andrewekhalel/MLQuestions#explain-over--and-under-fitting-and-how-to-combat-them)
2. **Write code for 20 minutes.** Implement softmax from scratch. Test it on large inputs, then add the same constant to every score and check that the output stays the same. [Deep-ML softmax exercise](https://www.deep-ml.com/problems/23)
3. **Spend 10 minutes on a follow-up.** For classic ML, explain how raising a threshold changes false positives and false negatives. For language models, sketch which tokens each position can see through a causal mask. [Precision vs. recall](/notes/precision-vs-recall/) · [GPT causal mask code](https://github.com/karpathy/ng-video-lecture/blob/52201428ed7b46804849dea0b3ccf0de9df1a5c3/gpt.py#L64-L93)

## ML foundations: explain the choice

- **[Probability Cheatsheet](https://github.com/wzchen/probability_cheatsheet)**\
  A compact reference for probability formulas and common distributions.\
  **Study** Use a two-stage sampling example to practice conditional probability and total expectation, then use the law of total variance to separate variation within groups from variation between groups. {{< new-tab "Conditional probability · PDF page 1" "https://www.wzchen.com/s/probability_cheatsheet.pdf#page=1" >}} · {{< new-tab "Total expectation and variance · PDF page 4" "https://www.wzchen.com/s/probability_cheatsheet.pdf#page=4" >}}

- **[ML-NLP](https://github.com/NLP-LOVE/ML-NLP)**\
  Chinese-language notes on algorithms with code and interview topics.\
  **Study** Compare logistic regression, decision trees, and random forests. Explain what result would make you replace a logistic regression baseline. {{< new-tab "Logistic regression" "https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/2.Logistics%20Regression/2.Logistics%20Regression.md" >}} · {{< new-tab "Decision trees" "https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/3.Desition%20Tree/Desition%20Tree.md" >}} · {{< new-tab "Random forests" "https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/3.1%20Random%20Forest/3.1%20Random%20Forest.md" >}}

- **[AIMLInterviews · ML Fundamentals](https://github.com/alirezadir/AIMLInterviews/blob/main/src/ml-fundamental.md)**\
  The fundamentals chapter of a broader technical interview guide.\
  **Study** Use training and validation performance to distinguish underfitting from overfitting. Explain how you would split the data. {{< new-tab "Bias and variance" "https://github.com/alirezadir/AIMLInterviews/blob/main/src/ml-fundamental.md#bias--variance-underfittingoverfitting" >}} · {{< new-tab "Data splitting code" "https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLC/problems/classic_ml/dataset_split.py" >}}

- **[Educatum · AI/ML Interview Playbook](https://www.educatum.com/the-ai-ml-interview-playbook)**\
  Study paths and classic interview questions organized by topic.\
  **Study** Explain how logistic regression can handle class imbalance. With only 100 alerts reviewed each day, choose a metric and explain how you would set a threshold on the validation set. {{< new-tab "Educatum's class imbalance question" "https://www.educatum.com/12355925845b81a98b05f8005cd3c9fd" >}} · [ROC and precision–recall curves](/notes/roc-auc-vs-pr-auc/)

- **[Data Science Interview Questions and Answers](https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers)**\
  A question bank covering statistics, probability, and modeling.\
  **Study** Use the central limit theorem to reason about uncertainty in a sample mean, then design an A/B test. If each user can visit several times, explain the unit of randomization and the primary metric. {{< new-tab "Q1 · Central limit theorem" "https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers/blob/main/Statistics%20Interview%20Questions%20%26%20Answers%20for%20Data%20Scientists.md#q1-explain-the-central-limit-theorem-and-give-examples-of-when-you-can-use-it-in-a-real-world-problem" >}} · {{< new-tab "Q2 · A/B testing" "https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers/blob/main/Statistics%20Interview%20Questions%20%26%20Answers%20for%20Data%20Scientists.md#q2-briefly-explain-the-ab-testing-and-its-application-what-are-some-common-pitfalls-encountered-in-ab-testing" >}}

- **[MLQuestions](https://github.com/andrewekhalel/MLQuestions)**\
  Technical interview questions on machine learning and computer vision.\
  **Study** Explain why generating a fresh random data split for each model makes the evaluation results harder to compare. {{< new-tab "MLQuestions on data splitting" "https://github.com/andrewekhalel/MLQuestions#given-that-we-want-to-evaluate-the-performance-of-n-different-machine-learning-models-on-the-same-data-why-would-the-following-splitting-mechanism-be-incorrect" >}}

## ML coding: from NumPy to a Transformer

- **[Deep-ML](https://www.deep-ml.com/problems)**\
  Python exercises you can run and test in the browser.\
  **Study** Work from matrix operations and gradient descent to multi-head attention. Write down the tensor shapes, then add a causal mask to your own implementation. {{< new-tab "Matrix–vector product" "https://www.deep-ml.com/problems/1" >}} · {{< new-tab "Gradient descent for linear regression" "https://www.deep-ml.com/problems/15" >}} · {{< new-tab "Multi-head attention" "https://www.deep-ml.com/problems/94" >}}

- **[AIMLInterviews · ML Coding](https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLC/ml-coding.md)**\
  Practice prompts for implementing algorithms and model components in interviews.\
  **Study** Implement K-means assignments and updates under a time limit. Test a small example by hand, then test an empty cluster. {{< new-tab "K-means notebook" "https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLC/notebooks/k_means_2.ipynb" >}} · {{< new-tab "Reference implementation" "https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLC/problems/classic_ml/kmeans.py" >}}

- **[Karpathy · Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero)**\
  A hands-on course that starts with automatic differentiation and builds up to a small language model.\
  **Study** Use micrograd to review computation graphs. Implement the forward and backward passes of a two-layer perceptron, then check one weight with finite differences. {{< new-tab "micrograd notebook" "https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd/micrograd_lecture_first_half_roughly.ipynb" >}} · {{< new-tab "Two-layer backpropagation exercise" "https://colab.research.google.com/drive/1WV2oi2fh9XXyldh02wupFQX0wh5ZC-z-?usp=sharing" >}} · {{< new-tab "Worked notebook" "https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb" >}}

- **[Stanford CS336 · Assignment 1](https://github.com/stanford-cs336/assignment1-basics)**\
  A course assignment that brings tokenization, Transformer components, and the training loop together.\
  **Study** Implement scaled dot-product attention. Use the tests to check masking and inputs with extra batch dimensions. {{< new-tab "Assignment problem · PDF page 24" "https://cdn.jsdelivr.net/gh/stanford-cs336/assignment1-basics@a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf#page=24" >}} · {{< new-tab "Attention tests" "https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/tests/test_model.py#L57-L71" >}}

## ML system design: put the model in a product

- **[AIMLInterviews · ML System Design](https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/ml-system-design.md)**\
  A framework for answering system design questions, with worked cases.\
  **Study** Turn the product goal into something you can measure. Define the label and offline metric, then say what you would check after launch. {{< new-tab "Problem formulation" "https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/ml-system-design.md#1-problem-formulation" >}} · {{< new-tab "Offline and online metrics" "https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/ml-system-design.md#2-metrics-offline-and-online" >}}

- **[Chip Huyen · Machine Learning Systems Design](https://huyenchip.com/machine-learning-systems-design/toc.html)**\
  A guide to data pipelines, training, and serving.\
  **Study** Check which training features are available at prediction time and how you would measure serving latency. {{< new-tab "Data pipeline" "https://huyenchip.com/machine-learning-systems-design/design-a-machine-learning-system.html#data-pipeline-698WI4R" >}} · {{< new-tab "Serving" "https://huyenchip.com/machine-learning-systems-design/design-a-machine-learning-system.html#serving-091rIYw" >}}

- **[Datawhale fun-rec](https://github.com/datawhalechina/fun-rec)**\
  A Chinese-language recommendation systems textbook and exercises, from retrieval through reranking.\
  **Study** Start with the two-tower model. Explain how many candidates each stage handles, then sketch how impressions become training feedback. {{< new-tab "Two-tower retrieval" "https://datawhalechina.github.io/fun-rec/chapter_1_retrieval/3.two_tower/index.html" >}} · {{< new-tab "Ranking and preference prediction" "https://datawhalechina.github.io/fun-rec/chapter_2_ranking/index.html" >}}

> **Interview follow-up** Offline ranking improves and the live click rate stays flat. Which check would you start with among experiment assignment, retrieval coverage, and serving latency? Explain what it could rule out.

## Language models and AI engineering: pick a specialty

### Model internals and training

- **[Dive into LLMs](https://github.com/Lordog/dive-into-llms)**\
  Chinese-language labs from Shanghai Jiao Tong University coursework, including fine-tuning and deployment.\
  **Study** Use the fine-tuning lab to identify the pretrained weights, the task data, and the parameters that get updated. Say how you would compare results before and after training. {{< new-tab "Fine-tuning modules" "https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter1/README.md#21-%E7%90%86%E8%A7%A3%E5%85%B3%E9%94%AE%E6%A8%A1%E5%9D%97" >}} · {{< new-tab "Fine-tuning notebook" "https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter1/dive-tuning.ipynb" >}}

- **[Educatum · Advanced language model and generative AI questions](https://www.educatum.com/llm-and-genai-advanced-interview-questions)**\
  A question bank for oral practice on language models and generative AI.\
  **Study** Explain how self-attention assigns weights to input positions. Sketch a causal mask, then explain why training can process positions in parallel and generation decodes one token at a time. {{< new-tab "Educatum's self-attention question" "https://www.educatum.com/12555925845b81fd92a5c48f91079203" >}} · {{< new-tab "Causal mask code" "https://github.com/karpathy/ng-video-lecture/blob/52201428ed7b46804849dea0b3ccf0de9df1a5c3/gpt.py#L64-L93" >}}

### Retrieval and agents

- **[Hands-On Large Language Models](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main/chapter08)**\
  Code from the book of the same name. Chapter 8 covers semantic search and retrieval-augmented generation (RAG).\
  **Study** Run a retrieval example from Chapter 8 and pick a question it gets wrong. Inspect the retrieved evidence and generated answer separately. {{< new-tab "Chapter 8 notebook" "https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter08/Chapter%208%20-%20Semantic%20Search.ipynb" >}} · {{< new-tab "Open in Google Colab" "https://colab.research.google.com/github/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter08/Chapter%208%20-%20Semantic%20Search.ipynb" >}}

- **[learn-harness-engineering](https://github.com/walkinglabs/learn-harness-engineering)**\
  Projects on agent workspaces, execution feedback, and verification.\
  **Study** Follow P05's generator and evaluator setup. Add a rubric check for whether the evidence supports the answer, then try it on an unsupported answer. {{< new-tab "P05 project brief" "https://github.com/walkinglabs/learn-harness-engineering/blob/main/docs/en/projects/project-05-grounded-qa-verification/index.md" >}} · {{< new-tab "Example evaluator rubric" "https://github.com/walkinglabs/learn-harness-engineering/blob/main/projects/project-05/solution/gen-eval/evaluator-rubric.md" >}}

- **[Educatum · AI engineering questions](https://www.educatum.com/ai-engineering-interview-questions)**\
  Questions and answers for artificial intelligence engineering interviews.\
  **Study** Use the ReAct question to explain how an agent alternates actions and observations. Then say what you would log to reproduce and diagnose a failed task. {{< new-tab "Educatum's ReAct question" "https://www.educatum.com/31655925845b80128df5de4b76d09b73" >}} · {{< new-tab "LangSmith on traces" "https://docs.langchain.com/langsmith/observability-concepts#traces" >}}

### Inference and post-training

- **[vLLM](https://github.com/vllm-project/vllm)**\
  An inference and serving engine for language models.\
  **Study** Read how PagedAttention stores the key–value (KV) cache in blocks. Write a small cache, compare token-by-token decoding with a full forward pass, then reason about memory and latency as context grows. {{< new-tab "PagedAttention block concepts" "https://docs.vllm.ai/en/latest/design/paged_attention/#concepts" >}} · {{< new-tab "Cache allocation function" "https://github.com/vllm-project/vllm/blob/d64414d044a898c6db646dcecd122213accd1732/vllm/v1/core/kv_cache_manager.py#L400" >}}

- **[verl](https://github.com/verl-project/verl)**\
  A framework for post-training language models with reinforcement learning.\
  **Study** Trace group relative policy optimization (GRPO). Follow several responses to the same prompt through reward scoring, advantage estimation, and the policy update. {{< new-tab "GRPO example configuration" "https://verl.readthedocs.io/en/latest/algo/grpo.html#reference-example" >}} · {{< new-tab "Advantage calculation function" "https://github.com/verl-project/verl/blob/61adb0c7eb90730306e7794c2bf248331211e657/verl/trainer/ppo/core_algos.py#L268-L331" >}}

## More questions and practice

- **[Cheng Zhang's ML engineering question bank](https://chengz.art/en/mle-prep)**\
  Interview questions organized by topic and difficulty.\
  **Study** Explain why cross-entropy works for classification, then pick two more questions on topics you find difficult. Write down the first follow-up that stops you. {{< new-tab "Cross-entropy question · jumps in browsers that support text fragments" "https://chengz.art/en/mle-prep#:~:text=%E8%A7%A3%E9%87%8A%E4%BA%A4%E5%8F%89%E7%86%B5%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E7%9A%84%E6%95%B0%E5%AD%A6%E5%8E%9F%E7%90%86%E5%8F%8A%E5%85%B6%E5%9C%A8%E5%88%86%E7%B1%BB%E4%BB%BB%E5%8A%A1%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8%E3%80%82" >}}

- **[Silvia Sapora's ML interview guide](https://silviasapora.github.io/blog/ml-interviews.html)**\
  A preparation guide with programming and debugging exercises.\
  **Study** Pick one exercise from the coding and debugging checklist and solve it under a time limit. Redo the part you missed the next day. {{< new-tab "ML coding and debugging checklist" "https://silviasapora.github.io/blog/ml-interviews.html#prep-technical:~:text=ML%20coding%20and%20debugging" >}}
