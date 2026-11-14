---
title: "ML 基础与 AI 面试复习路线"
date: 2026-11-13T17:29:06-06:00
description: "从经典机器学习题开始，接着手写模型，再按岗位选系统设计或大模型。每一节都有可直接打开的题库、课程和 GitHub 仓库。"
topics: [data-science, llm]
resourceGuide: true
---

这份笔记面向机器学习（ML）、机器学习工程师（MLE）和人工智能（AI）方向的面试。先做第一轮练习，再按面试内容选专题。数据科学多练数据切分和评估，应用型 MLE 加练系统设计，研究工程和大语言模型（LLM）岗位多练 Transformer 实现与调试。

## 第一轮：45 分钟

1. **15 分钟讲模型。** 选一道过拟合或模型评估题。合上答案，用英文讲一分钟，说明怎么发现问题、下一步怎么验证。 [MLQuestions 基础题](https://github.com/andrewekhalel/MLQuestions#machine-learning-fundamentals)
2. **20 分钟写代码。** 从空文件写出 softmax。用很大的输入测试数值稳定性，再给所有分数加同一个常数，检查输出是否保持不变。 [Deep-ML 的 softmax 练习](https://www.deep-ml.com/problems/23)
3. **10 分钟选一道追问。** 练经典 ML，就解释调高分类阈值会怎样影响误报和漏报。练大模型，就画出因果掩码允许每个 token 看到的位置。 [Precision 与 Recall](/zh/notes/precision-vs-recall/) · [Zero to Hero 的 GPT 课](https://karpathy.ai/zero-to-hero.html)

## 机器学习基础：能解释为什么

- **[Probability Cheatsheet](https://github.com/wzchen/probability_cheatsheet)**\
  概率速查表，包含公式和常见分布。\
  **这次学**　条件概率、全期望和方差。拿一个两阶段抽样例子，先条件化，再算总体期望。

- **[ML-NLP](https://github.com/NLP-LOVE/ML-NLP)**\
  中文算法笔记，配有代码和面试知识点。\
  **这次学**　对照逻辑回归、决策树和集成学习，解释什么情况下会换掉逻辑回归基线。

- **[AIMLInterviews · ML Fundamentals](https://github.com/alirezadir/AIMLInterviews/blob/main/src/ml-fundamental.md)**\
  综合面试指南里的模型与基础概念章节。\
  **这次学**　用验证集表现区分欠拟合与过拟合，再说明验证集该怎样切分。

- **[Educatum · AI/ML Interview Playbook](https://www.educatum.com/the-ai-ml-interview-playbook)**\
  按方向组织的学习路线和经典面试题。\
  **这次学**　挑数据泄漏与类别不平衡题。每天只能复核 100 条告警时，想清楚该看哪项指标、怎样在验证集上选阈值。 [ROC 曲线下面积与精确率–召回率曲线下面积](/zh/notes/roc-auc-vs-pr-auc/)

- **[Data Science Interview Questions and Answers](https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers)**\
  按统计、概率、建模等方向整理的数据科学题库。\
  **这次学**　从统计题复习抽样和不确定性，再设计一次 A/B 测试。同一用户会多次访问时，先说清随机分组单位和主指标。

- **[MLQuestions](https://github.com/andrewekhalel/MLQuestions)**\
  机器学习和计算机视觉的技术面试题库。\
  **这次学**　用一道评估题复盘第一轮口述。说出哪种训练集与验证集表现会让你怀疑过拟合。 [模型评估与验证题](https://github.com/andrewekhalel/MLQuestions#model-evaluation-and-validation)

## 手写实现：从 NumPy 到 Transformer

- **[Deep-ML](https://www.deep-ml.com/problems)**\
  可以在浏览器里写 Python 并运行测试的练习库。\
  **这次学**　从矩阵运算和梯度下降练到多头注意力。先标出张量形状，再给自己的版本加因果掩码。 [多头注意力练习](https://www.deep-ml.com/problems/94)

- **[AIMLInterviews · ML Coding](https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLC/ml-coding.md)**\
  用面试形式练算法和模型部件的实现。\
  **这次学**　挑一道熟悉的题限时写完，留一个手算例子和一个边界输入做验证。

- **[Karpathy · Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero)**\
  从自动求导讲到小型语言模型的动手课程。\
  **这次学**　从 micrograd 复习计算图，自己写两层感知机的前向和反向，用有限差分核对一个权重的梯度。

- **[Stanford CS336 · Assignment 1](https://github.com/stanford-cs336/assignment1-basics)**\
  从分词器、Transformer 部件做到训练循环的课程作业。\
  **这次学**　先实现一个注意力部件并跑通对应测试，再把它放回模型里。 [作业说明](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_assignment1_basics.pdf)

## ML 系统设计：把模型放进产品

- **[AIMLInterviews · ML System Design](https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/ml-system-design.md)**\
  系统设计答题框架，附案例。\
  **这次学**　先把业务目标写成可验证的问题，再确定标签、离线指标和上线后的检查。

- **[Chip Huyen · Machine Learning Systems Design](https://huyenchip.com/machine-learning-systems-design/toc.html)**\
  从数据管道、训练到在线服务的设计讲义。\
  **这次学**　读数据管道与服务两节，检查线上预测时能否取到训练用的特征，以及怎样测服务延迟。 [设计章节](https://huyenchip.com/machine-learning-systems-design/design-a-machine-learning-system.html)

- **[Datawhale fun-rec](https://github.com/datawhalechina/fun-rec)**\
  中文推荐系统教材与实验，覆盖召回、排序和重排。\
  **这次学**　从双塔模型入手，分别说明召回和排序各处理多少候选，然后画出展示到反馈的流程。 [双塔模型章节](https://datawhalechina.github.io/fun-rec/chapter_1_retrieval/3.two_tower/index.html)

> **面试追问**　离线排序指标变好，线上点击率没变。你会先查实验分流、召回覆盖还是服务延迟？说出第一步能排除什么。

## 大模型与 AI 工程：按岗位选专题

### 模型原理与训练

- **[动手学大模型](https://github.com/Lordog/dive-into-llms)**\
  上海交通大学课程衍生的中文实践教程，涵盖微调、部署和其他专题。\
  **这次学**　从微调示例区分预训练权重、任务数据和更新的参数，讲清实验前后怎样比较。 [微调与部署教程](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter1)

- **[Educatum · 大模型与生成式 AI 进阶题](https://www.educatum.com/llm-and-genai-advanced-interview-questions)**\
  面向大模型岗位的进阶口述题。\
  **这次学**　先手画一个因果掩码，再解释训练时为什么可以并行算各位置，生成时为什么逐 token 解码。

### 检索与智能体

- **[Hands-On Large Language Models](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main/chapter08)**\
  同名书的配套代码。第 8 章有语义搜索和检索增强生成（RAG）的 notebook。\
  **这次学**　跑一个检索示例，挑一个答错的问题，分开检查召回结果和生成答案。

- **[learn-harness-engineering](https://github.com/walkinglabs/learn-harness-engineering)**\
  用项目练习智能体的工作环境、执行反馈和结果校验。\
  **这次学**　做 P05。给出一条没有证据支撑的答案，看看验证步骤能否发现它。 [P05 项目说明](https://github.com/walkinglabs/learn-harness-engineering/blob/main/docs/en/projects/project-05-grounded-qa-verification/index.md)

- **[Educatum · AI 工程题](https://www.educatum.com/ai-engineering-interview-questions)**\
  人工智能工程面试用的问答集。\
  **这次学**　挑一题练口述，并准备一个追问。遇到智能体任务失败时，你会记录什么，才能复现并定位失败？

### 推理与后训练

- **[vLLM](https://github.com/vllm-project/vllm)**\
  大语言模型推理与服务引擎。\
  **这次学**　看 PagedAttention 如何按块管理键值（key–value，KV）缓存。先手写一个缓存，核对逐 token 解码和完整前向的结果，再解释上下文变长时显存和延迟会怎样变。 [PagedAttention 文档](https://docs.vllm.ai/en/latest/design/paged_attention/)

- **[verl](https://github.com/verl-project/verl)**\
  用强化学习给语言模型做后训练的框架。\
  **这次学**　沿着组相对策略优化（GRPO）的示例，追一遍同一提示的多份回答如何得到奖励，以及优势估计如何参与策略更新。 [GRPO 文档](https://verl.readthedocs.io/en/latest/algo/grpo.html)

## 更多题库与练习

- **[Cheng Zhang 的 MLE 题库](https://chengz.art/zh/mle-prep)**\
  按类别和难度筛选的面试题库。\
  **这次学**　挑一个薄弱主题，用英文答三题，记下第一次卡住的追问。

- **[Silvia Sapora 的面试准备指南](https://silviasapora.github.io/blog/ml-interviews.html)**\
  面向机器学习岗位的备考指南，涵盖编程、调试等练习。\
  **这次学**　挑一道手写或调试题做限时复盘，隔天重做第一次没答出来的部分。
