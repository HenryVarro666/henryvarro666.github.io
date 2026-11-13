---
title: "ML 基础与 AI 面试复习路线"
date: 2026-11-13T17:29:06-06:00
description: "从经典机器学习题开始，接着手写模型，再按岗位选系统设计或大模型。每一节都有可直接打开的题库、课程和 GitHub 仓库。"
topics: [data-science, llm]
---

这份索引面向机器学习（ML）、机器学习工程师（MLE）和人工智能（AI）方向的面试。先补基础，再从空文件写一个实现。之后按面试内容选一节深入：数据科学侧重数据切分、模型评估和 A/B 测试；应用型 MLE 加练系统设计；研究工程或大语言模型（LLM）岗位多练 Transformer 和代码调试。

## 第一轮：45 分钟

1. **15 分钟讲模型。** 打开 [MLQuestions](https://github.com/andrewekhalel/MLQuestions)，选一道过拟合或模型评估的题。先合上答案，用英文讲一分钟：你怎么发现问题，下一步怎么验证。
2. **20 分钟写代码。** 做 [Deep-ML 的 softmax 题](https://www.deep-ml.com/problems/23)。用一组很大的输入测试数值稳定性，再检查给所有分数加同一个常数后，输出是否保持不变。
3. **10 分钟选一道追问。** 做经典 ML 就读 [Precision 与 Recall](/zh/notes/precision-vs-recall/)，解释调高阈值会影响谁。做大模型就打开 [Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) 的 *Let's build GPT*，画出因果掩码允许每个 token 看到的位置。

## 机器学习基础：能解释为什么

- **概率与模型。** 忘了条件概率、期望或方差时查 [Probability Cheatsheet](https://github.com/wzchen/probability_cheatsheet)。用中文补课可从 [ML-NLP](https://github.com/NLP-LOVE/ML-NLP) 选逻辑回归、决策树和集成学习；再用 [AIMLInterviews 的 ML Fundamentals](https://github.com/alirezadir/AIMLInterviews/blob/main/src/ml-fundamental.md) 查漏。
- **验证与评估。** 用 [Educatum 的 AI/ML Interview Playbook](https://www.educatum.com/the-ai-ml-interview-playbook) 找经典题，重点练过拟合、数据泄漏和类别不平衡。再读[受试者工作特征曲线下面积（ROC-AUC）和精确率–召回率曲线下面积（PR-AUC）的对照笔记](/zh/notes/roc-auc-vs-pr-auc/)。假设阳性很少、每天只能复核 100 个样本，你会报哪项指标，怎样在验证集上选阈值？
- **统计与实验。** 用 [Data Science Interview Questions and Answers](https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers) 复习统计题。另练一道 A/B 测试：如果同一用户每天来多次，按访问分组可能出什么问题？先说清随机分组的单位和主指标，再讨论显著性。

这一轮的完成标准：拿到一张表格数据，能说出先跑什么基线、怎样按数据产生的方式切分，以及什么结果会让你换模型。

## 手写实现：从 NumPy 到 Transformer

- **短题。** 从 [Deep-ML 题库](https://www.deep-ml.com/problems) 依次挑矩阵运算、梯度下降和聚类题；再翻 [AIMLInterviews 的 ML Coding](https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLC/ml-coding.md) 补题。每道题写出输入输出形状，先找一个能手算的小例子，再试边界输入。
- **两层网络。** 在 [Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) 先看 *micrograd*，再看多层感知机的反向传播。自己写一个两层网络的前向和反向，用有限差分检查一个权重的梯度。
- **小语言模型。** 先用 [多头注意力练习](https://www.deep-ml.com/problems/94) 熟悉张量形状，再在自己的版本中加因果掩码。想把部件接起来，再做 [Stanford CS336 Assignment 1](https://github.com/stanford-cs336/assignment1-basics)：从分词、Transformer 部件做到训练循环。时间紧就只做一个部件并跑通对应测试。

## ML 系统设计：把模型放进产品

先用 [AIMLInterviews 的 ML System Design](https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/ml-system-design.md) 过一遍答题顺序，再用 [Chip Huyen 的 Machine Learning Systems Design](https://huyenchip.com/machine-learning-systems-design/toc.html) 补数据、部署和监控方面的缺口。

拿推荐场景练一次：[Datawhale fun-rec](https://github.com/datawhalechina/fun-rec) 先读召回和排序。画出召回、排序、展示、反馈的流程，说明训练标签什么时候可用。追问：离线排序指标变好了，线上点击率没动，你会先看实验分流、召回覆盖，还是服务延迟？说清你先看哪一项，以及它能帮你排除什么。

## 大模型与 AI 工程：按岗位选专题

- **模型原理。** [动手学大模型](https://github.com/Lordog/dive-into-llms) 用中文补 Transformer 和训练流程。再去 [Educatum 的大模型与生成式 AI 进阶题](https://www.educatum.com/llm-and-genai-advanced-interview-questions) 抽题口述。先试着解释：因果注意力怎样防止模型读到未来 token？
- **检索与智能体。** 跑 [Hands-On Large Language Models 第 8 章](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main/chapter08) 的语义搜索与检索增强生成（RAG）示例。用一个失败样本分别检查检索结果和生成答案。面试涉及智能体再做 [learn-harness-engineering](https://github.com/walkinglabs/learn-harness-engineering) 的工具调用与评估练习，配合 [Educatum 的 AI 工程题](https://www.educatum.com/ai-engineering-interview-questions) 练口述。
- **推理与后训练。** 推理方向先写一个键值（key–value，KV）缓存，验证逐 token 解码与完整前向的结果在浮点误差范围内一致；再读 [vLLM](https://github.com/vllm-project/vllm)，想清楚更长的上下文会怎样影响显存与延迟。后训练方向用 [verl](https://github.com/verl-project/verl) 查强化学习训练流程，沿着一次生成、奖励计算和策略更新走一遍。

要补题量，可以打开 [Cheng Zhang 的 MLE 题库](https://chengz.art/zh/mle-prep) 按类别挑题，或看 [Silvia Sapora 的面试准备指南](https://silviasapora.github.io/blog/ml-interviews.html) 找手写和调试的练习方向。复习时记下第一次答不出来的追问，隔一天只重做那道题。
