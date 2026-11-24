---
title: "ML 基础与 AI 面试复习路线"
date: 2026-11-13T17:29:06-06:00
description: "从经典机器学习题开始，接着手写模型，再按岗位选系统设计或大模型。每一节都有可直接打开的题库、课程和 GitHub 仓库。"
topics: [data-science, llm]
resourceGuide: true
---

这份笔记面向机器学习（ML）、机器学习工程师（MLE）和人工智能（AI）方向的面试。先做第一轮练习，再按面试内容选专题。数据科学多练数据切分和评估，应用型 MLE 加练系统设计，研究工程和大语言模型（LLM）岗位多练 Transformer 实现与调试。

## 第一轮：45 分钟

1. **15 分钟讲模型。** 用英文讲一分钟，说明怎么发现过拟合、下一步怎么验证。讲完再打开答案。 [MLQuestions 的过拟合与欠拟合题](https://github.com/andrewekhalel/MLQuestions#explain-over--and-under-fitting-and-how-to-combat-them)
2. **20 分钟写代码。** 从空文件写出 softmax。用很大的输入测试数值稳定性，再给所有分数加同一个常数，检查输出是否保持不变。 [Deep-ML 的 softmax 练习](https://www.deep-ml.com/problems/23)
3. **10 分钟选一道追问。** 练经典 ML，就解释调高分类阈值会怎样影响误报和漏报。练大模型，就画出因果掩码允许每个 token 看到的位置。 [Precision 与 Recall](/zh/notes/precision-vs-recall/) · [GPT 因果掩码代码](https://github.com/karpathy/ng-video-lecture/blob/52201428ed7b46804849dea0b3ccf0de9df1a5c3/gpt.py#L64-L93)

## 机器学习基础：能解释为什么

- **[Probability Cheatsheet](https://github.com/wzchen/probability_cheatsheet)**\
  概率速查表，包含公式和常见分布。\
  **这次学**　用两阶段抽样练习条件概率和全期望，再用全方差公式拆开组内与组间的变动。 {{< new-tab "条件概率 · PDF 第 1 页" "https://www.wzchen.com/s/probability_cheatsheet.pdf#page=1" >}} · {{< new-tab "全期望与全方差 · PDF 第 4 页" "https://www.wzchen.com/s/probability_cheatsheet.pdf#page=4" >}}

- **[ML-NLP](https://github.com/NLP-LOVE/ML-NLP)**\
  中文算法笔记，配有代码和面试知识点。\
  **这次学**　对照逻辑回归、决策树和随机森林，解释什么情况下会换掉逻辑回归基线。 {{< new-tab "逻辑回归" "https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/2.Logistics%20Regression/2.Logistics%20Regression.md" >}} · {{< new-tab "决策树" "https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/3.Desition%20Tree/Desition%20Tree.md" >}} · {{< new-tab "随机森林" "https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/3.1%20Random%20Forest/3.1%20Random%20Forest.md" >}}

- **[AIMLInterviews · ML Fundamentals](https://github.com/alirezadir/AIMLInterviews/blob/main/src/ml-fundamental.md)**\
  综合面试指南里的模型与基础概念章节。\
  **这次学**　用训练集和验证集的表现区分欠拟合与过拟合，再说明验证集该怎样切分。 {{< new-tab "偏差与方差章节" "https://github.com/alirezadir/AIMLInterviews/blob/main/src/ml-fundamental.md#bias--variance-underfittingoverfitting" >}} · {{< new-tab "无数据泄漏的切分代码" "https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLC/problems/classic_ml/dataset_split.py" >}}

- **[Educatum · AI/ML Interview Playbook](https://www.educatum.com/the-ai-ml-interview-playbook)**\
  按方向组织的学习路线和经典面试题。\
  **这次学**　先答逻辑回归怎么处理类别不平衡。每天只能复核 100 条告警时，再选指标，并说明怎样用验证集定阈值。 {{< new-tab "Educatum 的类别不平衡题" "https://www.educatum.com/12355925845b81a98b05f8005cd3c9fd" >}} · [ROC 与精确率–召回率曲线笔记](/zh/notes/roc-auc-vs-pr-auc/)

- **[Data Science Interview Questions and Answers](https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers)**\
  按统计、概率、建模等方向整理的数据科学题库。\
  **这次学**　用中央极限定理想清楚样本均值的不确定性，再设计一次 A/B 测试。同一用户会多次访问时，先说清随机分组单位和主指标。 {{< new-tab "第 1 题 · 中央极限定理" "https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers/blob/main/Statistics%20Interview%20Questions%20%26%20Answers%20for%20Data%20Scientists.md#q1-explain-the-central-limit-theorem-and-give-examples-of-when-you-can-use-it-in-a-real-world-problem" >}} · {{< new-tab "第 2 题 · A/B 测试" "https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers/blob/main/Statistics%20Interview%20Questions%20%26%20Answers%20for%20Data%20Scientists.md#q2-briefly-explain-the-ab-testing-and-its-application-what-are-some-common-pitfalls-encountered-in-ab-testing" >}}

- **[MLQuestions](https://github.com/andrewekhalel/MLQuestions)**\
  机器学习和计算机视觉的技术面试题库。\
  **这次学**　解释为什么给两个模型各随机切分一次，会让评估结果难以直接比较。 {{< new-tab "MLQuestions 的数据切分题" "https://github.com/andrewekhalel/MLQuestions#given-that-we-want-to-evaluate-the-performance-of-n-different-machine-learning-models-on-the-same-data-why-would-the-following-splitting-mechanism-be-incorrect" >}}

## 手写实现：从 NumPy 到 Transformer

- **[Deep-ML](https://www.deep-ml.com/problems)**\
  可以在浏览器里写 Python 并运行测试的练习库。\
  **这次学**　从矩阵运算和梯度下降练到多头注意力。先标出张量形状，再给自己的版本加因果掩码。 {{< new-tab "矩阵与向量相乘" "https://www.deep-ml.com/problems/1" >}} · {{< new-tab "线性回归的梯度下降" "https://www.deep-ml.com/problems/15" >}} · {{< new-tab "多头注意力" "https://www.deep-ml.com/problems/94" >}}

- **[AIMLInterviews · ML Coding](https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLC/ml-coding.md)**\
  用面试形式练算法和模型部件的实现。\
  **这次学**　限时实现 K-means 的分配与更新，用手算的小例子和空簇测试。 {{< new-tab "K-means notebook" "https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLC/notebooks/k_means_2.ipynb" >}} · {{< new-tab "参考实现" "https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLC/problems/classic_ml/kmeans.py" >}}

- **[Karpathy · Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero)**\
  从自动求导讲到小型语言模型的动手课程。\
  **这次学**　从 micrograd 复习计算图，手写两层感知机的前向和反向，用有限差分核对一个权重的梯度。 {{< new-tab "micrograd 计算图" "https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd/micrograd_lecture_first_half_roughly.ipynb" >}} · {{< new-tab "两层网络反向传播练习" "https://colab.research.google.com/drive/1WV2oi2fh9XXyldh02wupFQX0wh5ZC-z-?usp=sharing" >}} · {{< new-tab "讲解 notebook" "https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb" >}}

- **[Stanford CS336 · Assignment 1](https://github.com/stanford-cs336/assignment1-basics)**\
  从分词器、Transformer 部件做到训练循环的课程作业。\
  **这次学**　从缩放点积注意力开始。实现后，用测试核对掩码和多维批量输入。 {{< new-tab "作业题 · PDF 第 24 页" "https://cdn.jsdelivr.net/gh/stanford-cs336/assignment1-basics@a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf#page=24" >}} · {{< new-tab "注意力测试代码" "https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/tests/test_model.py#L57-L71" >}}

## ML 系统设计：把模型放进产品

- **[AIMLInterviews · ML System Design](https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/ml-system-design.md)**\
  系统设计答题框架，附案例。\
  **这次学**　先把业务目标写成可验证的问题，再确定标签、离线指标和上线后的检查。 {{< new-tab "问题定义" "https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/ml-system-design.md#1-problem-formulation" >}} · {{< new-tab "离线与在线指标" "https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/ml-system-design.md#2-metrics-offline-and-online" >}}

- **[Chip Huyen · Machine Learning Systems Design](https://huyenchip.com/machine-learning-systems-design/toc.html)**\
  从数据管道、训练到在线服务的设计讲义。\
  **这次学**　检查线上预测时能否取到训练用的特征，以及怎样测服务延迟。 {{< new-tab "数据管道" "https://huyenchip.com/machine-learning-systems-design/design-a-machine-learning-system.html#data-pipeline-698WI4R" >}} · {{< new-tab "在线服务" "https://huyenchip.com/machine-learning-systems-design/design-a-machine-learning-system.html#serving-091rIYw" >}}

- **[Datawhale fun-rec](https://github.com/datawhalechina/fun-rec)**\
  中文推荐系统教材与实验，覆盖召回、排序和重排。\
  **这次学**　从双塔模型入手，分别说明召回和排序各处理多少候选，然后画出展示到反馈的流程。 {{< new-tab "双塔召回" "https://datawhalechina.github.io/fun-rec/chapter_1_retrieval/3.two_tower/index.html" >}} · {{< new-tab "排序与偏好预测" "https://datawhalechina.github.io/fun-rec/chapter_2_ranking/index.html" >}}

> **面试追问**　离线排序指标变好，线上点击率没变。你会先查实验分流、召回覆盖还是服务延迟？说出第一步能排除什么。

## 大模型与 AI 工程：按岗位选专题

### 模型原理与训练

- **[动手学大模型](https://github.com/Lordog/dive-into-llms)**\
  上海交通大学课程衍生的中文实践教程，涵盖微调、部署和其他专题。\
  **这次学**　从微调示例区分预训练权重、任务数据和更新的参数，讲清实验前后怎样比较。 {{< new-tab "微调的关键模块" "https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter1/README.md#21-%E7%90%86%E8%A7%A3%E5%85%B3%E9%94%AE%E6%A8%A1%E5%9D%97" >}} · {{< new-tab "微调 notebook" "https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter1/dive-tuning.ipynb" >}}

- **[Educatum · 大模型与生成式 AI 进阶题](https://www.educatum.com/llm-and-genai-advanced-interview-questions)**\
  面向大模型岗位的进阶口述题。\
  **这次学**　先口述自注意力如何给输入位置分配权重，再手画因果掩码，解释为什么训练能并行计算各位置，生成时要逐 token 解码。 {{< new-tab "Educatum 的自注意力题" "https://www.educatum.com/12555925845b81fd92a5c48f91079203" >}} · {{< new-tab "因果掩码代码" "https://github.com/karpathy/ng-video-lecture/blob/52201428ed7b46804849dea0b3ccf0de9df1a5c3/gpt.py#L64-L93" >}}

### 检索与智能体

- **[Hands-On Large Language Models](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main/chapter08)**\
  同名书的配套代码。第 8 章有语义搜索和检索增强生成（RAG）的 notebook。\
  **这次学**　跑第 8 章的检索示例，挑一个答错的问题，分开检查召回结果和生成答案。 {{< new-tab "第 8 章 notebook" "https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter08/Chapter%208%20-%20Semantic%20Search.ipynb" >}} · {{< new-tab "在 Google Colab 打开" "https://colab.research.google.com/github/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter08/Chapter%208%20-%20Semantic%20Search.ipynb" >}}

- **[learn-harness-engineering](https://github.com/walkinglabs/learn-harness-engineering)**\
  用项目练习智能体的工作环境、执行反馈和结果校验。\
  **这次学**　对照 P05 的生成与评估分工，在评分规则里加入一条「答案有无证据支持」，再用一条无依据的答案试跑。 {{< new-tab "P05 中文实作说明" "https://github.com/walkinglabs/learn-harness-engineering/blob/main/docs/zh/projects/project-05-grounded-qa-verification/index.md" >}} · {{< new-tab "评分规则示例" "https://github.com/walkinglabs/learn-harness-engineering/blob/main/projects/project-05/solution/gen-eval/evaluator-rubric.md" >}}

- **[Educatum · AI 工程题](https://www.educatum.com/ai-engineering-interview-questions)**\
  人工智能工程面试用的问答集。\
  **这次学**　用 ReAct 题练口述，讲清智能体如何交替行动和读取结果。再追问一次，任务失败时该记录什么，才能复现并定位问题？ {{< new-tab "Educatum 的 ReAct 题" "https://www.educatum.com/31655925845b80128df5de4b76d09b73" >}} · {{< new-tab "LangSmith 的追踪说明" "https://docs.langchain.com/langsmith/observability-concepts#traces" >}}

### 推理与后训练

- **[vLLM](https://github.com/vllm-project/vllm)**\
  大语言模型推理与服务引擎。\
  **这次学**　看 PagedAttention 如何按块管理键值（key–value，KV）缓存。手写一个缓存，核对逐 token 解码和完整前向的结果，再解释上下文变长时显存与延迟会怎样变。 {{< new-tab "PagedAttention 的块概念" "https://docs.vllm.ai/en/latest/design/paged_attention/#concepts" >}} · {{< new-tab "缓存分配函数" "https://github.com/vllm-project/vllm/blob/d64414d044a898c6db646dcecd122213accd1732/vllm/v1/core/kv_cache_manager.py#L400" >}}

- **[verl](https://github.com/verl-project/verl)**\
  用强化学习给语言模型做后训练的框架。\
  **这次学**　沿着组相对策略优化（GRPO）的示例，追一遍同一提示的多份回答如何得到奖励，以及优势估计如何参与策略更新。 {{< new-tab "GRPO 示例配置" "https://verl.readthedocs.io/en/latest/algo/grpo.html#reference-example" >}} · {{< new-tab "组内优势计算函数" "https://github.com/verl-project/verl/blob/61adb0c7eb90730306e7794c2bf248331211e657/verl/trainer/ppo/core_algos.py#L268-L331" >}}

## 更多题库与练习

- **[Cheng Zhang 的 MLE 题库](https://chengz.art/zh/mle-prep)**\
  按类别和难度筛选的面试题库。\
  **这次学**　先口述交叉熵为何适合分类，再挑两道薄弱题。记下第一次卡住的追问。 {{< new-tab "交叉熵题 · 支持文字定位的浏览器可直达" "https://chengz.art/zh/mle-prep#:~:text=%E8%A7%A3%E9%87%8A%E4%BA%A4%E5%8F%89%E7%86%B5%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E7%9A%84%E6%95%B0%E5%AD%A6%E5%8E%9F%E7%90%86%E5%8F%8A%E5%85%B6%E5%9C%A8%E5%88%86%E7%B1%BB%E4%BB%BB%E5%8A%A1%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8%E3%80%82" >}}

- **[Silvia Sapora 的面试准备指南](https://silviasapora.github.io/blog/ml-interviews.html)**\
  面向机器学习岗位的备考指南，涵盖编程、调试等练习。\
  **这次学**　从技术准备里的手写与调试清单挑一题做限时练习，隔天重做第一次没答出来的部分。 {{< new-tab "ML 编程与调试清单" "https://silviasapora.github.io/blog/ml-interviews.html#prep-technical:~:text=ML%20coding%20and%20debugging" >}}
