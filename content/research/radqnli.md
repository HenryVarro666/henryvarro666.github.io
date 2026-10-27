---
title: "RadQNLI: radiology NLI for LLMs"
date: 2023-04-01
repo: "https://github.com/HenryVarro666/RadQNLI"
lang: Jupyter
stack: ["OpenAI API", "pandas", "NLTK", "Hugging Face Evaluate"]
topics: [llm, healthcare]
paper: "radiology-nli-tradeoffs"
---
Can one generic LLM read radiology reports as well as models fine-tuned on the task? Builds RadQNLI from RadQA's MIMIC-III reports by pairing each clinician question with every report sentence and keeping the negatives that share words with the question (ROUGE-1 ≥ 0.2): 9,838 pairs, rebuilt exactly by the repository's scripts. ChatGPT with three examples drops from 0.81 to 0.62 accuracy as the test negatives share more words with the question, but random negatives of the same count drop almost as far (0.83 to 0.64): most of the decline follows the rising share of entailment pairs. In the paper GPT-4 reaches 0.842 accuracy, against 0.847 for BERT-family models fine-tuned on all the training data. RadQA needs PhysioNet credentialing; a synthetic example shows every step.
