---
title: "vLLM vs TensorRT-LLM on a TTS model's LLM backbone"
date: 2026-08-01
repo: "https://github.com/HenryVarro666/tts-inference-engine-bench"
lang: Python
stack: ["PyTorch", "vLLM", "TensorRT-LLM", "Hugging Face Transformers", "Docker"]
topics: [speech, ml-systems]
---
Reproducible H100 benchmark of eager PyTorch, vLLM 0.17.1 and TensorRT-LLM on the autoregressive backbone of a multi-stage TTS model. vLLM reaches about 453 tok/s against 213 for TensorRT-LLM and 21 for eager, but the LLM is only about 21% of end-to-end latency; the flow-matching stage takes about 68%, which these engines do not cover.
