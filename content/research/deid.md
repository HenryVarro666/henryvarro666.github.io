---
title: "Open-LLM De-identification"
date: 2023-08-01
repo: "https://github.com/HenryVarro666/open-llm-deid"
lang: Python
stars: 1
topics: [llm, healthcare]
---
Can open 7B LLMs strip protected health information from clinical notes, zero-shot? Runs LLaMA 1 and 2, Falcon-7B and Flan-T5 on the 514-note n2c2 2014 test set under explicit, implicit and ask-then-delete prompts. A re-read of the 2023 runs shows the rewrite scores sit at the floor of an untouched note (0.597), because the scripts saved the prompt with the output; asking Flan-T5 for the PHI and deleting its answers gave the one real gain (0.737). A synthetic worked example shows every step without the restricted data.
