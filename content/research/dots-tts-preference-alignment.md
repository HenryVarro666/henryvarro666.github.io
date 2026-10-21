---
title: "When Likelihood Disagrees with Quality"
date: 2026-06-01
repo: "https://github.com/HenryVarro666/dots-tts-preference-alignment"
lang: Python
topics: [speech]
---
A diagnosis of Direct Preference Optimization on a continuous-latent flow-matching TTS model. The automatic-metric reward consistently prefers the candidate the model finds *less* likely, so standard DPO pushes the model the wrong way and quality degrades. Shown with a held-out benchmark, a training-free mechanistic probe and a cross-architecture replication, followed by the alignment routes that do work.
