---
title: "Spectator Sport"
date: 2022-05-17
repo: "https://github.com/HenryVarro666/spectatorSport"
lang: Jupyter
stack: ["pandas", "scikit-learn", "Gensim", "Selenium"]
status: "course project"
topics: [data-science, nlp]
---
DATA 641 team project (three people) on news coverage of Russia's invasion of Ukraine: 2,921 liveuamap posts serve as references, each turned into a dated Google query, giving 6,347 articles from 902 domains (Selenium and newspaper3k). Two LDA comparisons follow: per-outlet topic models of the references each outlet covered, compared by Jaccard distance, and the bigram partners of shared topic words in pairs of articles. Contribution: the proposal, zero-shot NLI tests with XLM-RoBERTa, LDA experiments on reference titles and an articles-per-outlet filter. No bias score or classifier is computed; conclusions are read from the plots.
