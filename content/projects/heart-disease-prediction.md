---
title: "Heart Disease Prediction"
date: 2022-12-01
repo: "https://github.com/HenryVarro666/heart-disease-prediction"
lang: Jupyter
stack: ["scikit-learn", "pandas", "NumPy", "Matplotlib"]
status: "course project"
topics: [data-science, healthcare]
---
Course 603 term project: five scikit-learn classifiers on the UCI Cleveland heart-disease data (303 patients, one-hot encoded, one 80/20 split). SVM and random forest reach ROC AUC 0.97 on the 61 held-out patients, ahead of logistic regression (0.96), decision tree (0.95) and k-NN (0.89); the scaler is fitted before the split, so test statistics leak into the scaling.
