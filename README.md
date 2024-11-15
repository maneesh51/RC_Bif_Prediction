# Predicting multi-parametric dynamics of an externally forced oscillator using reservoir computing and minimal data


- This repository contains source files and figures for the article **"Predicting multi-parametric dynamics of an externally forced oscillator using reservoir computing and minimal data"**. 
- Authors: Manish Yadav, Swati Chauhan, Manish Dev Shrimali and Merten Stender
- The preprint of the manuscript is available on ArXiv: https://arxiv.org/abs/2408.14987

<p align="center">
<img src="https://github.com/maneesh51/RC_Bif_Prediction/blob/main/Figures/Fig1.png">
</p>


## 1. Brief description
Mechanical systems exhibit diverse dynamics, ranging from harmonic oscillations to chaos, influenced by internal parameters like stiffness and external forces. Determining complete bifurcation diagrams is often resource-intensive or impractical. This study explores a data-driven approach to infer bifurcations from limited system response data, using reservoir computing (RC). As a proof of concept, the method is applied to a Duffing oscillator with harmonic forcing, trained on minimal data. Results show that RC effectively learns system dynamics for training conditions and robustly predicts qualitatively accurate responses, including higher-order periodic and chaotic dynamics, for unseen multi-parameter regimes.

## 2. Description of the files present in this repository
All the functions required to build the RC for multi-parametric predictions and to reproduce the manuscript results are present in `1.DufRC_1D_Reps_Figs1_2.py`. All the simulations can also be run using the Python notebook `1.DufRC_1D_Reps_Figs1_2.ipynb`.

## 3. Online executable code
An online executable version of the code is also provided in the Python notebook format on [Google Colab](https://colab.research.google.com/drive/10z6Bs2C83DwtnomQdFChYKzzjCUIL0xi?usp=sharing)

## 4. Specific Python libraries and their versions
- **NetworkX:** version 2.8.4 (https://networkx.org/documentation/stable/release/release_2.8.4.html)
- **Matplotlib:** version 3.8.0 (https://matplotlib.org/stable/users/prev_whats_new/whats_new_3.8.0.html)
- **Scipy:** version 1.10.1 (https://docs.scipy.org/doc/scipy-1.10.1/)
