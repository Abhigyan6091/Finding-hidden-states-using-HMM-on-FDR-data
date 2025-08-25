# Finding-hidden-states-using-HMM-on-FDR-data

📌 Overview

This project applies Hidden Markov Models (HMMs) to analyze Flight Data Recorder (FDR) time series from 5335 flights. The goal is to determine the optimal number of hidden states that best describe the flight dynamics using KL divergence as the model selection criterion.

The analysis leverages parallel computing (32-core CPU) to scale efficiently across thousands of flights, making it suitable for large-scale aviation safety and anomaly detection studies.

⚙️ Methodology

Data Input

FDR sequences from multiple flights (altitude, speed, pitch, roll, etc.).

Each flight is modeled using HMMs with varying hidden states (2–5).

Model Training

HMMs are trained on subsampled data with probabilities p = 0.85, 0.90, 0.95.

Transition (A), emission (B), and initial state (π) matrices are estimated.

Model Selection (KL Divergence)

KL divergence between fitted models is computed for each p.

The model with the minimum KL divergence across subsamples is chosen as the best fit.

Parallelization

MATLAB’s parpool and parfor are used to distribute flight computations across CPU cores.

Reduces computation time significantly for 5000+ flights.

📊 Results

For each flight, the best hidden state number (hSt) is stored.

A histogram visualization summarizes the distribution of selected hidden states across all flights.

Example histogram:

histogram(hstArray, 'BinEdges', 1.5:1:5.5);
xticks(2:5);
xlabel('Hidden States');
ylabel('Flights');
title('Best HMM Hidden States Across Flights');

🚀 Features

✅ Automatic HMM model selection via KL divergence.

✅ Parallelized training across thousands of flights.

✅ Histogram plots for global hidden state distribution.

✅ Modular MATLAB functions for easy extension.

🔧 Requirements

MATLAB R2022a or later

Statistics and Machine Learning Toolbox

Parallel Computing Toolbox

📂 Repository Structure
├── hmm_main.m         % Main parallelized script
├── compare_matrices.m % KL divergence comparison
├── hidden_state.m     % Hidden state selection logic
├── results/           % Stores results (hstAll, KL values)
├── plots/             % Generated histograms and figures
└── README.md          % Project documentation

📈 Future Work

Extend to more than 5 hidden states.

Explore alternative model selection metrics (AIC, BIC, CV likelihood).

Apply anomaly detection for flight safety analysis.

Deploy on HPC or cloud cluster for >10,000 flights.
