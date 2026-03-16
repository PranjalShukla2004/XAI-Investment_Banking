# 3. Methodology

## 3.1 Chapter Overview
This chapter presents the end-to-end methodology used to build and evaluate an explainable valuation model for investment-banking use cases. The implemented pipeline combines structured annual financial statement features with unstructured news information: raw company news is attached to each ticker-year observation, per-article sentiment is scored with FinBERT, and these scores are aggregated into numeric news features (for example, sentiment mean, dispersion, polarity shares, and coverage). The prediction target is `total_assets`, modeled in log-space to reduce heavy-tail effects.

The core modeling strategy is residual learning around an accounting identity baseline, where base assets are defined as `total_liabilities + total_equity`. Instead of predicting assets directly, the neural network predicts the residual relative to this baseline, which improves stability and keeps predictions grounded in financial structure. The implementation uses a regularized MLP with dropout, L1/L2 penalties, Huber loss, early stopping, learning-rate scheduling, feature standardization, optional correlation-based feature selection, and optional PCA. Additional guards in the final pipeline include quantile-based output clipping and a tiny-asset regime rule that reverts predictions to the identity baseline for very small firms to avoid extreme percentage-error blowups.

Evaluation is designed to test both temporal robustness and cross-company generalization. Internally, training uses a time-aware validation split by fiscal year. Externally, the final model is tested on two strict holdout settings: (i) out-of-time data and (ii) unseen-ticker data. Performance is reported against both the learned model and the identity baseline using RMSE, R², MAPE, non-zero MAPE, SMAPE, and tail-risk error diagnostics. To support reproducibility, each run stores configuration, selected features, scalers, training history, prediction outputs, and summary metrics as saved artifacts.

## 3.2 Research Design
### 3.2.1 Research Questions and Objectives
Define the specific research questions/hypotheses addressed in this chapter.

### 3.2.2 Methodological Approach
Justify the overall approach (e.g., quantitative, experimental, predictive modeling).

### 3.2.3 End-to-End Workflow
Summarize the full pipeline from data collection to evaluation and interpretation.

## 3.3 Data and Variables
### 3.3.1 Data Sources
Describe all data sources used (financial statements, market data, news/sentiment, etc.).

### 3.3.2 Sampling and Scope
Define population, sampling criteria, timeframe, and inclusion/exclusion rules.

### 3.3.3 Target and Predictor Variables
Define dependent variable(s), independent variables, and engineered features.

### 3.3.4 Data Preprocessing
Document cleaning, missing-value handling, outlier treatment, scaling/transforms, and encoding.

### 3.3.5 Dataset Splits
Specify train/validation/test design, temporal split logic, and leakage controls.

## 3.4 Model Development
### 3.4.1 Baseline Models
List and justify baseline approaches used for comparison.

### 3.4.2 Proposed Model Architecture
Describe the main model(s), structure, and rationale for design choices.

### 3.4.3 Training Procedure
Explain loss functions, optimization, regularization, early stopping, and convergence criteria.

### 3.4.4 Hyperparameter Tuning
Describe search strategy, parameter ranges, and model selection criteria.

## 3.5 Explainability Framework
### 3.5.1 Global Explainability
Describe methods for global feature importance and model-wide behavior.

### 3.5.2 Local Explainability
Describe methods for instance-level explanations and case-based interpretation.

### 3.5.3 Explanation Quality Checks
Define stability, consistency, and faithfulness checks for explanations.

## 3.6 Evaluation Methodology
### 3.6.1 Performance Metrics
Define primary and secondary metrics and justify why they match the task.

### 3.6.2 Validation Strategy
Explain cross-validation/time-split protocol and robustness checks.

### 3.6.3 Statistical Analysis
Include confidence intervals, significance testing, and effect-size reporting.

### 3.6.4 Error Analysis
Break down errors by subgroup, size, time period, or sector where relevant.

## 3.7 Reproducibility and Implementation Details
### 3.7.1 Experimental Environment
Document software versions, libraries, hardware, and runtime configuration.

### 3.7.2 Reproducibility Controls
Document random seeds, fixed splits, experiment logging, and artifact tracking.

## 3.8 Validity, Limitations, and Ethics
### 3.8.1 Threats to Validity
Discuss internal, external, construct, and statistical conclusion validity risks.

### 3.8.2 Methodological Limitations
State known limitations of data, model assumptions, and evaluation setup.

### 3.8.3 Ethical and Practical Considerations
Address bias, fairness, interpretability risks, and responsible use in finance.

## 3.9 Chapter Summary
Summarize methodological choices and transition to results/experiments chapter.

## Overleaf Copy-Paste (LaTeX)
Use this directly in Overleaf:

```latex
\chapter{Background Chapter}

\section{Chapter Overview}
\subsection{1}

\section{Research Design}
\subsection{Research Question and Objectives}
\subsection{Methodological Approach}
\subsection{End-to-End Workflow}

\section{Data and Variables}
\subsection{Data Sources}
\subsection{Sampling and Scope}
\subsection{Target and Predictor Variables}
\subsection{Data Preprocessing}
\subsection{Dataset Splits}

\section{Model Development}
\subsection{Baseline Models}
\subsection{Proposed Model Architecture}
\subsection{Training Procedure}
\subsection{Hyperparameter Tuning}

\section{Explainability Framework}
\subsection{Global Explainability}
\subsection{Local Explainability}
\subsection{Explanation Quality Checks}

\section{Evaluation Methodology}
\subsection{Performance Metrics}
\subsection{Validation Strategy}
\subsection{Statistical Analysis}
\subsection{Error Analysis}

\section{Reproducibility and Implementation Details}
\subsection{Experimental Environment}
\subsection{Reproducibility Controls}

\section{Validity, Limitations, and Ethics}
\subsection{Threats to Validity}
\subsection{Methodological Limitations}
\subsection{Ethical and Practical Considerations}

\section{Chapter Summary}
```
