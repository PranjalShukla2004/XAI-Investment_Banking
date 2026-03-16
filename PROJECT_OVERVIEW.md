# Project Overview

## 1. Project Purpose

This project develops an explainable machine-learning pipeline for investment-banking style analysis. The work is organized around two connected modelling tasks:

1. a valuation task, where the model estimates firm value using accounting fundamentals and news-derived features
2. a risk task, where the model estimates the firm's forward-looking downside risk using market data

The project is not only about predictive accuracy. A central requirement is interpretability. The models must produce outputs that are useful in a finance setting, where users need to understand why a company is being assessed in a particular way rather than receiving only a black-box score. For that reason, explainability is treated as a core part of the system design rather than as an optional add-on.


## 2. Problem Statement

Traditional corporate and market analysis often relies on large volumes of financial statement data, ratio analysis, price data, and qualitative judgment from news and events. In practice, these sources are difficult to combine consistently at scale. A purely manual process is slow and hard to standardize, while a purely predictive model without interpretation is difficult to defend in a financial context.

This project addresses that gap by building an integrated pipeline with three aims:

1. estimate valuation from structured fundamentals and news features
2. estimate future downside risk from market data
3. explain model behavior using SHAP so that both global and firm-level drivers can be inspected

The core research problem is therefore:

How can a machine-learning pipeline combine financial statement information, news signals, and market data to generate useful valuation and risk estimates while remaining interpretable enough for finance-oriented decision support?


## 3. High-Level System Design

The repository is organized around three major workstreams:

- **Valuation pipeline**: predicts `total_assets` as the current implementation target / valuation proxy from financial statement variables and engineered news features.
- **Risk pipeline**: is intended to predict forward 1-year drawdown using market-price histories for the same ticker universe.
- **Explainability pipeline**: generates SHAP-based explanations for the valuation models so the learned drivers can be analyzed globally and locally.

These workstreams are related, but they are intentionally kept as parallel modules so that valuation, risk, and explainability can be developed and evaluated without tightly coupling the code.


## 4. Valuation Workstream

### 4.1 Objective

The valuation side of the project predicts `total_assets` from a combination of:

- raw and engineered accounting features
- financial ratios
- selected transformed numeric features
- news-derived variables such as article counts and sentiment-related signals

In the current implementation, `total_assets` is used as the modelling target because it provides a stable, well-defined accounting quantity that can be linked to both firm scale and valuation-oriented analysis.


### 4.2 Main Valuation Models

The valuation pipeline contains multiple modelling files, but the active baseline focus is:

- `residual_mlp_valuation.py` as the main valuation pipeline
- `xgb_valuation.py` as the benchmark comparison model

The residual MLP is the primary model because it is designed to learn corrections around an accounting identity baseline rather than learning the entire target from scratch.


### 4.3 Residual MLP Design

The residual MLP uses an accounting baseline built from:

- `total_liabilities`
- `total_equity`

These are combined into an identity-based baseline feature:

- `identity_base_assets = total_liabilities + total_equity`

The model then learns a residual correction around that baseline. Conceptually, the architecture separates:

- the part of `total_assets` already implied by basic accounting structure
- the additional variation captured by other fundamentals and news features

This design is useful because it anchors the model in finance logic before allowing the neural network to learn non-trivial adjustments.


### 4.4 XGBoost Benchmark

The XGBoost model serves as a benchmark for comparison. It is useful because:

- it is strong on structured tabular data
- it provides a non-neural baseline
- it supports efficient native feature attributions through `pred_contribs`

This allows the project to compare a more structured residual-learning architecture against a high-performing gradient-boosted tree model.


### 4.5 Valuation Evaluation

The valuation codebase is separated into modelling modules and evaluation modules. Evaluation includes validation and test-set style analysis, including out-of-time and ticker-generalization scenarios for some model variants. This separation helps preserve a cleaner experimental workflow and avoids mixing model-training code with reporting code.

Valuation artifacts are stored under:

- `experiments/valuation/runs/`

Explainability artifacts for valuation are stored separately under:

- `experiments/valuation/SHAP/`


## 5. Risk Workstream

### 5.1 Objective

The risk side of the project is intended to predict future downside risk for each firm, specifically a forward 1-year drawdown measure. This extends the system beyond current company valuation into forward-looking market risk.


### 5.2 Why the Risk Model Matters

Valuation alone does not capture the full investment-banking or financial analysis problem. Two firms may appear similar on size or financial profile but face very different future downside risk. A useful system should therefore answer both:

- what is the firm's current valuation-related profile?
- what is the likely severity of future downside risk?

The risk model is meant to provide this second layer.


### 5.3 Risk Data Pipeline

The risk pipeline depends on daily market data. The project includes a market-data fetch stage that downloads daily flatfiles and a drawdown-target generation stage that converts those market histories into forward-looking labels.

The current intended flow is:

1. fetch daily market data across the required date range
2. organize the saved flatfiles in a layout that can be consumed downstream
3. build future drawdown targets from those market histories
4. train a separate risk model on the generated targets

The relevant market and risk modules include:

- `src/data_fetch/market/fetch_market_data.py`
- `src/models/risk/build_future_drawdown_targets.py`

This workstream is structurally planned but remains secondary to the valuation and explainability work.


## 6. Explainability and SHAP

### 6.1 Why Explainability Is Central

In finance and investment contexts, predictive performance alone is not sufficient. Analysts, researchers, and stakeholders need to understand:

- which features drive model predictions overall
- which factors push an individual firm upward or downward
- whether the model is learning economically meaningful structure or spurious artifacts

SHAP is used in this project to provide that interpretability layer.


### 6.2 SHAP Scope in This Project

SHAP is currently applied to the valuation pipeline. The explainability workflow compares:

- the main residual MLP valuation model
- the benchmark XGBoost valuation model

The explainability code lives under:

- `src/explainability/shap_valuation.py`
- `src/explainability/plot_shap_valuation.py`


### 6.3 XGBoost SHAP Strategy

For XGBoost, the project uses XGBoost's native contribution output via `pred_contribs`. This provides efficient SHAP-style attributions directly from the trained tree model.

This is appropriate for the benchmark because:

- tree models support fast contribution calculations
- the attribution is directly tied to the fitted structure of the booster
- the result can be used as a strong comparison baseline


### 6.4 Residual MLP SHAP Strategy

The residual MLP required more careful design. A full-pipeline SHAP explanation of the reconstructed prediction caused the explicit accounting baseline term to dominate the global importance plots. That made the explanation less useful, because it mostly restated the accounting identity instead of showing what the neural network had actually learned.

To address this, the project moved to a **residual-only SHAP** strategy for the main neural model.

That means the SHAP analysis for the residual MLP explains:

- the learned residual correction on the `log1p(total_assets)` scale

and not:

- the full baseline-plus-correction prediction

This is a key design choice in the project. It keeps the explainability aligned with the modelling objective of the residual network, and it prevents the accounting baseline from overwhelming the interpretation.


### 6.5 SHAP Outputs

The SHAP pipeline produces both machine-readable artifacts and visual outputs. These include:

- validation prediction tables
- SHAP value arrays
- feature importance tables
- beeswarm plots
- dependence plots
- local explanation plots for selected high-error cases
- summary JSON files describing the explainability run

These artifacts are saved under:

- `experiments/valuation/SHAP/residual_mlp/`
- `experiments/valuation/SHAP/xgb/`


### 6.6 What SHAP Adds to the Project

The explainability layer makes it possible to answer questions such as:

- Which accounting and news features matter most for valuation corrections?
- Does the model rely on intuitive financial variables such as liquidity, leverage, profitability, or capital expenditure?
- For a specific company-year observation, which features pushed the predicted valuation correction upward or downward?
- How does the neural model's behavior compare with the benchmark tree model?

This turns the project from a prediction exercise into an interpretable analytical system.


## 7. Data and Features

The project combines three main types of information:

- **Fundamental data**: accounting statement items such as liabilities, equity, revenue, debt, cash, profitability, and other firm-level financial variables.
- **Engineered financial ratios**: variables intended to standardize company comparisons across scale, including measures linked to liquidity, leverage, profitability, cash generation, and efficiency.
- **News-derived features**: variables derived from firm-related news coverage, such as volume of coverage and sentiment-oriented signals.

This mixed feature design reflects the practical reality that valuation and risk assessment are not driven by one source alone. Instead, the project tries to combine structured accounting information with a lighter layer of qualitative market information extracted from news.


## 8. Repository Structure

At a high level, the repository is organized as follows:

- `src/models/valuation/`: valuation model training code
- `src/evaluation/`: evaluation scripts for valuation experiments
- `src/models/risk/`: risk-target and risk-related modelling utilities
- `src/data_fetch/market/`: market-data ingestion scripts
- `src/explainability/`: SHAP generation and SHAP plotting utilities
- `experiments/valuation/runs/`: valuation run artifacts
- `experiments/valuation/SHAP/`: valuation explainability artifacts

This structure separates data ingestion, modelling, evaluation, and explainability into distinct layers, which makes the project easier to extend and document.


## 9. Current Project State

The project currently has a more mature valuation side than risk side.

The present status is:

- valuation modelling is implemented and actively evaluated
- residual MLP is treated as the main valuation pipeline
- XGBoost is retained as the benchmark model
- SHAP analysis is implemented for both valuation models
- residual-only SHAP has been introduced to improve interpretability for the neural model
- risk data ingestion and target-building are in progress, but the full risk-model training pipeline is still pending

This means the project already demonstrates the intended combined direction, but the development maturity is uneven across workstreams.


## 10. Research Contribution of the Project

The project aims to contribute in three ways:

1. **Methodological integration**: combining fundamentals, news features, market data, predictive modelling, and explainability in one pipeline
2. **Model design**: using a residual neural architecture anchored in accounting structure rather than relying only on unconstrained prediction
3. **Interpretability in finance**: using SHAP not just as a generic XAI add-on, but as a tailored explanation layer that had to be redesigned to reflect the structure of the residual model

The project is therefore not simply building a valuation model or a risk model in isolation. It is building an explainable analytical framework that can support deeper financial interpretation.


## 11. Summary

In summary, this repository is an explainable financial modelling project with three linked objectives:

- estimate valuation from fundamentals and news
- estimate forward downside risk from market data
- explain valuation-model behavior using SHAP

The valuation pipeline is currently the most developed component, with the residual MLP as the main model and XGBoost as the benchmark. The risk pipeline is being developed as a parallel workstream based on market flatfiles and future drawdown targets. SHAP is a core part of the project because the end goal is not only accurate prediction, but also transparent and defensible model reasoning.
