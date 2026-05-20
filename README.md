<div align="center">

# <span style="color:#0284c7">Supervised Istanbul Housing Evaluation (Random Forest Ensembles)</span>

![Python](https://img.shields.io/badge/Python-3.10%2B-1e293b?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine_Learning-f7931e?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Dataframe-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-059669?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-0284c7?style=for-the-badge)

<br/>

<h3>
  <em>"Predicting Real Estate Metrics via Bootstrap Aggregation, Random Subspace Sampling, and Dual Ensemble Topologies"</em><br/>
</h3>

<br/>

[📌 Overview](#-overview) • [✨ Features](#-features) • [🚀 Pipeline Flow](#-pipeline-flow) • [⚙️ Installation](#%EF%B8%8F-installation) • [🧠 Mathematical Foundations](#-mathematical-foundations) • [🆚 Single Trees vs Random Forests](#-architectural-comparison-single-decision-tree-vs-random-forest-ensemble) • [🤝 Contributing](#-contributing)

</div>

---

## 📌 Overview

**Supervised Istanbul Housing Evaluation** is a predictive machine learning repository that implements dual **Random Forest Ensembles** to analyze real estate valuation patterns. Processing structural attributes (*price per square meter*, *district location*, and *currency exchange index*), this pipeline runs side-by-side training tracks to map continuous valuation scales and discrete, quantile-split tier assignments (`Cheap`, `Average`, `Expensive`) simultaneously.

By deploying bootstrap bagging workflows, the system builds multiple independent decision trees to protect against overfitting, tests its parameters on a rigorous $20\%$ holdout data split, and renders clear prediction bar layouts and categorical classification pie charts using Matplotlib.

---

## ✨ Features

### 🎯 Core Capabilities
| Feature | Description |
|---|---|
| 🗂️ **One-Hot Subspace Encoding** | Expands spatial districts and currency flags into numeric vector tracking blocks via `pd.get_dummies`. |
| 🔢 **Quantile Target Quantization** | Automatically segments continuous values into uniform categorical tiers using `pd.qcut`. |
| 🛡️ **Holdout Data Partitioning** | Allocates an isolated $20\%$ test set using `train_test_split` to simulate out-of-sample data conditions. |
| 🌲 **Parallel Tree Bagging Engine** | Instantiates $50$ parallel estimators across both regression and classification structures. |
| 🎨 **Dual Inference Analytics** | Renders numeric bar charts for real-time item inferences alongside distribution pie charts for all unseen test samples. |

### 🌟 Design Highlights
- 🧠 **Ensemble Learning Framework** — Combines multiple weak predictors into a strong model using bagging mechanics.
- ⚡ **Out-of-Sample Verification** — Trains models strictly on training sets and evaluates accuracy using entirely unseen testing profiles.
- 📉 **Multi-Objective Modeling** — Maps continuous currency outputs alongside discrete classification buckets using the same baseline feature entries.

---

## 🚀 Pipeline Flow

┌─────────────────────────────────────────────────────────┐│               ISTANBUL PROPERTY DATASET INGESTION       ││         (Price, Square Meter Cost, District, Exchange)  │└─────────────────────────────────────────────────────────┘│▼[ One-Hot Matrix Expansion & Target Tier Quantization ]│▼┌──────────────────────┴──────────────────────┐▼                                             ▼[ Target: Continuous Price ]                 [ Target: Quantile Tier ]│                                             │▼                                             ▼┌───────────────────────────────────┐         ┌───────────────────────────────────┐│ RANDOM FOREST REGRESSOR           │         │ RANDOM FOREST CLASSIFIER          ││ - 50 Bagged Estimators            │         │ - 50 Bagged Estimators            ││ - Average Ensemble Voting Output  │         │ - Majority Consensus Label Vote   │└───────────────────────────────────┘         └───────────────────────────────────┘│                                             │└──────────────────────┬──────────────────────┘│▼┌─────────────────────────────────────────────────────────┐│                 OUT-OF-SAMPLE TEST PREDICTIONS          ││                                                         ││   Inference: Process unseen test samples (20% Split)    ││   Output 1: Matplotlib Bar Graph of Continuous Forecasts││   Output 2: Matplotlib Pie Chart of Tier Labels Spread  │└─────────────────────────────────────────────────────────┘
---

## ⚙️ Installation

### Prerequisites

Make sure your local computer has these foundational software components set up:
- Python 3.10+
- Git

---

### 🔧 Step-by-Step Setup

**1. Clone the Repository**
```bash
git clone [https://github.com/jaweriashakoor/Supervised-Istanbul-Housing-RandomForest.git](https://github.com/jaweriashakoor/Supervised-Istanbul-Housing-RandomForest.git)
cd Supervised-Istanbul-Housing-RandomForest
2. Create a Virtual EnvironmentBashpython -m venv myenv

# Windows (PowerShell)
.\myenv\Scripts\Activate.ps1

# Linux / macOS
source myenv/bin/activate
3. Install DependenciesBashpip install numpy pandas scikit-learn matplotlib
4. Execute the Ensemble ScriptBashpython istanbul_housing_rf.py
🧠 Mathematical FoundationsBootstrap Aggregation (Bagging)The Random Forest model trains multiple deep decision trees in parallel. For each individual tree $b \in \{1, \dots, B\}$, the algorithm samples a subset $X_b$ from our training data with replacement. This variance-reduction technique ensures every tree evaluates a unique sample layout.Random Subspace SamplingTo ensure the individual trees do not learn identical rules, Random Forests apply feature bagging. At each node split inside a tree, the algorithm chooses a random subset of features $m$ from our total feature pool $M$:$$m = \sqrt{M} \quad \text{(for classification)} \quad \Big| \quad m = \frac{M}{3} \quad \text{(for regression)}$$Ensemble Consensus ConsensusWhen an unseen sample $x^*$ enters the system for evaluation, the Random Forest passes it down all $B$ independent trees and calculates the final output based on the network's model layout:Regression Mode Output ($\hat{y}_{\text{reg}}$): Computes the numerical average of all individual tree predictions:$$\hat{y}_{\text{reg}} = \frac{1}{B}\sum_{b=1}^{B} f_b(x^*)$$Classification Mode Output ($\hat{y}_{\text{clf}}$): Evaluates a majority vote consensus across all trees, assigning the category label with the highest frequency:$$\hat{y}_{\text{clf}} = \text{mode}\big\{ f_1(x^*), f_2(x^*), \dots, f_B(x^*) \big\}$$🆚 Architectural Comparison: Single Decision Tree vs. Random Forest EnsembleArchitectural FeatureSingle Decision TreeRandom Forest EnsembleStructural CompositionA single isolated logical pathway tree.Multiple independent parallel trees ($n\_estimators=50$).Overfitting PronenessHigh; easily grows too deep and memorizes noise.Extremely low; averaging across random trees neutralizes individual variances.Feature SelectionEvaluates every input attribute at every node.Evaluates a random feature subset at each node.Statistical StyleDeterministic (gives the exact same result every run).Stochastic (uses random row and column sampling paths).🤝 ContributingContributions help keep our open-source tools robust, optimized, and ready for development!Bash# 1. Fork the Project Repository
# 2. Setup your feature branch
git checkout -b feature/feature-importance-tracking

# 3. Commit functional updates
git commit -m "Add: Plot Gini importance scores to discover key house price drivers"

# 4. Push updates to origin branch
git push origin feature/feature-importance-tracking

# 5. Open a Pull Request
📄 LicenseThis ensemble evaluation architecture is distributed as open-source software under the terms of the MIT License.
