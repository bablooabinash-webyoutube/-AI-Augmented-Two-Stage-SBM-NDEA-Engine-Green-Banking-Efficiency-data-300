# 🌿 AI-Augmented Two-Stage SBM-NDEA Engine: Green Banking Efficiency

An enterprise-grade analytical platform that combines Tone’s mathematical **Slack-Based Measure Two-Stage Network Data Envelopment Analysis (SBM-NDEA)** with a **Deep Learning Surrogate Engine** to evaluate, predict, and optimize green banking operations across commercial branch networks.

---

## 📌 Key Features

* **Two-Stage SBM Optimization Engine:** Implements non-radial Slack-Based Measure formulations (Input-Oriented, Output-Oriented, and Non-Oriented Charnes-Cooper models) using `scipy.optimize.linprog` with the HiGHS solver.
* **Carbon Credit Risk Penalization:** Directly incorporates carbon-heavy Non-Performing Loans ($y_2$) as undesirable output risk penalties within Stage 2 operations.
* **AI/DL Surrogate Layer:** Integrates Multi-Layer Perceptron (MLP/PyTorch) and Random Forest models to surrogate linear programs, achieving sub-millisecond efficiency scoring ($<1\text{ ms}$) for out-of-sample branches and live scenario modeling.
* **Explainable AI (XAI) Bottleneck Diagnostics:** Extracts global network drivers and feature importances to flag whether branches fail due to front-office operations or back-office ESG risk conversion.
* **Interactive Enterprise Dashboard:** Includes a full-stack Streamlit app featuring Plotly visualizations for individual branch audits, prescriptive target improvement charts, and dynamic custom health checks.

---

## 🏗️ Two-Stage Network Architecture

```text
[ Operational Inputs ]           [ Intermediate Measures ]                 [ Final Outputs ]
(Stage 1)                          (Stage 1 ➔ Stage 2)                    (Stage 2)

x1: Operational Overhead ──┐                                         ┌──► y1: Net Green Revenue (Desirable)
                           ├──► [ STAGE 1: GREEN CONVERSION ] ──► z1: Green Deposits ──┼──►
x2: FTE Staff Count      ──┘                                  ──► z2: ESG Accounts   ──┴──► y2: Carbon-Heavy NPLs (Risk Penalty)
