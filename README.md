# Causal Effects of Informal Caregiving on Community Residence

## Overview
Remaining community-dwelling rather than transitioning to institutional care is a key indicator of successful aging, associated with autonomy, quality of life, and reduced healthcare costs. Informal caregiving provided by family and friends plays a critical role in enabling older adults to remain in their homes, yet estimating its **causal effect** is challenging due to **time-varying confounding** between caregiving needs and health status.

This project estimates the **causal effect of informal caregiving intensity on community residence** among older adults using longitudinal data from the **National Health and Aging Trends Study (NHATS)**. To address time-varying confounders affected by prior exposure, the analysis employs a **Marginal Structural Model (MSM)** estimated via **stabilized inverse probability weighting (IPW)**.

---

## Features
- Longitudinal causal inference using Marginal Structural Models (MSMs)
- Stabilized inverse probability weighting to address time-varying confounding
- Multi-category time-varying treatment (caregiving intensity)
- Population-level causal effect estimation
- Weight diagnostics including overlap and effective sample size
- Reproducible Python-based analysis pipeline

---

## Requirements
- Python 3.9+
- NumPy
- pandas
- statsmodels
- scikit-learn
- matplotlib / seaborn (for diagnostics and plots)

---

## Dataset
This project uses data from the **National Health and Aging Trends Study (NHATS)**, a nationally representative longitudinal survey of U.S. Medicare beneficiaries aged 65 and older.

- Survey rounds: NHATS Rounds 1–14
- Unit of analysis: person–round
- Approximate sample size: ~80,000 person–round observations
- Data structure: unbalanced longitudinal panel

Key NHATS components used:
- Sample Person (SP) files
- Other Person (OP) caregiving files
- Residential status indicators
- Health and functional limitation measures

---

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/causal-effects-of-informal-caregiving.git
cd causal-effects-of-informal-caregiving
pip install -r requirements.txt
```

Ensure access to NHATS data files prior to running the analysis.

---

## Usage

Run the main analysis pipeline after preprocessing NHATS data:

```bash
python CIPr.py
```

Supporting scripts:
- `StD.py`: Data construction and variable harmonization
- `StMSM.py`: MSM estimation, stabilized IPW computation, and diagnostics

---

## Code Walkthrough

### Data Loading and Preprocessing
- Harmonizes NHATS rounds 1–14 into a person–round panel
- Aggregates monthly caregiving hours across helpers
- Constructs time-varying covariates:
  - Demographics
  - Socioeconomic status
  - Health and functional limitations
- Excludes observations with missing key variables
- Addresses implausible caregiving hour values

---

### LSTM Model Integration
Not applicable.  
This project uses **statistical causal inference models**, not machine learning predictors.

---

### User Input Processing
- User specifies NHATS file paths and analysis parameters
- Treatment categories and covariates defined programmatically
- Survey round indicators added automatically

---

### Similarity Calculation
Not applicable.  
Causal estimation is based on **counterfactual modeling**, not similarity measures.

---

### Recommendation
Not applicable.  
This project estimates **population-level causal effects**, not individual-level recommendations.

---

## Example Input and Output

### Input
- Longitudinal NHATS person–round dataset
- Key variables:
  - Informal caregiving hours per month
  - Community residence indicator
  - Time-varying health and functional covariates

---

### Output
- Stabilized inverse probability weights
- Effective sample size and weight diagnostics
- Estimated marginal probabilities of community residence under:
  - No caregiving
  - Low (1–20 hrs/month)
  - Moderate (21–80 hrs/month)
  - High (81+ hrs/month)
- MSM coefficient estimates with cluster-robust standard errors
- Tables and diagnostic plots

---

## Customization
- Modify caregiving intensity cutoffs
- Add additional time-varying covariates
- Adjust weight truncation thresholds
- Extend outcome definitions beyond community residence

---

## Limitations
- Causal validity depends on no unmeasured confounding
- Caregiving quality is not observed, only hours
- Residential status observed annually
- Effects are population-average, not heterogeneous

---

## Future Enhancements
- Incorporate inverse probability of censoring weights
- Explore effect heterogeneity by demographic subgroups
- Extend to joint modeling of caregiving and health trajectories
- Sensitivity analysis for unmeasured confounding

---

## Acknowledgments
This project was developed at **Syracuse University**.

**Author:**
- Neda Abdolrahimi

**Data Source:**
- National Health and Aging Trends Study (NHATS)

