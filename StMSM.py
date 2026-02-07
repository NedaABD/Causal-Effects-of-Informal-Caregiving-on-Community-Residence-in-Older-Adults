import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

PANEL_PATH = "/Users/nedaabdolrahimi/Documents/Fall25/Causal inference/Project/Data/nhats_panel_r1_r14.csv"
df = pd.read_csv(PANEL_PATH)

print("Loaded panel:", df.shape)
print(df.columns.tolist())

# Required columns from stacking
req = ["spid","round","Y_comm","H_cat","age_cat","income","adl_help_count","chronic_count","mobility_count","gender","racehisp"]
missing = [c for c in req if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}. Re-run stacking or adjust this list.")

# Keep essentials
df_msm = df[req].copy()

# Clean: NHATS special negative codes -> missing
for c in ["age_cat","income","adl_help_count","chronic_count","mobility_count","gender","racehisp"]:
    df_msm[c] = pd.to_numeric(df_msm[c], errors="coerce")
    df_msm.loc[df_msm[c] < 0, c] = np.nan

df_msm["round_str"] = df_msm["round"].astype(int).astype(str)
df_msm["H_cat"] = df_msm["H_cat"].astype(str)
df_msm["Y_comm"] = pd.to_numeric(df_msm["Y_comm"], errors="coerce")

# Minimal dropna (don’t nuke everything)
df_msm = df_msm.dropna(subset=["Y_comm","H_cat","round_str","age_cat","adl_help_count","chronic_count","mobility_count"])
print("Rows available for longitudinal MSM:", len(df_msm))
if len(df_msm) == 0:
    raise ValueError("df_msm is empty after filtering. Check missingness in harmonized variables.")

# Treatment model covariates
confounders = ["age_cat","income","adl_help_count","chronic_count","mobility_count","gender","racehisp","round_str"]

X = df_msm[confounders]
T = df_msm["H_cat"]

print("Treatment levels present:", sorted(T.unique().tolist()))

# Multinomial propensity model with one-hot encoding
preprocess = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), confounders)],
    remainder="drop",
)

prop_model = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=2000))
])

prop_model.fit(X, T)

ps = prop_model.predict_proba(X)
classes = prop_model.named_steps["clf"].classes_
ps_df = pd.DataFrame(ps, columns=classes, index=df_msm.index)

# Stabilized weights
p_denom = np.array([ps_df.loc[i, t] for i, t in zip(ps_df.index, T)])
p_marg = T.value_counts(normalize=True)
p_num = T.map(p_marg).astype(float).values

df_msm["sw_raw"] = p_num / p_denom
df_msm["sw"] = df_msm["sw_raw"].clip(0.01, 10)

print("\nWeight diagnostics (raw):")
print(df_msm["sw_raw"].describe(percentiles=[0.01,0.05,0.5,0.95,0.99]))
print("\nWeight diagnostics (trimmed):")
print(df_msm["sw"].describe(percentiles=[0.01,0.05,0.5,0.95,0.99]))

# Histogram of raw weights
plt.figure(figsize=(8, 5))
plt.hist(df_msm["sw_raw"], bins=60)
plt.axvline(10, linestyle="--", linewidth=2)
plt.xlabel("Stabilized IP Weight (raw)")
plt.ylabel("Frequency")
plt.title("Distribution of Stabilized IP Weights (Before Trimming)")
plt.tight_layout()
plt.show()

# Effective sample size
ess = (df_msm["sw"].sum() ** 2) / (df_msm["sw"] ** 2).sum()
print("\nEffective sample size (approx):", float(ess))

# Outcome MSM: treatment + round fixed effects
T_dum = pd.get_dummies(df_msm["H_cat"], drop_first=True)
R_dum = pd.get_dummies(df_msm["round_str"], drop_first=True)

X_out = sm.add_constant(pd.concat([T_dum, R_dum], axis=1)).astype(float)

glm = sm.GLM(
    df_msm["Y_comm"].astype(float),
    X_out,
    family=sm.families.Binomial(),
    freq_weights=df_msm["sw"].astype(float)
).fit(cov_type="cluster", cov_kwds={"groups": df_msm["spid"]})

import numpy as np

# Extract treatment coefficients
coef = glm.params

def logistic(x):
    return 1 / (1 + np.exp(-x))

# Baseline log-odds (no care, round 1)
baseline = coef["const"]

print("\nEstimated marginal P(Y_comm=1) by caregiving intensity (Round 1 baseline):")
for lvl in ["0", "1-20", "21-80", "81+"]:
    if lvl == "0":
        lp = baseline
    else:
        lp = baseline + coef[lvl]
    print(f"{lvl:>6}: {logistic(lp):.4f}")


print("\n=== Longitudinal MSM / IPW Results (cluster-robust by spid) ===")
print(glm.summary())
