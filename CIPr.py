import os
import pandas as pd
import numpy as np

DATA_DIR = "/Users/nedaabdolrahimi/Documents/Fall25/Causal inference/Project/NHATS_R14_Beta_Release_STATA"  

SP_PATH  = os.path.join(DATA_DIR, "NHATS_Round_14B_SP_File.dta")
OP_PATH  = os.path.join(DATA_DIR, "NHATS_Round_14B_OP_File.dta")
TRK_PATH = os.path.join(DATA_DIR, "NHATS_Round_14B_Tracker_File.dta")


# ----------------------------
# 1) Load SP (core outcomes + covariates)
# ----------------------------
sp = pd.read_stata(SP_PATH, convert_categoricals=False)

# Keep what you need (edit after you inspect available vars)
covars = [
    "r14d2intvrage", "r13dgender", "rl13dracehisp", "ia14totinc",
    "hc14disescn1", "hc14disescn2", "hc14disescn4", "hc14disescn6",
    "hc14disescn7", "hc14disescn8", "hc14disescn10",
    "sc14bathhlp", "sc14dreshlp", "sc14toilhlp", "sc14eathlp",
    "pc14walk3blks", "pc14walk6blks", "pc14up20stair",
]

sp_keep = ["spid", "r14dresid"] + covars
sp = sp[sp_keep].copy()


# ----------------------------
# 2) Load Tracker (eligibility + weights)
# ----------------------------
trk_cols = [
    "spid", "r14status", "r14spstat1", "r14spstat2",
    "w14varstrat", "w14varunit", "w14trfinwgt0"
]
trk = pd.read_stata(TRK_PATH, columns=trk_cols, convert_categoricals=False)

# Example: keep completed interviews (tune based on NHATS documentation)
trk = trk[trk["r14status"].isin([60, 61, 62, 63, 64])].copy()


# -------------------------------------------------------
# 3) Build treatment H_it from OP (chunked aggregation)
# -------------------------------------------------------
op_use = ["spid", "opid", "op14dhrsmth", "op14paidhelpr", "op14govpayhlp", "op14inspayhlp"]
chunksize = 200_000  # adjust if memory is tight

# We'll aggregate the *core* causal exposure correctly: total hours/month.
# We'll also compute "any paid" etc. as max across helpers (works with chunks).
agg_hours = []
agg_flags = []

for chunk in pd.read_stata(OP_PATH, columns=op_use, chunksize=chunksize, convert_categoricals=False):
    # hours
    chunk["op14dhrsmth"] = pd.to_numeric(chunk["op14dhrsmth"], errors="coerce")
    chunk.loc[chunk["op14dhrsmth"] < 0, "op14dhrsmth"] = np.nan


    # sum hours per SP in this chunk
    hours_part = chunk.groupby("spid", as_index=False)["op14dhrsmth"].sum()
    hours_part.rename(columns={"op14dhrsmth": "H_hours_mth_part"}, inplace=True)
    agg_hours.append(hours_part)

    # flags per SP in this chunk (max)
    def _nanmax01(s):
        s = pd.to_numeric(s, errors="coerce")
        return np.nanmax(s.values) if len(s) else np.nan

    flags_part = chunk.groupby("spid", as_index=False).agg(
        any_paid_part=("op14paidhelpr", _nanmax01),
        any_govpay_part=("op14govpayhlp", _nanmax01),
        any_inspay_part=("op14inspayhlp", _nanmax01),
    )
    agg_flags.append(flags_part)

# combine hours (sum across chunks)
op_hours = pd.concat(agg_hours, ignore_index=True).groupby("spid", as_index=False)["H_hours_mth_part"].sum()
op_hours.rename(columns={"H_hours_mth_part": "H_hours_mth"}, inplace=True)

# combine flags (max across chunks)
op_flags = pd.concat(agg_flags, ignore_index=True).groupby("spid", as_index=False).agg(
    any_paid=("any_paid_part", "max"),
    any_govpay=("any_govpay_part", "max"),
    any_inspay=("any_inspay_part", "max"),
)

op_agg = op_hours.merge(op_flags, on="spid", how="outer")


# ----------------------------
# 4) Merge SP + OP + Tracker
# ----------------------------
df = sp.merge(op_agg, on="spid", how="left").merge(trk, on="spid", how="inner")

# no helper rows => 0 caregiving hours
df["H_hours_mth"] = df["H_hours_mth"].fillna(0)

# Optional categorical dose (helps positivity + easier plots)
df["H_cat"] = pd.cut(
    df["H_hours_mth"],
    bins=[-0.1, 0, 20, 80, np.inf],
    labels=["0", "1-20", "21-80", "81+"]
)

print("Merged dataset shape:", df.shape)
print(df[["spid", "r14dresid", "H_hours_mth", "H_cat", "r14status"]].head())


# ----------------------------
# 5) Next: outcome coding
# ----------------------------
# r14dresid is categorical with value labels.
# You must map community vs facility correctly.
# Tip: inspect distribution first:
print(df["r14dresid"].value_counts(dropna=False).head(20))
df["Y_comm"] = (df["r14dresid"] == 1).astype(int)

# --- Treatment cleanup ---
df["H_hours_mth"] = pd.to_numeric(df["H_hours_mth"], errors="coerce")
df.loc[df["H_hours_mth"] < 0, "H_hours_mth"] = np.nan
df["H_hours_mth"] = df["H_hours_mth"].fillna(0)

# --- Dose categories ---
df["H_cat"] = pd.cut(
    df["H_hours_mth"],
    bins=[-0.1, 0, 20, 80, np.inf],
    labels=["0", "1-20", "21-80", "81+"]
)


print(df["H_hours_mth"].describe())
print(df["H_cat"].value_counts())
print(df.groupby("H_cat")["Y_comm"].mean())

# Save clean analytic dataset for causal modeling
df.to_csv("/Users/nedaabdolrahimi/Documents/Fall25/Causal inference/Project/NHATS_R14_Beta_Release_STATA/nhats_r14_analytic.csv", index=False)
print("Saved analytic dataset:", df.shape)



# After you identify which codes correspond to community vs facility, create Y:
# Example (PLACEHOLDER):
# community_codes = [1]  # <-- replace after you inspect labels
# df["Y_comm"] = df["r14dresid"].isin(community_codes).astype(int)
