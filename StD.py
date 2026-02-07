
import os
import numpy as np
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = "/Users/nedaabdolrahimi/Documents/Fall25/Causal inference/Project/Data"  # <-- EDIT
OUT_PATH = "/Users/nedaabdolrahimi/Documents/Fall25/Causal inference/Project/Data/nhats_panel_r1_r14.csv"
ROUNDS = list(range(1, 15))


# ----------------------------
# HELPERS
# ----------------------------
def first_present(df: pd.DataFrame, cols):
    """Return the first available column (cleaned) among cols; else all-NaN series."""
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            s = s.where(s >= 0, np.nan)  # NHATS negative codes -> missing
            return s
    return pd.Series(np.nan, index=df.index)

def row_sum_present(df: pd.DataFrame, cols):
    """Row-wise sum over available cols, treating NHATS negative codes as missing."""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    tmp = df[cols].apply(pd.to_numeric, errors="coerce")
    tmp = tmp.where(tmp >= 0, np.nan)
    return tmp.sum(axis=1, min_count=1)

def build_op_agg(op_path: str, r: int) -> pd.DataFrame:
    """
    Aggregate OP to SP level for a given round r:
      - H_hours_mth = sum of op{r}dhrsmth across helpers
      - any_paid/any_govpay/any_inspay = max across helpers
    Chunked read to be memory-safe.
    """
    hr_col   = f"op{r}dhrsmth"
    paid_col = f"op{r}paidhelpr"
    gov_col  = f"op{r}govpayhlp"
    ins_col  = f"op{r}inspayhlp"

    use_cols = ["spid", "opid", hr_col, paid_col, gov_col, ins_col]
    chunksize = 200_000

    hours_parts = []
    flags_parts = []

    def _nanmax01(s):
        s = pd.to_numeric(s, errors="coerce")
        s = s.where(s >= 0, np.nan)
        return np.nanmax(s.values) if len(s) else np.nan

    for chunk in pd.read_stata(
        op_path,
        columns=[c for c in use_cols if c],
        chunksize=chunksize,
        convert_categoricals=False
    ):
        if "spid" not in chunk.columns:
            raise ValueError(f"'spid' not found in OP file: {op_path}")

        # --- hours ---
        if hr_col in chunk.columns:
            chunk[hr_col] = pd.to_numeric(chunk[hr_col], errors="coerce")
            chunk.loc[chunk[hr_col] < 0, hr_col] = np.nan  # negative codes -> missing
            h_part = chunk.groupby("spid", as_index=False)[hr_col].sum()
            h_part.rename(columns={hr_col: "H_hours_mth_part"}, inplace=True)
            hours_parts.append(h_part)

        # --- flags ---
        agg_dict = {}
        if paid_col in chunk.columns: agg_dict["any_paid_part"]   = (paid_col, _nanmax01)
        if gov_col in chunk.columns:  agg_dict["any_govpay_part"] = (gov_col, _nanmax01)
        if ins_col in chunk.columns:  agg_dict["any_inspay_part"] = (ins_col, _nanmax01)

        if agg_dict:
            f_part = chunk.groupby("spid", as_index=False).agg(**agg_dict)
            flags_parts.append(f_part)

    # combine hours (sum across chunks)
    if hours_parts:
        op_hours = (
            pd.concat(hours_parts, ignore_index=True)
              .groupby("spid", as_index=False)["H_hours_mth_part"].sum()
              .rename(columns={"H_hours_mth_part": "H_hours_mth"})
        )
    else:
        op_hours = pd.DataFrame({"spid": pd.Series(dtype=float), "H_hours_mth": pd.Series(dtype=float)})

    # combine flags (max across chunks)
    if flags_parts:
        flags_all = pd.concat(flags_parts, ignore_index=True)
        keep_cols = ["spid"]
        if "any_paid_part" in flags_all.columns: keep_cols.append("any_paid_part")
        if "any_govpay_part" in flags_all.columns: keep_cols.append("any_govpay_part")
        if "any_inspay_part" in flags_all.columns: keep_cols.append("any_inspay_part")

        flags_all = flags_all[keep_cols]
        agg_map = {}
        if "any_paid_part" in flags_all.columns:   agg_map["any_paid"]   = ("any_paid_part", "max")
        if "any_govpay_part" in flags_all.columns: agg_map["any_govpay"] = ("any_govpay_part", "max")
        if "any_inspay_part" in flags_all.columns: agg_map["any_inspay"] = ("any_inspay_part", "max")

        op_flags = flags_all.groupby("spid", as_index=False).agg(**agg_map)
    else:
        op_flags = pd.DataFrame({"spid": pd.Series(dtype=float)})

    # merge
    op_agg = op_hours.merge(op_flags, on="spid", how="outer")
    return op_agg


def build_round_panel(r: int) -> pd.DataFrame:
    """Build one round (r) SP-level dataset with treatment, outcome, eligibility, and harmonized covariates."""
    sp_path  = os.path.join(DATA_DIR, f"NHATS_Round_{r}_SP_File.dta")
    op_path  = os.path.join(DATA_DIR, f"NHATS_Round_{r}_OP_File.dta")
    trk_path = os.path.join(DATA_DIR, f"NHATS_Round_{r}_Tracker_File.dta")

    # ----------------------------
    # SP: outcome + round-specific raw covariates
    # ----------------------------
    y_col = f"r{r}dresid"

    # Round-specific candidates used to derive harmonized covariates
    age_candidates = [f"r{r}d2intvrage", f"r{r}age"]
    inc_candidates = [f"ia{r}totinc", f"ia{r}income"]

    adl_cols = [f"sc{r}bathhlp", f"sc{r}dreshlp", f"sc{r}toilhlp", f"sc{r}eathlp"]
    chronic_cols = [
        f"hc{r}disescn1", f"hc{r}disescn2", f"hc{r}disescn4",
        f"hc{r}disescn6", f"hc{r}disescn7", f"hc{r}disescn8", f"hc{r}disescn10"
    ]
    mob_cols = [f"pc{r}walk3blks", f"pc{r}walk6blks", f"pc{r}up20stair"]

    # time-invariant (often present) covariates used in your R14 analysis
    invariant = ["r13dgender", "rl13dracehisp"]

    sp = pd.read_stata(sp_path, convert_categoricals=False)

    needed = ["spid", y_col] + age_candidates + inc_candidates + adl_cols + chronic_cols + mob_cols + invariant
    needed = [c for c in needed if c in sp.columns]
    sp = sp[needed].copy()

    if y_col not in sp.columns:
        raise ValueError(f"Outcome column {y_col} not found in SP file for round {r}.")

    # Outcome: community vs not
    sp["Y_comm"] = (pd.to_numeric(sp[y_col], errors="coerce") == 1).astype(int)

    # ----------------------------
    # Tracker: eligibility restriction (completed interviews)
    # ----------------------------
    status_col = f"r{r}status"
    trk = pd.read_stata(trk_path, convert_categoricals=False)
    if "spid" not in trk.columns or status_col not in trk.columns:
        raise ValueError(f"Tracker missing spid or {status_col} for round {r}.")

    trk = trk[["spid", status_col]].copy()
    trk[status_col] = pd.to_numeric(trk[status_col], errors="coerce")
    trk = trk[trk[status_col].isin([60, 61, 62, 63, 64])].copy()

    # ----------------------------
    # OP: treatment aggregation
    # ----------------------------
    op_agg = build_op_agg(op_path, r)

    # ----------------------------
    # Merge round components
    # ----------------------------
    df_r = sp.merge(op_agg, on="spid", how="left").merge(trk, on="spid", how="inner")

    # Treatment: hours/month (fill no-OP as 0)
    df_r["H_hours_mth"] = pd.to_numeric(df_r.get("H_hours_mth", np.nan), errors="coerce").fillna(0)

    # Dose categories
    df_r["H_cat"] = pd.cut(
        df_r["H_hours_mth"],
        bins=[-0.1, 0, 20, 80, np.inf],
        labels=["0", "1-20", "21-80", "81+"]
    ).astype(str)

    # ----------------------------
    # Harmonized covariates (same names across rounds)
    # ----------------------------
    df_r["age_cat"] = first_present(df_r, age_candidates)
    df_r["income"]  = first_present(df_r, inc_candidates)

    df_r["adl_help_count"] = row_sum_present(df_r, adl_cols)
    df_r["chronic_count"]  = row_sum_present(df_r, chronic_cols)
    df_r["mobility_count"] = row_sum_present(df_r, mob_cols)

    df_r["gender"]  = first_present(df_r, ["r13dgender"])
    df_r["racehisp"] = first_present(df_r, ["rl13dracehisp"])

    # Round + status
    df_r["round"] = r
    df_r.rename(columns={status_col: "status"}, inplace=True)

    # Keep a clean minimal set for longitudinal MSM
    keep = [
        "spid", "round", "status",
        "Y_comm", "H_hours_mth", "H_cat",
        "age_cat", "income", "adl_help_count", "chronic_count", "mobility_count",
        "gender", "racehisp",
        # optional flags if present
        "any_paid", "any_govpay", "any_inspay"
    ]
    keep = [c for c in keep if c in df_r.columns]
    df_r = df_r[keep].copy()

    return df_r


# ----------------------------
# MAIN: BUILD & STACK
# ----------------------------
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

panel_list = []
for r in ROUNDS:
    print(f"Building round {r}...")
    panel_list.append(build_round_panel(r))

panel = pd.concat(panel_list, ignore_index=True)

print("Stacked panel shape:", panel.shape)
print("Columns:", panel.columns.tolist())
print(panel.head())

panel.to_csv(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH}")
