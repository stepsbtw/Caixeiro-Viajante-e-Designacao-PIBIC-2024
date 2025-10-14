#!/usr/bin/env python3
import json, math, pandas as pd, numpy as np, pathlib, sys

# ------------- CONFIG -------------
JSON_PATH = "results/assign/assignment_detailed_results.json"
OUTPUT_CSV = "assignment_summary_solvers.csv"

# Status-based timeout / failure classification
TIMEOUT_STATUSES = {"timeout","time_limit","limit"}
FAIL_STATUSES = {"error","failed","infeasible","unbounded","binary_not_found"} | TIMEOUT_STATUSES
SUCCESS_STATUSES = None          # e.g. {"optimal","solved"}

# Hard runtime cutoff (seconds) after which a run counts as timeout even if status=solved
HARD_TIMEOUT_SECONDS = 100.0

# If True, only successful (non-timeout, non-fail, below threshold) runs used for timing/performance
USE_ONLY_SUCCESS_FOR_TIMES = True

# If True, treat objective==0 as invalid (drop for gap calculation)
DROP_ZERO_OBJECTIVE_IN_GAP = False

# If True, when a run exceeds HARD_TIMEOUT_SECONDS we cap (censor) its time at HARD_TIMEOUT_SECONDS
# (Useful if you decide to INCLUDE those runs in stats; currently we exclude them when USE_ONLY_SUCCESS_FOR_TIMES=True)
CENSOR_TIMEOUT_TIMES = True
# ----------------------------------

def geometric_mean(values):
    vals = [v for v in values if v > 0 and np.isfinite(v)]
    if not vals:
        return float('nan')
    return math.exp(sum(math.log(v) for v in vals)/len(vals))

# Load JSON
with open(JSON_PATH, "r") as f:
    raw = json.load(f)

records = []
for r in raw:
    if not isinstance(r, dict):
        continue
    inst = r.get("instance")
    solver = r.get("solver")
    t = r.get("elapsed_time")
    if inst is None or solver is None or t is None:
        continue
    try:
        elapsed = float(t)
    except:
        continue
    status = (r.get("status") or "").strip().lower()
    obj = r.get("objective")
    if isinstance(obj, str):
        try:
            obj = float(obj)
        except:
            obj = None
    records.append({
        "instance": inst.strip(),
        "instance_size": r.get("instance_size"),
        "solver": solver.strip(),
        "run": r.get("run"),
        "status": status,
        "elapsed_time": elapsed,
        "objective": obj
    })

df = pd.DataFrame(records)
if df.empty:
    sys.exit("No valid records found.")

# Hard timeout flag
df["is_hard_timeout"] = df["elapsed_time"] >= HARD_TIMEOUT_SECONDS

# Initial success classification
if SUCCESS_STATUSES is not None:
    succ_set = {s.lower() for s in SUCCESS_STATUSES}
    df["is_success"] = df["status"].isin(succ_set)
else:
    df["is_success"] = ~df["status"].isin(FAIL_STATUSES)

# Enforce hard timeout as failure
df.loc[df["is_hard_timeout"], "is_success"] = False

# Optionally censor timeout elapsed times (useful if later included)
if CENSOR_TIMEOUT_TIMES:
    df.loc[df["is_hard_timeout"], "elapsed_time"] = HARD_TIMEOUT_SECONDS

# DataFrame used for timing/performance aggregates
df_perf = df[df.is_success].copy() if USE_ONLY_SUCCESS_FOR_TIMES else df.copy()
if df_perf.empty:
    sys.exit("No successful runs after applying HARD_TIMEOUT_SECONDS and filters.")

# Aggregate per (solver, instance)
grp = df_perf.groupby(["solver","instance"], as_index=False)
agg = grp.agg(
    median_time=("elapsed_time","median"),
    mean_time=("elapsed_time","mean"),
    n_success_runs=("elapsed_time","count"),
    solver_best_obj=("objective", lambda x: np.nan if x.dropna().empty else np.nanmin(x))
)

# Objective set for gap computation (successful runs preferred)
df_for_obj = df[df.is_success].copy()
if df_for_obj.empty:
    df_for_obj = df.copy()
if DROP_ZERO_OBJECTIVE_IN_GAP:
    df_for_obj = df_for_obj[(df_for_obj.objective.notna()) & (df_for_obj.objective != 0)]
else:
    df_for_obj = df_for_obj[df_for_obj.objective.notna()]

global_best = df_for_obj.groupby("instance")["objective"].min().rename("global_best_obj")
agg = agg.merge(global_best, on="instance", how="left")
agg["gap_pct"] = 100.0 * (agg["solver_best_obj"] - agg["global_best_obj"]) / agg["global_best_obj"]
agg.loc[agg["global_best_obj"] <= 0, "gap_pct"] = np.nan

# Performance ratios & ranks
best_median = agg.groupby("instance")["median_time"].min().rename("best_median_time")
agg = agg.merge(best_median, on="instance", how="left")
agg["perf_ratio"] = agg["median_time"] / agg["best_median_time"]
agg["rank"] = agg.groupby("instance")["median_time"].rank(method="average")

summary_rows = []
for solver, sdata in agg.groupby("solver"):
    solver_all = df[df.solver == solver]
    instances_covered = solver_all.instance.nunique()
    successful_runs = solver_all[solver_all.is_success].shape[0]
    successful_instances = solver_all[solver_all.is_success].instance.nunique()
    total_runs = solver_all.shape[0]
    # Count timeouts: status-based OR hard threshold
    total_timeouts = solver_all[
        solver_all.status.isin(TIMEOUT_STATUSES) | (solver_all.is_hard_timeout)
    ].shape[0]

    summary_rows.append({
        "solver": solver,
        "instances_covered": instances_covered,
        "successful_instances": successful_instances,
        "successful_runs": successful_runs,
        "success_rate_instances": successful_instances / instances_covered if instances_covered else np.nan,
        "success_rate_runs": successful_runs / total_runs if total_runs else np.nan,
        "median_of_medians": sdata.median_time.median(),
        "mean_of_means": sdata.mean_time.mean(),
        "std_time_mean": sdata.mean_time.std(ddof=0),
        "geom_mean_perf_ratio": geometric_mean(sdata.perf_ratio.tolist()),
        "avg_gap_best_pct": sdata.gap_pct.mean(skipna=True),
        "total_timeouts": total_timeouts,
        "total_runs": total_runs,
        "avg_rank": sdata["rank"].mean()
    })

summary_df = pd.DataFrame(summary_rows)
col_order = [
    "solver","instances_covered","successful_instances","successful_runs",
    "success_rate_instances","success_rate_runs","median_of_medians","mean_of_means",
    "std_time_mean","geom_mean_perf_ratio","avg_gap_best_pct","total_timeouts",
    "total_runs","avg_rank"
]
summary_df = summary_df[col_order].sort_values("avg_rank")
summary_df.to_csv(OUTPUT_CSV, index=False)

print(summary_df.to_string(index=False))
print(f"Saved to {OUTPUT_CSV} (source {JSON_PATH})  (Hard timeout >= {HARD_TIMEOUT_SECONDS}s)")