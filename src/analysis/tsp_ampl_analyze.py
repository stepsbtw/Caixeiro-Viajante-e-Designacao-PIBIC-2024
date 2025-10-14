import json, math, pathlib
import pandas as pd
import numpy as np

# ------------- CONFIG -------------
JSON_PATH = "results/tsp_old/ampl/benchmark_detailed_results.json"
OUTPUT_CSV = "ampl_summary_solvers.csv"

# Status classification
TIMEOUT_STATUSES = {"timeout", "time_limit", "limit"}
FAIL_STATUSES = {"error", "failed", "infeasible", "unbounded", "binary_not_found"} | TIMEOUT_STATUSES
SUCCESS_STATUSES = None  # If not None, treat only these as success (case-insensitive)
# Use only successful runs for timing / performance metrics
USE_ONLY_SUCCESS_FOR_TIMES = True
# Treat objective 0 as invalid (drop for gap calc)
DROP_ZERO_OBJECTIVE_IN_GAP = False
# ----------------------------------

def geometric_mean(values):
    vals = [v for v in values if v > 0]
    if not vals:
        return float('nan')
    return math.exp(sum(math.log(v) for v in vals)/len(vals))

with open(JSON_PATH, "r") as f:
    raw = json.load(f)

records = []
for r in raw:
    if not isinstance(r, dict):
        continue
    inst = r.get("instance")
    solver = r.get("solver")
    t = r.get("elapsed_time")
    if solver is None or inst is None or t is None:
        continue
    status = (r.get("status") or "").strip().lower()
    obj = r.get("objective")
    if isinstance(obj, str):
        try: obj = float(obj)
        except: obj = None
    records.append({
        "instance": inst.strip(),
        "solver": solver.strip(),
        "run": r.get("run"),
        "status": status,
        "elapsed_time": float(t),
        "objective": obj
    })

df = pd.DataFrame(records)
if df.empty:
    raise SystemExit("No valid records found.")

# Classify success
if SUCCESS_STATUSES is not None:
    succ_set = {s.lower() for s in SUCCESS_STATUSES}
    df["is_success"] = df["status"].isin(succ_set)
else:
    df["is_success"] = ~df["status"].isin(FAIL_STATUSES)

# Dataframes for timing/performance metrics
if USE_ONLY_SUCCESS_FOR_TIMES:
    df_perf = df[df.is_success].copy()
else:
    df_perf = df.copy()

# Per (solver, instance) metrics (only instances with at least 1 successful run if USE_ONLY_SUCCESS_FOR_TIMES)
grp = df_perf.groupby(["solver", "instance"], as_index=False)
agg = grp.agg(
    median_time=("elapsed_time", "median"),
    mean_time=("elapsed_time", "mean"),
    n_success_runs=("elapsed_time", "count"),
    solver_best_obj=("objective", lambda x: np.nan if x.dropna().empty else np.nanmin(x))
)

# Global best objective (consider all successful runs; fallback to all if none)
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
# Avoid divide-by-zero / invalid global best
agg.loc[agg["global_best_obj"] <= 0, "gap_pct"] = np.nan
# Performance ratios & ranks (within instances present in agg)
best_median = agg.groupby("instance")["median_time"].min().rename("best_median_time")
agg = agg.merge(best_median, on="instance", how="left")
agg["perf_ratio"] = agg["median_time"] / agg["best_median_time"]
agg["rank"] = agg.groupby("instance")["median_time"].rank(method="average")

summary_rows = []
for solver, sdata in agg.groupby("solver"):
    # Instances with any run (coverage)
    solver_all = df[df.solver == solver]
    instances_covered = solver_all.instance.nunique()
    # Successful runs / instances
    successful_runs = solver_all[solver_all.is_success].shape[0]
    successful_instances = solver_all[solver_all.is_success].instance.nunique()
    total_runs = solver_all.shape[0]
    total_timeouts = solver_all[solver_all.status.isin(TIMEOUT_STATUSES)].shape[0]

    median_of_medians = sdata.median_time.median()
    mean_of_means = sdata.mean_time.mean()
    std_time_mean = sdata.mean_time.std(ddof=0)
    geom_mean_perf_ratio = geometric_mean(sdata.perf_ratio.tolist())
    avg_gap_best_pct = sdata.gap_pct.mean(skipna=True)
    # Use bracket access; sdata.rank is the DataFrame method, not the column
    avg_rank = sdata["rank"].mean()

    summary_rows.append({
        "solver": solver,
        "instances_covered": instances_covered,
        "successful_instances": successful_instances,
        "successful_runs": successful_runs,
        "success_rate_instances": successful_instances / instances_covered if instances_covered else np.nan,
        "success_rate_runs": successful_runs / total_runs if total_runs else np.nan,
        "median_of_medians": median_of_medians,
        "mean_of_means": mean_of_means,
        "std_time_mean": std_time_mean,
        "geom_mean_perf_ratio": geom_mean_perf_ratio,
        "avg_gap_best_pct": avg_gap_best_pct,
        "total_timeouts": total_timeouts,
        "total_runs": total_runs,
        "avg_rank": avg_rank
    })

summary_df = pd.DataFrame(summary_rows)

col_order = [
    "solver",
    "instances_covered",
    "successful_instances",
    "successful_runs",
    "success_rate_instances",
    "success_rate_runs",
    "median_of_medians",
    "mean_of_means",
    "std_time_mean",
    "geom_mean_perf_ratio",
    "avg_gap_best_pct",
    "total_timeouts",
    "total_runs",
    "avg_rank"
]
summary_df = summary_df[col_order].sort_values("avg_rank", ascending=True)
summary_df.to_csv(OUTPUT_CSV, index=False)
print(summary_df.to_string(index=False))
print(f"\nSaved to {OUTPUT_CSV}")