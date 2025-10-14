#!/usr/bin/env python3
"""
AMPL Solvers Benchmark for Assignment Problem

This script benchmarks various AMPL solvers on assignment problem instances.
Based on the TSP benchmark but adapted for assignment problems.
"""

import os
import sys
import time
import json
import re
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import numpy as np

# Configuration
MODEL_FILE = "INSTANCES/models/assignment_old.mod"
DATA_DIR = "dat/assign"
SOLVERS = [
    "gurobi", 
    "cplex", 
    "highs",
    "xpress",   
    "copt",
    "mosek", 
    "scip", 
    "cbc",     
]
RUNS = 3
TIME_LIMIT = int(os.environ.get("AMPL_TIME_LIMIT", 300))
MAX_INSTANCE_SIZE = int(os.environ.get("MAX_INSTANCE_SIZE", 800))
SMALL_INSTANCES = []
DISABLED_SOLVERS = set()
DROP_SOLVERS_ENV = os.environ.get('DROP_SOLVERS')
ONLY_SOLVERS_ENV = os.environ.get("ONLY_SOLVERS")
ONLY_INSTANCES_ENV = os.environ.get("ONLY_INSTANCES")

def get_instance_size(dat_file_path):
    try:
        with open(dat_file_path, 'r') as f:
            content = f.read()
        match = re.search(r'set WORKERS := ([^;]+);', content)
        if match:
            workers_str = match.group(1).strip()
            workers = workers_str.split()
            return len(workers)
        return None
    except Exception as e:
        print(f"Error reading {dat_file_path}: {e}")
        return None

def filter_instances_by_size(data_dir, max_size):
    all_instances = [f for f in os.listdir(data_dir) if f.endswith(".dat")]
    filtered_instances = []
    print(f"Filtering instances by size (max {max_size} workers/tasks)...")
    for instance in all_instances:
        instance_path = os.path.join(data_dir, instance)
        size = get_instance_size(instance_path)
        if size is not None:
            if size <= max_size:
                filtered_instances.append(instance)
                print(f"  ✓ {instance} ({size} workers/tasks)")
            else:
                print(f"  ✗ {instance} ({size} workers/tasks) - too large")
        else:
            print(f"  ? {instance} - could not determine size")
    print(f"Selected {len(filtered_instances)} instances out of {len(all_instances)} total")
    return filtered_instances

def run_solver(instance_path, solver):
    from amplpy import AMPL
    ampl = AMPL()
    try:
        ampl.setOption("solver", solver)
        # Try common time limit options
        for opt in ("timelim", "timelimit"):
            try:
                ampl.setOption(opt, TIME_LIMIT)
                break
            except Exception:
                continue
        ampl.read(MODEL_FILE)
        ampl.readData(instance_path)
        start = time.time()
        ampl.solve()
        elapsed = time.time() - start
        status = ampl.getValue("solve_result")
        try:
            obj = ampl.getObjective("Total_Cost").value()
        except Exception:
            obj = None
        try:
            gap = ampl.getValue("solve_result_gap")
        except Exception:
            gap = None
        ampl.close()
        return elapsed, None, obj, gap, status
    except Exception as e:
        print(f"Error with {solver} on {instance_path}: {str(e)}")
        try:
            ampl.close()
        except Exception:
            pass
        return None, None, None, None, "ERROR"

def benchmark():
    results = []
    detailed_results = []
    # Use small instances for testing, or filter by size if SMALL_INSTANCES is empty
    if SMALL_INSTANCES:
        test_instances = SMALL_INSTANCES
        print(f"Using predefined small instances: {test_instances}")
    else:
        test_instances = filter_instances_by_size(DATA_DIR, MAX_INSTANCE_SIZE)
    instances = [inst for inst in test_instances if os.path.exists(os.path.join(DATA_DIR, inst))]
    instances_with_size = []
    for inst in instances:
        inst_path = os.path.join(DATA_DIR, inst)
        sz = get_instance_size(inst_path)
        if sz is not None:
            instances_with_size.append((inst, sz))
    instances_with_size.sort(key=lambda x: x[1])
    instances = [inst for inst, _ in instances_with_size]
    active_solvers = SOLVERS
    if ONLY_SOLVERS_ENV:
        requested = [s.strip() for s in ONLY_SOLVERS_ENV.split(',') if s.strip()]
        active_solvers = [s for s in SOLVERS if s in requested]
        print(f"Env override ONLY_SOLVERS -> limiting to: {active_solvers}")
    if ONLY_INSTANCES_ENV:
        req_instances = {s.strip() for s in ONLY_INSTANCES_ENV.split(',') if s.strip()}
        instances = [i for i in instances if i in req_instances]
        instances_with_size = [t for t in instances_with_size if t[0] in req_instances]
        print(f"Env override ONLY_INSTANCES -> limiting to: {sorted(req_instances)}")
    print(f"\nTesting with {len(instances)} instances (sorted by size):")
    preview = instances_with_size[:10]
    for i, (inst, size) in enumerate(preview):
        print(f"  {i+1:2d}. {inst} ({size} workers/tasks)")
    if len(instances_with_size) > len(preview):
        print(f"  ... and {len(instances_with_size) - len(preview)} more instances")
    print(f"\nSolvers: {active_solvers}")
    print(f"Runs per instance: {RUNS}")
    print(f"Time limit: {TIME_LIMIT} seconds")
    print("=" * 80)
    dry_run = os.environ.get('DRY_RUN') == '1'
    if dry_run:
        print('\nDRY_RUN=1 set -> listing planned (instance, solver) pairs only, no solves executed.')
        for inst in instances:
            for solver in active_solvers:
                print(f"PLAN: {inst} :: {solver}")
        return pd.DataFrame([])
    for i, inst in enumerate(instances, 1):
        inst_path = os.path.join(DATA_DIR, inst)
        instance_size = get_instance_size(inst_path)
        print(f"\n[{i}/{len(instances)}] Instance: {inst} ({instance_size} workers/tasks)", flush=True)
        existing_runs_map = {}
        try:
            import json
            detailed_json_path = "results/assignment_detailed_results.json"
            if os.path.exists(detailed_json_path):
                with open(detailed_json_path, 'r') as f:
                    existing_data = json.load(f)
                if isinstance(existing_data, list):
                    for r in existing_data:
                        if r.get('instance') == inst:
                            key = (r.get('instance'), r.get('solver'))
                            existing_runs_map.setdefault(key, set()).add(r.get('run'))
        except Exception:
            pass
        for solver in active_solvers:
            if solver in DISABLED_SOLVERS:
                print(f"  Skipping {solver}: previously disabled (license/driver failure).")
                continue
            key = (inst, solver)
            have_runs = existing_runs_map.get(key, set())
            if len(have_runs) >= RUNS:
                print(f"  Skipping {solver}: already have {len(have_runs)}/{RUNS} runs recorded.")
                continue
            print(f"  Testing solver: {solver} (have {len(have_runs)} / need {RUNS})", flush=True)
            times_ext, objs, gaps, statuses = [], [], [], []
            run_details = []
            for r in range(RUNS):
                run_number = r + 1
                if run_number in have_runs:
                    continue
                print(f"    Run {run_number}/{RUNS}...", end=" ", flush=True)
                result = run_solver(inst_path, solver)
                if result[0] is None:
                    print("ERROR")
                    if r == 0:
                        DISABLED_SOLVERS.add(solver)
                        print(f"    → Disabling solver {solver} for remaining instances (initial error).")
                    break
                t_ext, _t_solver_unused, o, g, s = result
                if r == 0 and (s.startswith('FAILURE(') or (s.lower() in {'failure','failed'} and (o is None or o == 0) and t_ext < 1.0)):
                    print(f"FAIL ({s})")
                    DISABLED_SOLVERS.add(solver)
                    print(f"    → Disabling solver {solver} (likely unavailable / license). Skipping remaining runs.")
                    break
                run_detail = {
                    'instance': inst,
                    'instance_size': instance_size,
                    'solver': solver,
                    'run': r + 1,
                    'status': s,
                    'objective': o,
                    'elapsed_time': t_ext,
                    'gap': g,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                run_details.append(run_detail)
                print(f"{t_ext:.3f}s", flush=True)
                if t_ext is not None:
                    times_ext.append(t_ext)
                if o is not None:
                    objs.append(o)
                if g is not None:
                    gaps.append(g)
                statuses.append(s)
            if times_ext:
                result_summary = {
                    "instance": inst,
                    "instance_size": instance_size,
                    "solver": solver,
                    "time_ext_mean": float(np.mean(times_ext)),
                    "time_ext_median": float(np.median(times_ext)),
                    "time_ext_std": float(np.std(times_ext)),
                    "time_ext_min": float(np.min(times_ext)),
                    "time_ext_max": float(np.max(times_ext)),
                    "obj_best": float(np.min(objs)) if objs else None,
                    "obj_mean": float(np.mean(objs)) if objs else None,
                    "obj_std": float(np.std(objs)) if len(objs) > 1 else None,
                    "gap_mean": float(np.mean(gaps)) if gaps else None,
                    "status_last": statuses[-1] if statuses else None,
                    "successful_runs": len(times_ext),
                    "total_runs": RUNS
                }
                results.append(result_summary)
                detailed_results.extend(run_details)
                print(f"    → Saving results after {solver} completion...", flush=True)
                save_detailed_results(detailed_results)
                df_temp = pd.DataFrame(results)
                summary_file = "results/assignment_ampl_solvers_incremental.csv"
                df_temp.to_csv(summary_file, index=False)
                print(f"    → Incremental summary saved to {summary_file}", flush=True)
                if len(times_ext) > 0:
                    print(f"    → Summary: {len(times_ext)}/{RUNS} successful, avg time: {np.mean(times_ext):.3f}s, obj: {np.min(objs) if objs else 'N/A'}", flush=True)
            else:
                if solver in DISABLED_SOLVERS:
                    print(f"    → Solver {solver} disabled; no runs recorded.")
                else:
                    print(f"    → No successful runs for {solver} on {inst}")
    save_detailed_results(detailed_results)
    return pd.DataFrame(results)

def save_detailed_results(detailed_results):
    """Save detailed results to JSON and text files (incremental saves)"""
    
    # Save as JSON - load existing and append new results
    json_file = "results/assignment_detailed_results.json"
    os.makedirs("results", exist_ok=True)
    
    try:
        with open(json_file, 'r') as f:
            existing_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_results = []
    
    # Only add new results that aren't already saved
    existing_keys = set()
    for result in existing_results:
        key = (result['instance'], result['solver'], result['run'])
        existing_keys.add(key)
    
    new_results = []
    for result in detailed_results:
        key = (result['instance'], result['solver'], result['run'])
        if key not in existing_keys:
            new_results.append(result)
    
    if new_results:
        all_results = existing_results + new_results
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"✓ Added {len(new_results)} new results to {json_file}")
    
    # Save as readable text - append mode
    txt_file = "results/assignment_detailed_results.txt"
    if new_results:
        with open(txt_file, 'a') as f:
            if not existing_results:
                f.write("Assignment Problem Solver Benchmark - Detailed Results\n")
                f.write("=" * 60 + "\n\n")
            
            current_instance = None
            current_solver = None
            
            for result in new_results:
                if result['instance'] != current_instance:
                    f.write(f"\nInstance: {result['instance']} ({result['instance_size']} workers/tasks)\n")
                    f.write("-" * 50 + "\n")
                    current_instance = result['instance']
                    current_solver = None
                
                if result['solver'] != current_solver:
                    f.write(f"  Solver: {result['solver']}\n")
                    current_solver = result['solver']
                
                f.write(f"    Run {result['run']}: {result['elapsed_time']:.3f}s, "
                       f"obj={result['objective']}, status={result['status']}\n")
        
        print(f"✓ Appended {len(new_results)} new results to {txt_file}")

def stats_comparison(df):
    """Perform statistical comparison between solvers"""
    pivot = df.pivot(index="instance", columns="solver", values="time_ext_median").dropna()

    # Global comparison - Friedman test
    if len(pivot.columns) > 2:
        try:
            stat, p = friedmanchisquare(*[pivot[s] for s in pivot.columns])
            print(f"\nFriedman test (global): stat={stat:.3f}, p={p:.3e}")
        except Exception as e:
            print(f"Could not perform Friedman test: {e}")

    # Pairwise comparison - Wilcoxon test
    for i, s1 in enumerate(SOLVERS):
        for s2 in SOLVERS[i+1:]:
            if s1 in pivot.columns and s2 in pivot.columns:
                try:
                    stat, p = wilcoxon(pivot[s1], pivot[s2])
                    print(f"Wilcoxon {s1} vs {s2}: stat={stat:.3f}, p={p:.3e}")
                except Exception as e:
                    print(f"Could not compare {s1} vs {s2}: {e}")

if __name__ == "__main__":
    print("Starting Assignment Problem AMPL Solver Benchmark...")
    print(f"Solvers: {SOLVERS}")
    print(f"Time limit: {TIME_LIMIT} seconds")
    print(f"Runs per instance: {RUNS}")
    print("=" * 60)
    
    # Check if model file exists
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model file {MODEL_FILE} not found!")
        sys.exit(1)
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} not found!")
        print("Please run assignment_to_dat_converter.py first to generate .dat files")
        sys.exit(1)
    
    df = benchmark()
    
    # Save summary results
    os.makedirs("results", exist_ok=True)
    summary_file = "results/assignment_ampl_solvers.csv"
    df.to_csv(summary_file, index=False)
    print(f"\n✓ Summary results saved to {summary_file}")
    
    # Print summary statistics
    if not df.empty:
        print(f"\nSummary Statistics:")
        print(f"Instances tested: {df['instance'].nunique()}")
        print(f"Solvers tested: {df['solver'].nunique()}")
        print(f"Total successful runs: {df['successful_runs'].sum()}")
        
        # Show average times by solver
        print(f"\nAverage times by solver:")
        solver_avg = df.groupby('solver')['time_ext_mean'].mean().sort_values()
        for solver, avg_time in solver_avg.items():
            print(f"  {solver}: {avg_time:.3f}s")
        
        # Show average objective values by instance
        print(f"\nAverage objective values by instance:")
        instance_obj = df.groupby('instance')['obj_mean'].mean().sort_values()
        for instance, avg_obj in instance_obj.items():
            if pd.notna(avg_obj):
                print(f"  {instance}: {avg_obj:.1f}")
        
        # Perform statistical comparison
        try:
            stats_comparison(df)
        except Exception as e:
            print(f"Could not perform statistical comparison: {e}")
    else:
        print("No results obtained!")
