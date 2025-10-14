"""
GAP_AMPL_SOLVERS.py
Benchmarking script for GAP (Generalized Assignment Problem) using AMPL solvers.
Structure and logic adapted from TSP_AMPL_SOLVERS.py.
"""
import os
import time
import pandas as pd
import numpy as np
from multiprocessing import Process, Queue
from scipy.stats import friedmanchisquare, wilcoxon

# Configuration
MODEL_FILE_MAX = "../models/gap.mod"      # Maximization model (gap1-gap12)
MODEL_FILE_MIN = "../models/gap_min.mod"  # Minimization model (gapa-gapd)
DATA_DIR = "../dat/gap_others"  # Using instances from "others" folder
SOLVERS = [
    "gurobi", 
    "cplex", 
    "xpress",  
    "highs",
    "scip", 
    "cbc",     
]
RUNS = 1
TIME_LIMIT = int(os.environ.get("AMPL_TIME_LIMIT", 100))
MIN_INSTANCE_SIZE = int(os.environ.get("MIN_INSTANCE_SIZE", 0))   # 100x100 = 10,000
MAX_INSTANCE_SIZE = int(os.environ.get("MAX_INSTANCE_SIZE", 10000))  # 800x800 = 640,000
# Filter to only a-e class instances (minimization problems from "others" folder)
import glob
import os
GAP_ABCDE_INSTANCES = [os.path.basename(f) for f in glob.glob("../dat/gap_others/[a-e]*.dat")]
SMALL_INSTANCES = []
DISABLED_SOLVERS = set()
EXTERNAL_WALLCLOCK_ENFORCE = True

# Environment overrides
DROP_SOLVERS_ENV = os.environ.get('DROP_SOLVERS')
ONLY_SOLVERS_ENV = os.environ.get("ONLY_SOLVERS")
ONLY_INSTANCES_ENV = os.environ.get("ONLY_INSTANCES")

# Create results directory
os.makedirs("../results/gap", exist_ok=True)

def get_model_file(instance_name):
    """
    Determine which GAP model to use based on instance name:
    - All instances from "others" folder (a-e classes) use minimization model
    - gap[a-d] instances use minimization model  
    - gap[1-9] and gap1[0-2] instances use maximization model
    """
    import re
    # All instances from "others" folder are minimization problems
    if re.match(r'^[a-e]\d+\.dat$', instance_name):
        return MODEL_FILE_MIN
    elif re.match(r'^gap[a-d]_p\d+\.dat$', instance_name):
        return MODEL_FILE_MIN
    else:
        return MODEL_FILE_MAX

def get_instance_size(dat_file_path):
    """Extract the number of decision variables (workers × tasks) from a GAP .dat file"""
    try:
        with open(dat_file_path, 'r') as f:
            content = f.read()
            
        import re
        
        # Get number of workers
        workers_match = re.search(r'set WORKERS := ([^;]+);', content)
        if workers_match:
            workers_str = workers_match.group(1).strip()
            num_workers = len(workers_str.split())
        else:
            return None
            
        # Get number of tasks  
        tasks_match = re.search(r'set TASKS := ([^;]+);', content)
        if tasks_match:
            tasks_str = tasks_match.group(1).strip()
            num_tasks = len(tasks_str.split())
        else:
            return None
            
        # Return total decision variables (workers × tasks)
        return num_workers * num_tasks
        
    except Exception as e:
        print(f"Error reading {dat_file_path}: {e}")
        return None

def filter_instances_by_size(data_dir, min_size, max_size):
    """Filter .dat files to only include instances with min_size <= decision variables (workers × tasks) <= max_size"""
    all_instances = [f for f in os.listdir(data_dir) if f.endswith(".dat")]
    filtered_instances = []
    
    print(f"Filtering GAP instances by size ({min_size} <= variables <= {max_size})...")
    
    for instance in all_instances:
        instance_path = os.path.join(data_dir, instance)
        size = get_instance_size(instance_path)
        
        if size is not None:
            if min_size <= size <= max_size:
                filtered_instances.append(instance)
                print(f"  ✓ {instance} ({size} variables)")
            else:
                if size < min_size:
                    print(f"  ✗ {instance} ({size} variables) - too small")
                else:
                    print(f"  ✗ {instance} ({size} variables) - too large")
        else:
            print(f"  ? {instance} - could not determine size")
    
    print(f"Selected {len(filtered_instances)} instances out of {len(all_instances)} total")
    return filtered_instances

def _run_solver_core(instance_path, solver, limit, queue=None):
    """Core solver execution with AMPL"""
    from amplpy import AMPL
    ampl = AMPL()
    try:
        ampl.setOption("solver", solver)
        try:
            ampl.eval("option solver_msg 1;")
        except Exception:
            pass
        
        # Apply time limit
        def _apply_time_limit(ampl_obj, solver_name, time_limit):
            for generic_opt in ("timelim", "timelimit"):
                try:
                    ampl_obj.setOption(generic_opt, time_limit)
                    break
                except Exception:
                    continue
            
            solver_key = solver_name.lower()
            candidates = {
                "gurobi": [f"timelim={time_limit}", f"TimeLimit={time_limit}"],
                "cplex": [f"timelim={time_limit}", f"timelimit={time_limit}"],
                "highs": [f"timelimit={time_limit}", f"time_limit={time_limit}", f"timelim={time_limit}"],
                "xpress": [f"timelimit={time_limit}", f"TIMELIMIT={time_limit}", f"MAXTIME={time_limit}"],
                "copt": [f"timelim={time_limit}", f"TimeLimit={time_limit}"],
                "mosek": [f"timelim={time_limit}", f"timelimit={time_limit}"],
                "scip": [f"timelim={time_limit}", f"timelimit={time_limit}"],
                "cbc": [f"timelim={time_limit}", f"seconds={time_limit}", f"sec={time_limit}"],
            }.get(solver_key, [])
            if candidates:
                option_name = f"{solver_key}_options"
                for opt_string in candidates:
                    try:
                        ampl_obj.setOption(option_name, opt_string)
                        break
                    except Exception:
                        continue
        
        _apply_time_limit(ampl, solver, limit)
        
        # Choose the correct model based on instance name
        model_file = get_model_file(os.path.basename(instance_path))
        ampl.read(model_file)
        ampl.readData(instance_path)
        
        start = time.time()
        ampl.solve()
        elapsed = time.time() - start
        
        try:
            status = ampl.getValue("solve_result")
        except Exception:
            status = "UNKNOWN"
        try:
            obj = ampl.getObjective("Total_Cost").value()
        except Exception:
            obj = None
        try:
            gap = ampl.getValue("solve_result_gap")
        except Exception:
            gap = None
        
        ampl.close()
        
        if (status or '').lower() in {"failure", "failed"} and (obj is None or obj == 0):
            status = f"FAILURE({solver})"
        
        result = (elapsed, None, obj, gap, status)
        if queue is not None:
            queue.put(result)
        return result
        
    except Exception as e:
        try:
            ampl.close()
        except Exception:
            pass
        result = (None, None, None, None, f"ERROR: {e}")
        if queue is not None:
            queue.put(result)
        return result

def run_solver(instance_path, solver):
    """Run solver with external wall-clock enforcement"""
    if not EXTERNAL_WALLCLOCK_ENFORCE:
        return _run_solver_core(instance_path, solver, TIME_LIMIT)
    
    start_wall = time.time()
    q = Queue()
    p = Process(target=_run_solver_core, args=(instance_path, solver, TIME_LIMIT, q))
    p.start()
    p.join(TIME_LIMIT + 2)
    
    if p.is_alive():
        p.terminate()
        p.join()
        elapsed = time.time() - start_wall
        return (elapsed, None, None, None, "TIMEOUT")
    
    if not q.empty():
        result = q.get()
        if result[0] is not None and result[0] > TIME_LIMIT + 1:
            return (TIME_LIMIT, result[1], result[2], result[3], "TIMEOUT")
        return result
    
    elapsed = time.time() - start_wall
    return (elapsed, None, None, None, "ERROR")

def benchmark():
    results = []
    detailed_results = []
    
    # Use size-based filtering for instances in the specified range
    if SMALL_INSTANCES:
        test_instances = SMALL_INSTANCES
        print(f"Using predefined small instances: {test_instances}")
    else:
        # Filter instances by size range (100x100 to 800x800)
        test_instances = filter_instances_by_size(DATA_DIR, MIN_INSTANCE_SIZE, MAX_INSTANCE_SIZE)
        print(f"Using instances in size range {MIN_INSTANCE_SIZE}-{MAX_INSTANCE_SIZE}: {len(test_instances)} instances")
    
    # Filter to only use instances that actually exist
    instances = [inst for inst in test_instances if os.path.exists(os.path.join(DATA_DIR, inst))]
    
    # Build (instance,size) list then sort by size
    instances_with_size = []
    for inst in instances:
        inst_path = os.path.join(DATA_DIR, inst)
        sz = get_instance_size(inst_path)
        if sz is not None:
            instances_with_size.append((inst, sz))
    
    # Sort by size suffix (e.g., 05100, 10100) then by letter prefix (a, b, c, d, e)
    def sort_key(item):
        inst, sz = item
        # Extract size suffix from filename (e.g., "05100" from "a05100.dat")
        import re
        match = re.search(r'[a-e](\d+)\.dat', inst)
        if match:
            size_suffix = match.group(1)
            letter_prefix = inst[0]  # First character (a, b, c, d, e)
            return (size_suffix, letter_prefix)
        return (str(sz), inst)  # Fallback to numeric size
    
    instances_with_size.sort(key=sort_key)
    instances = [inst for inst, _ in instances_with_size]

    # Active solvers after optional env filter
    active_solvers = SOLVERS
    if ONLY_SOLVERS_ENV:
        requested = [s.strip() for s in ONLY_SOLVERS_ENV.split(',') if s.strip()]
        active_solvers = [s for s in SOLVERS if s in requested]
        print(f"Env override ONLY_SOLVERS -> limiting to: {active_solvers}")

    # Optional instance filtering via env var
    if ONLY_INSTANCES_ENV:
        req_instances = {s.strip() for s in ONLY_INSTANCES_ENV.split(',') if s.strip()}
        instances = [i for i in instances if i in req_instances]
        instances_with_size = [t for t in instances_with_size if t[0] in req_instances]
        print(f"Env override ONLY_INSTANCES -> limiting to: {sorted(req_instances)}")

    print(f"\nTesting with {len(instances)} GAP instances (sorted by size):")
    preview = instances_with_size[:10]
    for i, (inst, size) in enumerate(preview):
        print(f"  {i+1:2d}. {inst} ({size} tasks)")
    if len(instances_with_size) > len(preview):
        print(f"  ... and {len(instances_with_size) - len(preview)} more instances")

    print(f"\nSolvers: {active_solvers}")
    print(f"Runs per instance: {RUNS}")
    print(f"Time limit: {TIME_LIMIT} seconds")
    print("=" * 80)

    # Load existing results once for all instances
    existing_runs_map = {}
    try:
        import json
        detailed_json_path = "../results/gap/gap_detailed_results.json"
        if os.path.exists(detailed_json_path):
            with open(detailed_json_path, 'r') as f:
                existing_data = json.load(f)
            if isinstance(existing_data, list):
                for r in existing_data:
                    key = (r.get('instance'), r.get('solver'))
                    existing_runs_map.setdefault(key, set()).add(r.get('run'))
            print(f"Loaded existing results for {len(existing_runs_map)} (instance, solver) pairs")
    except Exception as e:
        print(f"Warning: Could not load existing results: {e}")

    dry_run = os.environ.get('DRY_RUN') == '1'
    if dry_run:
        print('\nDRY_RUN=1 set -> listing planned (instance, solver) pairs only.')
        for inst in instances:
            for solver in active_solvers:
                print(f"PLAN: {inst} :: {solver}")
        return pd.DataFrame([])

    for i, inst in enumerate(instances, 1):
        inst_path = os.path.join(DATA_DIR, inst)
        instance_size = get_instance_size(inst_path)
        print(f"\n[{i}/{len(instances)}] Instance: {inst} ({instance_size} variables)", flush=True)

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
                    "time_ext_mean": np.mean(times_ext),
                    "time_ext_median": np.median(times_ext),
                    "time_ext_std": np.std(times_ext),
                    "time_ext_min": np.min(times_ext),
                    "time_ext_max": np.max(times_ext),
                    "obj_best": np.min(objs) if objs else None,
                    "obj_mean": np.mean(objs) if objs else None,
                    "obj_std": np.std(objs) if len(objs) > 1 else None,
                    "gap_mean": np.mean(gaps) if gaps else None,
                    "status_last": statuses[-1] if statuses else None,
                    "successful_runs": len(times_ext),
                    "total_runs": RUNS
                }
                results.append(result_summary)
                detailed_results.extend(run_details)
                
                # Save results after each solver completion
                print(f"    → Saving results after {solver} completion...", flush=True)
                save_detailed_results(detailed_results)
                
                # Save summary results incrementally
                df_temp = pd.DataFrame(results)
                summary_file = "../results/gap/gap_ampl_solvers_incremental.csv"
                df_temp.to_csv(summary_file, index=False)
                print(f"    → Incremental summary saved to {summary_file}", flush=True)
                
                if len(times_ext) > 0:
                    print(f"    → Summary: {len(times_ext)}/{RUNS} successful, avg time: {np.mean(times_ext):.3f}s, obj: {np.min(objs) if objs else 'N/A'}", flush=True)
            else:
                if solver in DISABLED_SOLVERS:
                    print(f"    → Solver {solver} disabled; no runs recorded.")
                else:
                    print(f"    → No successful runs for {solver} on {inst}")

    # Save detailed results
    save_detailed_results(detailed_results)
    return pd.DataFrame(results)

def save_detailed_results(detailed_results):
    """Save detailed results to JSON and text files (incremental saves)"""
    import json
    
    # Save as JSON - load existing and append new results
    json_file = "../results/gap/gap_detailed_results.json"
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
    txt_file = "../results/gap/gap_detailed_results.txt"
    if new_results:
        with open(txt_file, 'a') as f:
            if not existing_results:
                f.write("GAP AMPL SOLVER BENCHMARK - DETAILED RESULTS\n")
                f.write("=" * 60 + "\n\n")
            
            current_instance = None
            current_solver = None
            
            for result in new_results:
                if result['instance'] != current_instance:
                    current_instance = result['instance']
                    f.write(f"\nInstance: {current_instance} ({result.get('instance_size', '?')} tasks)\n")
                    f.write("-" * 50 + "\n")
                
                if result['solver'] != current_solver or result['run'] == 1:
                    current_solver = result['solver']
                    f.write(f"\nSolver: {current_solver}\n")
                
                f.write(f"  Run {result['run']}: ")
                f.write(f"Status={result['status']}, ")
                f.write(f"Obj={result['objective']}, ")
                f.write(f"Time={result['elapsed_time']:.3f}s")
                f.write(f" [{result['timestamp']}]\n")
        
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
                x, y = pivot[s1], pivot[s2]
                common = x.index.intersection(y.index)
                if len(common) > 0:
                    try:
                        stat, p = wilcoxon(x.loc[common], y.loc[common])
                        print(f"Wilcoxon {s1} vs {s2}: stat={stat}, p={p:.3e}")
                    except Exception as e:
                        print(f"Could not perform Wilcoxon test for {s1} vs {s2}: {e}")

def main():
    results = benchmark()
    
    # Save summary results
    summary_file = "../results/gap/gap_ampl_solvers.csv"
    results.to_csv(summary_file, index=False)
    print(f"\n✓ Summary results saved to {summary_file}")
    
    # Print summary statistics
    if not results.empty:
        print(f"\nSummary Statistics:")
        print(f"Instances tested: {results['instance'].nunique()}")
        print(f"Solvers tested: {results['solver'].nunique()}")
        print(f"Total successful runs: {results['successful_runs'].sum()}")
        
        # Show average times by solver
        print(f"\nAverage times by solver:")
        solver_avg = results.groupby('solver')['time_ext_mean'].mean().sort_values()
        for solver, avg_time in solver_avg.items():
            print(f"  {solver}: {avg_time:.3f}s")
        
        # Show average objective values by instance
        print(f"\nBest objective values by instance:")
        instance_obj = results.groupby('instance')['obj_best'].mean().sort_values()
        for instance, best_obj in instance_obj.head(10).items():
            if pd.notna(best_obj):
                print(f"  {instance}: {best_obj:.1f}")
        
        # Perform statistical comparison
        try:
            stats_comparison(results)
        except Exception as e:
            print(f"Could not perform statistical comparison: {e}")
    else:
        print("No results obtained!")

if __name__ == "__main__":
    print("Starting GAP AMPL Solver Benchmark...")
    print(f"Solvers: {SOLVERS}")
    print(f"Time limit: {TIME_LIMIT} seconds")
    print(f"Runs per instance: {RUNS}")
    print(f"Instance size range: {MIN_INSTANCE_SIZE:,} to {MAX_INSTANCE_SIZE:,} variables (≈100×100 to 800×800)")
    print("=" * 60)

    # Purge existing detailed JSON entries for specified solvers
    if DROP_SOLVERS_ENV:
        to_drop = {s.strip() for s in DROP_SOLVERS_ENV.split(',') if s.strip()}
        json_file = "../results/gap/gap_detailed_results.json"
        if os.path.exists(json_file):
            try:
                import json
                with open(json_file, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    before = len(data)
                    data = [r for r in data if r.get('solver') not in to_drop]
                    after = len(data)
                    if after != before:
                        with open(json_file, 'w') as f:
                            json.dump(data, f, indent=2)
                        print(f"Purged {before-after} existing records for solvers: {', '.join(sorted(to_drop))}")
            except Exception as e:
                print(f"Warning: could not purge records: {e}")
    
    main()
