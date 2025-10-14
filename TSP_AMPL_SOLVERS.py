import os
import time
import pandas as pd
import numpy as np
from multiprocessing import Process, Queue
from scipy.stats import friedmanchisquare, wilcoxon

# Configurações - using amplpy with built-in solvers
MODEL_FILE = "models/tsp.mod"  # seu modelo AMPL
DATA_DIR = "dat"               # pasta com .dat para cada instância
SOLVERS = [
            "gurobi", 
            "cplex", 
            "highs",
            "xpress",   
            "copt",
            "mosek", 
            "scip", 
            "cbc",     
            #"cuopt",
            
        ]  # Available solvers from test
RUNS = 3                       # repetições por instância
TIME_LIMIT = 100               # hard wall-clock limit in seconds

EXTERNAL_WALLCLOCK_ENFORCE = True  # Hard kill if solver exceeds wall clock limit
# Filter instances by size - only instances with <= 100 nodes
MAX_INSTANCE_SIZE = 100  # default (can override via env MAX_INSTANCE_SIZE)

# Small instances for quick testing (uncomment to use only these)
# SMALL_INSTANCES = [
#     "burma14.dat", "ulysses16.dat", "gr17.dat", "gr21.dat", "gr24.dat"
# ]
SMALL_INSTANCES = []  # Empty means use all instances <= MAX_INSTANCE_SIZE

# Track solvers that are detected as unusable (e.g., missing license) to skip quickly
DISABLED_SOLVERS = set()

# Apply environment override for max instance size if provided
try:
    _env_max = os.environ.get('MAX_INSTANCE_SIZE')
    if _env_max:
        MAX_INSTANCE_SIZE = int(_env_max)
except Exception:
    pass

# Optional purge list of solvers before run (DROP_SOLVERS=solver1,solver2)
DROP_SOLVERS_ENV = os.environ.get('DROP_SOLVERS')

# Optional environment overrides for quick tests
ONLY_SOLVERS_ENV = os.environ.get("ONLY_SOLVERS")  # comma separated
ONLY_INSTANCES_ENV = os.environ.get("ONLY_INSTANCES")  # comma separated (exact .dat names)

def get_instance_size(dat_file_path):
    """Extract the number of nodes from a .dat file"""
    try:
        with open(dat_file_path, 'r') as f:
            content = f.read()
            
        # Look for the set NODES line
        import re
        match = re.search(r'set NODES := ([^;]+);', content)
        if match:
            nodes_str = match.group(1).strip()
            # Count the numbers in the nodes declaration
            nodes = nodes_str.split()
            return len(nodes)
        
        return None
    except Exception as e:
        print(f"Error reading {dat_file_path}: {e}")
        return None

def filter_instances_by_size(data_dir, max_size):
    """Filter .dat files to only include instances with <= max_size nodes"""
    all_instances = [f for f in os.listdir(data_dir) if f.endswith(".dat")]
    filtered_instances = []
    
    print(f"Filtering instances by size (max {max_size} nodes)...")
    
    for instance in all_instances:
        instance_path = os.path.join(data_dir, instance)
        size = get_instance_size(instance_path)
        
        if size is not None:
            if size <= max_size:
                filtered_instances.append(instance)
                print(f"  ✓ {instance} ({size} nodes)")
            else:
                print(f"  ✗ {instance} ({size} nodes) - too large")
        else:
            print(f"  ? {instance} - could not determine size")
    
    print(f"Selected {len(filtered_instances)} instances out of {len(all_instances)} total")
    return filtered_instances

def _run_solver_core(instance_path, solver, limit, queue=None):
    """Core solver execution (no external kill). Applies internal solver timelimits."""
    from amplpy import AMPL
    ampl = AMPL()
    try:
        ampl.setOption("solver", solver)
        # Ask AMPL to print solver messages (helps diagnosing failures)
        try:
            ampl.eval("option solver_msg 1;")
        except Exception:
            pass
        # Apply internal time limit attempts
        def _apply_time_limit(ampl_obj, solver_name, time_limit):
            # Generic attempt
            for generic_opt in ("timelim", "timelimit"):
                try:
                    ampl_obj.setOption(generic_opt, time_limit)
                    break
                except Exception:
                    continue
            # Solver specific synonyms
            solver_key = solver_name.lower()
            candidates = {
                "gurobi": [f"timelim={time_limit}", f"TimeLimit={time_limit}"],
                "cplex": [f"timelim={time_limit}", f"timelimit={time_limit}"],
                "highs": [f"timelimit={time_limit}", f"time_limit={time_limit}", f"timelim={time_limit}"],
                # Xpress: avoid lower-case maxtime (driver rejected); try common synonyms
                # Order tries more standard driver options first
                "xpress": [
                    f"timelimit={time_limit}",
                    f"TIMELIMIT={time_limit}",
                    f"MAXTIME={time_limit}",
                    f"TMAX={time_limit}",
                    f"timelim={time_limit}",  # fallback generic
                ],
                "copt": [f"timelim={time_limit}", f"TimeLimit={time_limit}", f"timelimit={time_limit}"],
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
        ampl.read(MODEL_FILE)
        ampl.readData(instance_path)
        start = time.time()
        cpu_start = time.process_time()
        ampl.solve()
        # Retrieve raw solve message if available for diagnostics
        solve_message = None
        try:
            # AMPL keeps last solve message in a string param _solve_message (implementation detail)
            solve_message = ampl.getValue('_solve_message')  # may fail silently
        except Exception:
            try:
                # Some versions expose solve_message
                solve_message = ampl.getValue('solve_message')
            except Exception:
                pass
        elapsed = time.time() - start
        cpu_elapsed = time.process_time() - cpu_start
        # Collect results
        try:
            status = ampl.getValue("solve_result")
        except Exception:
            status = "UNKNOWN"
        try:
            obj = ampl.getObjective("TotalDistance").value()
        except Exception:
            obj = None
        try:
            gap = ampl.getValue("solve_result_gap")
        except Exception:
            gap = None
        # Try to obtain solver reported time, else fallback to measured CPU time
        try:
            _ = ampl.getValue("solve_time")  # Ignored now
        except Exception:
            pass  # We no longer record solver_time
        ampl.close()
        # Normalize obvious failure patterns (e.g., solver not available / license)
        if (status or '').lower() in {"failure", "failed"} and (obj is None or obj == 0):
            # augment status so caller can react
            status = f"FAILURE({solver})"
        result = (elapsed, None, obj, gap, status, solve_message)
        if queue is not None:
            queue.put(result)
        return result
    except Exception as e:
        try:
            ampl.close()
        except Exception:
            pass
        result = (None, None, None, None, f"ERROR: {e}", None)
        if queue is not None:
            queue.put(result)
        return result

def run_solver(instance_path, solver):
    """Run solver with external wall-clock enforcement returning (elapsed_time, unused_solver_time, obj, gap, status)."""
    if not EXTERNAL_WALLCLOCK_ENFORCE:
        return _run_solver_core(instance_path, solver, TIME_LIMIT)
    start_wall = time.time()
    q = Queue()
    p = Process(target=_run_solver_core, args=(instance_path, solver, TIME_LIMIT, q))
    p.start()
    p.join(TIME_LIMIT + 2)  # small grace period
    if p.is_alive():
        p.terminate()
        p.join()
        elapsed = time.time() - start_wall
        return (elapsed, None, None, None, "TIMEOUT")
    if not q.empty():
        result = q.get()
        if result[0] is not None and result[0] > TIME_LIMIT + 1:
            # Treat as timeout if internal limit ignored
            return (TIME_LIMIT, result[1], result[2], result[3], "TIMEOUT")
        # Strip solve_message before returning (keep interface stable)
        if len(result) == 6:
            return result[:5]
        return result
    # Fallback error
    elapsed = time.time() - start_wall
    return (elapsed, None, None, None, "ERROR")


def benchmark():
    results = []
    detailed_results = []  # For more detailed output
    
    # Use small instances for testing, or filter by size if SMALL_INSTANCES is empty
    if SMALL_INSTANCES:
        test_instances = SMALL_INSTANCES
        print(f"Using predefined small instances: {test_instances}")
    else:
        test_instances = filter_instances_by_size(DATA_DIR, MAX_INSTANCE_SIZE)
    
    # Filter to only use instances that actually exist
    instances = [inst for inst in test_instances if os.path.exists(os.path.join(DATA_DIR, inst))]
    
    # Build (instance,size) list then sort by size
    instances_with_size = []
    for inst in instances:
        inst_path = os.path.join(DATA_DIR, inst)
        sz = get_instance_size(inst_path)
        if sz is not None:
            instances_with_size.append((inst, sz))
    instances_with_size.sort(key=lambda x: x[1])
    instances = [inst for inst, _ in instances_with_size]

    # Active solvers after optional env filter (apply BEFORE listing for clarity)
    active_solvers = SOLVERS
    if ONLY_SOLVERS_ENV:
        requested = [s.strip() for s in ONLY_SOLVERS_ENV.split(',') if s.strip()]
        active_solvers = [s for s in SOLVERS if s in requested]
        print(f"Env override ONLY_SOLVERS -> limiting to: {active_solvers}")

    # Optional instance filtering via env var (apply BEFORE listing)
    if ONLY_INSTANCES_ENV:
        req_instances = {s.strip() for s in ONLY_INSTANCES_ENV.split(',') if s.strip()}
        instances = [i for i in instances if i in req_instances]
        instances_with_size = [t for t in instances_with_size if t[0] in req_instances]
        print(f"Env override ONLY_INSTANCES -> limiting to: {sorted(req_instances)}")

    # Optional chunk controls
    try:
        inst_offset = int(os.environ.get('INSTANCE_OFFSET', '0'))
        inst_limit = os.environ.get('INSTANCE_LIMIT')
        if inst_limit is not None:
            inst_limit = int(inst_limit)
    except Exception:
        inst_offset = 0
        inst_limit = None
    if inst_offset or inst_limit is not None:
        sliced = instances_with_size[inst_offset: inst_offset + inst_limit if inst_limit is not None else None]
        print(f"Applying slice: offset={inst_offset} limit={inst_limit if inst_limit is not None else '∞'} (from {len(instances_with_size)} total)")
        instances_with_size = sliced
        instances = [inst for inst, _ in instances_with_size]

    print(f"\nTesting with {len(instances)} instances (sorted by size):")
    preview = instances_with_size[:10]
    for i, (inst, size) in enumerate(preview):
        print(f"  {i+1:2d}. {inst} ({size} nodes)")
    if len(instances_with_size) > len(preview):
        print(f"  ... and {len(instances_with_size) - len(preview)} more instances")

    print(f"\nSolvers: {active_solvers}")

    # Optional instance filtering via env var
    if ONLY_INSTANCES_ENV:
        req_instances = {s.strip() for s in ONLY_INSTANCES_ENV.split(',') if s.strip()}
        instances = [i for i in instances if i in req_instances]
        instances_with_size = [t for t in instances_with_size if t[0] in req_instances]
        print(f"Env override ONLY_INSTANCES -> limiting to: {sorted(req_instances)}")

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
        print(f"\n[{i}/{len(instances)}] Instance: {inst} ({instance_size} nodes)", flush=True)

        # Resume logic: look into detailed JSON for existing runs per solver
        existing_runs_map = {}
        try:
            import json
            detailed_json_path = "results/benchmark_detailed_results.json"
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
                    continue  # already present
                print(f"    Run {run_number}/{RUNS}...", end=" ", flush=True)
                result = run_solver(inst_path, solver)
                
                if result[0] is None:  # Error occurred
                    print("ERROR")
                    # Disable solver after repeated first-run failure
                    if r == 0:
                        DISABLED_SOLVERS.add(solver)
                        print(f"    → Disabling solver {solver} for remaining instances (initial error).")
                    break

                t_ext, _t_solver_unused, o, g, s = result
                # Detect fast systematic failures (license / driver) and disable solver early
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

            if times_ext:  # Only add results if we have some data
                result_summary = {
                    "instance": inst,
                    "instance_size": instance_size,
                    "solver": solver,
                    "time_ext_mean": np.mean(times_ext),
                    "time_ext_median": np.median(times_ext),
                    "time_ext_std": np.std(times_ext),
                    "time_ext_min": np.min(times_ext),
                    "time_ext_max": np.max(times_ext),
                    # solver_time removed
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
                
                # Also save summary results incrementally
                df_temp = pd.DataFrame(results)
                summary_file = "results_ampl_solvers_incremental.csv"
                df_temp.to_csv(summary_file, index=False)
                print(f"    → Incremental summary saved to {summary_file}", flush=True)
                
                # Print quick summary for this solver
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
    json_file = "results/benchmark_detailed_results.json"
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
    txt_file = "results/benchmark_detailed_results.txt"
    if new_results:
        with open(txt_file, 'a') as f:
            if not existing_results:  # First time writing
                f.write("AMPL SOLVER BENCHMARK - DETAILED RESULTS\n")
                f.write("=" * 60 + "\n\n")
            
            current_instance = None
            current_solver = None
            
            for result in new_results:
                if result['instance'] != current_instance:
                    current_instance = result['instance']
                    f.write(f"\nInstance: {current_instance} ({result.get('instance_size', '?')} nodes)\n")
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
    pivot = df.pivot(index="instance", columns="solver", values="time_ext_median").dropna()

    # Comparação global - Friedman
    if len(pivot.columns) > 2:
        try:
            stat, p = friedmanchisquare(*[pivot[s] for s in pivot.columns])
            print(f"\nFriedman test (global): stat={stat:.3f}, p={p:.3e}")
        except Exception as e:
            print(f"Could not perform Friedman test: {e}")

    # Comparação par a par - Wilcoxon
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


if __name__ == "__main__":
    print("Starting AMPL Solver Benchmark...")
    print(f"Solvers: {SOLVERS}")
    print(f"Time limit: {TIME_LIMIT} seconds")
    print(f"Runs per instance: {RUNS}")
    print("=" * 60)

    # Purge existing detailed JSON entries for specified solvers (cleanup prior failed runs)
    if DROP_SOLVERS_ENV:
        to_drop = {s.strip() for s in DROP_SOLVERS_ENV.split(',') if s.strip()}
        json_file = "results/benchmark_detailed_results.json"
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
    
    df = benchmark()
    
    # Save summary results
    summary_file = "results/results_ampl_solvers.csv"
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
        
        # Perform statistical comparison
        try:
            stats_comparison(df)
        except Exception as e:
            print(f"Could not perform statistical comparison: {e}")
    else:
        print("No results obtained!")
