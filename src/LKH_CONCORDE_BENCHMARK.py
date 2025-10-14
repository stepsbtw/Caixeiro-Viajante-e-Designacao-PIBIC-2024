import os
import time
import subprocess
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
import re
import tempfile
import shutil
import argparse

# Configurações
TSP_DIR = "INSTANCES/TSP"          # pasta com arquivos .tsp
LKH_CMD = "/home/paco/LKH-2.0.11/LKH"           # binário LKH
CONCORDE_CMD = "/home/paco/concorde/TSP/concorde"  # binário Concorde
SOLVERS = ["concorde", "LKH"]               # Both solvers
RUNS = 3                           # repetições por instância
TIME_LIMIT = 1800                     # 30 minutes for large instances

# Filter instances by size
MAX_INSTANCE_SIZE = 10000                 # Run on ALL instances up to 10k nodes

# Results directory
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
DETAILED_JSON_PATH = None  # set later when CLI args parsed

def get_active_results_dir():
    """Directory where outputs should be written (follows detailed JSON path if provided)."""
    global DETAILED_JSON_PATH
    if DETAILED_JSON_PATH:
        return os.path.dirname(DETAILED_JSON_PATH) or RESULTS_DIR
    return RESULTS_DIR

# Create temporary directory for solver files
TEMP_DIR = tempfile.mkdtemp(prefix="tsp_benchmark_")
print(f"Using temporary directory: {TEMP_DIR}")

def get_tsp_instance_size(tsp_file_path):
    """Extract the number of nodes from a .tsp file"""
    try:
        with open(tsp_file_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if line.startswith('DIMENSION'):
                dimension = int(line.split(':')[1].strip())
                return dimension
        
        return None
    except Exception as e:
        print(f"Error reading {tsp_file_path}: {e}")
        return None

def filter_tsp_instances_by_size(tsp_dir, max_size):
    """Filter .tsp files to only include instances with <= max_size nodes"""
    all_instances = [f for f in os.listdir(tsp_dir) if f.endswith(".tsp")]
    filtered_instances = []
    
    print(f"Filtering TSP instances by size (max {max_size} nodes)...")
    
    for instance in all_instances:
        instance_path = os.path.join(tsp_dir, instance)
        size = get_tsp_instance_size(instance_path)
        
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

def run_lkh(instance_path, run_id, timeout=TIME_LIMIT):
    """Executa LKH gerando parameter file com seed fixa."""
    instance_name = os.path.basename(instance_path)
    par_file = os.path.join(TEMP_DIR, f"temp_{instance_name}_run{run_id}.par")
    tour_file = os.path.join(TEMP_DIR, f"temp_{instance_name}_run{run_id}.tour")

    with open(par_file, "w") as f:
        f.write(f"PROBLEM_FILE = {instance_path}\n")
        f.write(f"OUTPUT_TOUR_FILE = {tour_file}\n")
        f.write("RUNS = 1\n")
        f.write(f"SEED = {run_id+1}\n")

    start = time.time()
    try:
        output = subprocess.check_output(
            [LKH_CMD, par_file],
            stderr=subprocess.STDOUT,
            timeout=timeout,
            universal_newlines=True
        )
        elapsed = time.time() - start
        cost = None
        
        # Parse LKH output for cost
        for line in output.splitlines():
            if "Cost.min" in line:
                try:
                    # Extract from lines like "Cost.min = 3323, Cost.avg = 3323.00, Cost.max = 3323"
                    if "Cost.min" in line and "=" in line:
                        parts = line.split("Cost.min")[1].strip()
                        if "=" in parts:
                            cost_part = parts.split("=")[1].split(",")[0].strip()
                            cost = float(cost_part)
                            break
                except:
                    pass
        
        return elapsed, cost, "solved" if cost is not None else "no_solution"
        
    except subprocess.TimeoutExpired:
        return timeout, None, "timeout"
    except subprocess.CalledProcessError as e:
        print(f"LKH error: {e}")
        return None, None, "error"
    except FileNotFoundError:
        print(f"LKH binary not found at {LKH_CMD}")
        return None, None, "binary_not_found"
    finally:
        # Cleanup temp files
        for temp_file in [par_file, tour_file]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

def run_concorde(instance_path, timeout=TIME_LIMIT):
    """Executa Concorde direto na linha de comando."""
    start = time.time()
    try:
        output = subprocess.check_output(
            [CONCORDE_CMD, instance_path],
            stderr=subprocess.STDOUT,
            timeout=timeout,
            universal_newlines=True
        )
        elapsed = time.time() - start
        cost = None
        
        # Parse Concorde output for cost
        for line in output.splitlines():
            if "Optimal" in line or "Cost" in line or "Length" in line:
                try:
                    # Extract number from line
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        cost = float(numbers[-1])
                        break
                except:
                    pass
        
        return elapsed, cost, "solved" if cost is not None else "no_solution"
        
    except subprocess.TimeoutExpired:
        return timeout, None, "timeout"
    except subprocess.CalledProcessError as e:
        print(f"Concorde error: {e}")
        return None, None, "error"
    except FileNotFoundError:
        print(f"Concorde binary not found at {CONCORDE_CMD}")
        return None, None, "binary_not_found"
    finally:
        # Cleanup Concorde temp files
        instance_name = os.path.splitext(os.path.basename(instance_path))[0]
        concorde_files = [
            f"{instance_name}.mas", f"{instance_name}.pul", 
            f"{instance_name}.sav", f"{instance_name}.sol",
            f"{instance_name}.res",  # Add .res files
            f"O{instance_name}.mas", f"O{instance_name}.pul", 
            f"O{instance_name}.sav", f"O{instance_name}.res"  # Sometimes Concorde creates files with "O" prefix
        ]
        for temp_file in concorde_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

def run_solver(instance_path, solver):
    """Run a specific solver on an instance"""
    if solver.lower() == "lkh":
        # Run LKH with random seed based on time
        run_id = int(time.time() * 1000) % 10000
        return run_lkh(instance_path, run_id)
    elif solver.lower() == "concorde":
        return run_concorde(instance_path)
    else:
        raise ValueError(f"Unknown solver: {solver}")

def save_detailed_results(detailed_results):
    """Save detailed results (JSON + text) in active results directory."""
    import json
    results_dir = get_active_results_dir()
    os.makedirs(results_dir, exist_ok=True)
    json_file = DETAILED_JSON_PATH or os.path.join(results_dir, "lkh_concorde_detailed_results.json")
    try:
        with open(json_file, 'r') as f:
            existing_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_results = []
    existing_keys = {(r['instance'], r['solver'], r['run']) for r in existing_results}
    new_results = [r for r in detailed_results if (r['instance'], r['solver'], r['run']) not in existing_keys]
    if not new_results:
        return
    all_results = existing_results + new_results
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Added {len(new_results)} new results to {json_file}")
    txt_file = os.path.join(results_dir, "lkh_concorde_detailed_results.txt")
    with open(txt_file, 'a') as f:
        if not existing_results:
            f.write("LKH vs CONCORDE BENCHMARK - DETAILED RESULTS\n")
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
                f.write(f"\nSolver: {current_solver.upper()}\n")
            f.write(f"  Run {result['run']}: Status={result['status']}, Cost={result['objective']}, Time={result['elapsed_time']:.3f}s [{result['timestamp']}]\n")
    print(f"✓ Appended {len(new_results)} new results to {txt_file}")

def benchmark():
    results = []
    detailed_results = []
    
    # Filter instances by size
    test_instances = filter_tsp_instances_by_size(TSP_DIR, MAX_INSTANCE_SIZE)
    
    # Filter to only use instances that actually exist
    instances = [inst for inst in test_instances if os.path.exists(os.path.join(TSP_DIR, inst))]
    
    # Sort instances by size (smallest first)
    instances_with_size = []
    for inst in instances:
        inst_path = os.path.join(TSP_DIR, inst)
        size = get_tsp_instance_size(inst_path)
        if size is not None:
            instances_with_size.append((inst, size))
        else:
            print(f"Warning: Could not determine size for {inst}")
    
    # Sort by size and extract instance names
    instances_with_size.sort(key=lambda x: x[1])
    instances = [inst for inst, size in instances_with_size]
    
    print(f"\nTesting with {len(instances)} instances (sorted by size):")
    for i, (inst, size) in enumerate(instances_with_size[:10]):  # Show first 10
        print(f"  {i+1:2d}. {inst} ({size} nodes)")
    if len(instances_with_size) > 10:
        print(f"  ... and {len(instances_with_size) - 10} more instances")
    
    print(f"\nSolvers: {SOLVERS}")
    print(f"Runs per instance: {RUNS}")
    print(f"Time limit: {TIME_LIMIT} seconds")
    print("=" * 80)

    for i, inst in enumerate(instances, 1):
        inst_path = os.path.join(TSP_DIR, inst)
        instance_size = get_tsp_instance_size(inst_path)
        print(f"\n[{i}/{len(instances)}] Instance: {inst} ({instance_size} nodes)")
        
        # Check if this instance has already been run for all solvers
        if check_instance_already_run(inst):
            print(f"  ✓ Already completed - skipping {inst}")
            continue

        for solver in SOLVERS:
            print(f"  Testing solver: {solver.upper()}")
            times, costs, statuses = [], [], []
            run_details = []

            for r in range(RUNS):
                print(f"    Run {r+1}/{RUNS}...", end=" ", flush=True)
                start_total = time.time()
                result = run_solver(inst_path, solver)
                total_run_time = time.time() - start_total
                
                if result[0] is None:  # Error occurred
                    print("ERROR")
                    continue
                    
                t_elapsed, cost, status = result
                
                run_detail = {
                    'instance': inst,
                    'instance_size': instance_size,
                    'solver': solver,
                    'run': r + 1,
                    'status': status,
                    'objective': cost,
                    'elapsed_time': t_elapsed,
                    'total_run_time': total_run_time,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                run_details.append(run_detail)
                print(f"{t_elapsed:.3f}s (cost: {cost})")
                
                if t_elapsed is not None:
                    times.append(t_elapsed)
                if cost is not None:
                    costs.append(cost)
                statuses.append(status)

            if times:  # Only add results if we have some data
                result_summary = {
                    "instance": inst,
                    "instance_size": instance_size,
                    "solver": solver,
                    "time_mean": np.mean(times),
                    "time_median": np.median(times),
                    "time_std": np.std(times),
                    "time_min": np.min(times),
                    "time_max": np.max(times),
                    "cost_best": np.min(costs) if costs else None,
                    "cost_mean": np.mean(costs) if costs else None,
                    "cost_std": np.std(costs) if len(costs) > 1 else None,
                    "status_last": statuses[-1] if statuses else None,
                    "successful_runs": len(times),
                    "total_runs": RUNS
                }
                results.append(result_summary)
                detailed_results.extend(run_details)
                
                # Save results after each solver completion
                print(f"    → Saving results after {solver.upper()} completion...")
                save_detailed_results(detailed_results)
                
                # Also save summary results incrementally
                df_temp = pd.DataFrame(results)
                summary_file = os.path.join(get_active_results_dir(), "results_lkh_concorde_incremental.csv")
                df_temp.to_csv(summary_file, index=False)
                print(f"    → Incremental summary saved to {summary_file}")
                
                # Print quick summary for this solver
                if len(times) > 0:
                    best_cost = np.min(costs) if costs else 'N/A'
                    print(f"    → Summary: {len(times)}/{RUNS} successful, avg time: {np.mean(times):.3f}s, best cost: {best_cost}")
            else:
                print(f"    → No successful runs for {solver.upper()} on {inst}")

    return pd.DataFrame(results)

def stats_comparison(df):
    """Statistical comparison between solvers (writes to active results dir)."""
    stats_file = os.path.join(get_active_results_dir(), "lkh_concorde_statistics.txt")
    
    with open(stats_file, 'w') as f:
        f.write("LKH vs CONCORDE BENCHMARK - STATISTICAL ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        
        # Basic summary
        f.write(f"Benchmark Summary:\n")
        f.write(f"  Instances tested: {df['instance'].nunique()}\n")
        f.write(f"  Solvers tested: {df['solver'].nunique()}\n")
        f.write(f"  Total successful runs: {df['successful_runs'].sum()}\n")
        f.write(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Solver performance summary
        f.write("Solver Performance Summary:\n")
        for solver in df['solver'].unique():
            solver_data = df[df['solver'] == solver]
            f.write(f"  {solver.upper()}:\n")
            f.write(f"    Average time: {solver_data['time_mean'].mean():.3f}s\n")
            f.write(f"    Median time: {solver_data['time_median'].mean():.3f}s\n")
            best_costs = solver_data['cost_best'].dropna()
            if not best_costs.empty:
                f.write(f"    Average best cost: {best_costs.mean():.1f}\n")
            f.write(f"    Successful runs: {solver_data['successful_runs'].sum()}\n")
            f.write(f"    Success rate: {solver_data['successful_runs'].sum() / (len(solver_data) * RUNS) * 100:.1f}%\n\n")
        
        if len(df['solver'].unique()) < 2:
            f.write("Need at least 2 solvers for statistical comparison\n")
            print("Need at least 2 solvers for comparison")
            return
        
        pivot = df.pivot(index="instance", columns="solver", values="time_median").dropna()
        
        f.write("Statistical Tests:\n")
        f.write("-" * 20 + "\n")
        
        # Global comparison - Friedman test
        if len(pivot.columns) > 2:
            try:
                stat, p = friedmanchisquare(*[pivot[s] for s in pivot.columns])
                result = f"Friedman test (global): stat={stat:.3f}, p={p:.3e}"
                f.write(result + "\n")
                print(f"\n{result}")
            except Exception as e:
                error_msg = f"Could not perform Friedman test: {e}"
                f.write(error_msg + "\n")
                print(error_msg)

        # Pairwise comparison - Wilcoxon test
        f.write("\nPairwise Comparisons (Wilcoxon):\n")
        solvers = df['solver'].unique()
        for i, s1 in enumerate(solvers):
            for s2 in solvers[i+1:]:
                if s1 in pivot.columns and s2 in pivot.columns:
                    x, y = pivot[s1], pivot[s2]
                    common = x.index.intersection(y.index)
                    if len(common) > 0:
                        try:
                            stat, p = wilcoxon(x.loc[common], y.loc[common])
                            result = f"Wilcoxon {s1.upper()} vs {s2.upper()}: stat={stat}, p={p:.3e}"
                            f.write(f"  {result}\n")
                            print(result)
                        except Exception as e:
                            error_msg = f"Could not perform Wilcoxon test for {s1} vs {s2}: {e}"
                            f.write(f"  {error_msg}\n")
                            print(error_msg)
        
        # Performance comparison table
        f.write(f"\nDetailed Comparison Table:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Instance':<15}")
        for solver in pivot.columns:
            f.write(f"{solver.upper():<12}")
        f.write("Winner\n")
        f.write("-" * 40 + "\n")
        
        for instance in pivot.index:
            f.write(f"{instance:<15}")
            times = {}
            for solver in pivot.columns:
                time_val = pivot.loc[instance, solver]
                times[solver] = time_val
                f.write(f"{time_val:<12.3f}")
            
            # Find winner (fastest)
            if times:
                winner = min(times.keys(), key=lambda s: times[s])
                f.write(f"{winner.upper()}\n")
            else:
                f.write("N/A\n")
    
    print(f"✓ Statistical analysis saved to {stats_file}")

def cleanup_remaining_files():
    """Move stray solver result files into the active results directory."""
    import glob
    results_dir = get_active_results_dir()
    os.makedirs(results_dir, exist_ok=True)
    result_extensions = ['*.csv', '*.res', '*.mas', '*.pul', '*.sav', '*.sol', '*.tour', '*.par']
    moved_files = []
    for pattern in result_extensions:
        for file_path in glob.glob(pattern):
            if os.path.isfile(file_path):
                if not file_path.startswith(results_dir + '/'):
                    dest_path = os.path.join(results_dir, os.path.basename(file_path))
                    try:
                        shutil.move(file_path, dest_path)
                        moved_files.append(os.path.basename(file_path))
                    except Exception as e:
                        print(f"Warning: Could not move {file_path}: {e}")
    if moved_files:
        print(f"✓ Moved {len(moved_files)} result files to {results_dir}/: {', '.join(moved_files)}")

def check_instance_already_run(instance_name):
    """Check if an instance has already been run for both solvers"""
    import json
    json_file = DETAILED_JSON_PATH or os.path.join(get_active_results_dir(), "lkh_concorde_detailed_results.json")
    if not os.path.exists(json_file):
        return False
    
    try:
        with open(json_file, 'r') as f:
            existing_results = json.load(f)
    except:
        return False
    
    # Check if we have results for both solvers for this instance
    solvers_found = set()
    for result in existing_results:
        if result['instance'] == instance_name:
            solvers_found.add(result['solver'])
    
    # Return True if we have results for all configured solvers
    return len(solvers_found) >= len(SOLVERS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LKH vs Concorde Benchmark Utilities")
    parser.add_argument('--rerun-lkh-zero', action='store_true', help='Rerun ONLY LKH on instances where previous objective was zero (or below threshold) (remove old faulty runs first).')
    parser.add_argument('--list-lkh-zero', action='store_true', help='List instances where LKH objective was zero (or below threshold) and exit.')
    parser.add_argument('--zero-threshold', type=float, default=0.0, help='Threshold <= value considered zero (default 0).')
    parser.add_argument('--detailed-file', type=str, default=None, help='Path to detailed JSON results file (overrides default).')
    args = parser.parse_args()

    def resolve_detailed_file():
        """Determine which detailed JSON file to use.
        Priority: CLI provided -> standard location -> common alternative subpaths."""
        if args.detailed_file:
            return args.detailed_file
        candidates = [
            os.path.join(RESULTS_DIR, "lkh_concorde_detailed_results.json"),
            os.path.join(RESULTS_DIR, 'tsp', 'lkh_concorde_detailed_results.json'),
            os.path.join(RESULTS_DIR, 'tsp', 'lkh_concorde_detailed_results_old.json')
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        # fallback to first default path
        return candidates[0]

    DETAILED_JSON_PATH = resolve_detailed_file()
    if not os.path.exists(DETAILED_JSON_PATH):
        # Ensure directory exists if we later write
        os.makedirs(os.path.dirname(DETAILED_JSON_PATH), exist_ok=True)

    print(f"Using detailed results file: {DETAILED_JSON_PATH}")

    def load_detailed_json():
        import json
        json_file = DETAILED_JSON_PATH
        if not os.path.exists(json_file):
            return []
        try:
            with open(json_file, 'r') as f:
                return json.load(f)
        except Exception:
            return []

    def save_detailed_json(data):
        import json
        json_file = DETAILED_JSON_PATH
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)

    def rebuild_summary_from_detailed():
        """Recompute summary CSV from detailed JSON runs (after modifications/reruns)."""
        detailed = load_detailed_json()
        if not detailed:
            return pd.DataFrame()
        # group by instance, solver
        rows = []
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in detailed:
            grouped[(r['instance'], r['solver'])].append(r)
        for (inst, solver), runs in grouped.items():
            try:
                runs_sorted = sorted(runs, key=lambda x: x.get('run', 0))
            except Exception:
                runs_sorted = runs
            times = [r['elapsed_time'] for r in runs_sorted if r.get('elapsed_time') is not None]
            costs = [r['objective'] for r in runs_sorted if r.get('objective') is not None]
            if not times:
                continue
            row = {
                'instance': inst,
                'instance_size': runs_sorted[0].get('instance_size'),
                'solver': solver,
                'time_mean': float(np.mean(times)),
                'time_median': float(np.median(times)),
                'time_std': float(np.std(times)),
                'time_min': float(np.min(times)),
                'time_max': float(np.max(times)),
                'cost_best': float(np.min(costs)) if costs else None,
                'cost_mean': float(np.mean(costs)) if costs else None,
                'cost_std': float(np.std(costs)) if len(costs) > 1 else None,
                'status_last': runs_sorted[-1].get('status'),
                'successful_runs': len(times),
                'total_runs': len(runs_sorted)
            }
            rows.append(row)
        df_summary = pd.DataFrame(rows)
        summary_file_final = os.path.join(get_active_results_dir(), "results_lkh_concorde_final.csv")
        df_summary.to_csv(summary_file_final, index=False)
        summary_file_incr = os.path.join(get_active_results_dir(), "results_lkh_concorde_incremental.csv")
        df_summary.to_csv(summary_file_incr, index=False)
        print(f"✓ Rebuilt summary with {len(df_summary)} (instance,solver) pairs")
        return df_summary

    def objective_is_zero(val: object) -> bool:
        if val is None:
            return False
        try:
            fval = float(val)
        except (ValueError, TypeError):
            return False
        return fval <= args.zero_threshold

    def list_lkh_zero_instances():
        detailed = load_detailed_json()
        zero_instances = sorted({
            r['instance'] for r in detailed
            if r.get('solver','').lower()== 'lkh' and objective_is_zero(r.get('objective'))
        })
        if zero_instances:
            print(f"Instances with LKH objective <= {args.zero_threshold}:")
            for inst in zero_instances:
                print(f"  - {inst}")
        else:
            print(f"No instances found with LKH objective <= {args.zero_threshold}")
        return zero_instances

    def rerun_lkh_zero_instances():
        detailed = load_detailed_json()
        if not detailed:
            print("No detailed results file present. Run a full benchmark first.")
            return
        zero_instances = list_lkh_zero_instances()
        if not zero_instances:
            print("Nothing to rerun.")
            return
        # Remove all LKH runs for affected instances (so they get clean stats)
        before = len(detailed)
        detailed = [r for r in detailed if not (r.get('solver','').lower()=='lkh' and r['instance'] in zero_instances)]
        removed = before - len(detailed)
        print(f"Removed {removed} faulty LKH run records.")
        save_detailed_json(detailed)

        new_runs = []
        for inst in zero_instances:
            inst_path = os.path.join(TSP_DIR, inst)
            if not os.path.exists(inst_path):
                print(f"Skipping missing instance file: {inst}")
                continue
            instance_size = get_tsp_instance_size(inst_path)
            print(f"Re-running LKH for {inst} ({instance_size} nodes)...")
            for r_id in range(RUNS):
                start_total = time.time()
                t_elapsed, cost, status = run_lkh(inst_path, r_id)
                total_run_time = time.time() - start_total
                run_record = {
                    'instance': inst,
                    'instance_size': instance_size,
                    'solver': 'LKH',
                    'run': r_id + 1,
                    'status': status,
                    'objective': cost,
                    'elapsed_time': t_elapsed,
                    'total_run_time': total_run_time,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                new_runs.append(run_record)
                print(f"  Run {r_id+1}/{RUNS}: {t_elapsed:.3f}s cost={cost} status={status}")
            # Persist after each instance
            save_detailed_results(new_runs)
            new_runs.clear()
        # After reruns rebuild summary
        df_new = rebuild_summary_from_detailed()
        if not df_new.empty:
            try:
                stats_comparison(df_new)
            except Exception as e:
                print(f"Stats comparison failed after rerun: {e}")

    # Mode handling
    if args.list_lkh_zero:
        list_lkh_zero_instances()
        exit(0)
    if args.rerun_lkh_zero:
        print(f"=== Rerun LKH for zero-objective instances mode (threshold <= {args.zero_threshold}) ===")
        rerun_lkh_zero_instances()
        cleanup_remaining_files()
        try:
            shutil.rmtree(TEMP_DIR)
            print(f"\n✓ Cleaned up temporary directory: {TEMP_DIR}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory: {e}")
        exit(0)

    # Default: full benchmark
    print("Starting LKH vs Concorde Benchmark...")
    print(f"Solvers: {[s.upper() for s in SOLVERS]}")
    print(f"Time limit: {TIME_LIMIT} seconds")
    print(f"Runs per instance: {RUNS}")
    print("=" * 60)

    # Check if solver binaries exist
    for solver in SOLVERS:
        if solver.lower() == "lkh" and not os.path.exists(LKH_CMD):
            print(f"WARNING: LKH binary not found at {LKH_CMD}")
        elif solver.lower() == "concorde" and not os.path.exists(CONCORDE_CMD):
            print(f"WARNING: Concorde binary not found at {CONCORDE_CMD}")

    df = benchmark()

    # Save summary results
    summary_file = os.path.join(get_active_results_dir(), "results_lkh_concorde_final.csv")
    df.to_csv(summary_file, index=False)
    print(f"\n✓ Final summary results saved to {summary_file}")

    # Print summary statistics
    if not df.empty:
        print(f"\nSummary Statistics:")
        print(f"Instances tested: {df['instance'].nunique()}")
        print(f"Solvers tested: {df['solver'].nunique()}")
        print(f"Total successful runs: {df['successful_runs'].sum()}")

        # Show average times by solver
        print(f"\nAverage times by solver:")
        solver_avg = df.groupby('solver')['time_mean'].mean().sort_values()
        for solver, avg_time in solver_avg.items():
            print(f"  {solver.upper()}: {avg_time:.3f}s")

        # Show best costs by solver
        print(f"\nBest costs by solver:")
        for solver in df['solver'].unique():
            solver_data = df[df['solver'] == solver]
            best_costs = solver_data['cost_best'].dropna()
            if not best_costs.empty:
                print(f"  {solver.upper()}: avg best = {best_costs.mean():.1f}")

        # Perform statistical comparison
        try:
            stats_comparison(df)
        except Exception as e:
            print(f"Could not perform statistical comparison: {e}")
    else:
        print("No results obtained!")

    # Move any remaining result files to results directory
    cleanup_remaining_files()

    # Cleanup temporary directory
    try:
        shutil.rmtree(TEMP_DIR)
        print(f"\n✓ Cleaned up temporary directory: {TEMP_DIR}")
    except Exception as e:
        print(f"Warning: Could not clean up temporary directory: {e}")
