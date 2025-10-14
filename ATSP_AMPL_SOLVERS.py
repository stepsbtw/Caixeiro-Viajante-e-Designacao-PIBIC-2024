import os
import time
import json
import numpy as np
import pandas as pd
from multiprocessing import Process, Queue
from scipy.stats import friedmanchisquare, wilcoxon

# Configurações - using amplpy with built-in solvers
MODEL_FILE = "models/atsp.mod"  # ATSP model AMPL
DATA_DIR = "dat/atsp"  # ATSP data files directory
RESULTS_DIR = "results/atsp"  # Results directory
OUTPUT_PREFIX = "atsp_builtin"

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# AMPL-compatible solvers available in amplpy
SOLVERS = {
    'highs': {
        'name': 'HiGHS',
        'type': 'linear',
        'time_limit': 300,
        'options': {
            'time_limit': 300,
            'presolve': 'on',
            'mip_feasibility_tolerance': 1e-6,
            'mip_gap_abs': 1e-6,
            'mip_gap_rel': 1e-4
        }
    },
    'gurobi': {
        'name': 'Gurobi',
        'type': 'general',
        'time_limit': 300,
        'options': {
            'TimeLimit': 300,
            'MIPGap': 1e-4,
            'MIPGapAbs': 1e-6,
            'Presolve': 2,
            'Cuts': 2
        }
    },
    'cplex': {
        'name': 'CPLEX',
        'type': 'general',
        'time_limit': 300,
        'options': {
            'timelimit': 300,
            'mipgap': 1e-4,
            'absmipgap': 1e-6,
            'presolve': 1,
            'cuts': 2
        }
    },
    'cbc': {
        'name': 'CBC',
        'type': 'linear',
        'time_limit': 300,
        'options': {
            'sec': 300,
            'ratio': 1e-4,
            'allow': 1e-6,
            'cuts': 'on',
            'preprocess': 'on'
        }
    }
}


def solve_atsp_instance(instance_file, solver_name, run_number=1, results_queue=None):
    """
    Solve a single ATSP instance with specified solver.
    
    Args:
        instance_file: Path to .dat file
        solver_name: Name of solver to use
        run_number: Run number for multiple runs
        results_queue: Queue for multiprocessing results
    
    Returns:
        Dictionary with solve results
    """
    try:
        from amplpy import AMPL
        
        # Initialize AMPL
        ampl = AMPL()
        
        # Read model
        ampl.read(MODEL_FILE)
        
        # Read data
        ampl.readData(instance_file)
        
        # Configure solver
        solver_config = SOLVERS[solver_name]
        ampl.setOption('solver', solver_name)
        
        # Set solver options
        for option, value in solver_config['options'].items():
            ampl.setOption(f"{solver_name}_{option}", value)
        
        # Get instance info
        n = ampl.getParameter('n').value()
        instance_name = os.path.basename(instance_file).replace('.dat', '')
        
        print(f"[{solver_name}] Solving {instance_name} (n={n}) - Run {run_number}")
        
        # Solve
        start_time = time.time()
        ampl.solve()
        solve_time = time.time() - start_time
        
        # Get results
        solve_result = ampl.get_value('solve_result')
        solve_message = ampl.get_value('solve_message')
        
        result = {
            'instance': instance_name,
            'instance_file': instance_file,
            'instance_size': int(n),
            'solver': solver_name,
            'solver_name': solver_config['name'],
            'run': run_number,
            'solve_result': solve_result,
            'solve_message': solve_message,
            'runtime': solve_time
        }
        
        # Check if solved optimally or feasibly
        if solve_result in ['solved', 'optimal']:
            objective = ampl.getObjective('minimize_tour').value()
            result.update({
                'status': 'optimal' if solve_result == 'optimal' else 'feasible',
                'objective': objective,
                'solved': True
            })
            
            # Get additional solver statistics if available
            try:
                result['nodes'] = ampl.get_value('solve_elapsed_time') or 0
                result['iterations'] = ampl.get_value('solve_iterations') or 0
            except:
                result['nodes'] = 0
                result['iterations'] = 0
            
            print(f"[{solver_name}] {instance_name}: SOLVED - Obj: {objective:.2f}, Time: {solve_time:.3f}s")
            
        else:
            result.update({
                'status': 'infeasible' if 'infeasible' in solve_result.lower() else 'time_limit',
                'objective': float('inf'),
                'solved': False,
                'nodes': 0,
                'iterations': 0
            })
            
            print(f"[{solver_name}] {instance_name}: {solve_result.upper()} - Time: {solve_time:.3f}s")
        
        # Clean up
        ampl.close()
        
        # Add to queue if provided (for multiprocessing)
        if results_queue:
            results_queue.put(result)
        
        return result
        
    except Exception as e:
        error_result = {
            'instance': os.path.basename(instance_file).replace('.dat', ''),
            'instance_file': instance_file,
            'instance_size': 0,
            'solver': solver_name,
            'solver_name': SOLVERS.get(solver_name, {}).get('name', solver_name),
            'run': run_number,
            'status': 'error',
            'objective': float('inf'),
            'runtime': 0.0,
            'solved': False,
            'nodes': 0,
            'iterations': 0,
            'solve_result': 'error',
            'solve_message': str(e),
            'error': str(e)
        }
        
        print(f"[{solver_name}] ERROR in {instance_file}: {str(e)}")
        
        if results_queue:
            results_queue.put(error_result)
        
        return error_result


def solve_all_instances(solvers_to_test=None, max_workers=4, runs_per_instance=1, 
                       instance_pattern="*.dat", parallel=True):
    """
    Solve all ATSP instances with specified solvers.
    
    Args:
        solvers_to_test: List of solver names to test (None = all available)
        max_workers: Maximum number of parallel processes
        runs_per_instance: Number of runs per instance-solver combination
        instance_pattern: Pattern to match instance files
        parallel: Whether to use parallel processing
    
    Returns:
        List of result dictionaries
    """
    import glob
    
    # Get available solvers
    if solvers_to_test is None:
        solvers_to_test = list(SOLVERS.keys())
    
    # Filter to available solvers
    available_solvers = []
    for solver in solvers_to_test:
        if solver in SOLVERS:
            available_solvers.append(solver)
        else:
            print(f"Warning: Solver {solver} not available")
    
    if not available_solvers:
        print("No available solvers found!")
        return []
    
    # Get instance files
    instance_files = glob.glob(os.path.join(DATA_DIR, instance_pattern))
    instance_files.sort()
    
    if not instance_files:
        print(f"No instance files found in {DATA_DIR} matching {instance_pattern}")
        return []
    
    print(f"Found {len(instance_files)} instances")
    print(f"Testing {len(available_solvers)} solvers: {available_solvers}")
    print(f"Runs per instance: {runs_per_instance}")
    print(f"Parallel processing: {parallel} (max {max_workers} workers)")
    print(f"Total combinations: {len(instance_files) * len(available_solvers) * runs_per_instance}")
    
    all_results = []
    
    if parallel and max_workers > 1:
        # Parallel processing
        from multiprocessing import Process, Queue
        
        # Create task queue
        tasks = []
        for instance_file in instance_files:
            for solver in available_solvers:
                for run in range(1, runs_per_instance + 1):
                    tasks.append((instance_file, solver, run))
        
        # Process tasks in batches
        batch_size = max_workers
        results_queue = Queue()
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            processes = []
            
            print(f"\nProcessing batch {i//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size}")
            
            # Start processes for this batch
            for instance_file, solver, run in batch:
                p = Process(target=solve_atsp_instance, 
                          args=(instance_file, solver, run, results_queue))
                p.start()
                processes.append(p)
            
            # Collect results from this batch
            for _ in range(len(processes)):
                result = results_queue.get()
                all_results.append(result)
            
            # Wait for all processes to complete
            for p in processes:
                p.join()
    
    else:
        # Sequential processing
        task_count = 0
        total_tasks = len(instance_files) * len(available_solvers) * runs_per_instance
        
        for instance_file in instance_files:
            for solver in available_solvers:
                for run in range(1, runs_per_instance + 1):
                    task_count += 1
                    print(f"\nTask {task_count}/{total_tasks}")
                    
                    result = solve_atsp_instance(instance_file, solver, run)
                    all_results.append(result)
    
    return all_results


def save_results(results, output_file=None):
    """Save results to JSON and CSV files."""
    if output_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"{RESULTS_DIR}/{OUTPUT_PREFIX}_results_{timestamp}"
    
    # Save as JSON
    json_file = f"{output_file}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {json_file}")
    
    # Save as CSV
    csv_file = f"{output_file}.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    print(f"Results saved to {csv_file}")
    
    return json_file, csv_file


def analyze_results(results):
    """Perform basic analysis of results."""
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("RESULTS ANALYSIS")
    print("="*60)
    
    # Overall statistics
    print(f"Total instances: {df['instance'].nunique()}")
    print(f"Total solvers: {df['solver'].nunique()}")
    print(f"Total runs: {len(df)}")
    
    # Solve statistics
    solved_df = df[df['solved'] == True]
    print(f"Successfully solved: {len(solved_df)}/{len(df)} ({len(solved_df)/len(df)*100:.1f}%)")
    
    if len(solved_df) == 0:
        print("No instances solved successfully!")
        return
    
    # Solver comparison
    print("\nSOLVER PERFORMANCE:")
    print("-" * 40)
    
    solver_stats = []
    for solver in df['solver'].unique():
        solver_data = df[df['solver'] == solver]
        solved_data = solver_data[solver_data['solved'] == True]
        
        stats = {
            'solver': solver,
            'instances': len(solver_data),
            'solved': len(solved_data),
            'solve_rate': len(solved_data) / len(solver_data) * 100,
            'avg_objective': solved_data['objective'].mean() if len(solved_data) > 0 else float('inf'),
            'avg_runtime': solver_data['runtime'].mean(),
            'max_runtime': solver_data['runtime'].max()
        }
        solver_stats.append(stats)
    
    solver_df = pd.DataFrame(solver_stats)
    solver_df = solver_df.sort_values('solve_rate', ascending=False)
    
    print(solver_df.round(3))
    
    # Best results per instance
    print("\nBEST RESULTS PER INSTANCE:")
    print("-" * 40)
    
    if len(solved_df) > 0:
        best_results = solved_df.loc[solved_df.groupby('instance')['objective'].idxmin()]
        best_summary = best_results[['instance', 'solver', 'objective', 'runtime']].copy()
        best_summary = best_summary.sort_values('objective')
        
        print(best_summary.head(10))
        
        # Solver wins
        print(f"\nSolver wins:")
        wins = best_results['solver'].value_counts()
        for solver, count in wins.items():
            print(f"  {solver}: {count}")
    
    # Runtime analysis
    print(f"\nRUNTIME STATISTICS:")
    print("-" * 30)
    runtime_stats = df.groupby('solver')['runtime'].agg(['mean', 'std', 'min', 'max']).round(3)
    print(runtime_stats)
    
    return solver_df


def statistical_analysis(results):
    """Perform statistical significance tests."""
    df = pd.DataFrame(results)
    solved_df = df[df['solved'] == True]
    
    if len(solved_df) == 0:
        print("No solved instances for statistical analysis")
        return
    
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    # Find instances solved by multiple solvers
    instance_solver_count = solved_df.groupby('instance')['solver'].nunique()
    common_instances = instance_solver_count[instance_solver_count > 1].index
    
    if len(common_instances) == 0:
        print("No instances solved by multiple solvers")
        return
    
    print(f"Analyzing {len(common_instances)} instances solved by multiple solvers")
    
    # Prepare data for statistical tests
    solvers = solved_df['solver'].unique()
    
    if len(solvers) < 2:
        print("Need at least 2 solvers for comparison")
        return
    
    # Create comparison matrix
    comparison_data = {}
    for solver in solvers:
        solver_data = solved_df[
            (solved_df['solver'] == solver) & 
            (solved_df['instance'].isin(common_instances))
        ]
        objectives = solver_data.set_index('instance')['objective']
        comparison_data[solver] = objectives
    
    # Align data
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.dropna()
    
    if len(comparison_df) < 3:
        print("Insufficient data for statistical tests")
        return
    
    print(f"Testing on {len(comparison_df)} instances")
    
    # Friedman test (non-parametric ANOVA)
    if len(solvers) > 2:
        try:
            stat, p_value = friedmanchisquare(*[comparison_df[solver] for solver in solvers])
            print(f"\nFriedman test:")
            print(f"  Statistic: {stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
        except Exception as e:
            print(f"Friedman test failed: {e}")
    
    # Pairwise Wilcoxon tests
    print(f"\nPairwise Wilcoxon signed-rank tests:")
    for i, solver1 in enumerate(solvers):
        for solver2 in solvers[i+1:]:
            try:
                data1 = comparison_df[solver1].dropna()
                data2 = comparison_df[solver2].dropna()
                
                # Find common indices
                common_idx = data1.index.intersection(data2.index)
                
                if len(common_idx) > 1:
                    stat, p_value = wilcoxon(data1[common_idx], data2[common_idx])
                    significant = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    print(f"  {solver1:>8} vs {solver2:<8}: p={p_value:.4f} {significant}")
            except Exception as e:
                print(f"  {solver1:>8} vs {solver2:<8}: Test failed")


def main():
    """Main function to run ATSP solver comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Solve ATSP instances with multiple solvers")
    parser.add_argument("--solvers", nargs='+', choices=list(SOLVERS.keys()),
                       help="Solvers to test (default: all available)")
    parser.add_argument("--pattern", default="*.dat", 
                       help="Instance file pattern (default: *.dat)")
    parser.add_argument("--runs", type=int, default=1,
                       help="Number of runs per instance (default: 1)")
    parser.add_argument("--workers", type=int, default=4,
                       help="Maximum parallel workers (default: 4)")
    parser.add_argument("--sequential", action="store_true",
                       help="Use sequential processing instead of parallel")
    parser.add_argument("--output", 
                       help="Output file prefix (default: auto-generated)")
    parser.add_argument("--analyze-only", 
                       help="Analyze existing results file instead of solving")
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Load and analyze existing results
        print(f"Loading results from {args.analyze_only}")
        with open(args.analyze_only, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results")
        
        analyze_results(results)
        statistical_analysis(results)
        
    else:
        # Solve instances
        print("ATSP SOLVER COMPARISON")
        print("=" * 40)
        print(f"Model file: {MODEL_FILE}")
        print(f"Data directory: {DATA_DIR}")
        print(f"Results directory: {RESULTS_DIR}")
        
        # Check if model file exists
        if not os.path.exists(MODEL_FILE):
            print(f"Error: Model file {MODEL_FILE} not found!")
            return
        
        # Check if data directory exists
        if not os.path.exists(DATA_DIR):
            print(f"Error: Data directory {DATA_DIR} not found!")
            return
        
        # Solve all instances
        results = solve_all_instances(
            solvers_to_test=args.solvers,
            max_workers=args.workers,
            runs_per_instance=args.runs,
            instance_pattern=args.pattern,
            parallel=not args.sequential
        )
        
        if not results:
            print("No results obtained!")
            return
        
        # Save results
        json_file, csv_file = save_results(results, args.output)
        
        # Analyze results
        analyze_results(results)
        statistical_analysis(results)
        
        print(f"\nResults saved to:")
        print(f"  {json_file}")
        print(f"  {csv_file}")


if __name__ == "__main__":
    main()
