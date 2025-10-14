#!/usr/bin/env python3
"""
Benchmark GAP Heuristics
Comprehensive benchmarking script for GAP heuristic methods.
"""

import os
import time
import glob
import pandas as pd
import numpy as np
from gap_heuristic import GAPParser, GAPSolver, GAPResult


def benchmark_gap_heuristics():
    """Benchmark different GAP heuristic combinations."""
    
    # Configuration
    INSTANCES_DIR = "INSTANCES/gap"
    RESULTS_DIR = "benchmark_results"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Heuristic methods to test
    construction_methods = [
        'greedy_ratio',
        'lsa'  # Linear Sum Assignment (Hungarian algorithm)
    ]
    
    improvement_methods = [
        'none',
        'local_search',
        'two_opt',
        'vns',
        'simulated_annealing'
    ]
    
    # Discover all instance files and their problem indices
    all_files = sorted([f for f in os.listdir(INSTANCES_DIR) if f.endswith('.txt')])
    instance_problem_list = []  # list of (filename, problem_index)
    for fname in all_files:
        fpath = os.path.join(INSTANCES_DIR, fname)
        try:
            # Read first non-empty line to get number of problems
            with open(fpath, 'r') as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]
            num_problems = int(lines[0]) if lines else 1
        except Exception:
            num_problems = 1
        for pi in range(max(1, num_problems)):
            instance_problem_list.append((fname, pi))
    
    results = []
    total_combinations = len(construction_methods) * len(improvement_methods) * len(instance_problem_list)
    current_combination = 0
    
    print(f"Starting GAP heuristics benchmark...")
    print(f"Testing {len(construction_methods)} construction × {len(improvement_methods)} improvement methods")
    print(f"on {len(instance_problem_list)} instance-problems = {total_combinations} total combinations\n")
    
    for construction in construction_methods:
        for improvement in improvement_methods:
            print(f"\nTesting: {construction} + {improvement}")
            print(f"{'='*50}")
            
            for instance_file, problem_index in instance_problem_list:
                current_combination += 1
                instance_path = os.path.join(INSTANCES_DIR, instance_file)
                
                if not os.path.exists(instance_path):
                    print(f"  ⚠️  Instance not found: {instance_file}")
                    continue
                
                try:
                    # Parse instance
                    instance = GAPParser.parse_gap_file(instance_path, problem_index)
                    
                    # Solve with current method combination
                    start_time = time.time()
                    solver = GAPSolver(instance, seed=42)  # Fixed seed for reproducibility
                    result = solver.solve(construction, improvement, multi_start=1)
                    total_time = time.time() - start_time
                    
                    result.instance_name = f"{instance_file}:p{problem_index}"
                    result.total_runtime = total_time
                    
                    # Store results
                    results.append({
                        'instance': result.instance_name,
                        'construction': construction,
                        'improvement': improvement,
                        'method': f"{construction}+{improvement}",
                        'n_workers': instance.n_workers,
                        'n_tasks': instance.n_tasks,
                        'problem_type': 'MAX' if instance.is_maximization else 'MIN',
                        'objective_value': result.objective_value,
                        'is_feasible': result.is_feasible,
                        'workers_used': result.workers_used,
                        'max_capacity_utilization': result.max_capacity_utilization,
                        'construction_value': result.construction_value,
                        'improvement_value': result.objective_value - result.construction_value,
                        'improvement_iterations': result.improvement_iterations,
                        'total_runtime': result.total_runtime,
                        'construction_time': result.construction_time,
                        'improvement_time': result.improvement_time
                    })
                    
                    # Progress indicator
                    status = "✅" if result.is_feasible else "❌"
                    label = f"{instance_file}:p{problem_index}"
                    print(f"  {status} {label:20} | Obj: {result.objective_value:8.1f} | "
                          f"Time: {result.total_runtime:6.3f}s | "
                          f"Iter: {result.improvement_iterations:3d} | "
                          f"({current_combination}/{total_combinations})")
                    
                except Exception as e:
                    print(f"  ❌ {instance_file:12} | Error: {str(e)}")
                    continue
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    if df.empty:
        print("No results collected!")
        return
    
    # Save detailed results
    detailed_results_file = os.path.join(RESULTS_DIR, "gap_heuristics_benchmark_results.csv")
    df.to_csv(detailed_results_file, index=False)
    print(f"\n📊 Detailed results saved to: {detailed_results_file}")
    
    # Analyze results
    analyze_results(df, RESULTS_DIR)


def analyze_results(df, results_dir):
    """Analyze and summarize benchmark results."""
    
    # Filter only feasible solutions for most analyses
    feasible_df = df[df['is_feasible'] == True]
    
    if feasible_df.empty:
        print("⚠️  No feasible solutions found!")
        return
    
    print(f"\n📈 ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Total combinations tested: {len(df)}")
    print(f"Feasible solutions: {len(feasible_df)} ({len(feasible_df)/len(df)*100:.1f}%)")
    
    # 1. Method Ranking Analysis
    print(f"\n🏆 METHOD RANKING (by average objective value)")
    print(f"{'='*60}")
    
    method_performance = feasible_df.groupby('method').agg({
        'objective_value': ['mean', 'std', 'count'],
        'total_runtime': 'mean',
        'improvement_iterations': 'mean',
        'max_capacity_utilization': 'mean'
    }).round(3)
    
    method_performance.columns = ['avg_objective', 'std_objective', 'num_instances', 
                                'avg_runtime', 'avg_iterations', 'avg_utilization']
    method_performance = method_performance.sort_values('avg_objective', ascending=False)
    
    print(method_performance.to_string())
    
    # Save method ranking
    method_ranking_file = os.path.join(results_dir, "gap_method_ranking.csv")
    method_performance.to_csv(method_ranking_file)
    
    # 2. Construction Methods Comparison
    print(f"\n🔨 CONSTRUCTION METHODS COMPARISON")
    print(f"{'='*50}")
    
    construction_perf = feasible_df.groupby('construction').agg({
        'construction_value': ['mean', 'std'],
        'construction_time': 'mean',
        'is_feasible': 'count'
    }).round(3)
    
    construction_perf.columns = ['avg_construction_value', 'std_construction_value', 
                               'avg_construction_time', 'num_solutions']
    construction_perf = construction_perf.sort_values('avg_construction_value', ascending=False)
    
    print(construction_perf.to_string())
    
    # Save construction comparison
    construction_file = os.path.join(results_dir, "gap_construction_comparison.csv")
    construction_perf.to_csv(construction_file)
    
    # 3. Improvement Methods Comparison
    print(f"\n⚡ IMPROVEMENT METHODS COMPARISON")
    print(f"{'='*50}")
    
    improvement_perf = feasible_df.groupby('improvement').agg({
        'improvement_value': ['mean', 'std'],
        'improvement_time': 'mean',
        'improvement_iterations': 'mean'
    }).round(3)
    
    improvement_perf.columns = ['avg_improvement', 'std_improvement', 
                              'avg_improvement_time', 'avg_iterations']
    improvement_perf = improvement_perf.sort_values('avg_improvement', ascending=False)
    
    print(improvement_perf.to_string())
    
    # Save improvement comparison
    improvement_file = os.path.join(results_dir, "gap_improvement_comparison.csv")
    improvement_perf.to_csv(improvement_file)
    
    # 4. Instance-wise Best Results
    print(f"\n🎯 BEST RESULTS PER INSTANCE")
    print(f"{'='*40}")
    
    instance_best = feasible_df.loc[feasible_df.groupby('instance')['objective_value'].idxmax()]
    instance_summary = instance_best[['instance', 'method', 'objective_value', 'total_runtime', 
                                    'workers_used', 'max_capacity_utilization']].round(3)
    
    print(instance_summary.to_string(index=False))
    
    # Save instance best results
    instance_best_file = os.path.join(results_dir, "gap_instance_best_results.csv")
    instance_summary.to_csv(instance_best_file, index=False)
    
    # 5. Problem Type Analysis (MAX vs MIN)
    print(f"\n📊 PROBLEM TYPE ANALYSIS")
    print(f"{'='*30}")
    
    type_analysis = feasible_df.groupby('problem_type').agg({
        'objective_value': ['mean', 'std', 'count'],
        'total_runtime': 'mean'
    }).round(3)
    
    type_analysis.columns = ['avg_objective', 'std_objective', 'num_instances', 'avg_runtime']
    print(type_analysis.to_string())
    
    # 6. Summary Statistics
    print(f"\n📋 SUMMARY STATISTICS")
    print(f"{'='*30}")
    print(f"Average objective value: {feasible_df['objective_value'].mean():.2f}")
    print(f"Average runtime: {feasible_df['total_runtime'].mean():.3f}s")
    print(f"Average workers used: {feasible_df['workers_used'].mean():.1f}")
    print(f"Average capacity utilization: {feasible_df['max_capacity_utilization'].mean():.1%}")
    print(f"Average improvement iterations: {feasible_df['improvement_iterations'].mean():.1f}")
    
    # Best overall method
    best_method = method_performance.index[0]
    best_avg_obj = method_performance.iloc[0]['avg_objective']
    print(f"\n🥇 BEST OVERALL METHOD: {best_method}")
    print(f"   Average objective value: {best_avg_obj:.2f}")
    
    # Create summary file
    summary_file = os.path.join(results_dir, "gap_heuristics_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("GAP HEURISTICS BENCHMARK SUMMARY\n")
        f.write("="*40 + "\n\n")
        f.write(f"Total combinations tested: {len(df)}\n")
        f.write(f"Feasible solutions: {len(feasible_df)} ({len(feasible_df)/len(df)*100:.1f}%)\n")
        f.write(f"Best method: {best_method}\n")
        f.write(f"Best average objective: {best_avg_obj:.2f}\n\n")
        
        f.write("TOP 5 METHODS:\n")
        f.write("-" * 20 + "\n")
        for i, (method, row) in enumerate(method_performance.head().iterrows(), 1):
            f.write(f"{i}. {method}: {row['avg_objective']:.2f} (±{row['std_objective']:.2f})\n")
    
    print(f"\n📄 Summary saved to: {summary_file}")


if __name__ == "__main__":
    benchmark_gap_heuristics()
