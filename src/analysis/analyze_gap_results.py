"""
GAP (Generalized Assignment Problem) Results Analyzer
Parses detailed JSON results (multiple solvers, multiple runs) and provides
summary tables, textual reports, and optional plots.

Input file format (JSON list of objects):
[
  {
    "instance": "gap1_p5.dat",
    "instance_size": 75,
    "solver": "highs",
    "run": 1,
    "status": "optimal",
    "objective": 580.0,
    "runtime": 0.025,
    "gap": 0.0,
    "nodes": 0,
    "iterations": 10,
    "presolve_time": 0.001,
    "solve_time": 0.024
  },
  ...
]

This script generates:
1. Summary statistics per solver
2. Performance comparison tables
3. Statistical significance tests
4. Best results per instance
5. Runtime analysis
6. Optional visualizations
"""

import json
import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Statistical tests
try:
    from scipy.stats import friedmanchisquare, wilcoxon, mannwhitneyu
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Statistical tests will be skipped.")

# Plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    plt.style.use('default')
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Plots will be skipped.")


class GAPResultsAnalyzer:
    """Analyzer for GAP solver results."""
    
    def __init__(self, results_file: str):
        """Initialize analyzer with results file."""
        self.results_file = results_file
        self.df = None
        self.load_results()
    
    def load_results(self):
        """Load results from JSON file."""
        try:
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("Results file must contain a list of result objects")
            
            self.df = pd.DataFrame(data)
            print(f"Loaded {len(self.df)} results from {self.results_file}")
            
            # Validate required columns
            required_cols = ['instance', 'solver', 'objective', 'runtime', 'status']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert numeric columns
            numeric_cols = ['objective', 'runtime', 'gap', 'nodes', 'iterations']
            for col in numeric_cols:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Add derived columns
            self.df['solved'] = self.df['status'].isin(['optimal', 'feasible'])
            self.df['instance_name'] = self.df['instance'].str.replace('.dat', '', regex=False)
            
            if 'instance_size' not in self.df.columns:
                # Try to extract size from instance name
                self.df['instance_size'] = self.df['instance_name'].str.extract(r'(\d+)').astype(float)
            
        except Exception as e:
            print(f"Error loading results: {e}")
            sys.exit(1)
    
    def print_overview(self):
        """Print overview of loaded results."""
        print("\n" + "="*60)
        print("RESULTS OVERVIEW")
        print("="*60)
        
        print(f"Total results: {len(self.df)}")
        print(f"Instances: {self.df['instance'].nunique()}")
        print(f"Solvers: {list(self.df['solver'].unique())}")
        
        if 'run' in self.df.columns:
            print(f"Runs per instance-solver: {self.df['run'].nunique()}")
        
        print(f"Status distribution:")
        status_counts = self.df['status'].value_counts()
        for status, count in status_counts.items():
            pct = count / len(self.df) * 100
            print(f"  {status}: {count} ({pct:.1f}%)")
    
    def analyze_solvers(self) -> pd.DataFrame:
        """Analyze solver performance."""
        print("\n" + "="*60)
        print("SOLVER PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Group by solver and calculate statistics
        solver_stats = []
        
        for solver in self.df['solver'].unique():
            solver_data = self.df[self.df['solver'] == solver]
            solved_data = solver_data[solver_data['solved']]
            
            stats = {
                'solver': solver,
                'total_instances': len(solver_data),
                'solved': len(solved_data),
                'solve_rate': len(solved_data) / len(solver_data) * 100,
                'avg_objective': solved_data['objective'].mean(),
                'std_objective': solved_data['objective'].std(),
                'avg_runtime': solver_data['runtime'].mean(),
                'std_runtime': solver_data['runtime'].std(),
                'max_runtime': solver_data['runtime'].max(),
                'timeout_count': len(solver_data[solver_data['status'] == 'time_limit'])
            }
            
            if 'gap' in solved_data.columns:
                stats['avg_gap'] = solved_data['gap'].mean()
                stats['max_gap'] = solved_data['gap'].max()
            
            if 'nodes' in solved_data.columns:
                stats['avg_nodes'] = solved_data['nodes'].mean()
                stats['max_nodes'] = solved_data['nodes'].max()
            
            solver_stats.append(stats)
        
        solver_df = pd.DataFrame(solver_stats)
        
        # Sort by solve rate, then by average objective
        solver_df = solver_df.sort_values(['solve_rate', 'avg_objective'], 
                                         ascending=[False, False])
        
        print(solver_df.round(3))
        
        return solver_df
    
    def find_best_results(self) -> pd.DataFrame:
        """Find best result for each instance."""
        print("\n" + "="*60)
        print("BEST RESULTS PER INSTANCE")
        print("="*60)
        
        # Filter to solved instances only
        solved_df = self.df[self.df['solved']].copy()
        
        if len(solved_df) == 0:
            print("No solved instances found!")
            return pd.DataFrame()
        
        # Find best objective for each instance
        best_results = solved_df.loc[solved_df.groupby('instance')['objective'].idxmax()]
        
        # Select relevant columns
        columns = ['instance', 'solver', 'objective', 'runtime', 'status']
        if 'gap' in best_results.columns:
            columns.append('gap')
        if 'nodes' in best_results.columns:
            columns.append('nodes')
        
        best_df = best_results[columns].copy()
        best_df = best_df.sort_values('objective', ascending=False)
        
        print(best_df.round(3))
        
        # Summary statistics
        print(f"\nSummary:")
        print(f"Average best objective: {best_df['objective'].mean():.2f}")
        print(f"Average runtime: {best_df['runtime'].mean():.3f}s")
        
        solver_wins = best_df['solver'].value_counts()
        print(f"\nSolver wins:")
        for solver, wins in solver_wins.items():
            print(f"  {solver}: {wins} instances")
        
        return best_df
    
    def runtime_analysis(self):
        """Analyze runtime patterns."""
        print("\n" + "="*60)
        print("RUNTIME ANALYSIS")
        print("="*60)
        
        runtime_stats = self.df.groupby('solver')['runtime'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(3)
        
        print("Runtime statistics by solver:")
        print(runtime_stats)
        
        # Identify slow instances
        if len(self.df) > 0:
            runtime_threshold = self.df['runtime'].quantile(0.9)
            slow_instances = self.df[self.df['runtime'] > runtime_threshold]
            
            if len(slow_instances) > 0:
                print(f"\nSlowest instances (runtime > {runtime_threshold:.3f}s):")
                slow_summary = slow_instances[['instance', 'solver', 'runtime', 'status']].sort_values('runtime', ascending=False)
                print(slow_summary.head(10))
    
    def statistical_tests(self):
        """Perform statistical significance tests."""
        if not SCIPY_AVAILABLE:
            print("\nStatistical tests skipped (scipy not available)")
            return
        
        print("\n" + "="*60)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("="*60)
        
        # Get common instances solved by all solvers
        solved_df = self.df[self.df['solved']].copy()
        
        if len(solved_df) == 0:
            print("No solved instances for statistical tests")
            return
        
        # Find instances solved by multiple solvers
        instance_solver_counts = solved_df.groupby('instance')['solver'].nunique()
        common_instances = instance_solver_counts[instance_solver_counts > 1].index
        
        if len(common_instances) == 0:
            print("No instances solved by multiple solvers")
            return
        
        print(f"Testing on {len(common_instances)} instances solved by multiple solvers")
        
        # Prepare data for statistical tests
        solvers = solved_df['solver'].unique()
        if len(solvers) < 2:
            print("Need at least 2 solvers for comparison")
            return
        
        # Friedman test for multiple solvers
        if len(solvers) > 2:
            try:
                solver_objectives = []
                for solver in solvers:
                    solver_data = solved_df[
                        (solved_df['solver'] == solver) & 
                        (solved_df['instance'].isin(common_instances))
                    ]
                    if len(solver_data) > 0:
                        objectives = solver_data.set_index('instance')['objective']
                        solver_objectives.append(objectives)
                
                if len(solver_objectives) > 2:
                    # Align data
                    aligned_data = pd.concat(solver_objectives, axis=1)
                    aligned_data.columns = solvers[:len(solver_objectives)]
                    aligned_data = aligned_data.dropna()
                    
                    if len(aligned_data) > 0:
                        statistic, p_value = friedmanchisquare(*[aligned_data[col] for col in aligned_data.columns])
                        print(f"\nFriedman test (multiple solvers):")
                        print(f"  Statistic: {statistic:.4f}")
                        print(f"  p-value: {p_value:.4f}")
                        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
            except Exception as e:
                print(f"Friedman test failed: {e}")
        
        # Pairwise Wilcoxon tests
        print(f"\nPairwise Wilcoxon signed-rank tests:")
        for i, solver1 in enumerate(solvers):
            for solver2 in solvers[i+1:]:
                try:
                    data1 = solved_df[
                        (solved_df['solver'] == solver1) & 
                        (solved_df['instance'].isin(common_instances))
                    ]
                    data2 = solved_df[
                        (solved_df['solver'] == solver2) & 
                        (solved_df['instance'].isin(common_instances))
                    ]
                    
                    # Find common instances between these two solvers
                    common_inst = set(data1['instance']) & set(data2['instance'])
                    
                    if len(common_inst) > 1:
                        obj1 = data1[data1['instance'].isin(common_inst)].set_index('instance')['objective']
                        obj2 = data2[data2['instance'].isin(common_inst)].set_index('instance')['objective']
                        
                        # Align by instance
                        aligned = pd.concat([obj1, obj2], axis=1).dropna()
                        
                        if len(aligned) > 1:
                            statistic, p_value = wilcoxon(aligned.iloc[:, 0], aligned.iloc[:, 1])
                            print(f"  {solver1} vs {solver2}: p={p_value:.4f} ({'*' if p_value < 0.05 else ''})")
                
                except Exception as e:
                    print(f"  {solver1} vs {solver2}: Test failed ({e})")
    
    def create_plots(self, output_dir: str):
        """Create visualization plots."""
        if not PLOTTING_AVAILABLE:
            print("\nPlots skipped (matplotlib not available)")
            return
        
        print(f"\nCreating plots in {output_dir}/...")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Solver performance comparison
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Solve rates
        plt.subplot(2, 2, 1)
        solver_stats = self.df.groupby('solver').agg({
            'solved': 'sum',
            'instance': 'count'
        })
        solver_stats['solve_rate'] = solver_stats['solved'] / solver_stats['instance'] * 100
        
        plt.bar(solver_stats.index, solver_stats['solve_rate'])
        plt.title('Solve Rate by Solver')
        plt.ylabel('Solve Rate (%)')
        plt.xticks(rotation=45)
        
        # Subplot 2: Runtime distribution
        plt.subplot(2, 2, 2)
        for solver in self.df['solver'].unique():
            solver_data = self.df[self.df['solver'] == solver]
            plt.hist(solver_data['runtime'], alpha=0.6, label=solver, bins=20)
        plt.xlabel('Runtime (s)')
        plt.ylabel('Frequency')
        plt.title('Runtime Distribution')
        plt.legend()
        plt.yscale('log')
        
        # Subplot 3: Objective values
        plt.subplot(2, 2, 3)
        solved_df = self.df[self.df['solved']]
        if len(solved_df) > 0:
            solved_df.boxplot(column='objective', by='solver', ax=plt.gca())
            plt.title('Objective Values by Solver')
            plt.suptitle('')  # Remove default title
        
        # Subplot 4: Runtime vs Instance Size
        plt.subplot(2, 2, 4)
        if 'instance_size' in self.df.columns:
            for solver in self.df['solver'].unique():
                solver_data = self.df[self.df['solver'] == solver]
                plt.scatter(solver_data['instance_size'], solver_data['runtime'], 
                          alpha=0.6, label=solver)
            plt.xlabel('Instance Size')
            plt.ylabel('Runtime (s)')
            plt.title('Runtime vs Instance Size')
            plt.legend()
            plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/solver_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Performance profile
        if len(self.df['solver'].unique()) > 1:
            self._create_performance_profile(output_dir)
    
    def _create_performance_profile(self, output_dir: str):
        """Create performance profile plot."""
        solved_df = self.df[self.df['solved']].copy()
        
        if len(solved_df) == 0:
            return
        
        # Create performance profile based on runtime
        plt.figure(figsize=(10, 6))
        
        solvers = solved_df['solver'].unique()
        tau_max = 10  # Maximum ratio
        tau_range = np.logspace(0, np.log10(tau_max), 100)
        
        for solver in solvers:
            solver_data = solved_df[solved_df['solver'] == solver]
            
            # For each instance, find the best runtime among all solvers
            rho_values = []
            
            for tau in tau_range:
                count = 0
                total = 0
                
                for instance in solver_data['instance'].unique():
                    instance_data = solved_df[solved_df['instance'] == instance]
                    if len(instance_data) > 1:  # Multiple solvers solved this instance
                        best_runtime = instance_data['runtime'].min()
                        solver_runtime = solver_data[solver_data['instance'] == instance]['runtime']
                        
                        if len(solver_runtime) > 0:
                            ratio = solver_runtime.iloc[0] / best_runtime
                            if ratio <= tau:
                                count += 1
                            total += 1
                
                rho = count / total if total > 0 else 0
                rho_values.append(rho)
            
            plt.plot(tau_range, rho_values, label=solver, linewidth=2)
        
        plt.xlabel('Performance ratio τ')
        plt.ylabel('ρ(τ) - Fraction of instances')
        plt.title('Performance Profile (Runtime)')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(1, tau_max)
        plt.ylim(0, 1)
        
        plt.savefig(f"{output_dir}/performance_profile.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, output_file: str):
        """Generate comprehensive text report."""
        print(f"\nGenerating report: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("GAP SOLVER RESULTS ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Overview
            f.write("OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Results file: {self.results_file}\n")
            f.write(f"Total results: {len(self.df)}\n")
            f.write(f"Instances: {self.df['instance'].nunique()}\n")
            f.write(f"Solvers: {', '.join(self.df['solver'].unique())}\n")
            
            if 'run' in self.df.columns:
                f.write(f"Runs per instance-solver: {self.df['run'].nunique()}\n")
            
            f.write(f"\nStatus distribution:\n")
            status_counts = self.df['status'].value_counts()
            for status, count in status_counts.items():
                pct = count / len(self.df) * 100
                f.write(f"  {status}: {count} ({pct:.1f}%)\n")
            
            # Solver performance
            f.write("\n\nSOLVER PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            
            solver_stats = self.analyze_solvers()
            f.write(solver_stats.to_string())
            
            # Best results
            f.write("\n\nBEST RESULTS PER INSTANCE\n")
            f.write("-" * 35 + "\n")
            
            best_results = self.find_best_results()
            if len(best_results) > 0:
                f.write(best_results.to_string())
                f.write(f"\n\nAverage best objective: {best_results['objective'].mean():.2f}\n")
                f.write(f"Average runtime: {best_results['runtime'].mean():.3f}s\n")
            
            # Runtime analysis
            f.write("\n\nRUNTIME ANALYSIS\n")
            f.write("-" * 25 + "\n")
            
            runtime_stats = self.df.groupby('solver')['runtime'].agg([
                'count', 'mean', 'std', 'min', 'max', 'median'
            ]).round(3)
            f.write(runtime_stats.to_string())
    
    def run_full_analysis(self, output_dir: str = "analysis_results"):
        """Run complete analysis and generate all outputs."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Running full GAP results analysis...")
        
        # Print all analyses
        self.print_overview()
        solver_stats = self.analyze_solvers()
        best_results = self.find_best_results()
        self.runtime_analysis()
        self.statistical_tests()
        
        # Generate outputs
        self.generate_report(f"{output_dir}/analysis_report.txt")
        
        # Save dataframes
        solver_stats.to_csv(f"{output_dir}/solver_performance.csv", index=False)
        if len(best_results) > 0:
            best_results.to_csv(f"{output_dir}/best_results.csv", index=False)
        
        # Create plots
        self.create_plots(output_dir)
        
        print(f"\nAnalysis complete! Results saved to {output_dir}/")
        print(f"Key files:")
        print(f"  - analysis_report.txt: Comprehensive text report")
        print(f"  - solver_performance.csv: Solver statistics")
        print(f"  - best_results.csv: Best result per instance")
        if PLOTTING_AVAILABLE:
            print(f"  - solver_comparison.png: Performance plots")
            print(f"  - performance_profile.png: Performance profile")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze GAP solver results")
    parser.add_argument("results_file", help="JSON file with solver results")
    parser.add_argument("--output-dir", default="analysis_results", 
                       help="Output directory for analysis results")
    parser.add_argument("--no-plots", action="store_true", 
                       help="Skip generating plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: Results file {args.results_file} not found")
        sys.exit(1)
    
    # Create analyzer and run analysis
    analyzer = GAPResultsAnalyzer(args.results_file)
    
    if args.no_plots:
        global PLOTTING_AVAILABLE
        PLOTTING_AVAILABLE = False
    
    analyzer.run_full_analysis(args.output_dir)


if __name__ == "__main__":
    main()
