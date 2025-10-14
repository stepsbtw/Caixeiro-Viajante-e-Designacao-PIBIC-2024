#!/usr/bin/env python3
"""
TSP Construction Benchmark Analysis

This script analyzes the TSP construction benchmark results and generates
summary statistics similar to the GAP analysis:
- Method summaries (construction + improvement combinations)
- Construction summaries (averaged across improvements)
- Improvement summaries (averaged across constructions)
"""

import argparse
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def load_benchmark_data(results_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three CSV files from benchmark results"""
    
    construction_only = pd.read_csv(os.path.join(results_dir, "construction_only.csv"))
    construction_2opt = pd.read_csv(os.path.join(results_dir, "construction_2opt.csv"))
    construction_lk = pd.read_csv(os.path.join(results_dir, "construction_lk.csv"))
    
    return construction_only, construction_2opt, construction_lk


def prepare_method_data(construction_only: pd.DataFrame, 
                       construction_2opt: pd.DataFrame,
                       construction_lk: pd.DataFrame) -> pd.DataFrame:
    """Prepare unified dataset with all methods"""
    
    # Add method column for each dataset
    construction_only_clean = construction_only.copy()
    construction_only_clean['method'] = construction_only_clean['construction_method'] + '+none'
    construction_only_clean['final_cost'] = construction_only_clean['cost']
    construction_only_clean['total_time'] = construction_only_clean['construction_time']
    construction_only_clean['improvement_method'] = 'none'
    
    construction_2opt_clean = construction_2opt.copy()
    construction_2opt_clean['method'] = construction_2opt_clean['construction_method'] + '+2opt'
    construction_2opt_clean['improvement_method'] = '2opt'
    
    construction_lk_clean = construction_lk.copy()
    construction_lk_clean['method'] = construction_lk_clean['construction_method'] + '+lk'
    construction_lk_clean['improvement_method'] = 'lk'
    
    # Select common columns
    common_cols = ['instance', 'n', 'construction_method', 'run', 'final_cost', 
                   'gap_percent', 'total_time', 'bks', 'method', 'improvement_method']
    
    # Ensure all dataframes have the required columns
    for df in [construction_only_clean, construction_2opt_clean, construction_lk_clean]:
        for col in common_cols:
            if col not in df.columns:
                df[col] = np.nan
    
    combined = pd.concat([
        construction_only_clean[common_cols],
        construction_2opt_clean[common_cols], 
        construction_lk_clean[common_cols]
    ], ignore_index=True)
    
    # Clean gap_percent - convert to numeric and handle missing BKS
    combined['gap_percent'] = pd.to_numeric(combined['gap_percent'], errors='coerce')
    combined['gap'] = combined['gap_percent'] / 100.0  # Convert percentage to ratio
    
    return combined


def calculate_summary_stats(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Calculate summary statistics for given grouping columns"""
    
    # Filter out rows with missing gap values (no BKS available)
    df_with_gap = df.dropna(subset=['gap'])
    
    # Group and calculate statistics
    summary = df_with_gap.groupby(group_cols).agg({
        'gap': ['mean', 'std', 'median', 'count'],
        'total_time': ['mean', 'std', 'median']
    }).round(6)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={
        'gap_mean': 'mean_gap',
        'gap_std': 'std_gap', 
        'gap_median': 'median_gap',
        'gap_count': 'count',
        'total_time_mean': 'mean_time',
        'total_time_std': 'std_time',
        'total_time_median': 'median_time'
    })
    
    # Reset index to make grouping columns regular columns
    summary = summary.reset_index()
    
    # Fill NaN std values with 0 (when count=1)
    summary['std_gap'] = summary['std_gap'].fillna(0)
    summary['std_time'] = summary['std_time'].fillna(0)
    
    return summary


def generate_summaries(results_dir: str, output_dir: str) -> None:
    """Generate all summary files"""
    
    print(f"Loading benchmark data from {results_dir}")
    construction_only, construction_2opt, construction_lk = load_benchmark_data(results_dir)
    
    print("Preparing unified dataset...")
    combined_data = prepare_method_data(construction_only, construction_2opt, construction_lk)
    
    # Filter out instances without BKS for gap calculations
    instances_with_bks = combined_data.dropna(subset=['bks'])['instance'].unique()
    print(f"Found {len(instances_with_bks)} instances with BKS values")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Method-level summaries (construction + improvement)
    print("Generating method summaries...")
    method_summary = calculate_summary_stats(combined_data, ['method'])
    method_summary = method_summary.sort_values('mean_gap')
    method_file = os.path.join(output_dir, "tsp_method_summary.csv")
    method_summary.to_csv(method_file, index=False)
    print(f"  ✓ Written to {method_file}")
    
    # 2. Construction-level summaries (averaged across improvements)
    print("Generating construction summaries...")
    construction_summary = calculate_summary_stats(combined_data, ['construction_method'])
    construction_summary = construction_summary.sort_values('mean_gap') 
    construction_file = os.path.join(output_dir, "tsp_construction_summary.csv")
    construction_summary.to_csv(construction_file, index=False)
    print(f"  ✓ Written to {construction_file}")
    
    # 3. Improvement-level summaries (averaged across constructions)
    print("Generating improvement summaries...")
    improvement_summary = calculate_summary_stats(combined_data, ['improvement_method'])
    improvement_summary = improvement_summary.sort_values('mean_gap')
    improvement_file = os.path.join(output_dir, "tsp_improvement_summary.csv")
    improvement_summary.to_csv(improvement_file, index=False)
    print(f"  ✓ Written to {improvement_file}")
    
    # 4. Detailed summary with both construction and improvement
    print("Generating detailed construction+improvement summaries...")
    detailed_summary = calculate_summary_stats(combined_data, ['construction_method', 'improvement_method'])
    detailed_summary = detailed_summary.sort_values('mean_gap')
    detailed_file = os.path.join(output_dir, "tsp_detailed_summary.csv")
    detailed_summary.to_csv(detailed_file, index=False)
    print(f"  ✓ Written to {detailed_file}")
    
    # Print some key insights
    print("\n" + "="*60)
    print("TSP BENCHMARK SUMMARY STATISTICS")
    print("="*60)
    
    print("\nBest Methods (by mean gap):")
    print(method_summary[['method', 'mean_gap', 'mean_time', 'count']].head(5).to_string(index=False))
    
    print("\nConstruction Methods (by mean gap):")
    print(construction_summary[['construction_method', 'mean_gap', 'mean_time', 'count']].to_string(index=False))
    
    print("\nImprovement Methods (by mean gap):")
    print(improvement_summary[['improvement_method', 'mean_gap', 'mean_time', 'count']].to_string(index=False))
    
    print(f"\nAll detailed results written to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Analyze TSP construction benchmark results")
    parser.add_argument('--results-dir', default='results/construction_benchmark_full',
                       help='Directory containing benchmark CSV files')
    parser.add_argument('--output-dir', default='results/tsp_analysis',
                       help='Output directory for summary files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory {args.results_dir} not found")
        return
    
    # Check if required files exist
    required_files = ["construction_only.csv", "construction_2opt.csv", "construction_lk.csv"]
    for file in required_files:
        file_path = os.path.join(args.results_dir, file)
        if not os.path.exists(file_path):
            print(f"Error: Required file {file_path} not found")
            return
    
    generate_summaries(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
