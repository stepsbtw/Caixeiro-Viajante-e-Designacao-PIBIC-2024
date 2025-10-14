#!/usr/bin/env python3
"""
Combine and analyze TSP benchmark results from all three size categories
"""

import pandas as pd
import os
import argparse

def combine_results(base_dir="results"):
    """Combine results from small, medium, and large instance benchmarks"""
    
    parts = ["tsp_small", "tsp_medium", "tsp_large"]
    all_detailed = []
    all_final = []
    
    for part in parts:
        part_dir = os.path.join(base_dir, part)
        
        # Load detailed results
        detailed_file = os.path.join(part_dir, "heuristic_detailed_results.json")
        if os.path.exists(detailed_file):
            import json
            with open(detailed_file, 'r') as f:
                detailed_data = json.load(f)
                for record in detailed_data:
                    record['size_category'] = part.replace('tsp_', '')
                all_detailed.extend(detailed_data)
        
        # Load final summary
        final_file = os.path.join(part_dir, "heuristic_results_final.csv")
        if os.path.exists(final_file):
            df = pd.read_csv(final_file)
            df['size_category'] = part.replace('tsp_', '')
            all_final.append(df)
    
    # Save combined results
    combined_dir = os.path.join(base_dir, "tsp_combined")
    os.makedirs(combined_dir, exist_ok=True)
    
    if all_detailed:
        import json
        with open(os.path.join(combined_dir, "combined_detailed_results.json"), 'w') as f:
            json.dump(all_detailed, f, indent=2)
        print(f"✓ Combined detailed results saved: {len(all_detailed)} records")
    
    if all_final:
        combined_df = pd.concat(all_final, ignore_index=True)
        combined_df.to_csv(os.path.join(combined_dir, "combined_final_results.csv"), index=False)
        print(f"✓ Combined final results saved: {len(combined_df)} records")
        
        # Print summary by size category
        print("\nSummary by size category:")
        print("=" * 50)
        
        summary = combined_df.groupby(['size_category', 'method_name']).agg({
            'cost_best': 'mean',
            'runtime_mean': 'mean',
            'successes': 'sum',
            'runs': 'sum'
        }).round(3)
        
        print(summary)
        
        # Save size category summary
        summary.to_csv(os.path.join(combined_dir, "summary_by_size.csv"))
        print(f"\n✓ Size category summary saved")
    
    return combined_dir

def main():
    parser = argparse.ArgumentParser(description="Combine TSP benchmark results from all size categories")
    parser.add_argument('--base-dir', default='results', help='Base results directory')
    args = parser.parse_args()
    
    print("Combining TSP benchmark results from all size categories...")
    combined_dir = combine_results(args.base_dir)
    print(f"\nAll combined results saved in: {combined_dir}/")

if __name__ == "__main__":
    main()
