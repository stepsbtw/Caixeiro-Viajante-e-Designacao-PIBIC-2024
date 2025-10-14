"""
MATLAB TSP GA Results Analyzer
Parses CSV results from MATLAB and provides analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional


class TSPResultsAnalyzer:
    def __init__(self, results_file: str = "results/tsp_ga_results.csv"):
        """
        Initialize analyzer with results file from MATLAB benchmark
        
        Args:
            results_file: Path to CSV results file from MATLAB
        """
        self.results_file = results_file
        self.df = None
        
        if os.path.exists(results_file):
            self.load_results()
        else:
            print(f"Results file not found: {results_file}")
    
    def load_results(self):
        """Load results from CSV file"""
        try:
            self.df = pd.read_csv(self.results_file)
            print(f"Loaded {len(self.df)} results from {self.results_file}")
        except Exception as e:
            print(f"Error loading results: {e}")
    
    def get_summary_stats(self) -> Optional[pd.DataFrame]:
        """Get summary statistics by instance"""
        if self.df is None:
            return None
        
        # Filter successful runs only
        successful = self.df[self.df['Success'] == True].copy()
        
        if len(successful) == 0:
            print("No successful results found")
            return None
        
        # The MATLAB CSV already has summary stats per instance
        summary = successful[['Instance', 'Cities', 'Best_Cost', 'Worst_Cost', 
                            'Avg_Cost', 'Std_Cost', 'Avg_Time', 'Std_Time']].copy()
        
        return summary
    
    def print_summary(self):
        """Print detailed summary of results"""
        if self.df is None:
            print("No results loaded")
            return
        
        summary = self.get_summary_stats()
        
        print(f"\\n{'='*80}")
        print("TSP GENETIC ALGORITHM RESULTS SUMMARY")
        print(f"{'='*80}")
        
        print(f"Total instances processed: {len(summary)}")
        print(f"Total runs: {len(self.df)}")
        print(f"Average runs per instance: {len(self.df) / len(summary):.1f}")
        
        print(f"\\nInstance size distribution:")
        size_dist = summary['Cities'].value_counts().sort_index()
        for cities, count in size_dist.items():
            print(f"  {cities:2d} cities: {count:2d} instances")
        
        print(f"\\n{'Instance':<15} {'Cities':<7} {'Best':<10} {'Avg':<10} {'Std':<8} {'Time':<8}")
        print("-" * 65)
        
        # Sort by number of cities
        for _, row in summary.sort_values('Cities').iterrows():
            print(f"{row['Instance']:<15} {row['Cities']:<7.0f} "
                  f"{row['Best_Cost']:<10.0f} {row['Avg_Cost']:<10.1f} "
                  f"{row['Std_Cost']:<8.1f} {row['Avg_Time']:<8.3f}")
        
        # Overall statistics
        print(f"\\nOverall Statistics:")
        print(f"  Average cost improvement (best vs avg): {((summary['Avg_Cost'] - summary['Best_Cost']) / summary['Best_Cost'] * 100).mean():.1f}%")
        print(f"  Average execution time: {summary['Avg_Time'].mean():.3f} ± {summary['Avg_Time'].std():.3f} seconds")
        
        # Performance by size
        print(f"\\nPerformance by problem size:")
        size_groups = summary.groupby(pd.cut(summary['Cities'], bins=[0, 30, 50, 100, float('inf')], 
                                           labels=['Small (≤30)', 'Medium (31-50)', 'Large (51-100)', 'Very Large (>100)']))
        
        for size_range, group in size_groups:
            if len(group) > 0:
                print(f"  {size_range}: {len(group)} instances, avg time {group['Avg_Time'].mean():.3f}s")
    
    def plot_results(self, save_plots: bool = True):
        """Create visualization plots of the results"""
        if self.df is None:
            print("No results to plot")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            summary = self.get_summary_stats()
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('TSP Genetic Algorithm Results Analysis', fontsize=16)
            
            # 1. Cost vs Problem Size
            axes[0, 0].scatter(summary['Cities'], summary['Best_Cost'], alpha=0.7, color='blue', label='Best Cost')
            axes[0, 0].scatter(summary['Cities'], summary['Avg_Cost'], alpha=0.7, color='red', label='Avg Cost')
            axes[0, 0].set_xlabel('Number of Cities')
            axes[0, 0].set_ylabel('Tour Cost')
            axes[0, 0].set_title('Tour Cost vs Problem Size')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Execution Time vs Problem Size
            axes[0, 1].scatter(summary['Cities'], summary['Avg_Time'], alpha=0.7, color='green')
            axes[0, 1].set_xlabel('Number of Cities')
            axes[0, 1].set_ylabel('Execution Time (seconds)')
            axes[0, 1].set_title('Execution Time vs Problem Size')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Cost Distribution Histogram
            axes[1, 0].hist(self.df['Cost'], bins=30, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].set_xlabel('Tour Cost')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Distribution of Tour Costs')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Solution Quality (Best vs Avg Cost)
            improvement = (summary['Avg_Cost'] - summary['Best_Cost']) / summary['Best_Cost'] * 100
            axes[1, 1].scatter(summary['Cities'], improvement, alpha=0.7, color='purple')
            axes[1, 1].set_xlabel('Number of Cities')
            axes[1, 1].set_ylabel('Improvement from Avg to Best (%)')
            axes[1, 1].set_title('Solution Quality Consistency')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                plots_dir = "results"
                os.makedirs(plots_dir, exist_ok=True)
                plt.savefig(os.path.join(plots_dir, 'tsp_ga_analysis.png'), dpi=300, bbox_inches='tight')
                print(f"Plots saved to {os.path.join(plots_dir, 'tsp_ga_analysis.png')}")
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Cannot create plots.")
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    def export_summary(self, filename: str = "results/tsp_ga_summary.csv"):
        """Export summary statistics to CSV"""
        if self.df is None:
            print("No results to export")
            return
        
        summary = self.get_summary_stats()
        summary.to_csv(filename, index=False)
        print(f"Summary exported to {filename}")
    
    def compare_with_known_optimal(self, optimal_file: str = None):
        """Compare results with known optimal solutions if available"""
        # This would require a file with known optimal solutions
        # For now, just placeholder
        print("Optimal solution comparison not implemented yet")
        print("To implement this, provide a file with known optimal costs for TSPLIB instances")


def main():
    """Main function for results analysis"""
    print("TSP GA Results Analyzer")
    print("=" * 30)
    
    # Check if results exist
    results_file = "results/tsp_ga_matlab_results.csv"
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        print("\\nTo generate results:")
        print("1. Open MATLAB")
        print("2. Navigate to the project directory")
        print("3. Run: tsp_ga_matlab_benchmark")
        print("   OR")
        print("   Run: test_ga_small  (for quick test)")
        return
    
    # Analyze results
    analyzer = TSPResultsAnalyzer(results_file)
    analyzer.print_summary()
    analyzer.export_summary()
    
    # Create plots if matplotlib is available
    try:
        analyzer.plot_results()
    except:
        print("\\nCould not create plots (matplotlib may not be available)")


if __name__ == "__main__":
    main()
