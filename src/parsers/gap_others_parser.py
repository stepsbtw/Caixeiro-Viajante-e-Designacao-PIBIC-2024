#!/usr/bin/env python3
"""
GAP Others Parser - Parser for GAP instances in the 'others' folder format

This script extends the existing GAP parser to handle the format used in 
INSTANCES/gap/others/ folder, which has a simpler format without multiple
problems per file.

Usage:
    python gap_others_parser.py --file others/a05100 --output a05100.dat
    python gap_others_parser.py --dir INSTANCES/gap/others --batch
"""

import argparse
import os
import numpy as np
from typing import Optional
from dataclasses import dataclass

@dataclass
class GAPInstance:
    """Represents a GAP instance."""
    n_workers: int
    n_tasks: int
    costs: np.ndarray        # [workers x tasks] cost matrix
    weights: np.ndarray      # [workers x tasks] weight/requirement matrix  
    capacities: np.ndarray   # [workers] capacity vector
    is_maximization: bool    # True for maximization, False for minimization


class GAPOthersParser:
    """Parser for GAP instances in the 'others' folder format."""
    
    @staticmethod
    def parse_gap_others_file(filename: str) -> GAPInstance:
        """
        Parse a GAP file from the 'others' folder format.
        
        Format:
        - Line 1: m n (workers, tasks)
        - Next m*n values: cost matrix (row by row)
        - Next m*n values: weight matrix (row by row)  
        - Next m values: capacity vector
        """
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # Determine if this is maximization or minimization based on filename
        # Convention: a,b,c,d,e instances are minimization problems
        basename = os.path.basename(filename).lower()
        is_maximization = False  # Others folder instances are typically minimization
        
        # Parse first line: m (workers), n (tasks)
        m, n = map(int, lines[0].split())
        line_idx = 1
        
        # Parse cost matrix (m*n values, organized as m rows of n values each)
        cost_matrix = []
        cost_values = []
        
        # Collect all cost values
        while len(cost_values) < m * n and line_idx < len(lines):
            numbers = list(map(int, lines[line_idx].split()))
            cost_values.extend(numbers)
            line_idx += 1
        
        # Reshape into m x n matrix
        for i in range(m):
            row_start = i * n
            row_end = row_start + n
            cost_matrix.append(cost_values[row_start:row_end])
        
        # Parse weight matrix (m*n values)
        weight_matrix = []
        weight_values = []
        
        # Collect all weight values
        while len(weight_values) < m * n and line_idx < len(lines):
            numbers = list(map(int, lines[line_idx].split()))
            weight_values.extend(numbers)
            line_idx += 1
        
        # Reshape into m x n matrix
        for i in range(m):
            row_start = i * n
            row_end = row_start + n
            weight_matrix.append(weight_values[row_start:row_end])
        
        # Parse capacities (m values)
        cap_values = []
        while len(cap_values) < m and line_idx < len(lines):
            numbers = list(map(int, lines[line_idx].split()))
            cap_values.extend(numbers)
            line_idx += 1
        
        capacities = cap_values[:m]
        
        # Convert to numpy arrays
        costs = np.array(cost_matrix)
        weights = np.array(weight_matrix)
        capacities = np.array(capacities)
        
        return GAPInstance(m, n, costs, weights, capacities, is_maximization)
    
    @staticmethod
    def convert_to_dat_format(instance: GAPInstance, output_file: str):
        """Convert a GAPInstance to AMPL .dat format."""
        
        with open(output_file, 'w') as f:
            # Write sets
            f.write(f"set WORKERS := {' '.join(map(str, range(1, instance.n_workers + 1)))};\n")
            f.write(f"set TASKS := {' '.join(map(str, range(1, instance.n_tasks + 1)))};\n\n")
            
            # Write cost matrix
            f.write("param cost :\n")
            f.write("    " + " ".join(f"{j:6d}" for j in range(1, instance.n_tasks + 1)) + " :=\n")
            for i in range(instance.n_workers):
                f.write(f"{i+1:2d} " + " ".join(f"{instance.costs[i,j]:6d}" for j in range(instance.n_tasks)) + "\n")
            f.write(";\n\n")
            
            # Write weight matrix
            f.write("param weight :\n")
            f.write("    " + " ".join(f"{j:6d}" for j in range(1, instance.n_tasks + 1)) + " :=\n")
            for i in range(instance.n_workers):
                f.write(f"{i+1:2d} " + " ".join(f"{instance.weights[i,j]:6d}" for j in range(instance.n_tasks)) + "\n")
            f.write(";\n\n")
            
            # Write capacities
            f.write("param capacity :=\n")
            for i in range(instance.n_workers):
                f.write(f"{i+1:2d} {instance.capacities[i]:6d}\n")
            f.write(";\n")


def main():
    parser = argparse.ArgumentParser(description="Parse GAP instances from 'others' folder format")
    parser.add_argument("--file", help="Path to single GAP file to parse")
    parser.add_argument("--dir", help="Directory containing GAP files to parse")
    parser.add_argument("--output", help="Output .dat file (for single file mode)")
    parser.add_argument("--batch", action="store_true", help="Batch convert all files in directory")
    parser.add_argument("--output-dir", default="dat/gap_others", help="Output directory for batch mode")
    
    args = parser.parse_args()
    
    if args.file:
        # Single file mode
        if not os.path.exists(args.file):
            print(f"Error: File {args.file} not found")
            return
        
        try:
            instance = GAPOthersParser.parse_gap_others_file(args.file)
            print(f"Parsed {args.file}:")
            print(f"  Workers: {instance.n_workers}")
            print(f"  Tasks: {instance.n_tasks}")
            print(f"  Optimization: {'Maximization' if instance.is_maximization else 'Minimization'}")
            
            if args.output:
                os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
                GAPOthersParser.convert_to_dat_format(instance, args.output)
                print(f"  Converted to: {args.output}")
        
        except Exception as e:
            print(f"Error parsing {args.file}: {e}")
    
    elif args.dir and args.batch:
        # Batch mode
        if not os.path.exists(args.dir):
            print(f"Error: Directory {args.dir} not found")
            return
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        files = [f for f in os.listdir(args.dir) if not f.startswith('.')]
        print(f"Found {len(files)} files in {args.dir}")
        
        successful = 0
        failed = 0
        
        for filename in sorted(files):
            file_path = os.path.join(args.dir, filename)
            output_path = os.path.join(args.output_dir, f"{filename}.dat")
            
            try:
                instance = GAPOthersParser.parse_gap_others_file(file_path)
                GAPOthersParser.convert_to_dat_format(instance, output_path)
                print(f"✓ {filename} -> {filename}.dat ({instance.n_workers}x{instance.n_tasks})")
                successful += 1
            except Exception as e:
                print(f"✗ {filename}: {e}")
                failed += 1
        
        print(f"\nCompleted: {successful} successful, {failed} failed")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
