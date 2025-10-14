#!/usr/bin/env python3
"""
Assignment Problem Sparse Parser
Parser for sparse assignment problem instances (assignp format)

This script parses assignment problems in sparse format where only non-zero
cost entries are listed, and converts them to AMPL .dat format.

Format:
  n                    # Problem size (n×n matrix)
  worker task cost     # One line per non-zero entry
  worker task cost
  ...

Usage:
    python assignment_sparse_parser.py --file assignp800.txt --output assignp800.dat
    python assignment_sparse_parser.py --dir INSTANCES/assign --pattern "assignp*.txt" --batch
"""

import argparse
import os
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class AssignmentInstance:
    """Represents an Assignment Problem instance."""
    n: int                    # Problem size (n×n)
    costs: Dict[Tuple[int, int], int]  # Sparse cost matrix: (worker, task) -> cost
    is_sparse: bool = True    # Always true for this format
    

class AssignmentSparseParser:
    """Parser for sparse assignment problem instances."""
    
    @staticmethod
    def parse_assignment_sparse_file(filename: str) -> AssignmentInstance:
        """
        Parse a sparse assignment problem file.
        
        Format:
        - Line 1: n (problem size)
        - Lines 2+: worker task cost (1-indexed)
        """
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # Parse problem size
        n = int(lines[0])
        
        # Parse sparse cost entries
        costs = {}
        for line in lines[1:]:
            parts = line.split()
            if len(parts) == 3:
                worker, task, cost = map(int, parts)
                # Convert to 0-indexed for internal use
                costs[(worker-1, task-1)] = cost
        
        return AssignmentInstance(n, costs)
    
    @staticmethod
    def convert_to_dat_format(instance: AssignmentInstance, output_file: str):
        """Convert an AssignmentInstance to AMPL .dat format."""
        
        with open(output_file, 'w') as f:
            # Write sets
            f.write(f"set WORKERS := {' '.join(map(str, range(1, instance.n + 1)))};\n")
            f.write(f"set TASKS := {' '.join(map(str, range(1, instance.n + 1)))};\n\n")
            
            # Write sparse cost matrix
            f.write("param cost default 0 :\n")
            f.write("    " + " ".join(f"{j:6d}" for j in range(1, instance.n + 1)) + " :=\n")
            
            # Group costs by worker for efficient writing
            costs_by_worker = {}
            for (worker, task), cost in instance.costs.items():
                if worker not in costs_by_worker:
                    costs_by_worker[worker] = {}
                costs_by_worker[worker][task] = cost
            
            # Write costs for each worker (only non-zero entries)
            for worker in range(instance.n):
                if worker in costs_by_worker:
                    worker_costs = costs_by_worker[worker]
                    # Write as [task,cost] pairs for sparse representation
                    cost_pairs = " ".join(f"[{task+1},{cost}]" for task, cost in worker_costs.items())
                    f.write(f"{worker+1:2d} {cost_pairs}\n")
            f.write(";\n")
    
    @staticmethod
    def get_statistics(instance: AssignmentInstance) -> Dict:
        """Get statistics about the instance."""
        total_entries = instance.n * instance.n
        nonzero_entries = len(instance.costs)
        sparsity = (total_entries - nonzero_entries) / total_entries * 100
        
        costs_list = list(instance.costs.values())
        return {
            'size': instance.n,
            'total_entries': total_entries,
            'nonzero_entries': nonzero_entries,
            'sparsity_percent': sparsity,
            'min_cost': min(costs_list) if costs_list else 0,
            'max_cost': max(costs_list) if costs_list else 0,
            'avg_cost': sum(costs_list) / len(costs_list) if costs_list else 0
        }


def main():
    parser = argparse.ArgumentParser(description="Parse sparse assignment problem instances")
    parser.add_argument("--file", help="Path to single assignment file to parse")
    parser.add_argument("--dir", help="Directory containing assignment files to parse")
    parser.add_argument("--pattern", default="assignp*.txt", help="File pattern to match (default: assignp*.txt)")
    parser.add_argument("--output", help="Output .dat file (for single file mode)")
    parser.add_argument("--batch", action="store_true", help="Batch convert all files in directory")
    parser.add_argument("--output-dir", default="dat/assign_sparse", help="Output directory for batch mode")
    parser.add_argument("--stats", action="store_true", help="Show detailed statistics")
    
    args = parser.parse_args()
    
    if args.file:
        # Single file mode
        if not os.path.exists(args.file):
            print(f"Error: File {args.file} not found")
            return
        
        try:
            instance = AssignmentSparseParser.parse_assignment_sparse_file(args.file)
            stats = AssignmentSparseParser.get_statistics(instance)
            
            print(f"Parsed {args.file}:")
            print(f"  Size: {stats['size']}×{stats['size']}")
            print(f"  Non-zero entries: {stats['nonzero_entries']:,} / {stats['total_entries']:,}")
            print(f"  Sparsity: {stats['sparsity_percent']:.1f}%")
            if args.stats:
                print(f"  Cost range: {stats['min_cost']} - {stats['max_cost']} (avg: {stats['avg_cost']:.1f})")
            
            if args.output:
                os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
                AssignmentSparseParser.convert_to_dat_format(instance, args.output)
                print(f"  Converted to: {args.output}")
        
        except Exception as e:
            print(f"Error parsing {args.file}: {e}")
    
    elif args.dir and args.batch:
        # Batch mode
        if not os.path.exists(args.dir):
            print(f"Error: Directory {args.dir} not found")
            return
        
        import glob
        pattern_path = os.path.join(args.dir, args.pattern)
        files = glob.glob(pattern_path)
        
        if not files:
            print(f"No files found matching pattern: {pattern_path}")
            return
            
        os.makedirs(args.output_dir, exist_ok=True)
        
        print(f"Found {len(files)} files matching {args.pattern}")
        
        successful = 0
        failed = 0
        
        for file_path in sorted(files):
            filename = os.path.basename(file_path)
            output_name = filename.replace('.txt', '.dat')
            output_path = os.path.join(args.output_dir, output_name)
            
            try:
                instance = AssignmentSparseParser.parse_assignment_sparse_file(file_path)
                AssignmentSparseParser.convert_to_dat_format(instance, output_path)
                stats = AssignmentSparseParser.get_statistics(instance)
                
                print(f"✓ {filename} -> {output_name} ({stats['size']}×{stats['size']}, {stats['nonzero_entries']:,} entries, {stats['sparsity_percent']:.1f}% sparse)")
                successful += 1
            except Exception as e:
                print(f"✗ {filename}: {e}")
                failed += 1
        
        print(f"\nCompleted: {successful} successful, {failed} failed")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
