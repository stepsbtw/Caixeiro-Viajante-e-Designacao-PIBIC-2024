#!/usr/bin/env python3
"""
GAP (Generalized Assignment Problem) Parser - Convert .txt files to AMPL .dat format

This script converts GAP instances from the format used in gap1.txt:
- First line: number of problems
- For each problem:
  - Line: m n (workers, tasks)
  - m lines: cost matrix rows (n values each)
  - m lines: requirement matrix rows (n values each)  
  - Line: capacity values (m values)

To AMPL .dat format compatible with gap.mod
"""

import os
import sys
import glob

def parse_gap_file(file_path):
    """
    Parse GAP file and return list of problems
    
    Args:
        file_path (str): Path to the GAP .txt file
        
    Returns:
        list: List of (m, n, cost_matrix, req_matrix, capacities) tuples
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Remove empty lines and strip whitespace
        lines = [line.strip() for line in lines if line.strip()]
        
        # First line is number of problems
        num_problems = int(lines[0])
        print(f"  Found {num_problems} problems in file")
        
        problems = []
        line_idx = 1
        
        for prob_num in range(num_problems):
            print(f"  Parsing problem {prob_num + 1}...")
            
            # Parse m and n
            m, n = map(int, lines[line_idx].split())
            print(f"    Workers (m): {m}, Tasks (n): {n}")
            line_idx += 1
            
            # Parse cost matrix (m rows of n values each)
            cost_matrix = []
            for i in range(m):
                row_data = []
                while len(row_data) < n and line_idx < len(lines):
                    # Get numbers from current line
                    numbers = list(map(int, lines[line_idx].split()))
                    row_data.extend(numbers)
                    line_idx += 1
                
                # Take exactly n values for this row
                cost_matrix.append(row_data[:n])
                
                # If we got more than n values, we need to handle overflow
                if len(row_data) > n:
                    # Put back excess numbers by reconstructing the line
                    excess = row_data[n:]
                    line_idx -= 1
                    lines[line_idx] = ' '.join(map(str, excess))
            
            # Parse requirement matrix (m rows of n values each)
            req_matrix = []
            for i in range(m):
                row_data = []
                while len(row_data) < n and line_idx < len(lines):
                    # Get numbers from current line
                    numbers = list(map(int, lines[line_idx].split()))
                    row_data.extend(numbers)
                    line_idx += 1
                
                # Take exactly n values for this row
                req_matrix.append(row_data[:n])
                
                # If we got more than n values, handle overflow
                if len(row_data) > n:
                    excess = row_data[n:]
                    line_idx -= 1
                    lines[line_idx] = ' '.join(map(str, excess))
            
            # Parse capacities (m values)
            cap_data = []
            while len(cap_data) < m and line_idx < len(lines):
                numbers = list(map(int, lines[line_idx].split()))
                cap_data.extend(numbers)
                line_idx += 1
            
            capacities = cap_data[:m]
            
            # If we got more than m values, handle overflow
            if len(cap_data) > m:
                excess = cap_data[m:]
                line_idx -= 1
                lines[line_idx] = ' '.join(map(str, excess))
            
            problems.append((m, n, cost_matrix, req_matrix, capacities))
            print(f"    ✓ Problem {prob_num + 1} parsed successfully")
        
        return problems
        
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def write_gap_dat_file(output_path, problem_num, m, n, cost_matrix, req_matrix, capacities):
    """
    Write GAP data in AMPL .dat format
    
    Args:
        output_path (str): Path for output .dat file
        problem_num (int): Problem number
        m (int): Number of workers
        n (int): Number of tasks
        cost_matrix (list): m x n cost matrix
        req_matrix (list): m x n requirement matrix
        capacities (list): m capacity values
    """
    try:
        with open(output_path, 'w') as f:
            # Write header
            f.write("# GAP Problem Data File\n")
            f.write(f"# Generated from problem {problem_num}\n\n")
            
            # Workers and tasks sets (following assignment pattern)
            workers = " ".join(str(i) for i in range(1, m+1))
            f.write(f"set WORKERS := {workers};\n")
            
            tasks = " ".join(str(j) for j in range(1, n+1))
            f.write(f"set TASKS := {tasks};\n\n")
            
            # Cost matrix
            f.write("param cost :\n")
            
            # Header row with task indices
            f.write("          ")
            for j in range(1, n+1):
                f.write(f"{j:>5}")
            f.write(" :=\n")
            
            # Data rows
            for i in range(m):
                f.write(f"   {i+1}")
                for j in range(n):
                    f.write(f"{cost_matrix[i][j]:>5}")
                f.write("\n")
            
            f.write(";\n\n")
            
            # Requirement matrix (weight/resource requirements)
            f.write("param weight :\n")
            
            # Header row
            f.write("          ")
            for j in range(1, n+1):
                f.write(f"{j:>5}")
            f.write(" :=\n")
            
            # Data rows
            for i in range(m):
                f.write(f"   {i+1}")
                for j in range(n):
                    f.write(f"{req_matrix[i][j]:>5}")
                f.write("\n")
            
            f.write(";\n\n")
            
            # Capacities
            f.write("param capacity :=\n")
            for i in range(m):
                f.write(f" {i+1} {capacities[i]}\n")
            f.write(";\n")
            
        print(f"  ✓ Created: {output_path}")
        
    except Exception as e:
        print(f"Error writing {output_path}: {e}")

def convert_gap_files(input_dir="INSTANCES/gap", output_dir="dat/gap"):
    """
    Convert all GAP .txt files to .dat format
    
    Args:
        input_dir (str): Directory containing .txt files
        output_dir (str): Directory for .dat files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .txt files in input directory
    pattern = os.path.join(input_dir, "*.txt")
    txt_files = glob.glob(pattern)
    
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return
    
    print(f"Found {len(txt_files)} GAP files to convert:")
    
    converted = 0
    total_problems = 0
    
    for txt_file in sorted(txt_files):
        base_name = os.path.basename(txt_file)
        instance_name = os.path.splitext(base_name)[0]
        
        print(f"\nConverting {base_name}...")
        
        # Parse the file
        problems = parse_gap_file(txt_file)
        
        if problems is not None:
            # Create separate .dat file for each problem
            for prob_idx, (m, n, cost_matrix, req_matrix, capacities) in enumerate(problems):
                dat_file = os.path.join(output_dir, f"{instance_name}_p{prob_idx + 1}.dat")
                
                # Skip if file already exists
                if os.path.exists(dat_file):
                    print(f"  ⚬ Skipped: {dat_file} (already exists)")
                    continue
                    
                write_gap_dat_file(dat_file, prob_idx + 1, m, n, cost_matrix, req_matrix, capacities)
                converted += 1
                total_problems += 1
        else:
            print(f"  ✗ Failed to parse {base_name}")
    
    print(f"\n{'='*60}")
    print(f"Conversion complete: {total_problems} problems from {len(txt_files)} files")
    print(f"Generated {converted} .dat files")
    print(f"Output directory: {output_dir}")

def main():
    """Main function with command line argument support"""
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "dat/gap"
    else:
        input_dir = "INSTANCES/gap"
        output_dir = "dat/gap"
    
    print("GAP Problem to AMPL .dat Converter")
    print("="*50)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print("="*50)
    
    convert_gap_files(input_dir, output_dir)

if __name__ == "__main__":
    main()
