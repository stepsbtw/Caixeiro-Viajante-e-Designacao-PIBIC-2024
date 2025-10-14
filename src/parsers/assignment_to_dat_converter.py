#!/usr/bin/env python3
"""
Assignment Problem Parser - Convert .txt files to AMPL .dat format

This script converts assignment problem instances from the format:
- First line: n (number of workers/tasks)  
- Following lines: cost matrix (n x n)

To AMPL .dat format compatible with assignment.mod
"""

import os
import sys
import glob

def parse_assignment_file(file_path):
    """
    Parse assignment problem file and return n and cost matrix
    
    Args:
        file_path (str): Path to the assignment .txt file
        
    Returns:
        tuple: (n, cost_matrix) where cost_matrix is list of lists
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Remove empty lines and strip whitespace
        lines = [line.strip() for line in lines if line.strip()]
        
        # First line is n
        n = int(lines[0])
        print(f"  Problem size: {n}x{n}")
        
        # Parse cost matrix
        cost_matrix = []
        current_row = []
        
        for line in lines[1:]:
            # Split the line into numbers
            numbers = [int(x) for x in line.split()]
            current_row.extend(numbers)
            
            # If we have n numbers, complete the row
            if len(current_row) >= n:
                cost_matrix.append(current_row[:n])
                current_row = current_row[n:]  # Keep any remaining numbers for next row
        
        # Add any remaining numbers as the last row
        if current_row:
            if len(current_row) < n:
                # Pad with zeros if needed (shouldn't happen in well-formed files)
                current_row.extend([0] * (n - len(current_row)))
            cost_matrix.append(current_row[:n])
        
        # Verify we have exactly n rows
        if len(cost_matrix) != n:
            print(f"  Warning: Expected {n} rows, got {len(cost_matrix)}")
            cost_matrix = cost_matrix[:n]  # Take first n rows
            
        # Verify each row has exactly n elements
        for i, row in enumerate(cost_matrix):
            if len(row) != n:
                print(f"  Warning: Row {i+1} has {len(row)} elements, expected {n}")
                
        return n, cost_matrix
        
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None, None

def write_dat_file(output_path, n, cost_matrix):
    """
    Write assignment data in AMPL .dat format
    
    Args:
        output_path (str): Path for output .dat file
        n (int): Number of workers/tasks
        cost_matrix (list): n x n cost matrix
    """
    try:
        with open(output_path, 'w') as f:
            # Write sets
            f.write("# Assignment Problem Data File\n")
            f.write("# Generated from assignment instance\n\n")
            
            # Workers and tasks (1 to n)
            workers = " ".join(str(i) for i in range(1, n+1))
            f.write(f"set WORKERS := {workers};\n")
            f.write(f"set TASKS := {workers};\n\n")
            
            # Cost matrix
            f.write("param cost :\n")
            
            # Header row with task indices
            f.write("     ")
            for j in range(1, n+1):
                f.write(f"{j:>6}")
            f.write(" :=\n")
            
            # Data rows
            for i in range(n):
                f.write(f"{i+1:>4}")
                for j in range(n):
                    f.write(f"{cost_matrix[i][j]:>6}")
                f.write("\n")
            
            f.write(";\n")
            
        print(f"  ✓ Created: {output_path}")
        
    except Exception as e:
        print(f"Error writing {output_path}: {e}")

def convert_assignment_files(input_dir="INSTANCES/assign", output_dir="dat/assign"):
    """
    Convert all assignment .txt files to .dat format
    
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
    
    print(f"Found {len(txt_files)} assignment files to convert:")
    
    converted = 0
    for txt_file in sorted(txt_files):
        base_name = os.path.basename(txt_file)
        instance_name = os.path.splitext(base_name)[0]
        
        print(f"\nConverting {base_name}...")
        
        # Parse the file
        n, cost_matrix = parse_assignment_file(txt_file)
        
        if n is not None and cost_matrix is not None:
            # Create output file
            dat_file = os.path.join(output_dir, f"{instance_name}.dat")
            write_dat_file(dat_file, n, cost_matrix)
            converted += 1
        else:
            print(f"  ✗ Failed to parse {base_name}")
    
    print(f"\n{'='*60}")
    print(f"Conversion complete: {converted}/{len(txt_files)} files converted")
    print(f"Output directory: {output_dir}")

def main():
    """Main function with command line argument support"""
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "dat/assign"
    else:
        input_dir = "INSTANCES/assign"
        output_dir = "dat/assign"
    
    print("Assignment Problem to AMPL .dat Converter")
    print("="*50)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print("="*50)
    
    convert_assignment_files(input_dir, output_dir)

if __name__ == "__main__":
    main()
