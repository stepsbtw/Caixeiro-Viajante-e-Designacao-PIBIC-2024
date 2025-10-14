import os
import math
import numpy as np

def parse_atsp_file(filename):
    """
    Parse an ATSP file and return the distance matrix and number of nodes.
    ATSP files typically use explicit distance matrices with FULL_MATRIX format.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse header information
    dimension = None
    edge_weight_type = None
    edge_weight_format = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('DIMENSION'):
            dimension = int(line.split(':')[1].strip())
        elif line.startswith('EDGE_WEIGHT_TYPE'):
            edge_weight_type = line.split(':')[1].strip()
        elif line.startswith('EDGE_WEIGHT_FORMAT'):
            edge_weight_format = line.split(':')[1].strip()
    
    if dimension is None:
        raise ValueError(f"Could not find DIMENSION in {filename}")
    
    # Initialize distance matrix
    dist_matrix = np.zeros((dimension, dimension))
    
    if edge_weight_type == 'EXPLICIT':
        # Explicit distance matrix (most common for ATSP)
        dist_matrix = parse_explicit_distances(lines, dimension, edge_weight_format)
    else:
        raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE for ATSP: {edge_weight_type}")
    
    return dist_matrix, dimension

def parse_explicit_distances(lines, dimension, edge_weight_format):
    """Parse explicit distance matrix for ATSP (asymmetric)."""
    dist_matrix = np.zeros((dimension, dimension))
    
    # Find EDGE_WEIGHT_SECTION
    in_weight_section = False
    weight_data = []
    
    for line in lines:
        line = line.strip()
        if line == 'EDGE_WEIGHT_SECTION':
            in_weight_section = True
            continue
        elif line == 'EOF':
            break
        elif in_weight_section and line:
            # Split the line and add all numbers to weight_data
            numbers = line.split()
            for num in numbers:
                try:
                    # Handle large values (9999 often represents infinity/no connection)
                    value = int(num)
                    weight_data.append(value)
                except ValueError:
                    pass  # Skip non-numeric values
    
    # Fill the distance matrix based on format
    # For ATSP, we typically see FULL_MATRIX format
    if edge_weight_format == 'FULL_MATRIX':
        idx = 0
        for i in range(dimension):
            for j in range(dimension):
                if idx < len(weight_data):
                    dist_matrix[i][j] = weight_data[idx]
                    idx += 1
    elif edge_weight_format == 'LOWER_DIAG_ROW':
        # Less common for ATSP, but included for completeness
        idx = 0
        for i in range(dimension):
            for j in range(i + 1):
                if idx < len(weight_data):
                    dist_matrix[i][j] = weight_data[idx]
                    # Note: For ATSP, we don't mirror the matrix (asymmetric)
                idx += 1
    elif edge_weight_format == 'UPPER_DIAG_ROW':
        idx = 0
        for i in range(dimension):
            for j in range(i, dimension):
                if idx < len(weight_data):
                    dist_matrix[i][j] = weight_data[idx]
                idx += 1
    elif edge_weight_format == 'UPPER_ROW':
        idx = 0
        for i in range(dimension):
            for j in range(i + 1, dimension):
                if idx < len(weight_data):
                    dist_matrix[i][j] = weight_data[idx]
                idx += 1
    else:
        raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {edge_weight_format}")
    
    return dist_matrix

def write_ampl_dat_file(dist_matrix, dimension, output_filename):
    """Write the asymmetric distance matrix to an AMPL .dat file."""
    with open(output_filename, 'w') as f:
        # Write the set of nodes
        f.write("set NODES := ")
        for i in range(1, dimension + 1):
            f.write(f"{i} ")
        f.write(";\n\n")
        
        # Write the distance matrix
        f.write("param dist :\n")
        f.write("     ")
        for j in range(1, dimension + 1):
            f.write(f"{j:8}")
        f.write(" :=\n")
        
        for i in range(1, dimension + 1):
            f.write(f"{i:4}")
            for j in range(1, dimension + 1):
                # Convert large values (like 9999) to a reasonable large number
                value = int(dist_matrix[i-1][j-1])
                if value >= 9999:
                    value = 99999  # Use a large but manageable number
                f.write(f"{value:8}")
            f.write("\n")
        f.write(";\n")

def convert_atsp_to_dat(atsp_file, dat_file):
    """Convert a single ATSP file to AMPL .dat format."""
    try:
        dist_matrix, dimension = parse_atsp_file(atsp_file)
        write_ampl_dat_file(dist_matrix, dimension, dat_file)
        print(f"Converted {atsp_file} -> {dat_file} (dimension: {dimension})")
        return True
    except Exception as e:
        print(f"Error converting {atsp_file}: {str(e)}")
        return False

def convert_all_atsp_files(atsp_dir="INSTANCES/ATSP", dat_dir="dat/atsp"):
    """Convert all ATSP files in a directory to .dat format."""
    # Create output directory if it doesn't exist
    os.makedirs(dat_dir, exist_ok=True)
    
    success_count = 0
    total_count = 0
    
    # Get all .atsp files
    atsp_files = [f for f in os.listdir(atsp_dir) if f.endswith('.atsp')]
    atsp_files.sort()  # Sort for consistent ordering
    
    print(f"Found {len(atsp_files)} ATSP files to convert...")
    
    for atsp_filename in atsp_files:
        atsp_path = os.path.join(atsp_dir, atsp_filename)
        dat_filename = atsp_filename.replace('.atsp', '.dat')
        dat_path = os.path.join(dat_dir, dat_filename)
        
        total_count += 1
        if convert_atsp_to_dat(atsp_path, dat_path):
            success_count += 1
    
    print(f"\nConversion complete!")
    print(f"Successfully converted: {success_count}/{total_count} files")
    print(f"Output directory: {dat_dir}")

def main():
    """Main function to run the converter."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert ATSP files to AMPL .dat format')
    parser.add_argument('--input', '-i', default='INSTANCES/ATSP', 
                        help='Input directory containing .atsp files (default: INSTANCES/ATSP)')
    parser.add_argument('--output', '-o', default='dat/atsp', 
                        help='Output directory for .dat files (default: dat/atsp)')
    parser.add_argument('--file', '-f', 
                        help='Convert a single ATSP file (provide full path)')
    parser.add_argument('--output-file', 
                        help='Output file for single file conversion')
    
    args = parser.parse_args()
    
    if args.file:
        # Single file conversion
        output_file = args.output_file if args.output_file else args.file.replace('.atsp', '.dat')
        convert_atsp_to_dat(args.file, output_file)
    else:
        # Batch conversion
        convert_all_atsp_files(args.input, args.output)

if __name__ == "__main__":
    main()
