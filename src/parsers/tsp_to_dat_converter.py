import os
import math
import numpy as np

def parse_tsp_file(filename):
    """
    Parse a TSP file and return the distance matrix and number of nodes.
    Handles both coordinate-based and explicit distance matrix formats.
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
    
    if edge_weight_type == 'GEO':
        # Geographic coordinates - need to compute distances
        coords = parse_coordinates(lines)
        dist_matrix = compute_geo_distances(coords)
    elif edge_weight_type == 'EUC_2D':
        # Euclidean 2D coordinates
        coords = parse_coordinates(lines)
        dist_matrix = compute_euclidean_distances(coords)
    elif edge_weight_type == 'ATT':
        # Pseudo-Euclidean distance (ATT)
        coords = parse_coordinates(lines)
        dist_matrix = compute_att_distances(coords)
    elif edge_weight_type == 'CEIL_2D':
        # Euclidean 2D with ceiling
        coords = parse_coordinates(lines)
        dist_matrix = compute_ceil_euclidean_distances(coords)
    elif edge_weight_type == 'EXPLICIT':
        # Explicit distance matrix
        dist_matrix = parse_explicit_distances(lines, dimension, edge_weight_format)
    else:
        raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE: {edge_weight_type}")
    
    return dist_matrix, dimension

def parse_coordinates(lines):
    """Parse NODE_COORD_SECTION and return coordinates."""
    coords = []
    in_coord_section = False
    
    for line in lines:
        line = line.strip()
        if line == 'NODE_COORD_SECTION':
            in_coord_section = True
            continue
        elif line == 'EOF' or line.startswith('EDGE_WEIGHT_SECTION'):
            break
        elif in_coord_section and line:
            parts = line.split()
            if len(parts) >= 3:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coords.append((x, y))
    
    return coords

def compute_geo_distances(coords):
    """Compute geographical distances using the formula from TSPLIB."""
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                lat1, lon1 = coords[i]
                lat2, lon2 = coords[j]
                
                # Convert to radians
                lat1_rad = math.pi * lat1 / 180.0
                lon1_rad = math.pi * lon1 / 180.0
                lat2_rad = math.pi * lat2 / 180.0
                lon2_rad = math.pi * lon2 / 180.0
                
                # TSPLIB geographical distance formula
                q1 = math.cos(lon1_rad - lon2_rad)
                q2 = math.cos(lat1_rad - lat2_rad)
                q3 = math.cos(lat1_rad + lat2_rad)
                
                distance = 6378.388 * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3))
                dist_matrix[i][j] = int(distance + 0.5)  # Round to nearest integer
    
    return dist_matrix

def compute_euclidean_distances(coords):
    """Compute Euclidean distances."""
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                dist_matrix[i][j] = int(distance + 0.5)  # Round to nearest integer
    
    return dist_matrix

def compute_att_distances(coords):
    """Compute ATT (pseudo-Euclidean) distances."""
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                rij = math.sqrt(((x1 - x2)**2 + (y1 - y2)**2) / 10.0)
                tij = int(rij + 0.5)
                if tij < rij:
                    dist_matrix[i][j] = tij + 1
                else:
                    dist_matrix[i][j] = tij
    
    return dist_matrix

def compute_ceil_euclidean_distances(coords):
    """Compute ceiling of Euclidean distances."""
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                dist_matrix[i][j] = math.ceil(distance)
    
    return dist_matrix

def parse_explicit_distances(lines, dimension, edge_weight_format):
    """Parse explicit distance matrix."""
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
                    weight_data.append(int(num))
                except ValueError:
                    pass  # Skip non-numeric values
    
    # Fill the distance matrix based on format
    if edge_weight_format == 'LOWER_DIAG_ROW':
        idx = 0
        for i in range(dimension):
            for j in range(i + 1):
                if idx < len(weight_data):
                    dist_matrix[i][j] = weight_data[idx]
                    dist_matrix[j][i] = weight_data[idx]  # Symmetric
                idx += 1
    elif edge_weight_format == 'UPPER_DIAG_ROW':
        idx = 0
        for i in range(dimension):
            for j in range(i, dimension):
                if idx < len(weight_data):
                    dist_matrix[i][j] = weight_data[idx]
                    dist_matrix[j][i] = weight_data[idx]  # Symmetric
                idx += 1
    elif edge_weight_format == 'UPPER_ROW':
        idx = 0
        for i in range(dimension):
            for j in range(i + 1, dimension):
                if idx < len(weight_data):
                    dist_matrix[i][j] = weight_data[idx]
                    dist_matrix[j][i] = weight_data[idx]  # Symmetric
                idx += 1
    elif edge_weight_format == 'FULL_MATRIX':
        idx = 0
        for i in range(dimension):
            for j in range(dimension):
                if idx < len(weight_data):
                    dist_matrix[i][j] = weight_data[idx]
                idx += 1
    else:
        raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {edge_weight_format}")
    
    return dist_matrix

def write_ampl_dat_file(dist_matrix, dimension, output_filename):
    """Write the distance matrix to an AMPL .dat file."""
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
                f.write(f"{int(dist_matrix[i-1][j-1]):8}")
            f.write("\n")
        f.write(";\n")

def convert_tsp_to_dat(tsp_file, dat_file):
    """Convert a single TSP file to AMPL .dat format."""
    try:
        dist_matrix, dimension = parse_tsp_file(tsp_file)
        write_ampl_dat_file(dist_matrix, dimension, dat_file)
        print(f"Converted {tsp_file} -> {dat_file} (dimension: {dimension})")
        return True
    except Exception as e:
        print(f"Error converting {tsp_file}: {str(e)}")
        return False

def convert_all_tsp_files(tsp_dir="INSTANCES/TSP", dat_dir="dat"):
    """Convert all TSP files in a directory to .dat files."""
    # Create output directory if it doesn't exist
    os.makedirs(dat_dir, exist_ok=True)
    
    # Get all TSP files
    tsp_files = [f for f in os.listdir(tsp_dir) if f.endswith('.tsp')]
    
    successful = 0
    failed = 0
    
    for tsp_file in tsp_files:
        tsp_path = os.path.join(tsp_dir, tsp_file)
        dat_file = tsp_file.replace('.tsp', '.dat')
        dat_path = os.path.join(dat_dir, dat_file)
        
        if convert_tsp_to_dat(tsp_path, dat_path):
            successful += 1
        else:
            failed += 1
    
    print(f"\nConversion complete:")
    print(f"Successfully converted: {successful} files")
    print(f"Failed: {failed} files")

if __name__ == "__main__":
    convert_all_tsp_files()
