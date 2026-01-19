"""
Utilities for handling OBJ files with multiple materials for PyBullet.
"""
import os
from pathlib import Path


def simplify_obj_single_material(input_obj, output_obj, remove_materials=True):
    """
    Simplify OBJ file to work better with PyBullet by removing material references.
    
    Args:
        input_obj: Path to input OBJ file
        output_obj: Path to output OBJ file
        remove_materials: If True, removes all usemtl and mtllib lines
    """
    with open(input_obj, 'r') as f:
        lines = f.readlines()
    
    with open(output_obj, 'w') as f:
        for line in lines:
            if remove_materials:
                # Skip material lines
                if line.startswith('mtllib') or line.startswith('usemtl'):
                    continue
            f.write(line)
    
    print(f"Simplified OBJ saved to: {output_obj}")


def split_obj_by_materials(input_obj, output_dir=None):
    """
    Split a multi-material OBJ file into separate OBJ files per material.
    
    Args:
        input_obj: Path to input OBJ file
        output_dir: Directory to save split files (default: same as input)
    
    Returns:
        List of created file paths
    """
    input_path = Path(input_obj)
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    # Read the entire file
    with open(input_obj, 'r') as f:
        lines = f.readlines()
    
    # Parse the file
    vertices = []
    normals = []
    texcoords = []
    materials = {}
    current_material = None
    mtllib_line = None
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('mtllib'):
            mtllib_line = line
        elif line.startswith('v '):
            vertices.append(line)
        elif line.startswith('vn '):
            normals.append(line)
        elif line.startswith('vt '):
            texcoords.append(line)
        elif line.startswith('usemtl'):
            current_material = line.split()[1]
            if current_material not in materials:
                materials[current_material] = []
        elif line.startswith('f ') and current_material:
            materials[current_material].append(line)
    
    # If only one material, no need to split
    if len(materials) <= 1:
        print(f"Only {len(materials)} material(s) found. No need to split.")
        return []
    
    # Create separate files for each material
    created_files = []
    base_name = input_path.stem
    
    for mat_name, faces in materials.items():
        output_file = output_dir / f"{base_name}_{mat_name}.obj"
        
        with open(output_file, 'w') as f:
            # Write header
            f.write(f"# Split from {input_path.name}\n")
            f.write(f"# Material: {mat_name}\n")
            if mtllib_line:
                f.write(f"{mtllib_line}\n")
            f.write("\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"{v}\n")
            
            # Write texture coordinates
            for vt in texcoords:
                f.write(f"{vt}\n")
            
            # Write normals
            for vn in normals:
                f.write(f"{vn}\n")
            
            # Write material and faces
            f.write(f"\nusemtl {mat_name}\n")
            for face in faces:
                f.write(f"{face}\n")
        
        created_files.append(str(output_file))
        print(f"Created: {output_file}")
    
    return created_files


def analyze_obj_materials(input_obj):
    """
    Analyze and print material information from an OBJ file.
    
    Args:
        input_obj: Path to input OBJ file
    """
    with open(input_obj, 'r') as f:
        lines = f.readlines()
    
    materials = {}
    current_material = None
    mtllib = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('mtllib'):
            mtllib = line.split()[1]
        elif line.startswith('usemtl'):
            mat_name = line.split()[1]
            if mat_name not in materials:
                materials[mat_name] = 0
            current_material = mat_name
        elif line.startswith('f ') and current_material:
            materials[current_material] += 1
    
    print(f"\n=== Analysis of {input_obj} ===")
    print(f"MTL file: {mtllib}")
    print(f"Number of materials: {len(materials)}")
    print(f"\nMaterial breakdown:")
    for mat, count in materials.items():
        print(f"  - {mat}: {count} faces")
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Analyze:  python obj_material_utils.py analyze <input.obj>")
        print("  Simplify: python obj_material_utils.py simplify <input.obj> <output.obj>")
        print("  Split:    python obj_material_utils.py split <input.obj> [output_dir]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "analyze":
        analyze_obj_materials(sys.argv[2])
    
    elif command == "simplify":
        if len(sys.argv) < 4:
            print("Error: Need input and output file paths")
            sys.exit(1)
        simplify_obj_single_material(sys.argv[2], sys.argv[3])
    
    elif command == "split":
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        split_obj_by_materials(sys.argv[2], output_dir)
    
    else:
        print(f"Unknown command: {command}")
        print("Available commands: analyze, simplify, split")
