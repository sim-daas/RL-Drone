import re

def sanitize_mtl(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    fixed_lines = []
    for line in lines:
        # Match map_Kd, map_Bump, map_Ke lines
        if line.strip().startswith(('map_Kd', 'map_Bump', 'map_Ke')):
            # Remove any flags like -s, -bm, -o and their subsequent values
            # We only want the last string (the actual filename)
            parts = line.split()
            filename = parts[-1]
            
            # Reconstruct the line as a clean 'map_Kd filename.jpg'
            tag = parts[0]
            fixed_lines.append(f"{tag} {filename}\n")
        else:
            fixed_lines.append(line)

    with open(output_file, 'w') as f:
        f.writelines(fixed_lines)
    print(f"✅ Sanitized MTL saved to: {output_file}")

if __name__ == "__main__":
    sanitize_mtl("base_visual.mtl", "base_visual_fixed.mtl")