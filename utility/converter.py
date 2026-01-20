import os

def process_warehouse_model(obj_filename, mtl_filename):
    if not os.path.exists(obj_filename):
        print(f"Error: {obj_filename} not found.")
        return

    with open(obj_filename, 'r') as f:
        lines = f.readlines()

    header = []
    materials = {}
    current_mtl = None

    # Parse OBJ for geometry and materials
    for line in lines:
        if line.startswith(('v ', 'vt ', 'vn ')):
            header.append(line)
        elif line.startswith('usemtl '):
            current_mtl = line.split()[1]
            if current_mtl not in materials:
                materials[current_mtl] = []
        elif line.startswith('f ') and current_mtl:
            materials[current_mtl].append(line)

    xml_output = ""
    
    # Generate splitted OBJ files and XML strings
    for mtl, faces in materials.items():
        part_name = f"part_{mtl}.obj"
        with open(part_name, 'w') as f:
            f.write(f"mtllib {mtl_filename}\n")
            f.writelines(header)
            f.write(f"usemtl {mtl}\n")
            f.writelines(faces)
        
        # Create the URDF visual block
        xml_output += f"""    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{part_name}" scale="1 1 1"/>
      </geometry>
    </visual>\n"""
        print(f"✅ Created: {part_name}")

    # Save the XML block to a text file for easy copy-pasting
    with open("urdf_visual_block.txt", "w") as f:
        f.write(xml_output)
    
    print("\n🚀 SUCCESS: Copy the contents of 'urdf_visual_block.txt' into your URDF.")

if __name__ == "__main__":
    process_warehouse_model("base_visual.obj", "base_visual.mtl")