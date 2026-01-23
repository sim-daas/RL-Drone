import xml.etree.ElementTree as ET

def convert_sdf_to_urdf_collision(sdf_file):
    tree = ET.parse(sdf_file)
    root = tree.getroot()
    
    urdf_blocks = []
    
    # Iterate through all collision elements in the SDF [cite: 33, 35, 42]
    for collision in root.findall(".//collision"):
        name = collision.get("name", "collision_part")
        pose_text = collision.find("pose").text if collision.find("pose") is not None else "0 0 0 0 0 0"
        size_text = collision.find(".//size").text if collision.find(".//size") is not None else "1 1 1"
        
        # SDF Pose: x y z r p y [cite: 33, 35]
        coords = pose_text.split()
        xyz = f"{coords[0]} {coords[1]} {coords[2]}"
        rpy = f"{coords[3]} {coords[4]} {coords[5]}"
        
        # Generate URDF block 
        block = f"""    <collision name="{name}">
      <origin xyz="{xyz}" rpy="{rpy}"/>
      <geometry>
        <box size="{size_text}"/>
      </geometry>
    </collision>"""
        urdf_blocks.append(block)
    
    with open("urdf_collision_blocks.txt", "w") as f:
        f.write("\n".join(urdf_blocks))
    
    print(f"✅ Converted {len(urdf_blocks)} collision elements to urdf_collision_blocks.txt")

if __name__ == "__main__":
    convert_sdf_to_urdf_collision("model.sdf")