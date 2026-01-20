import xml.etree.ElementTree as ET
from pathlib import Path
import os

def combine_urdf_files(input_dir, output_file, root_urdf="planer.urdf"):
    """
    Combine all URDF files in input_dir into a single URDF file.
    Uses the floor link from root_urdf as the base.
    
    Args:
        input_dir: Directory containing URDF files
        output_file: Output combined URDF file path
        root_urdf: Base URDF file (default: planer.urdf)
    """
    input_path = Path(input_dir)
    urdf_files = sorted(input_path.glob("*.urdf"))
    
    # Remove the output file and root file from the list if present
    urdf_files = [f for f in urdf_files if f.name != Path(output_file).name and f.name != root_urdf]
    
    # Parse root URDF
    root_path = input_path / root_urdf
    if not root_path.exists():
        print(f"Error: Root URDF {root_urdf} not found!")
        return
    
    root_tree = ET.parse(root_path)
    root_robot = root_tree.getroot()
    root_floor_link = root_robot.find(".//link[@name='floor']")
    
    if root_floor_link is None:
        print(f"Error: No 'floor' link found in {root_urdf}")
        return
    
    # Create new combined robot
    combined_robot = ET.Element("robot", name="combined_warehouse")
    
    # Add root floor link
    combined_robot.append(root_floor_link)
    
    print(f"✅ Added root link 'floor' from {root_urdf}")
    
    # Process each URDF file
    for urdf_file in urdf_files:
        print(f"\n📄 Processing {urdf_file.name}...")
        tree = ET.parse(urdf_file)
        robot = tree.getroot()
        
        # Get base name for prefixing
        base_name = urdf_file.stem
        
        # Extract all links
        links = robot.findall("link")
        
        for link in links:
            original_name = link.get("name")
            
            # Rename link to avoid collision
            new_link_name = f"{base_name}_{original_name}"
            link.set("name", new_link_name)
            
            # Add link to combined robot
            combined_robot.append(link)
            
            # Create fixed joint connecting to root floor
            joint = ET.Element("joint", name=f"floor_to_{new_link_name}", type="fixed")
            
            parent = ET.SubElement(joint, "parent", link="floor")
            child = ET.SubElement(joint, "child", link=new_link_name)
            origin = ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
            
            combined_robot.append(joint)
            
            print(f"  ✅ Added link '{new_link_name}' with fixed joint to floor")
    
    # Create output tree
    tree = ET.ElementTree(combined_robot)
    ET.indent(tree, space="  ")
    
    # Write to file
    output_path = Path(output_file)
    with open(output_path, 'wb') as f:
        f.write(b'<?xml version="1.0"?>\n')
        tree.write(f, encoding='utf-8', xml_declaration=False)
    
    print(f"\n🚀 SUCCESS: Combined URDF saved to {output_file}")
    print(f"   Total files merged: {len(urdf_files) + 1} (including {root_urdf})")

if __name__ == "__main__":
    # Combine all URDF files in converted_assets folder
    combine_urdf_files(
        input_dir="../converted_assets",
        output_file="../converted_assets/full.urdf",
        root_urdf="planer.urdf"
    )
