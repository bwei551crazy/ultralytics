"""
Scrip to inspect and compare the number of frames and object annotations in xml files. 
"""
import xml.etree.ElementTree as ET

#place within the dataset folder that uses .xml files for annotations

def inspect_xml_structure(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    print("XML Structure:")
    print(f"Root tag: {root.tag}")
    
    # Check all child elements
    for child in root:
        print(f"Child: {child.tag} - Attributes: {child.attrib}")
        
    # Count frames and objects
    frames = root.findall('.//frame')
    print(f"\nTotal frames in XML: {len(frames)}")
    
    return tree

if __name__ == "__main__":
# Usage
    xml_tree = inspect_xml_structure('/home/yanjiaqi/own_ultralytics/ultralytics/datasets/ua-detrac/DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML/MVI_63544.xml')