
import yaml
import json

INPUT_FILE = "/root/Athena/caliper-deploy-tool/action-best.yaml"
OUTPUT_FILE = "/root/Athena/caliper-deploy-tool/action.yaml"

def convert():
    with open(INPUT_FILE, 'r') as f:
        # It's actually JSON content in a .yaml file based on my previous Write
        data = json.load(f)
        
    flat_data = {}
    
    for section, params in data.items():
        flat_data[section] = {}
        for key, val_dict in params.items():
            value = val_dict['value']
            unit = val_dict.get('unit')
            
            if unit:
                flat_data[section][key] = f"{value}{unit}"
            else:
                flat_data[section][key] = value
                
    # Save as YAML
    with open(OUTPUT_FILE, 'w') as f:
        yaml.dump(flat_data, f, default_flow_style=False)
    
    print(f"Converted {INPUT_FILE} to {OUTPUT_FILE}")

if __name__ == "__main__":
    convert()
