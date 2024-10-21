"""
Author: Arno Rüegg
Date: 2024-10-21
Description: Global Ionospheric Maps from GNSS and VLBI data
"""

from utils.config_parser import parse_config

def pipeline(config):
    # calles all function to load data, model and train
    pass

def main():
    
    config = parse_config()

    # Example usage: Accessing parameters
    print(f"Project Name: {config['project_name']}")
    print(f"Model: {config['model']['model_type']}")
    print(f"Debugging Enabled: {config['debugging']['enable_debugging']}")

    pipeline(config)

if __name__ == "__main__":
    main()

