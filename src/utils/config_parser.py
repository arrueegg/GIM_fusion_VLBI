import yaml
import argparse

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Global Ionospheric Maps from GNSS and VLBI data")
    parser.add_argument('--config_path', type=str, default='config/config.yaml', help='Path to the config file')
    parser.add_argument('--year', type=int, help='year of data to process')
    parser.add_argument('--doy', type=int, help='day of year of data to process')
    parser.add_argument('--model_type', type=str, help='Override model type from config')
    parser.add_argument('--enable_debugging', type=bool, help='Override debugging setting from config')
    return parser.parse_args()

def parse_config():
    args = parse_args()
    config = load_config(args.config_path)

    # Override config parameters with arguments if provided
    if args.year:
        config['year'] = args.year
    if args.doy:
        config['doy'] = args.doy
    if args.model_type:
        config['model']['model_type'] = args.model_type
    if args.enable_debugging is not None:
        config['debugging']['enable_debugging'] = args.enable_debugging

    config['year'] = str(config['year'])
    config["doy"] = str(config['doy']).zfill(3)
    return config