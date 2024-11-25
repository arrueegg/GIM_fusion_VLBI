import yaml
import argparse
import os
import hashlib
import json

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Global Ionospheric Maps from GNSS and VLBI data")
    parser.add_argument('--config_path', type=str, default='config/config.yaml', help='Path to the config file')
    parser.add_argument('--year', type=int, help='year of data to process')
    parser.add_argument('--doy', type=int, help='day of year of data to process')
    parser.add_argument('--mode', type=str, help='Override training mode from config')
    parser.add_argument('--model_type', type=str, help='Override model type from config')
    parser.add_argument('--loss_fn', type=str, help='Override Loss function from config')
    parser.add_argument('--vlbi_loss_weight', type=float, help='Override VLBI loss weight from config')
    parser.add_argument('--vlbi_sampling_weight', type=float, help='Override VLBI sampling weight from config')
    parser.add_argument('--debug', type=str, help='Enable debugging mode') 
    return parser.parse_args() 

def get_hash(config):
    # Convert the config to a string in a consistent order
    config_string = yaml.dump(config, sort_keys=True)
    # Generate an MD5 hash of the config string
    hash = hashlib.md5(config_string.encode()).hexdigest()
    return hash

def create_experiment(config):
    hash = get_hash(config)
    # Ensure the experiment directory exists
    config['output_dir'] = f"experiments/{hash}"
    os.makedirs(f"experiments/{hash}", exist_ok=True)
    config_path = os.path.join(config['output_dir'], 'config.yaml')

    # Save the config to a YAML file
    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file, default_flow_style=False)

def parse_config():
    args = parse_args()
    config = load_config(args.config_path)

    # Override config parameters with arguments if provided
    if args.year:
        config['year'] = args.year
    if args.doy:
        config['doy'] = args.doy
    if args.mode:
        config['data']['mode'] = args.mode
    if args.model_type:
        config['model']['model_type'] = args.model_type
    if args.debug is not None:
        config['debugging']['debug'] = args.debug.lower() in ["true", "1", "yes"]
    if args.loss_fn:
        config['training']['loss_function'] = args.loss_fn
    if args.vlbi_loss_weight:
        config['training']['vlbi_loss_weight'] = args.vlbi_loss_weight
    if args.vlbi_sampling_weight:
        config['training']['vlbi_sampling_weight'] = args.vlbi_sampling_weight
    
    if config["training"]["loss_function"] == 'LaplaceLoss':
        config["model"]["output_size"] = 2
        config["model"]["apply_softplus"] = True

    if config["training"]["loss_function"] == 'GaussianNLLLoss':
        config["model"]["output_size"] = 2

    config['year'] = str(config['year'])
    config["doy"] = str(config['doy']).zfill(3)

    # calc input_size
    config["model"]["input_size"] = 3
    if config["preprocessing"]["SH_encoding"]:
        config["model"]["input_size"] += config["preprocessing"]["SH_degree"]**2 
    else:
        config["model"]["input_size"] += 2

    create_experiment(config)
    return config


