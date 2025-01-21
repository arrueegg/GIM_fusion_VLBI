import os
import yaml
import shutil

# Define the base directory where all experiment folders are located
base_dir = "/cluster/work/igp_psr/arrueegg/WP2/GIM_fusion_VLBI/experiments/" 
config_file_name = "config.yaml" 

# Define the YAML keys to use for renaming
def rename_experiment_folders(base_dir, config_file_name):
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)

        # Ensure it's a directory
        if not os.path.isdir(folder_path):
            continue

        config_path = os.path.join(folder_path, config_file_name)
        if not os.path.exists(config_path):
            print(f"Config file not found in {folder_path}, skipping...")
            continue

        # Read the config file
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        except Exception as e:
            print(f"Error reading {config_path}: {e}")
            continue

        # Build the new folder name based on specified keys
        mode = config['data']['mode']
        year = config['year']
        doy = config['doy']
        sw = config['training']['vlbi_sampling_weight']
        lw = config['training']['vlbi_loss_weight']

        new_folder_name = f'{mode}_{year}_{doy}_SW{sw}_LW{lw}'
        new_folder_path = os.path.join(base_dir, new_folder_name)

        # Rename the folder if the new name doesn't already exist
        if not os.path.exists(new_folder_path):
            shutil.move(folder_path, new_folder_path)
            print(f"Renamed {folder} -> {new_folder_name}")
        else:
            print(f"Target folder {new_folder_name} already exists, skipping...")

def remove_point0(base_dir):
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)

        # Check if the folder name contains ".0"
        if ".0" in folder:
            new_folder_name = folder.replace(".0", "")
            new_folder_path = os.path.join(base_dir, new_folder_name)

            # Ensure the new folder name does not already exist
            if not os.path.exists(new_folder_path):
                os.rename(folder_path, new_folder_path)
                print(f"Renamed: {folder} -> {new_folder_name}")
            else:
                print(f"Skipping {folder}, target name already exists: {new_folder_name}")


# Run the renaming function
#rename_experiment_folders(base_dir, config_file_name)
#remove_point0(base_dir)
