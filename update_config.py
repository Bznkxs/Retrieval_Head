import os
import json
import argparse

def update_config(directory, max_position_embeddings, rope_theta, window_size):
    # Traverse the directory for config.json files
    for root, _, files in os.walk(directory):
        for file in files:
            if file == "config.json":
                config_path = os.path.join(root, file)
                try:
                    # Load the config.json file
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    
                    # Update the values
                    if "max_position_embeddings" in config_data:
                        config_data["max_position_embeddings"] = max_position_embeddings
                    if "rope_theta" in config_data:
                        config_data["rope_theta"] = rope_theta
                    if len(window_size) > 0:
                        config_data["sliding_window"] = int(window_size)
                    elif "sliding_window" in config_data:
                        config_data.pop("sliding_window")
                    # Write back the updated config
                    with open(config_path, 'w') as f:
                        json.dump(config_data, f, indent=2)
                    
                    print(f"Updated {config_path}")
                except Exception as e:
                    print(f"Failed to update {config_path}: {e}")

if __name__ == "__main__":
    # Parse arguments from the command line
    parser = argparse.ArgumentParser(description='Update max_position_embeddings and rope_theta in config.json files.')
    parser.add_argument('directory', type=str, help='The directory to search for config.json files.')
    parser.add_argument('max_position_embeddings', type=int, help='The value for max_position_embeddings.')
    parser.add_argument('rope_theta', type=float, help='The value for rope_theta.')
    parser.add_argument('window_size', nargs='?', type=str, default="", help='sliding window size')

    args = parser.parse_args()

    # Call the function to update the config files
    update_config(args.directory, args.max_position_embeddings, args.rope_theta, args.window_size)

