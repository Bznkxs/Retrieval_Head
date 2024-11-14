import sys
import json
import os
import os.path as osp
def bos_force_to_true(model_path):
    tokenizer_config_path = model_path
    if tokenizer_config_path.find("tokenizer_config.json") == -1:
        tokenizer_config_path = f"{osp.join(model_path, 'tokenizer_config.json')}"
    tokenizer_config_path = tokenizer_config_path.strip()
    if not osp.exists(tokenizer_config_path):
        raise FileNotFoundError(f"'{tokenizer_config_path}' not found")
    # open "tokenizer_config.json" and set "add_bos_token" to "true"
    with open(tokenizer_config_path, "r") as f:
        tokenizer_config = json.load(f)
    tokenizer_config["add_bos_token"] = True
    with open(tokenizer_config_path, "w") as f:
        json.dump(tokenizer_config, f)



if __name__ == "__main__":
    for model_path in sys.argv[1:]:
        try:
            bos_force_to_true(model_path)
            print("Modified", model_path)
        except Exception as e:
            print(f"Warning: Skipped {model_path}: {e}")
