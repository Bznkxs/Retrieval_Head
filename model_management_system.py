import sys
import math
import os
import os.path as osp
sys.path.append("./")
from bos_force_to_true import bos_force_to_true
from update_config import update_config
from typing import Optional
class KBasedNumber:
    def __init__(self, number_input):
        if isinstance(number_input, int):
            self.number = number_input
        elif isinstance(number_input, str):
            # is number
            if number_input.isdigit():
                self.number = int(number_input)
            # is float
            elif number_input.replace(".", "", 1).isdigit():
                self.number = float(number_input)
            else:
                number_input = number_input.lower()
                if number_input[-1] == "k":
                    self.number = int(number_input[:-1]) * 1024
                elif number_input[-1] == "m":
                    self.number = int(number_input[:-1]) * 1024 * 1024
                elif number_input == "none":
                    self.number = -1
        elif isinstance(number_input, KBasedNumber):
            self.number = number_input.number
        elif number_input is None:
            self.number = -1
        if hasattr(self, "number") is False:
            raise ValueError("Invalid number input format")

    def get_short_name(self):
        if self.number % (1024 * 1024) == 0:
            return str(self.number // (1024 * 1024)) + "m"
        if self.number % 1024 == 0:
                return str(self.number // 1024) + "k"
        return str(self.number)

    def __str__(self):
        return self.get_short_name()

    def __repr__(self):
        return "KBasedNumber(" + self.get_short_name() + ")"

    def is_good_long_context_name(self):
        return 32 * 1024 <= self.number < 2 * 1024 * 1024 and self.number % 1024 == 0

    def is_good_short_context_name(self):
        return 512 <= self.number <= 32 * 1024 and self.number % 128 == 0

    def is_good_batch_size_name(self):
        return self.number >= 2 * 1024 * 1024 and self.number % 1024 % 1024 == 0

    def is_good_sliding_window_name(self):
        return self.is_good_short_context_name() or self.number == -1

    def is_good_rotary_base_name(self):
        return self.number >= 10000 and (int(self.number) != self.number or self.number % 10000 == 0)

    def __int__(self):
        return int(self.number)

    def __float__(self):
        return float(self.number)

def get_model_information(model_path, save=True, **additional_information):
    # read information from a model path.
    default_model_info = {
        "window_size": KBasedNumber(-1),
    }
    model_info = additional_information
    model_path = model_path.strip()
    model_info["model_path"] = model_path

    def softset(key, value):
        if key not in model_info:
            model_info[key] = value
            print("True:", key, value)
            return True
        print("False:", key, value)
        return False
    model_raw_name = model_path.strip("/").split("/")[-1].lower()
    split_by_underscore = model_raw_name.split("_")
    numbers_buffer_context = []
    preserved_list = []
    print("Split by underscore:", split_by_underscore)
    for split in split_by_underscore:
        try:
            cl_test = KBasedNumber(split)
            long_context_possible = cl_test.is_good_long_context_name()
            sliding_window_possible = cl_test.is_good_sliding_window_name()
            batch_size_possible = cl_test.is_good_batch_size_name()
            rotary_base_possible = cl_test.is_good_rotary_base_name()
            if rotary_base_possible:
                print("Rotary base possible")
                if softset("rope_theta", cl_test):
                    continue
            if long_context_possible and not sliding_window_possible and not batch_size_possible:
                print("Cl possible")
                if softset("context_length", cl_test):
                    continue
            if sliding_window_possible and not long_context_possible and not batch_size_possible:
                print("Ws possible")
                if softset("window_size", cl_test):
                    continue
            if batch_size_possible and not long_context_possible and not sliding_window_possible:
                print("Bs possible")
                if softset("batch_size", cl_test):
                    continue
            if long_context_possible and sliding_window_possible and not batch_size_possible:
                numbers_buffer_context.append(cl_test)
                continue
        except ValueError:
            pass
        if split == "llama" or split == "mistral":
            model_info["model_type"] = "LLaMA" if split == "llama" else "Mistral"
            continue
        # preserve rest of the name
        preserved_list.append(split)

    if len(numbers_buffer_context) >= 2:
        print("Multiple context lengths detected:", numbers_buffer_context)
        numbers_buffer_context.sort(key=lambda x: x.number)
        preserved_list.extend(numbers_buffer_context)
    elif len(numbers_buffer_context) == 1:
        print("Single context length detected:", numbers_buffer_context[0])
        for key in ["context_length", "window_size", "nothing"]:
            if key in model_info:
                break
        if key == "nothing":
            preserved_list.extend(numbers_buffer_context)
        else:
            for key in ["context_length", "window_size"]:
                softset(key, numbers_buffer_context[0])
    print("Preserved list:", preserved_list)
    model_info["preserved_list"] = preserved_list
    model_info["print_warning"] = lambda: None
    standardized_name = []
    for k in ["model_type", "context_length", "window_size", "rope_theta", "batch_size", "preserved_list"]:

        if k not in model_info:
            if k in default_model_info:
                model_info[k] = default_model_info[k]
            else:
                model_info["Warning"] = model_info.get("Warning", []) + [f"Missing {k}"]
                model_info["print_warning"] = lambda: print("\n".join(["Warning: "+warning for warning in model_info.get("Warning", [])]))
                standardized_name.append("None")
                continue
        if k == "preserved_list":
            standardized_name.extend([str(x) for x in model_info[k]])
        else:
            standardized_name.append(str(model_info[k]))
    model_info["standardized_name"] = "_".join(standardized_name)
    if "training_step" in model_info:
        model_info["standardized_name"] += "_" + str(model_info["training_step"])
    if "save_base_path" not in model_info:
        model_info["save_base_path"] = model_path.rstrip("/").rsplit("/", 1)[0] + "/"
    if save:
        model_info["save_path"] = osp.join(model_info["save_base_path"], model_info["standardized_name"])

    return model_info

MEGATRON_PATH = None

def _set_megatron_path(megatron_path):
    global MEGATRON_PATH
    MEGATRON_PATH = megatron_path


def _get_megatron_path() -> Optional[str]:
    return MEGATRON_PATH


def convert_megatron_model_to_hf_model(megatron_model_path, *, hf_config_path=None, exit_on_existence=True, **additional_information):
    model_info = get_model_information(megatron_model_path, **additional_information)
    # if save path is not empty
    print(f"Save path: '{model_info['save_path']}'")
    if osp.exists(model_info["save_path"]) and osp.isdir(model_info["save_path"]) and len(os.listdir(model_info["save_path"])) > 0:
        if exit_on_existence:
            print(f"Model {model_info['save_path']} already exists.")
            return model_info, 0
        else:
            print(f"Warning: Model {model_info['save_path']} already exists. Overwriting.")

    training_step_file = osp.join(model_info["model_path"], "latest_checkpointed_iteration.txt")
    training_step = model_info.get("training_step", None)
    if training_step is not None and osp.exists(training_step_file):
        with open(training_step_file, "w") as f:
            f.write(str(training_step))
    model_info["print_warning"]()
    megatron_path = _get_megatron_path()
    assert megatron_path is not None, ValueError("Megatron path is not set.")
    if "model_type" not in model_info:
        raise ValueError("Model type is not set.")
    if hf_config_path is None:
        if model_info["model_type"].lower() == "llama":
            hf_config_path = "meta-llama/Llama-2-7b-hf"
        elif model_info["model_type"].lower() == "mistral":
            hf_config_path = "mistralai/Mistral-7B-v0.1"

    conversion_script = f"python {osp.join(megatron_path, 'tools/checkpoint/convert.py')}"
    #"--model-type GPT --loader mcore --saver llama3_hf --megatron-path {megatron_path} --load-dir {model_info['model_path']} --save-dir {model_info['save_path']} --hf-config-path {hf_config_path}"
    args_dict = {
        "model-type": "GPT",
        "loader": "mcore",
        "saver": "llama3_hf",
        "megatron-path": megatron_path,
        "load-dir": model_info["model_path"],
        "save-dir": model_info["save_path"],
        "hf-config-path": hf_config_path
    }
    for k, v in args_dict.items():
        conversion_script += f" --{k} {v}"


    print(conversion_script)
    rv = os.system(conversion_script)
    return model_info, rv

RETRIEVAL_HEAD_PATH = None

def _set_retrieval_head_path(retrieval_head_path):
    global RETRIEVAL_HEAD_PATH
    RETRIEVAL_HEAD_PATH = retrieval_head_path

def _get_retrieval_head_path() -> Optional[str]:
    return RETRIEVAL_HEAD_PATH

def run_NIHS_test(model_info, *, s_len=4096, e_len=131072, context_limit=131072, num_intervals=32,
                  auto_range=True, auto_interval=True, full_test=False, exp_name_suffix="", **additional_information):
    if e_len < s_len:
        return model_info, 0
    retrieval_head_path = _get_retrieval_head_path()
    assert retrieval_head_path is not None, ValueError("Retrieval head path is not set.")
    if "model_type" not in model_info:
        raise ValueError("Model type is not set.")
    model_type = "LLaMA" if model_info["model_type"].lower() == "llama" else "Mistral"
    # nihs script example: python needle_in_haystack_with_mask.py --mask_top 0 --s 4096 --e 131072 --num_intervals 32 --model_provider LLaMA --model_path "$MODEL" --model_name_suffix "true"

    if auto_range:
        context_length = model_info.get("context_length", KBasedNumber(0))
        if int(context_length) > 0:
            e_len = min(int(context_length) * 2, int(context_limit))
            print("E len:", e_len)
            if not full_test:
                s_len = int(context_length) // 2
    if auto_interval:
        step = 2048
        while s_len % step == 0 and e_len % step == 0 and step <= 8192:
            step *= 2
        step //= 2
        num_intervals = (e_len - s_len) // step + 1

    # nihs_script = (f"python {osp.join(retrieval_head_path, 'needle_in_haystack_with_mask.py')} --mask_top 0 "
    #                f"--s {s_len} --e {e_len} --num_intervals {num_intervals} --model_provider {model_type} "
    #                f"--model_path {model_info['save_path']} --model_name_suffix {exp_name_suffix}")
    nihs_script = f"python {osp.join(retrieval_head_path, 'needle_in_haystack_with_mask.py')}"
    args_dict = {
        "mask_top": 0,
        "s": s_len,
        "e": e_len,
        "num_intervals": num_intervals,
        "model_provider": model_type,
        "model_path": model_info["save_path"],
        "model_name_suffix": exp_name_suffix
    }
    for k, v in args_dict.items():
        if v != "":
            nihs_script += f" --{k} {v}"
    print(nihs_script)
    # run the script
    rv = os.system(nihs_script)
    return model_info, rv

def beautify(name):
    if name == "llama":
        return "LLaMA"
    if name == "mistral":
        return "Mistral"
    return name

def visualize(model_info, *, exp_name_suffix="", **additional_information):
    retrieval_head_path = _get_retrieval_head_path()
    assert retrieval_head_path is not None, ValueError("Retrieval head path is not set.")
    record_path = osp.join(retrieval_head_path, "results/graph", model_info["standardized_name"] + "_" + exp_name_suffix + "_" + beautify(model_info["model_type"]) + "/")
    script = f"python {osp.join(retrieval_head_path, 'visualize.py')} {record_path}"
    print(script)
    rv = os.system(script)
    return model_info, rv

def full_pipeline_for_one_model(model_path, skip_step=0, model_info=None, *, model_format=None, **additional_information):
    if skip_step == 0:
        if model_format is None:
            # try to see if there is a config.json there
            if osp.exists(osp.join(model_path, "config.json")):
                model_format = "hf"
            else:
                model_format = "megatron"
            print("Inferring model format:", model_format, "from model path.")
        if model_format == "megatron":
            print("Converting Megatron model to HF model.")
            megatron_model_path = model_path
            model_info, rv = convert_megatron_model_to_hf_model(megatron_model_path, **additional_information)
        else:
            model_info = get_model_information(model_path, **additional_information)
            rv = 0
        if rv != 0:
            return None, rv, skip_step
        skip_step = 1
    if model_info is None:
        print("Error: model_info is None while trying to skip reading model info.")
        return None, 1, 0
    if skip_step == 1:
        print("Modifying model tokenizer config to set add_bos_token to True.")
        try:
            bos_force_to_true(model_info["save_path"])
        except Exception as e:
            print("Error in modifying model tokenizer config:", e)
            return model_info, 1, skip_step
        print("Updating model config.")
        try:
            update_config(model_info["save_path"], 131072, float(model_info["rope_theta"]), "None")
        except Exception as e:
            print("Error in updating model config:", e)
            return model_info, 1, skip_step
        skip_step = 2
    if skip_step == 2:
        print("Running NIHS test.")
        model_info, rv = run_NIHS_test(model_info, **additional_information)
        if rv != 0:
            return model_info, rv, skip_step
        skip_step = 3
    if skip_step == 3:
        print("Visualizing results.")
        model_info, rv = visualize(model_info, **additional_information)
        if rv != 0:
            return model_info, rv, skip_step
        skip_step = 4
        return model_info, 0, skip_step

def step_explanation():
    return """    Step 0: Convert Megatron model to HF model. 
    Step 1: Modify model tokenizer config to set add_bos_token to True.
    Step 2: Run NIHS test."""

import yaml

def full_pipeline_for_multiple_models(experiment_yaml_file):
    skip_step = 0
    proceeded_models = []
    resume_from = (None, None, None)
    finished = False
    while not finished:
        with open(experiment_yaml_file, "r") as f:
            experiment_dict = yaml.load(f, Loader=yaml.FullLoader)
        experiment_list = experiment_dict["list"]
        global_args = experiment_dict.get("global_args", {})
        error_occurred = False
        for all_information in experiment_list:
            model_path = all_information["model_path"].strip()
            training_step = all_information.get("training_step", None)
            additional_information = dict()
            for k, v in all_information.items():
                k = k.strip()
                if k != "model_path":
                    additional_information[k] = v
                    if isinstance(v, str):
                        additional_information[k] = v.strip()
            for k, v in global_args.items():
                if k not in additional_information:
                    additional_information[k] = v

            if resume_from[0] != model_path:
                if (model_path, training_step) in proceeded_models:
                    print("Skipping", model_path, "as it has been processed.")
                    continue
            print("Processing", model_path, "with args:")
            for k, v in additional_information.items():
                print(f"    {k:20}: {v}")
            if resume_from[0] == model_path:
                skip_step = resume_from[1]
                model_info = resume_from[2]
            else:
                skip_step = 0
                model_info = None
            model_info, rv, skip_step = full_pipeline_for_one_model(model_path, skip_step, model_info, **additional_information)
            if rv != 0:
                print(f"Error in processing {model_path}: step {skip_step} failed. Please modify the yaml file "
                      f"Before proceeding. Then, press Enter to continue from step {skip_step} or "
                      f"input the number of a specific step to start over. Explanation of steps: {step_explanation()}")
                option = input()
                if option.isdigit():
                    resume_from = (model_path, int(option))
                error_occurred = True
                break
            else:
                proceeded_models.append((model_path, training_step))
        if not error_occurred:
            finished = True

    return 0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Full pipeline for converting Megatron models to HF models and running NIHS tests.")
    parser.add_argument("--megatron_path", type=str, help="Path to Megatron-LM repo.")
    parser.add_argument("--retrieval_head_path", type=str, help="Path to retrieval head scripts.")
    parser.add_argument("--experiment_yaml_file", type=str, help="Path to experiment yaml file.")
    args = parser.parse_args()
    if args.megatron_path:
        _set_megatron_path(args.megatron_path)
    if args.retrieval_head_path:
        _set_retrieval_head_path(args.retrieval_head_path)
    if args.experiment_yaml_file:
        full_pipeline_for_multiple_models(args.experiment_yaml_file)