
import json
import glob
import math
import sys
import os
# FOLDER_PATH = "results/mask_0.2_0.2_rescale_True/"
MODEL_NAME = ""

# "LLaMA 2 7B continue-trained on 5B tokens 80K length Per-source length upsampled data"
PRETRAINED_LEN=300000

def main(folder_path):
    # Path to the directory containing JSON results
    if folder_path[-1] != "/":
        folder_path += "/"
    if("/" in folder_path):
        # extract the last part of the path
        model_name = os.path.basename(os.path.normpath(folder_path))
    else: model_name = MODEL_NAME
    print()
    print()
    print("====================================================")
    print("model_name = %s" % model_name)

    # Using glob to find all json files in the directory
    json_files = glob.glob(f"{folder_path}*.json")
    print(f"{len(json_files)} files found")
    # print(json_files)
    # import ipdb; ipdb.set_trace()

    # List to hold the data
    data = []
    hor_score_map = {}
    ver_score_map = {}

    score_divisions = {}
    divisions_h = [32768, 65536, 131072]
    divisions_v = [2048, 4096]

    # Iterating through each file and extract the 3 columns we need
    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            # Extracting the required fields
            document_depth = json_data.get("depth_percent", None)
            context_length = json_data.get("context_length", None)
            score = json_data.get("score", None)
            if context_length not in hor_score_map:
                hor_score_map[context_length] = {"sum": 0, "count": 0}
            hor_score_map[context_length]["sum"] += score
            hor_score_map[context_length]["count"] += 1
            if document_depth not in ver_score_map:
                ver_score_map[document_depth] = {"sum": 0, "count": 0}
            ver_score_map[document_depth]["sum"] += score
            ver_score_map[document_depth]["count"] += 1

            for i_h in divisions_h:
                if context_length <= i_h:
                    for j_v in divisions_v:
                        if document_depth <= j_v:
                            if (i_h, j_v) not in score_divisions:
                                score_divisions[(i_h, j_v)] = {"sum": 0, "count": 0}
                            score_divisions[(i_h, j_v)]["sum"] += score
                            score_divisions[(i_h, j_v)]["count"] += 1
                            break
                    break

    for k in score_divisions:
        if score_divisions[k]["count"] == 0:
            score_divisions[k] = math.nan
        else:
            score_divisions[k] = score_divisions[k]["sum"] / score_divisions[k]["count"]

    print(end="\t", )
    for h in divisions_h:
        print(f"{h//1024}k\t", end="")
    print()
    print("-----------------------------------------------------------")
    for v in divisions_v:
        print(f"{v//1024}k:\t", end="")
        for h in divisions_h:
            print(f"{score_divisions.get((h, v), math.nan)}\t", end="")
        print()



    # calculate average score of 0~32768, 32769~65536, 65537~131072
    avg_score = {'32k': 0., '64k': 0., '128k': 0., 'overall': 0}
    count = {'32k': 0., '64k': 0., '128k': 0., 'overall': 0}

    for k, v in hor_score_map.items():
        avg_score['overall'] += v['sum']
        count['overall'] += v['count']

        if k <= 32768:
            avg_score['32k'] += v['sum']
            count['32k'] += v['count']
        elif k <= 65536:
            avg_score['64k'] += v['sum']
            count['64k'] += v['count']
        else:
            avg_score['128k'] += v['sum']
            count['128k'] += v['count']
    if count['32k'] == 0: count['32k'] = 1
    if count['64k'] == 0: count['64k'] = 1
    if count['128k'] == 0: count['128k'] = 1
    if count['overall'] == 0: count['overall'] = 1
    avg_score['32k'] /= count['32k']
    avg_score['64k'] /= count['64k']
    avg_score['128k'] /= count['128k']
    avg_score['overall'] /= count['overall']
    print(avg_score)

    avg_ver_score = {'2k': 0, '4k': 0, '8k': 0., '12k': 0., '16k': 0.}
    count = {'2k': 0, '4k': 0, '8k': 0., '12k': 0., '16k': 0.}
    for k, v in ver_score_map.items():
        if k <= 2048:
            avg_ver_score['2k'] += v['sum']
            count['2k'] += v['count']
        elif k <= 4096:
            avg_ver_score['4k'] += v['sum']
            count['4k'] += v['count']
        elif k <= 8192:
            avg_ver_score['8k'] += v['sum']
            count['8k'] += v['count']
        elif k <= 12288:
            avg_ver_score['12k'] += v['sum']
            count['12k'] += v['count']
        else:
            avg_ver_score['16k'] += v['sum']
            count['16k'] += v['count']
    if count['2k'] == 0: count['2k'] = 1
    if count['4k'] == 0: count['4k'] = 1
    if count['8k'] == 0: count['8k'] = 1
    if count['12k'] == 0: count['12k'] = 1
    if count['16k'] == 0: count['16k'] = 1
    avg_ver_score['2k'] /= count['2k']
    avg_ver_score['4k'] /= count['4k']
    avg_ver_score['8k'] /= count['8k']
    avg_ver_score['12k'] /= count['12k']
    avg_ver_score['16k'] /= count['16k']
    print(avg_ver_score)



    print("====================================================")
    # save_path = "img/%s.1.png" % model_name
    # print("saving at %s" % save_path)
    # plt.savefig(save_path, dpi=150)
    # save_path = "img/%s.2.png" % model_name
    # print("saving at %s" % save_path)
    # plt.savefig(save_path, dpi=150)
if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder_paths = sys.argv[1:]
        folder_paths = [f for folder_path in folder_paths for f in glob.glob(folder_path)]
    else:
        folder_paths = [
        # "results/graph/lco_0_keep_1024_rescale_True_Mistral/",
        # "results/graph/lco_0.0001_keep_284_rescale_True_Mistral/",
        # "results/graph/lco_0.001_keep_254_rescale_True_Mistral/",
        # "results/graph/lco_0.005_keep_199_rescale_True_Mistral/",
        # "results/graph/lco_0.01_keep_168_rescale_True_Mistral/",
        # "results/graph/lco_0.02_rescale_True_Mistral/",
        #"results/graph/MistralLite_Mistral/",
        # "results/graph/lco_0_keep_1024_rescale_True_substitute_io_False_substitute_linear_False_Mistral/",
        # "results/graph/lco_0_keep_1024_rescale_True_substitute_io_False_substitute_linear_True_Mistral/",
        # "results/graph/lco_0_keep_1024_rescale_True_substitute_io_True_substitute_linear_False_Mistral/",
        # "results/graph/lco_0_keep_1024_rescale_True_substitute_io_True_substitute_linear_True_Mistral/",
        # "results/graph/lco_0_keep_1024_rescale_True_substitute_io_False_substitute_linear_False_resized_False_Mistral/",
        # "results/graph/lco_0_keep_1024_rescale_True_substitute_io_False_substitute_linear_True_resized_False_Mistral/",
        # "results/graph/lco_0.1_keep_34_rescale_True_substitute_io_False_substitute_linear_True_resized_False_lkr_0.125_linear_rescale_True_Mistral/",
        # "results/graph/lco_0.05_keep_81_rescale_True_substitute_io_False_substitute_linear_True_resized_False_lkr_0.125_linear_rescale_True_Mistral/",
        # "results/graph/lco_0.03_keep_120_rescale_True_substitute_io_False_substitute_linear_True_resized_False_lkr_0.125_linear_rescale_True_Mistral/",
        # "results/graph/lco_0.02_keep_144_rescale_True_substitute_io_False_substitute_linear_True_resized_False_lkr_0.125_linear_rescale_True_Mistral/",
        # "results/graph/lco_0.01_keep_168_rescale_True_substitute_io_False_substitute_linear_True_resized_False_lkr_0.125_linear_rescale_True_Mistral/",
        # "results/graph/lco_0.001_keep_254_rescale_True_substitute_io_False_substitute_linear_True_resized_False_lkr_0.125_linear_rescale_True_Mistral/",
        # "results/graph/m_0.001_keep_254_rescale_True_substitute_io_False_substitute_linear_True_resized_False_lkr_0.125_linear_rescale_True_Mistral/",
        # "results/graph/m_0.001_keep_254_rescale_True_substitute_io_False_substitute_linear_True_resized_False_lkr_0.1_linear_rescale_True_Mistral/",
        # "results/graph/m_0.001_keep_254_rescale_True_substitute_io_False_substitute_linear_True_resized_False_lkr_0.2_linear_rescale_True_Mistral/",
        # "results/graph/m_0.01_keep_168_rescale_True_substitute_io_False_substitute_linear_True_resized_False_lkr_0.125_linear_rescale_True_Mistral/",
        # "results/graph/m_0.01_keep_168_rescale_True_substitute_io_False_substitute_linear_True_resized_False_lkr_0.1_linear_rescale_True_Mistral/",
        # "results/graph/m_0.01_keep_168_rescale_True_substitute_io_False_substitute_linear_True_resized_False_lkr_0.2_linear_rescale_True_Mistral/",
        # "results/graph/m_0.03_keep_120_rescale_True_substitute_io_False_substitute_linear_True_resized_False_lkr_0.125_linear_rescale_True_Mistral/",
        # "results/graph/m_0.03_keep_120_rescale_True_substitute_io_False_substitute_linear_True_resized_False_lkr_0.2_linear_rescale_True_Mistral/",
        # "results/graph/lco_0.03_keep_120_rescale_True_substitute_io_False_substitute_linear_True_resized_False_lkr_0.1_linear_rescale_True_simple_delta_linear_merging_method_simple_delta_mlpk_False_False_False_Mistral/",
        # "results/graph/lco_0.03_keep_120_rescale_True_substitute_io_False_substitute_linear_True_resized_False_lkr_0.1_linear_rescale_True_simple_delta_linear_merging_method_simple_delta_mlpk_False_False_True_Mistral/",
        #"results/graph/lco_0.03_keep_120_rescale_True_substitute_io_False_substitute_linear_True_resized_False_lkr_0.1_linear_rescale_True_simple_delta_linear_merging_method_simple_delta_mlpk_True_False_False_Mistral/",
        # "results/graph/lco_0.03_keep_120_rescale_True_substitute_io_False_substitute_linear_True_resized_False_lkr_0.1_linear_rescale_True_simple_delta_linear_merging_method_simple_delta_mlpk_True_False_True_Mistral/",
        # "results/graph/lco_0.03_keep_120_rescale_True_substitute_io_False_substitute_linear_True_resized_False_lkr_0.1_linear_rescale_True_simple_delta_linear_merging_method_simple_delta_mlpk_False_True_False_Mistral/",
        # "results/graph/lco_0.03_keep_120_rescale_True_substitute_io_False_substitute_linear_True_resized_False_lkr_0.1_linear_rescale_True_simple_delta_linear_merging_method_simple_delta_mlpk_False_True_True_Mistral/",
        # "results/graph/lco_0.03_keep_120_rescale_True_substitute_io_False_substitute_linear_True_resized_False_lkr_0.1_linear_rescale_True_simple_delta_linear_merging_method_simple_delta_mlpk_True_True_False_Mistral/",
        # "results/graph/lco_0.03_keep_120_rescale_True_substitute_io_False_substitute_linear_True_resized_False_lkr_0.1_linear_rescale_True_simple_delta_linear_merging_method_simple_delta_mlpk_True_True_True_Mistral/",
        "results/graph/_save_checkpoint_150_Mistral/"
        ]
    print(folder_paths)
    for folder_path in folder_paths:
        main(folder_path)
