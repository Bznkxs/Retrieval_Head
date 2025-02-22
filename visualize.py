
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import json
import glob
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

            model_response = json_data.get("model_response", None).lower()
            needle = json_data.get("needle", None).lower()
            expected_answer = "eat a sandwich and sit in Dolores Park on a sunny day.".lower().split()
            # score = len(set(model_response.split()).intersection(set(expected_answer))) / len(expected_answer)
            # Appending to the list
            data.append({
                "Document Depth": document_depth,
                "Context Length": context_length,
                "Score": score
            })

    # calculate average score of 0~32768, 32769~65536, 65537~131072
    avg_score = {'32k': 0., '64k': 0., '128k': 0.}
    count = {'32k': 0., '64k': 0., '128k': 0.}
    for k, v in hor_score_map.items():
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
    avg_score['32k'] /= count['32k']
    avg_score['64k'] /= count['64k']
    avg_score['128k'] /= count['128k']
    print(avg_score)

    avg_ver_score = {'8k': 0., '12k': 0., '16k': 0.}
    count = {'8k': 0., '12k': 0., '16k': 0.}
    for k, v in ver_score_map.items():
        if k <= 8192:
            avg_ver_score['8k'] += v['sum']
            count['8k'] += v['count']
        elif k <= 12288:
            avg_ver_score['12k'] += v['sum']
            count['12k'] += v['count']
        else:
            avg_ver_score['16k'] += v['sum']
            count['16k'] += v['count']
    if count['8k'] == 0: count['8k'] = 1
    if count['12k'] == 0: count['12k'] = 1
    if count['16k'] == 0: count['16k'] = 1
    avg_ver_score['8k'] /= count['8k']
    avg_ver_score['12k'] /= count['12k']
    avg_ver_score['16k'] /= count['16k']
    print(avg_ver_score)



    # Creating a DataFrame
    df = pd.DataFrame(data)
    # print(df)
    try:
        locations = list(df["Context Length"].unique())
    except Exception as e:
        print(e)
        # exit()
    locations.sort()
    for li, l in enumerate(locations):
        print(li, l)
        if(l > PRETRAINED_LEN): break
    pretrained_len = li

    # print(df.head())
    print("Overall score %.3f" % df["Score"].mean())



    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
    print(pivot_table.iloc[:5, :50])

    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with better aesthetics
    f = plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    heatmap = sns.heatmap(
        pivot_table,
        vmin=0, vmax=100,
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        linewidths=0.5,  # Adjust the thickness of the grid lines here
        linecolor='grey',  # Set the color of the grid lines
        linestyle='--'
    )


    # More aesthetics
    model_name_ = model_name
    print("model_name_ = %s" % model_name_)
    title = f'Pressure Testing {model_name_} \nFact Retrieval Across Context Lengths ("Needle In A HayStack")'
    print("Title = %s" % title)
    plt.title(title)  # Adds a title
    plt.xlabel('Token Limit')  # X-axis label
    plt.ylabel('Depth')  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area

    # Add a vertical line at the desired column index
    plt.axvline(x=pretrained_len + 0.8, color='white', linestyle='--', linewidth=4)

    # add average score below the figure
    plt.text(0, -1, f"Average score of 0~32k: {avg_score['32k']:.3f}", fontsize=12)
    plt.text(0, -2, f"Average score of 32k~64k: {avg_score['64k']:.3f}", fontsize=12)
    plt.text(0, -3, f"Average score of 64k~128k: {avg_score['128k']:.3f}", fontsize=12)


    save_path = "img/%s.png" % model_name
    print("saving at %s" % save_path)
    plt.savefig(save_path, dpi=150)
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
        print(folder_path)
        main(folder_path)
