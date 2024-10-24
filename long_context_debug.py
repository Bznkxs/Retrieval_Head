import os
import sys
import glob
import json
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import random
import numpy as np
import argparse
from rouge_score import rouge_scorer
import torch

needle = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
haystack_dir = "PaulGrahamEssays"
retrieval_question = "What is the best thing to do in San Francisco?"
results_version = 1
context_lengths_min = 1000
context_lengths_max = 128000
context_lengths_num_intervals = 40
context_lengths = None
document_depth_percent_min = 0
document_depth_percent_max = 100
document_depth_percent_intervals = 10
document_depth_percents = None
document_depth_percent_interval_type = "linear"
mask_topk = 0
num_concurrent_requests = 1
save_results = True
save_contexts = True
final_context_length_buffer = 200
seconds_to_sleep_between_completions = None
print_ongoing_status = True

#model_name = '/work/nvme/bcbw/mtian8/converted_HF/llama2_7B_128k_-1_1_nolocal'
# model_name = "/u/yufengd4/projects_bbzy/models/Llama-2-7b-hf"
# model_name = "/u/yufengd4/projects_bbzy/models/Meta-Llama-3-8B"
# model_name = "meta-llama/Llama-2-7b-hf"
model_name = 'yaofu/llama-2-7b-80k'
device = 'cuda:0'
max_context_length = 4096

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
sliding_window = getattr(config, "sliding_window", None)
model_to_test = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2",
                                                     torch_dtype=torch.bfloat16,
                                                     ).eval()
model_to_test.to(device)

def set_sliding_window(new_sliding_window):
    for decode_layer in model_to_test.model.layers:
        decode_layer.self_attn.sliding_window = new_sliding_window
        print("Setting sliding window to" , decode_layer.self_attn.sliding_window)

sliding_window = 2048
set_sliding_window(sliding_window)

context = ""
def generate_context(context_length):
    def get_context_length_in_tokens(context):
        return len(tokenizer.encode(context))

    def decode_tokens(tokens, context_length=None):
        return tokenizer.decode(tokens[:context_length])


    # Load up tiktoken so we navigate tokens more easily

    # Get your Paul Graham files loaded into a string
    global context
    while get_context_length_in_tokens(context) < max_context_length:
        for file in glob.glob(f"{haystack_dir}/*.txt"):
            with open(file, 'r') as f:
                context += f.read()

    # Truncate the Paul Graham essays to the context length you desire
    tokens = tokenizer.encode(context)
    if len(tokens) > context_length:
        context = decode_tokens(tokens, context_length)

generate_context(context_length=max_context_length)

question = ". Mary is a chemist. She works for BlueSky Chemicals. Bob is a shop attendant at a Dollar General right next to BlueSky Chemicals. Question: What is Mary's occupation? Answer: "
question = ". Mary is a chemist. She works for BlueSky Chemicals. Bob is a shop attendant at a Dollar General right next to BlueSky Chemicals. Question: What is Bob's occupation? Answer: "
with torch.no_grad():
    torch.cuda.empty_cache()

offset = 500
model_input = tokenizer.encode(context[:-offset] + question, return_tensors="pt").to(device)
print(len(model_input[0]))
output = model_to_test.generate(model_input, max_new_tokens=100, do_sample=False, temperature=0, pad_token_id=tokenizer.eos_token_id)
print(len(output[0]))
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(tokenizer.decode(output[0][len(model_input[0]):]))

print(decoded_output[len(context) - offset - 10:])

