import sys

import requests
import json
sys.path.append("./faiss_attn")
# sys.path.append("../Long-Context-Data-Engineering/eval/needle/Retrieval_Head/faiss_attn/")
from source.modeling_llama import LlamaForCausalLM, LlamaConfig
from source.modeling_qwen2 import Qwen2ForCausalLM
from source.modeling_mixtral import MixtralForCausalLM
from source.modeling_mistral import MyMistralForCausalLM
from source.modeling_phi3 import Phi3ForCausalLM
from transformers import AutoTokenizer
import numpy as np
import argparse
import torch
from test_lcs import longest_common_subsequence

def lcs(string_a, string_b):
    # longest common substring.
    # len_a = len(string_a)
    # len_b = len(string_b)
    # # brute-force O(len_a * len_b) DP
    # longest_cs = [[0] * len_b, [0] * len_b]
    # answer = (0, 0, 0)
    #
    # for i in range(len_a):
    #     for j in range(len_b):
    #         first_dim = i % 2
    #         previous_first_dim = (i+1) % 2
    #         if i == 0 or j == 0:
    #             longest_cs[first_dim][j] = 0
    #             continue
    #         if string_a[i] == string_b[j]:
    #             longest_cs[first_dim][j] = longest_cs[previous_first_dim][j - 1] + 1
    #             if answer[2] < longest_cs[first_dim][j]:
    #                 answer = (i, j, longest_cs[first_dim][j])
    # common_string = longest_common_subsequence(string_a, string_b)
    # answer_a = string_a.find(common_string) + len(common_string) - 1
    # answer_b = string_b.find(common_string) + len(common_string) - 1
    # return answer_a, answer_b, len(common_string)
    print(f"[{string_a[:100]}]")
    print(f"[{string_b[:100]}]")
    print(f"[{string_a[-100:]}]")
    print(f"[{string_b[-100:]}]")
    for offset_b in range(0, len(string_b)):
        print("B:", offset_b)
        for offset_a in range(0, offset_b + 1):
            print(offset_a, offset_b, end="; ")
            l = 0
            for l in range(0, len(string_b) - offset_b):
                if string_a[offset_a+l] != string_b[offset_b+l]:
                    break
            # print("Offsets:", offset_a, offset_b, l)
            # input()
            if l > len(string_b) * 0.5:
                return offset_a + l - 1, offset_b + l - 1, l

def megatron_client_generate(url, prompt_list, tokens_to_generate, window_size=None, logprobs=True):
    if prompt_list is None:
        return None
    headers = {'Content-Type': 'application/json'}
    # print("Generate", len(prompt_list))
    data = {"prompts": prompt_list, "tokens_to_generate": tokens_to_generate,
            "ignore_special_tokens": True, "add_BOS": False, "random_seed": 0, "top_k": 1, "logprobs": logprobs,
            "window_size": window_size, "stop_on_eol": True, "prevent_newline_after_colon": True}  # for future implementation
    response = requests.put(url, data=json.dumps(data), headers=headers)

    if response.status_code != 200:
        raise ValueError(f"Error {response.status_code}: {response.json()}")
    else:
        try:
            return response.json()
        except:
            raise ValueError("Unclassified error in response:", response.json())

def megatron_client_tokenize(url, text, **kwargs):
    """
    Tokenize one sequence.
    url: url of backend
    text: the sequence to tokenize
    """
    headers = {'Content-Type': 'application/json'}
    # print("Tokenize 1")
    data = {"texts": [text], "add_BOS": False}
    # add kwargs to data
    for key, value in kwargs.items():
        data[key] = value

    response = requests.put(url, data=json.dumps(data), headers=headers)

    if response.status_code != 200:
        raise ValueError(f"Error {response.status_code}: {response.json()}")
    else:
        return response.json()['token_ids'][0]

def megatron_client_detokenize(url, tokens, **kwargs):
    headers = {'Content-Type': 'application/json'}
    # print("Detokenize 1")
    data = {"tokens": [tokens], "no_log": False}
    data.update(kwargs)
    response = requests.put(url, data=json.dumps(data), headers=headers)

    if response.status_code != 200:
        raise ValueError(f"Error {response.status_code}: {response.json()}")
    else:
        return response.json()['texts'][0]

def megatron_client_modify_window_size(url, window_size):
    headers = {'Content-Type': 'application/json'}
    data = {"window_size": window_size}
    response = requests.put(url, data=json.dumps(data), headers=headers)

    if response.status_code != 200:
        raise ValueError(f"Error {response.status_code}: {response.json()}")
    else:
        return

def get_url(base_url, request_type):
    base_url = base_url.strip()
    if request_type == "generate":
        return f"http://{base_url}/api"
    elif request_type == "tokenize":
        return f"http://{base_url}/api/tokenize"
    elif request_type == "detokenize":
        return f"http://{base_url}/api/detokenize"
    elif request_type == "modify_window_size":
        return f"http://{base_url}/api/modify_window_size"
    else:
        raise ValueError("Invalid request type. Must be 'generate', 'tokenize', or 'detokenize', or 'modify_window_size'.")

url = sys.argv[1]
# model_path = "/u/yufengd4/work/models/llama_128k_2k_5000000_None_checkpoint_long_noclip_1000"
model_path = "/work/nvme/bcbw/mtian8/converted_HF/mistral_7B_128k_4096_clip0.0_cyclic"
def test_megatron_client_generate(input_string, window_size=None):
    while True:
        try:
            tokens = megatron_client_tokenize(get_url(url, "tokenize"), input_string)
            break
        except requests.exceptions.ConnectionError:
            import time
            time.sleep(1)
            pass
    megatron_client_modify_window_size(get_url(url, "modify_window_size"), (window_size, 0) if window_size is not None else None)
    generation = megatron_client_generate(get_url(url, "generate"), [input_string], 10)
    print("Megatron:")
    print("Input tokens:", tokens[:10], tokens[-10:])
    generation_text = generation["text"][0]
    input_string = input_string.replace("<s>", "").strip()
    generation_text = generation_text.replace("<s>", "").strip()
    # generation_text = generation_text.replace(input_string, "")
    end_a, end_b, common_len = lcs(generation_text, input_string)
    print((end_a, end_b, common_len))
    print(len(input_string))
    print(len(generation_text))
    # input()
    common_string = generation_text[end_a - common_len + 1: end_a + 1]
    front_b = input_string[:end_b - common_len + 1]
    front_a = generation_text[:end_a - common_len + 1]
    back_b = input_string[end_b + 1: ][:100]
    back_a = generation_text[end_a + 1: ][:100]
    raw_generation_text = generation_text
    generation_text = back_a
    #
    # print(f"Generated: [{generation_text}]")
    # print(f"Front_a: [{front_a}]")
    # print(f"Front_b: [{front_b}]")
    # print(f"back_b: [{back_b}]")
    # x = int(input())
    # while x != -1:
    #     print(f"[{generation_text[x: x+100]}]")
    #     print(f"[{input_string[x: x+100]}]")
    #     x = int(input())
    # print("Log probs:")
    # print(generation["logits"])
    # for log_prob in generation["log_probs"][:10]:
    #     print(log_prob[:10], log_prob[-10:])
    return generation_text, generation["logits"][0]

def test_faiss_attn(input_string, window_size=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = LlamaForCausalLM.from_pretrained(model_path,
    #                 use_flash_attention_2="flash_attention_2", torch_dtype=torch.float16,device_map='auto').eval()
    model = MyMistralForCausalLM.from_pretrained(model_path,
                    use_flash_attention_2="flash_attention_2", torch_dtype=torch.float16,device_map='auto').eval()
    print(dir(model))
    print(MyMistralForCausalLM.__dict__)
    # print(dir(model.model))
    # print(model.model.__class__, id(model.model.__class__))
    # from source.modeling_mistral import MistralModel
    # print(MistralModel, id(MistralModel))

    model.call_mistral_model_modify_window_size(window_size)
    tokens = tokenizer(input_string, return_tensors="pt")["input_ids"]
    logprobs = []
    def decode(outputs, inp, decode_len):
        output, retrieval_score = [], [[[0, ''] for _ in range(32)] for _ in range(32)]
        past_kv = outputs.past_key_values
        if inp is None:
            # use q_outputs to generate the first token
            logprobs.append([outputs.logits[0, -1]])
            inp = outputs.logits[0, -1].argmax()
            output.append(inp.item())
        for step_i in range(decode_len):
            inp = inp.view(1, 1)
            outputs = model(input_ids=inp, past_key_values=past_kv, use_cache=True, output_attentions=False, block_list=None)
            past_kv = outputs.past_key_values
            logprobs[-1].append(outputs.logits[0, -1])
            inp = outputs.logits[0, -1].argmax()
            step_token = tokenizer.convert_ids_to_tokens(inp.item())
            output.append(inp.item())
            #self.retrieval_calculate(outputs.attentions, retrieval_score, inp, step_token)
            if step_token=='<0x0A>' or inp.item()==144: break

        return output, retrieval_score

    with torch.no_grad():
        q_outputs = model(input_ids=tokens, use_cache=True, return_dict=True)
        output, retrieval_score = decode(q_outputs, None, 10)
        response = tokenizer.decode(output, skip_special_tokens=True).strip()

    print("Faiss Attn:")
    print("Input tokens:", tokens[0][:10], tokens[0][-10:])
    print("Generated:", response)
    # print("Log probs:")
    # print(logprobs)
    # print("Log probs:")
    # for log_prob in logprobs[:10]:
    #     print(log_prob[:10], log_prob[-10:])
    return response, logprobs[0]


def comparison(input_string, window_size=None):
    faiss_response, faiss_logprobs = test_faiss_attn(input_string, window_size)
    megatron_response, megatron_logprobs = test_megatron_client_generate(input_string, window_size)

    megatron_logprobs = megatron_logprobs[:10]
    while len(megatron_logprobs) < 10:
        megatron_logprobs.append([0] * len(faiss_logprobs[0]))
    faiss_logprobs = faiss_logprobs[:10]
    while len(faiss_logprobs) < 10:
        faiss_logprobs.append([0] * len(faiss_logprobs[0]))
    print("Summary:")
    print("Megatron response:", megatron_response)
    print("Faiss response:", faiss_response)
    print("Log probs:")
    # assert megatron_response == faiss_response
    i = 0
    for megatron_logprob, faiss_logprob in zip(megatron_logprobs, faiss_logprobs):
        diff = np.abs(np.array(megatron_logprob) - np.array(faiss_logprob))
        l1_diff = np.sum(diff)
        l2_diff = np.sqrt(np.sum(diff**2))
        print("Token", i, ":")
        print("L1 diff:", l1_diff)
        print("L2 diff:", l2_diff)
        i += 1

import json
with open("/u/yufengd4/work/Retrieval_Head/results/graph/mistral_7b_128k_4k_clip0_cyclic_fp16_window_None_Megatron/mistral_7b_128k_4k_clip0_cyclic_fp16_window_None_Megatron_len_16384_depth_5600_results.json", "r") as f:
    record = json.load(f)
record_input = record["input"]
# comparison("Question: What is the capital of France?\nAnswer:", 4096)
comparison(record_input, 4096)
while True:
    input_string = input("Enter a prompt: ")
    comparison(input_string, 4096)
    print("\n\n")