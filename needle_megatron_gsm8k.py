"""
This script is adapted from
https://github.com/gkamradt/LLMTest_NeedleInAHaystack

# GPT-4
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider OpenAI\
    --model_name gpt-4-1106-preview
    --api_key $OPENAI_API_KEY
) 2>&1  | tee logs/eval_gpt_4_128k.log

# LLaMA 2 32K. Remember to download the model first
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider LLaMA\
    --model_path ../../../Llama-2-7B-32K-Instruct
) 2>&1  | tee logs/eval_llama2_32k_instruct.log

# LongChat. Remember to download the model first
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider LLaMA\
    --model_path /ML-A800/models/longchat-7b-v1.5-32k
) 2>&1  | tee logs/eval_longchat.log

# Our llama-2-7b-80k, requires 4*80G A100
# require you to download the model first
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider LLaMA\
    --model_path ../../../llama-2-7b-80k
) 2>&1  | tee logs/eval_llama-2-7b-80k.log
"""
import math
#import tiktoken
import os
import glob
import json
from functools import partial
import sys
import random
import re
import numpy as np

# from datasets import load_dataset


import numpy as np
import argparse
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

from datetime import datetime, timezone
from collections import defaultdict
import time
import requests

questions_raw = [
    {
        "_question": "A bakery produces 173 loaves of bread each day. The head baker sets aside three loaves for breakfast and uses four loaves to prepare sandwiches for the staff. The remaining loaves are sold at the bakery for $2 each. How much in dollars does the bakery make every day from selling the bread?",
        "_answer": "How many loaves does the bakery sell? ** The bakery sells 173 - 3 - 4 = <<173-3-4=166>>166 loaves a day. How much does the bakery make from selling bread? ** They make 166 * 2 = $<<166*2=332>>332 every day from selling bread. #### 332",
        "description": "A bakery produces 173 loaves of bread each day. The head baker sets aside three loaves for breakfast and uses four loaves to prepare sandwiches for the staff. The remaining loaves are sold at the bakery for $2 each.",
        "question_only": "How much in dollars does the bakery make every day from selling the bread?",
        "cot": [
            "How many loaves does the bakery sell?",
            "The bakery sells 173 - 3 - 4 = <<173-3-4=166>>166 loaves a day.",
            "How much does the bakery make from selling bread?",
            "They make 166 * 2 = $<<166*2=332>>332 every day from selling bread."
        ],
        "answer_only": "332"
    },
    {
        "_question": "A robe takes 230 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "_answer": "How many bolts of white fiber does it take? ** It takes 230/2=<<230/2=115>>115 bolts of white fiber. How many bolts in total does it take? ** So the total amount of fabric is 230+115=<<230+115=345>>345 bolts of fabric. #### 345",
        "description": "A robe takes 230 bolts of blue fiber and half that much white fiber.",
        "question_only": "How many bolts in total does it take?",
        "cot": [
            "How many bolts of white fiber does it take?",
            "It takes 230/2=<<230/2=115>>115 bolts of white fiber.",
            "How many bolts in total does it take?",
            "So the total amount of fabric is 230+115=<<230+115=345>>345 bolts of fabric."
        ],
        "answer_only": "345"
    },
    {
        "_question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
        "_answer": "How much did the house cost? ** The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000 How much did the repairs increase the value of the house? ** He increased the value of the house by 80,000*1.5=<<80000*1.5=120000>>120,000 What is the new value of the house? ** So the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000 How much profit did he make? ** So he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000 #### 70000",
        "description": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%.",
        "question_only": "How much profit did he make?",
        "cot": [
            "How much did the house cost?",
            "The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000",
            "How much did the repairs increase the value of the house?",
            "He increased the value of the house by 80,000*1.5=<<80000*1.5=120000>>120,000",
            "What is the new value of the house?",
            "So the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000",
            "How much profit did he make?",
            "So he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000"
        ],
        "answer_only": "70000"
    },
    {
        "_question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
        "_answer": "How many sprints does James run in a week? ** He sprints 3*3=<<3*3=9>>9 times How many meters does James run in a week? ** So he runs 9*60=<<9*60=540>>540 meters #### 540",
        "description": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint.",
        "question_only": "How many total meters does he run a week?",
        "cot": [
            "How many sprints does James run in a week?",
            "He sprints 3*3=<<3*3=9>>9 times",
            "How many meters does James run in a week?",
            "So he runs 9*60=<<9*60=540>>540 meters"
        ],
        "answer_only": "540"
    },
    {
        "_question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 23 cups of feed. In the afternoon, she gives her chickens another 31 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 29 chickens?",
        "_answer": "How many cups of feed does Wendi need to give her chickens in the final meal of the day? ** If each chicken eats 3 cups of feed per day, then for 29 chickens they would need 3*29=<<3*29=87>>87 cups of feed per day. How many cups of feed does she need to give her chickens in the final meal of the day? ** If she feeds the flock 23 cups of feed in the morning, and 31 cups in the afternoon, then the final meal would require 87-23-31=<<87-23-31=33>>33 cups of chicken feed. #### 33",
        "description": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 23 cups of feed. In the afternoon, she gives her chickens another 31 cups of feed.",
        "question_only": "How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 29 chickens?",
        "cot": [
            "How many cups of feed does Wendi need to give her chickens in the final meal of the day?",
            "If each chicken eats 3 cups of feed per day, then for 29 chickens they would need 3*29=<<3*29=87>>87 cups of feed per day.",
            "How many cups of feed does she need to give her chickens in the final meal of the day?",
            "If she feeds the flock 23 cups in the morning and 31 cups in the afternoon, then the final meal would require 87-23-31=<<87-23-31=33>>33 cups of chicken feed."
        ],
        "answer_only": "33"
    }
]

# process questions
# questions = []
# for q in questions_raw:
#     last_sentence_in_question = q["question"].split(". ")[-1]
#     cot = q["answer"].split("####")[0].split("**")

def megatron_client_generate(url, prompt_list, tokens_to_generate, window_size=None,
                             needle_positions=None, oracle_mode="off", distance_between_positions=0,
                             attention_save_file=""):
    """
    masked_tokens: None or list[list[bool]]. Masked tokens for each example. True means not masked. False means masked.
    """
    if prompt_list is None:
        return None
    headers = {'Content-Type': 'application/json'}
    # print("Generate", len(prompt_list))
    data = {"prompts": prompt_list, "tokens_to_generate": tokens_to_generate,

            "ignore_special_tokens": True, "add_BOS": False, "random_seed": 0, "top_k": 1,
            "window_size": window_size, "stop_on_eol": True, "prevent_newline_after_colon": True,
            "oracle_mode": oracle_mode, "distance_between_positions": distance_between_positions,
            "attention_save_file": attention_save_file}  # for future implementation
    if needle_positions:
        data["oracle_positions"] = needle_positions
    response = requests.put(url, data=json.dumps(data), headers=headers)

    if response.status_code != 200:
        raise ValueError(f"Error {response.status_code}: {response.json()}")
    else:
        try:
            return response.json()['text']
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

def reset_rope(model, model_max_train_len, scaling_factor):
    for l in model.model.layers:
        l.self_attn.rotary_emb.scaling_factor = scaling_factor
        l.self_attn.rotary_emb._set_cos_sin_cache(seq_len=model_max_train_len,
                                                  device=l.self_attn.rotary_emb.inv_freq.device, dtype=torch.float32)
    return


class GSM8kNeedle:
    def __init__(self, _idx, questions, setting="retrieval", selection="fixed", cot_level=2, show_few_shot_answer=False):
        """
        cot_level:
        - 0: only the question and answer
        - 1: the question, and cot steps except the last one
        - 2: the question, and all cot steps after the socratic questions are removed, except the last one
        - 3: the question, and all cot steps
        """
        self.target = questions[_idx]
        self.needle = self.target["description"]
        self.cot = ""
        self.cot_2 = " ".join(self.target["cot"][1:-1:2])
        if show_few_shot_answer:
            cot_level = 3
        if cot_level:
            # self.cot += " Let's think step by step: "
            if cot_level == 1:
                self.cot += " ".join(self.target["cot"][:-1])
            elif cot_level == 2:
                self.cot += " ".join(self.target["cot"][1:-1:2])
            elif cot_level == 3:
                self.cot += " ".join(self.target["cot"])
            elif cot_level == 4:
                self.cot = self.target["description"] + self.target["question_only"] + " ".join(self.target["cot"][:-1])
            elif cot_level == 5:
                self.cot += " ".join(self.target["cot"][:-1]) + " " + self.target["cot"][-1].split("=")[0] + "..."
        if show_few_shot_answer:
            self.cot += "\n Question: " + self.target["question_only"] + " Answer: Let's first repeat the problem description and reasoning." + self.needle + ". " + self.cot + ". Finally, the answer is " + self.target["answer_only"] + ".\n"
        # self.needle += self.cot
        self.question = self.target["question_only"]
        self.answer = self.target["answer_only"]

class GSM8kNeedleGenerator:
    def __init__(self, setting="retrieval", selection="fixed", cot_level=2, default_needle=0,
                 show_few_shot_answer=False):
        """
        Setting: "retrieval" or "local" or "[setting]-0"
        - retrieval: The question, cot and distractors are inserted
        - local: The question and cot are inserted in a close context
        """
        print("Show few shot answer", show_few_shot_answer)
        self.questions = questions_raw
        setting_parts = setting.split('-', maxsplit=2)
        self.setting = setting_parts[0]
        self.max_distractors = None
        self.max_examples = None
        if len(setting_parts) > 1:
            self.max_distractors = int(setting_parts[1])
            if len(setting_parts) > 2:
                self.max_examples = int(setting_parts[2])

        self.selection = selection
        self.pool = None
        self.cot_level = cot_level
        self.default_needle = default_needle
        self.show_few_shot_answer = show_few_shot_answer


    def generate(self):
        self.pool = {}
        if self.selection == "fixed":
            idx = self.default_needle
        else:
            idx = random.randint(0, len(self.questions)-1)

        distractors = list(range(0, len(self.questions)))
        distractors.remove(idx)
        # distractors = []
        examples = []

        if self.max_distractors is not None:
            print("Max distractors", self.max_distractors, ", All distractors", len(distractors))
            examples = distractors[self.max_distractors:]
            examples = examples[:self.max_examples]
            distractors = distractors[:self.max_distractors]
        else:
            print("All distractors", len(distractors))
            examples = []
        print("Distractors and examples:", len(distractors), len(examples))
        self.pool["needle"] = GSM8kNeedle(idx, self.questions, cot_level=self.cot_level)
        self.pool["distractors"] = [GSM8kNeedle(d, self.questions, cot_level=self.cot_level,
                                                show_few_shot_answer=self.show_few_shot_answer) for d in distractors]
        self.pool["examples"] = [GSM8kNeedle(e, self.questions, cot_level=self.cot_level,
                                             show_few_shot_answer=self.show_few_shot_answer) for e in examples]



    def positions_iter(self, document_depth_percents, depth_percent):
        """
        Generates distractor positions for a given set of possible positions and the needle depth. See usage for details
        """
        j = 0
        if len(self.pool["distractors"]) == 1:
            space = 1
        else:
            if len(document_depth_percents) <= 1:
                space = -1
            space = (len(document_depth_percents) - 2) / (len(self.pool["distractors"]) - 1)
        for i in range(len(self.pool["distractors"])):
            int_j = int(j + 0.5)
            if abs(document_depth_percents[int_j] - depth_percent) < 1e-5 and int_j + 1 < len(document_depth_percents):
                j += 1
                int_j += 1
            # print("Positions iter yielding ", i, int_j)
            yield i, int_j
            j += space






class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """

    def __init__(self,
                 needle=None,
                 haystack_dir="PaulGrahamEssays",
                 retrieval_question="What is the best thing to do in San Francisco?",
                 results_version=1,
                 context_lengths_min=1000,
                 context_lengths_max=128000,
                 context_lengths_num_intervals=40,
                 context_lengths=None,
                 document_depth_percent_min=0,
                 document_depth_percent_max=100,
                 document_depth_percent_intervals=10,
                 document_depth_percents=None,
                 document_depth_percent_interval_type="linear",
                 model_provider="Megatron",
                 mask_topk=0,
                 anthropic_api_key=None,
                 model_name='',
                 service_url='',
                 model_name_suffix=None,
                 num_concurrent_requests = 1,
                 save_results = True,
                 save_contexts = True,
                 final_context_length_buffer = 200,
                 seconds_to_sleep_between_completions = None,
                 print_ongoing_status = True,
                 device = "auto",
                 batch_size = 1,
                 window_size="None",
                 setting="retrieval",
                selection="fixed",
                 cot_level=2,
                 default_needle=0,
                 show_few_shot_answer=False,
                 split_cot=False,
                 send_needle_positions=False,
                 oracle_mode="off",
                 space_mode=False,
                 attention_save_file=""
                 ):
        """
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_provider: The provider of the model. Must be either 'OpenAI' or 'Anthropic'. Default is 'OpenAI'.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param anthropic_api_key: The API key for Anthropic. Default is None.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        self.needle_generator = GSM8kNeedleGenerator(setting=setting, selection=selection, cot_level=cot_level,
                                                     default_needle=default_needle,
                                                     show_few_shot_answer=show_few_shot_answer)
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        if model_provider != 'Megatron':
            raise ValueError("Model provider must be Megatron")
        self.testing_results = []
        self.head_counter = defaultdict(list)
        self.mask_topk = mask_topk
        self.service_url = service_url
        self.batch_size = batch_size
        self.oracle_mode = oracle_mode
        self.internal_space_mode = False
        self.space_mode = (not self.internal_space_mode) and space_mode
        self.attention_save_file = attention_save_file
        if window_size == "None":
            self.window_size = None
        else:
            try:
                self.window_size = [int(window_size), 0]
            except ValueError:
                self.window_size = [int(x) if x != "None" else None for x in window_size.split(",")]
        # self.window_size = [int(window_size), 0] if window_size != "None" else None
        megatron_client_modify_window_size(get_url(self.service_url, "modify_window_size"), self.window_size)
        # NOTICE: it should be clear that each time an evaluator is created, the backend window size will change!
        if ("/" in model_name):
            self.model_version = model_name.split("/")[-1]
        else:
            self.model_version = model_name
        if (model_name_suffix is not None): self.model_version += "_" + model_name_suffix
        if (window_size is not None):
            if len(window_size) < 32:
                self.model_version += f"_window_{window_size}"
            else:
                window_size = ""
                previous_size = "~"
                previous_cnt = 0
                for size in self.window_size:
                    if size != previous_size:
                        if previous_cnt > 0:
                            window_size += str(previous_size) + "x" + str(previous_cnt) + ","
                        previous_size = size
                        previous_cnt = 1
                    else:
                        previous_cnt += 1
                if previous_cnt:
                    window_size += str(previous_size) + "x" + str(previous_cnt)
                self.model_version += f"_window_{window_size}"

        if (setting): self.model_version += f"_{setting}"
        if (selection): self.model_version += f"_{selection}"
        if selection == "fixed" and default_needle != 0: self.model_version += f"-{default_needle}"
        if (cot_level): self.model_version += f"_cot_{cot_level}"
        self.model_version = "GSM8k_" + self.model_version
        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            self.model_version += f"_{document_depth_percent_interval_type}"
        self.model_version += f"_oracle_{oracle_mode}"
        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError(
                    "Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(
                    np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals,
                                endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths
        self.document_depth_percent_interval_type = document_depth_percent_interval_type
        self.document_depth_last_tokens = None
        self.document_depth_percent_min = document_depth_percent_min
        self.document_depth_percent_max = document_depth_percent_max
        self.document_depth_percent_intervals = document_depth_percent_intervals
        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError(
                    "Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(
                        np.linspace(document_depth_percent_min * 4, document_depth_percent_max * 4,
                                    num=document_depth_percent_intervals, endpoint=True)) / 4
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
                elif document_depth_percent_interval_type.startswith("last"):
                    self.document_depth_last_tokens = int(document_depth_percent_interval_type.split("-")[-1])
                    self.document_depth_percents = np.round(
                        np.linspace(0, self.document_depth_last_tokens,
                                    num=document_depth_percent_intervals, endpoint=True)).astype(int)

        else:
            self.document_depth_percents = document_depth_percents

        # if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
        #     raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        self.device = device
        self.model_name = model_name
        class WebTokenizer:
            def __init__(self, url):
                self.url = url

            def tokenize(self, text, **kwargs):
                return megatron_client_tokenize(get_url(self.url, "tokenize"), text, **kwargs)

            def detokenize(self, tokens, **kwargs):
                return megatron_client_detokenize(get_url(self.url, "detokenize"), tokens, **kwargs)
        self.enc = WebTokenizer(self.service_url)

        self.model_version += "_" + self.model_provider

        def model_wrapping_for_batch_pretender(prompt_list, tokens_to_generate, needle_positions=None, attention_save_file=""):
            if prompt_list is not None and len(getattr(self, "_model_output_buffer", [])) == 0:
                generation = megatron_client_generate(get_url(self.service_url, "generate"), prompt_list, tokens_to_generate,
                                                      window_size=self.window_size, needle_positions=needle_positions if send_needle_positions else None,
                                                      oracle_mode=oracle_mode, distance_between_positions=1437 if self.internal_space_mode else 1,
                                                      attention_save_file=attention_save_file)  # for future
                self._model_output_buffer = generation
            first_sample, self._model_output_buffer = self._model_output_buffer[0], self._model_output_buffer[1:]
            return first_sample

        self.model_to_test = model_wrapping_for_batch_pretender

        self.model_to_test_description = model_name

        self.evaluation_model = None

        model_name = model_name.split('/')[-1]

        self.block_list = []
        self.split_cot = split_cot

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

    def bound_evaluate_and_log(self, *args):
        self.evaluate_and_log(*args)

    def run_test(self, args):

        # Run through each iteration of context_lengths and depths
        tasks = []
        for context_length in self.context_lengths:
            if context_length < args.s_len or context_length > args.e_len: continue
            # if self.document_depth_percent_interval_type.startswith("last"):
                # document_depth_percent_min = max(0., 100 * (1 - self.document_depth_last_tokens / context_length))
                # self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, 100,
                #                      num=self.document_depth_percent_intervals, endpoint=True)).astype(int)
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(context_length, depth_percent)

    def generate_anthropic_prompt(self, context):
        # Generate the prompt for the Anthropic model
        # Replace the following line with the appropriate prompt structure
        test_format = f"This is a very long story book: <book> {context} </book>.\n"
        if self.model_version in ["Mistral-7B-Instruct-v0.2"]:
            prompt = [
                {"role": "user",
                 "content": f"<book>{context}</book>\nBased on the content of the book, Question: {self.retrieval_question}\nAnswer: The answer is "}, ]
            return prompt

    def retrieval_calculate(self, attention_maxtrix, retrieval_score, inp, step_token, topk=1):
        for layer_idx in range(32):
            for head_idx in range(32):
                values, idx = attention_maxtrix[layer_idx][0][head_idx][-1].topk(topk)
                for v, i in zip(values, idx):
                    if self.needle_start <= i < self.needle_end and inp.item() == self.prompt_ids[i].item():
                        retrieval_score[layer_idx][head_idx][0] += 1 / (self.needle_end - self.needle_start)
                        retrieval_score[layer_idx][head_idx][1] += step_token
                        break

    def retrieval_head_accumulate(self, retrieval_score):
        for layer_idx in range(32):
            for head_idx in range(32):
                self.head_counter[f"{layer_idx}-{head_idx}"].append(retrieval_score[layer_idx][head_idx][0])

    def decode(self, q_outputs, inp, decode_len):
        return q_outputs, None

    def find_needle_idx(self, needle):
        needle_ids = self.enc.tokenize(needle)
        #print( self.enc.decode(needle_ids, skip_special_tokens=False))
        span_len = len(needle_ids)
        for i in range(len(self.prompt_ids)):

            token_span = self.prompt_ids[i: i + span_len]
            if not isinstance(token_span, list):
                token_span = token_span.tolist()
            span_ids = set(token_span)
            overlap = float(len(span_ids.intersection(set(needle_ids)))) / len(set(needle_ids))
            if (overlap > 0.9):
                return i, i + span_len
        return -1, -1

    def construct_random_head(self, n):
        results = []
        seed_list = [i for i in range(32)]
        random.shuffle(seed_list)
        while len(results) < n:
            l, h = random.choices(seed_list, k=2)
            if (l, h) in results or (l, h) in self.block_list:
                continue
            else:
                results.append((l, h))
        return results

    def evaluate_and_log(self, context_length, depth_percent):

        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        # Go generate the required length context and place your needle statement in
        if self.mask_topk > 0:
            block_list = self.block_list[:self.mask_topk]
            save_name = f"{self.model_version}_block_top{self.mask_topk}"
        elif self.mask_topk == 0:
            block_list = None
            save_name = self.model_version
        else:
            block_list = self.construct_random_head(-self.mask_topk)
            save_name = f"{self.model_version}_block_random{-self.mask_topk}"
        context, input_ids, needle_positions = self.generate_input_ids_pretender(context_length, depth_percent)
        context_str = [c[1] for c in context]
        context_token = [c[0] for c in context]
        # context = self.generate_context(context_length, depth_percent)
        # question = f"Based on the content of the book, Question: {self.retrieval_question}\nAnswer:"
        input_context = context_token
        # if abs(depth_percent - 0) < 1e-5:
        #     return
        # input_ids = context
        test_start_time = time.time()

        # self.real_needle = "eat a sandwich and sit in Dolores Park on a sunny day"
        self.prompt_ids = input_ids

        output = self.model_to_test(prompt_list=input_context, tokens_to_generate=200, needle_positions=needle_positions,
                                    attention_save_file=self.attention_save_file + "_" + str(context_length) + "_" + str(depth_percent))
        # print("Response:", output)
        question = f"Based on the content of the book"
        # question += "The answer is"
        response = output.split(question)[1].split("Let's first repeat the problem description and reasoning.")[1].strip()
        self.needle2 = self.needle_generator.pool["needle"].needle + self.needle_generator.pool["needle"].cot_2
        response = ''.join(re.split(r'<<.*?>>', response))
        self.needle2 = ''.join(re.split(r'<<.*?>>', self.needle2))
        rouge_score = scorer.score(self.needle2, response)['rouge1'].recall*100



        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        # format the response
        # split the first sentence
        # response = re.split("\.[\s$]", response, re.M)[0]
        # print("Response sentence:", response)
        # all_numbers = re.findall(r'(\.\d+|\d[\d,]*\.?\d*)', response)
        # if len(all_numbers) == 0:
        #     score = 0
        # else:
        #     response = all_numbers[-1].replace(",", "")
        #     error = abs((float(response) - float(self.real_needle)) / float(self.real_needle))
        #     magnitude = math.log10(max(error, 1e-10))
        #     mapped_loss = (magnitude + 3) / 2  # 0.1% -> 0, 10% -> 1
        #     score = min(100., max(0., 100 - 100 * mapped_loss))
        if response.find(str(self.real_needle)) != -1:
            score = 100
        else:
            if response.find("166") != -1 or response.find("173") != -1:
                score = 50
            else:
                score = 0
        score = rouge_score
        results = {
            'model': self.model_to_test_description,
            'context_length': int(context_length),
            'depth_percent': float(depth_percent),
            'version': self.results_version,
            'needle': self.needle,
            'needle_positions': needle_positions,
            'model_response': response,
            'score': score,
            'test_duration_seconds': test_elapsed_time,
            'test_timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z'),
            'input': input_ids,
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print(f"-- Test Summary -- ")
            print(f"Duration: {test_elapsed_time:.1f} seconds")
            print(f"Context: {context_length} tokens")
            print(f"Depth: {depth_percent}" + ("%" if self.document_depth_last_tokens is None else " tokens") + ("" if len(needle_positions[0]) == 1 else " " + str(needle_positions[0][1][0])))
            print(f"Score: {score}")
            print(f"Response: {response}")
            print(f"Needle2: {self.needle2}")
        # input("Waiting for input.")
        context_file_location = f'{self.model_version.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent * 100)}'

        if self.save_results:
            # Save the context to file for retesting
            if not os.path.exists(f'results/graph/{save_name}'):
                os.makedirs(f'results/graph/{save_name}')

            # Save the result to file for retesting
            p = f'results/graph/{save_name}/{context_file_location}_results.json'
            print("Writing at %s" % p)
            with open(p, 'w') as f:
                json.dump(results, f)

    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = 'results/' + self.model_version
        print("Searching existing results at %s" % results_dir)
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    # import ipdb; ipdb.set_trace()
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    def generate_needle(self):
        self.needle_generator.generate()
        self.needle = self.needle_generator.pool["needle"].needle
        self.cot = self.needle_generator.pool["needle"].cot
        if not self.split_cot:
            self.needle += self.cot
        self.distractors = self.needle_generator.pool["distractors"]
        self.examples = self.needle_generator.pool["examples"]
        self.retrieval_question = self.needle_generator.pool["needle"].question
        self.real_needle = self.needle_generator.pool["needle"].answer
        print("Gen needle => example", len(self.examples), ", distractor", len(self.distractors))

    def get_examples(self):
        if hasattr(self, '_examples'):
            return self._examples
        examples = ""
        if hasattr(self, "examples") and len(self.examples) > 0:
                # examples += "Here are some example questions. You don't have to answer them:\n"
                for example in self.examples:
                    examples += f"{example.needle} {example.cot}\n"
                examples += "End of examples. "
        if len(examples) == 0:
            tokens_examples = []
        else:
            tokens_examples = self.encode_text_to_tokens(examples)
        print("Examples:", len(tokens_examples), "tokens", examples,)
        self._examples = examples, tokens_examples
        return examples, tokens_examples

    def _batch_generate_input_ids(self, context_length, depth_percent=0, max_length=-1):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()
        examples, tokens_examples = self.get_examples()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length - len(tokens_examples))

        # print("Context:", context[:100])
        # Insert your random statement according to your depth percent
        generated_contexts = []
        generated_ids = []
        needle_positions = []
        print("Generating at", self.document_depth_percents)
        for _depth_percent in self.document_depth_percents:
            if _depth_percent < depth_percent - 1e-5:
                continue
            # print("Generate", _depth_percent)
            question = f"\nBased on the content of the book, Question: {self.retrieval_question}\nAnswer: Let's first repeat the problem description and reasoning."
            # question += "The answer is "
            modified_context_tokens, context_string, needle_position = self.insert_needle(context, _depth_percent, context_length, question)

            # input_context = context_string + examples + question

            needle_positions.append(needle_position)
            # print(f"Input: *{[input_context]}*")
            # print(f"Context: *{[modified_context]}*")
            # print(f"Question: *{[question]}*")
            # input_ids = self.enc.tokenize(input_context, add_BOS=True)  # set to true
            generated_contexts.append((modified_context_tokens, context_string))
            # generated_ids.append(input_ids)
            # print("Generated contexts:", len(generated_contexts))
            if len(generated_contexts) == max_length:
                break
        return generated_contexts, needle_positions

    def generate_input_ids_pretender(self, context_length, depth_percent):
        """
        Pretends to generate input tokens based on the given context length and depth percent.
        Actually, the inputs of a given context length are generated at the same time at depth_percent = 0
        """
        # find the index of the current depth percent
        i = 0
        for _depth_percent in self.document_depth_percents:
            # print("Checking %f" % _depth_percent)
            # print("Against %f" % depth_percent)
            if _depth_percent >= depth_percent - 1e-5:
                break
            i += 1
        # print("Got i=", i)
        if i == 0:  # if this is the first depth, generate all depth percents
            self.batch_input, self.needle_positions = self._batch_generate_input_ids(context_length)
        if i % self.batch_size == 0:  # return a batch of input ids
            return self.batch_input[i: i + self.batch_size], self.batch_input[i][1], self.needle_positions[i: i + self.batch_size]
        return None, self.batch_input[i][1], None

    def encode_text_to_tokens(self, text, add_BOS=False, ignore_special_tokens=True, **kwargs):
        return self.enc.tokenize(text, add_BOS=add_BOS, ignore_special_tokens=ignore_special_tokens, **kwargs)

    def insert_needle(self, context, depth_percent, context_length, question=None):

        # print(f"Context: {[context]}")
        # print(f"Needle: {self.needle}")
        # tokens_needle = self.encode_text_to_tokens("\n!!! THE FOLLOWING PART IS VERY IMPORTANT. IT CONTAINS PROBLEM INFORMATION: **" + self.needle + "**\n")
        tokens_needle = self.encode_text_to_tokens(self.needle)  # [xxx]
        # tokens_needle = self.encode_text_to_tokens(self.needle, ignore_special_tokens=False)  # [1, xxx]
        print("Needle:", tokens_needle)
        tokens_distractors = [self.encode_text_to_tokens(d.needle) for d in self.distractors]
        # print("Distractors:", [d.needle for d in self.distractors])
        # print(f"Tokens_needle: {tokens_needle[:10]}")
        tokens_context_with_bos = self.encode_text_to_tokens(context, ignore_special_tokens=False)
        # print("Tokens_context:", tokens_context_with_bos[:10])
        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer
        if question:
            _tokens_question = self.encode_text_to_tokens(question)
        else:
            _tokens_question = []
        examples, tokens_examples = self.get_examples()
        tokens_examples_question = tokens_examples + _tokens_question
        # print("Len of example + question:", len(tokens_question))

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length

        max_length = context_length - len(tokens_needle) - len(tokens_examples_question)
        if self.split_cot:
            tokens_cot = self.encode_text_to_tokens(self.cot)
            max_length -= len(tokens_cot)
        for i, int_j in self.needle_generator.positions_iter(document_depth_percents=self.document_depth_percents,
                                                          depth_percent=depth_percent):
            max_length -= len(tokens_distractors[i])

        max_length = max(max_length, 0)
        tokens_context_with_bos = tokens_context_with_bos[:max_length]

        def change_tokens_context_to_space(tokens_context, skip_special_tokens=True):
            if getattr(self, "filler", None) is None:
                self.filler = "\n!!! THE FOLLOWING PART IS VERY IMPORTANT. IT CONTAINS PROBLEM INFORMATION: **IMPORTANT INFORMATION**\n"
                self.filler_tokens = self.encode_text_to_tokens(self.filler)
            print("??", tokens_context[:10])
            space_token = memoize_space(self.enc)[0]
            return [space_token if i >= 1 or skip_special_tokens else t for i, t
                    in enumerate(tokens_context)]  # space

            # very important message filler
            # return [self.filler_tokens[i % len(self.filler_tokens)] if t >= 3 or not skip_special_tokens else t for i, t in enumerate(tokens_context)]

        def insert_needle_inner(tokens_context, depth_percent, tokens_needle, lower_bound=0, print_info=""):

            if self.document_depth_last_tokens is not None:
                insertion_point = (len(tokens_context) + len(tokens_needle) + len(tokens_examples_question)
                                   - self.document_depth_last_tokens + int(depth_percent))
                if insertion_point > len(tokens_context):  # redundant
                    insertion_point = len(tokens_context)
            else:
                insertion_point = int(len(tokens_context) * (depth_percent / 100) + 0.5)
            if insertion_point < lower_bound:
                insertion_point = lower_bound
            if insertion_point <= 1:  # BOS
                insertion_point = 1
                if self.space_mode:
                    tokens_context = change_tokens_context_to_space(tokens_context, skip_special_tokens=False)
                print("!!!", tokens_context[:10])
                tokens_new_context = tokens_context[:1] + tokens_needle + tokens_context[1:]
            else:
                # Go get the position (in terms of tokens) to insert your needle


                # print(print_info, "Insertion point initial:", insertion_point)
                # import ipdb; ipdb.set_trace()

                # tokens_new_context represents the tokens before the needle
                tokens_new_context = tokens_context[:insertion_point]

                # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
                period_tokens = get_period_memoization(self.enc)
                # print("Period tokens:", period_tokens)



                # Then we iterate forward until we find the first period
                while len(tokens_context) > insertion_point and tokens_new_context[-1] not in period_tokens:
                    insertion_point += 1
                    tokens_new_context = tokens_context[:insertion_point]

                if len(tokens_context) > insertion_point:  # then we found a period. Insert after the period
                    insertion_point += 1
                # print(print_info, "Insertion at", insertion_point)

                if self.space_mode:
                    tokens_new_context = change_tokens_context_to_space(tokens_new_context, skip_special_tokens=False)
                    print("~!!!", tokens_new_context[:10])
                    tokens_context = change_tokens_context_to_space(tokens_context)

                # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
                # Now we have a needle in a haystack
                tokens_new_context += tokens_needle + tokens_context[insertion_point:]
                # print("Tokens_new_context:", tokens_new_context[:10])
            # print("Context length after needle insertion:", len(tokens_new_context))
            return tokens_new_context, insertion_point

        tokens_new_context = tokens_context_with_bos
        needle_position = []

        # insert distractors
        for i, int_j in self.needle_generator.positions_iter(document_depth_percents=self.document_depth_percents,
                                                          depth_percent=depth_percent):

            # print("Old length:", len(tokens_new_context))
            tokens_new_context, insertion_point = insert_needle_inner(tokens_new_context, self.document_depth_percents[int_j], tokens_distractors[i], print_info="[distractor" + str(i) + "]")
            print("_+_", insertion_point, tokens_new_context[:10])
            needle_position.append([insertion_point, insertion_point + len(tokens_distractors[i])])
            # print("Inserting distractor", i, "at", int_j, self.document_depth_percents[int_j], insertion_point)
            # print("New length:", len(tokens_new_context))

        # insert needle
        # if not self.split_cot:
        #     tokens_needle += tokens_cot
        print("_ss_", tokens_new_context[:10])
        tokens_new_context, insertion_point = insert_needle_inner(tokens_new_context, depth_percent, tokens_needle, print_info="[needle]")
        print("_s_", insertion_point, tokens_new_context[:10])
        needle_position.append([insertion_point, insertion_point + len(tokens_needle)])
        print("Inserting needle at", insertion_point, "with total length", len(tokens_new_context), "and needle length",
              len(tokens_needle), "and question length", len(tokens_examples_question))

        if self.split_cot:
            block_size = self.window_size
            if not block_size:
                block_size = 4096
            if self.document_depth_last_tokens:
                upper_bound = self.document_depth_last_tokens // block_size - 1  # if possible, avoid the last block
                lo = (depth_percent) // block_size + 1
                if lo > upper_bound:
                    target_depth_percent = depth_percent
                else:
                    target_depth_percent = np.random.randint(lo, upper_bound + 1) * block_size + (
                                block_size // 2)  # 2048 is the middle of a block
            else:
                upper_bound = 100
                lo = int(depth_percent + 1)
                if lo > upper_bound:
                    target_depth_percent = depth_percent
                else:
                    target_depth_percent = np.random.randint(lo, upper_bound + 1)
            print("Splitting cot: [", lo, ",", upper_bound, "]", target_depth_percent)
            tokens_new_context, insertion_point_2 = insert_needle_inner(tokens_new_context, target_depth_percent, tokens_cot, lower_bound=insertion_point + len(tokens_needle), print_info="[cot]")
            needle_position.append([insertion_point_2, insertion_point_2 + len(tokens_cot)])
            print("Inserting cot at", insertion_point_2, "with total length", len(tokens_new_context), "and cot length",
                  len(tokens_cot), "and question length", len(tokens_examples_question))

        # Convert back to a string and return it
        # print("")
        tokens_new_context += tokens_examples_question
        new_context_display = self.decode_tokens(tokens_new_context, ignore_special_tokens=False)
        print("New context:", [new_context_display[:100]], tokens_new_context[:20])
        return tokens_new_context, new_context_display, needle_position

    def get_context_length_in_tokens(self, context):
        # print("Getting context length in tokens")
        return len(self.encode_text_to_tokens(context, ignore_special_tokens=False))

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)
        predefined_file_sequence_text="""Open file PaulGrahamEssays/sun.txt
Open file PaulGrahamEssays/web20.txt
Open file PaulGrahamEssays/avg.txt
Open file PaulGrahamEssays/foundervisa.txt
Open file PaulGrahamEssays/laundry.txt
Open file PaulGrahamEssays/langdes.txt
Open file PaulGrahamEssays/vcsqueeze.txt
Open file PaulGrahamEssays/love.txt
Open file PaulGrahamEssays/worked.txt
Open file PaulGrahamEssays/gh.txt
Open file PaulGrahamEssays/unions.txt
Open file PaulGrahamEssays/addiction.txt
Open file PaulGrahamEssays/want.txt
Open file PaulGrahamEssays/hubs.txt
Open file PaulGrahamEssays/apple.txt
Open file PaulGrahamEssays/rss.txt
Open file PaulGrahamEssays/startuplessons.txt
Open file PaulGrahamEssays/newideas.txt
Open file PaulGrahamEssays/boss.txt
Open file PaulGrahamEssays/todo.txt
Open file PaulGrahamEssays/before.txt
Open file PaulGrahamEssays/goodtaste.txt
Open file PaulGrahamEssays/siliconvalley.txt
Open file PaulGrahamEssays/island.txt
Open file PaulGrahamEssays/pow.txt
Open file PaulGrahamEssays/rootsoflisp.txt
Open file PaulGrahamEssays/popular.txt
Open file PaulGrahamEssays/desres.txt
Open file PaulGrahamEssays/superangels.txt
Open file PaulGrahamEssays/weird.txt
Open file PaulGrahamEssays/philosophy.txt
Open file PaulGrahamEssays/bias.txt
Open file PaulGrahamEssays/corpdev.txt
Open file PaulGrahamEssays/mod.txt
Open file PaulGrahamEssays/gap.txt
Open file PaulGrahamEssays/vb.txt
Open file PaulGrahamEssays/aord.txt
Open file PaulGrahamEssays/useful.txt
Open file PaulGrahamEssays/copy.txt
Open file PaulGrahamEssays/ecw.txt
Open file PaulGrahamEssays/founders.txt
Open file PaulGrahamEssays/iflisp.txt
Open file PaulGrahamEssays/vw.txt
Open file PaulGrahamEssays/gba.txt
Open file PaulGrahamEssays/submarine.txt
Open file PaulGrahamEssays/wisdom.txt
Open file PaulGrahamEssays/know.txt
Open file PaulGrahamEssays/diff.txt
Open file PaulGrahamEssays/nft.txt"""
        predefined_file_sequence = []
        for line in predefined_file_sequence_text.split("\n"):
            predefined_file_sequence.append(line.split(" ")[-1])
        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in predefined_file_sequence:
            #for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                # print("Reading", file)
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        return self.encode_text_to_tokens(context)

    def decode_tokens(self, tokens, context_length=None, **kwargs):
        return self.enc.detokenize(tokens[:context_length], **kwargs)

    def encode_and_trim(self, context, context_length):
        # print("encode_and_trim")
        # print("get tokens")
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            # print("decode_tokens")
            context = self.decode_tokens(tokens, context_length, ignore_special_tokens=False)
        return context

    def get_results(self):
        return self.testing_results

    def print_start_test_summary(self):
        print("\n")
        print("Starting Needle In A Haystack Testing...")
        print(f"- Model: {self.model_name}")
        print(
            f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print(
            f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print(f"- Needle: {self.needle.strip()}")
        print("\n\n")

    def start_test(self, args):
        self.generate_needle()
        if self.print_ongoing_status:
            self.print_start_test_summary()
        #asyncio.run(self.run_test())
        self.run_test(args)


token_dict = {}


def memoize_period(enc):
    if '.' in token_dict:
        return token_dict['.']
    print("get token by calling multiple cases")
    sentence1 = enc.tokenize("Good.", ignore_special_tokens=True)
    sentence2 = enc.tokenize("This is the last chance.", ignore_special_tokens=True)
    sentence3 = enc.tokenize("The answer is 2.", ignore_special_tokens=True)

    token_dict['.'] = list({sentence1[-1]} | {sentence2[-1]} | {sentence3[-1]})
    return token_dict['.']

def memoize_space(enc):
    if ' ' in token_dict:
        return token_dict[' ']
    token_dict[' '] = enc.tokenize("  ", ignore_special_tokens=True)
    print("Space token:", token_dict[' '])
    print("Space token", enc.tokenize("  "))
    print("Space token", enc.tokenize("    "))
    print("Space token", enc.tokenize("     "))
    print("Space token", enc.tokenize("s b"))
    print("S", enc.tokenize("s"))
    print("B", enc.tokenize("b"))
    print("Space token", enc.tokenize("\n"))
    print("Space token", enc.tokenize("s\nb"))

    print("Space token:", f"[{enc.detokenize(token_dict[' '] + token_dict[' '])}]")
    print("----", enc.tokenize(enc.detokenize(token_dict[' '] + token_dict[' ']), ignore_special_tokens=True))
    return token_dict[' ']

def get_period_memoization(enc):


    period_tokens = memoize_period(enc)
    period_token = period_tokens[0]
    if period_token in [29889, 869]:
        period_tokens = [29889, 869]
    elif period_token in [88946, 13]:
        period_tokens = [88946, 13]
    elif period_token in [842, 28723]:
        period_tokens = [842, 28723]
    elif period_token in [918, 30930]:
        period_tokens = [918, 30930]
    return period_tokens


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('--s', '--s_len', metavar='N', type=int, help='a number')
    parser.add_argument('--e', '--e_len', metavar='N', type=int, help='a number')
    parser.add_argument('--model_path', type=str, default=None, help='path to model')
    parser.add_argument('--model_name', type=str, default=None, help='name of model')
    parser.add_argument('--model_name_suffix', type=str, default=None, help='name of model')
    parser.add_argument('--model_provider', type=str, default="Megatron", help='which model to use')
    parser.add_argument('--api_key', type=str, default="", help='OpenAI API Key')
    parser.add_argument('--mask_topk', type=int, default=0, help='mask topk heads, input a negative value to mask random heads')
    parser.add_argument('--num_intervals', type=int, default=40, help='number of intervals of the test')
    parser.add_argument('--device', type=str, default="auto", help="device")
    parser.add_argument('--url', type=str, default="localhost:5000", help="service url")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--window_size", type=str, default="None", help="window size")
    parser.add_argument("--discard", action="store_true", help="discard the results")
    parser.add_argument("--cot_level", type=int, default=2, help="cot level")
    parser.add_argument("--setting", type=str, default="local", help="setting")
    parser.add_argument("--selection", type=str, default="fixed", help="selection")
    parser.add_argument("--document_depth_percent_interval_type", type=str, default="linear",
                        help="interval type")  # sigmoid; last-4096
    parser.add_argument("--show_few_shot_answer", action="store_true", help="show few shot answer")
    parser.add_argument("--send_needle_positions", action="store_true", help="send needle positions")
    parser.add_argument("--split_cot", action="store_true", help="show few shot answer")
    parser.add_argument("--default_needle", type=int, default=0, help="The serial number of the default needle.")
    parser.add_argument("--oracle_mode", type=str, default="off", help=".")
    parser.add_argument("--document_depth_percent_intervals", type=int, default=10, help="number of tested depths")
    parser.add_argument("--space_mode", action="store_true", help="")
    parser.add_argument("--attention_save_file", type=str, default="", help="")
    # parser = add_args(parser)
    args = parser.parse_args()

    if (args.model_path is not None):
        assert (args.model_name is None)
        model_name = args.model_path
    else:
        assert (args.model_name is not None)
        model_name = args.model_name

    if not hasattr(args, 's_len'):
        args.s_len = args.s
    if not hasattr(args, 'e_len'):
        args.e_len = args.e

    ht = LLMNeedleHaystackTester(model_name=model_name,
                                 model_name_suffix=args.model_name_suffix,
                                 model_provider=args.model_provider,
                                 save_contexts=True,
                                 save_results=not args.discard,
                                 mask_topk=args.mask_topk,
                                context_lengths_min=args.s_len,
                                context_lengths_max=args.e_len,
                                context_lengths_num_intervals=args.num_intervals,
                                 document_depth_percent_intervals=args.document_depth_percent_intervals,
                                device=args.device,
                                service_url=args.url,
                                 batch_size=args.batch_size,
                                 window_size=args.window_size,
                                    setting=args.setting,
                                    selection=args.selection,
                                    cot_level=args.cot_level,
                                 document_depth_percent_interval_type=args.document_depth_percent_interval_type,
                                 default_needle=args.default_needle,
                                 show_few_shot_answer=args.show_few_shot_answer,
                                 split_cot=args.split_cot,
                                 send_needle_positions=args.send_needle_positions,
                                 oracle_mode=args.oracle_mode,
                                 space_mode=args.space_mode,
                                 attention_save_file=args.attention_save_file
      )

    ht.start_test(args)
