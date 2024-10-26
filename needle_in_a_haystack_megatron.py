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

#import tiktoken
import os
import glob
import json
from functools import partial

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import sys
import random


import numpy as np
import argparse
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

from datetime import datetime, timezone
from collections import defaultdict
import time
import torch
import requests

def megatron_client_generate(url, prompt, tokens_to_generate):
    headers = {'Content-Type': 'application/json'}

    data = {"prompts": [prompt], "tokens_to_generate": tokens_to_generate}
    response = requests.put(url, data=json.dumps(data), headers=headers)

    if response.status_code != 200:
        raise ValueError(f"Error {response.status_code}: {response.json()['message']}")
    else:
        return response.json()['text'][0]

def megatron_client_tokenize(url, text):
    headers = {'Content-Type': 'application/json'}

    data = {"texts": [text]}
    response = requests.put(url, data=json.dumps(data), headers=headers)

    if response.status_code != 200:
        raise ValueError(f"Error {response.status_code}: {response.json()['message']}")
    else:
        return response.json()['token_ids'][0]

def megatron_client_detokenize(url, tokens):
    headers = {'Content-Type': 'application/json'}

    data = {"tokens": [tokens]}
    response = requests.put(url, data=json.dumps(data), headers=headers)

    if response.status_code != 200:
        raise ValueError(f"Error {response.status_code}: {response.json()['message']}")
    else:
        return response.json()['texts'][0]

def get_url(base_url, request_type):
    if request_type == "generate":
        return f"http://{base_url}/api"
    elif request_type == "tokenize":
        return f"http://{base_url}/api/tokenize"
    elif request_type == "detokenize":
        return f"http://{base_url}/api/detokenize"
    else:
        raise ValueError("Invalid request type. Must be 'generate', 'tokenize', or 'detokenize'.")

def reset_rope(model, model_max_train_len, scaling_factor):
    for l in model.model.layers:
        l.self_attn.rotary_emb.scaling_factor = scaling_factor
        l.self_attn.rotary_emb._set_cos_sin_cache(seq_len=model_max_train_len,
                                                  device=l.self_attn.rotary_emb.inv_freq.device, dtype=torch.float32)
    return


class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """

    def __init__(self,
                 needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
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
                 device = "auto"):
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
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")

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

        if ("/" in model_name):
            self.model_version = model_name.split("/")[-1]
        else:
            self.model_version = model_name
        if (model_name_suffix is not None): self.model_version += "_" + model_name_suffix

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

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError(
                    "Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(
                        np.linspace(document_depth_percent_min, document_depth_percent_max,
                                    num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in
                                                    np.linspace(document_depth_percent_min, document_depth_percent_max,
                                                                document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        self.device = device
        self.model_name = model_name
        class WebTokenizer:
            def __init__(self, url):
                self.url = url

            def tokenize(self, text):
                return megatron_client_tokenize(get_url(self.url, "tokenize"), text)

            def detokenize(self, tokens):
                return megatron_client_detokenize(get_url(self.url, "detokenize"), tokens)
        self.enc = WebTokenizer(self.service_url)

        self.model_version += "_" + self.model_provider
        self.model_to_test = partial(megatron_client_generate, get_url(self.service_url, "generate"))

        self.model_to_test_description = model_name

        self.evaluation_model = None

        model_name = model_name.split('/')[-1]

        self.block_list = []

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
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(context_length, depth_percent)

    def generate_anthropic_prompt(self, context):
        # Generate the prompt for the Anthropic model
        # Replace the following line with the appropriate prompt structure
        test_format = f"This is a very long story book: <book> {context} </book>.\n"
        if self.model_version in ["Mistral-7B-Instruct-v0.2"]:
            prompt = [
                {"role": "user",
                 "content": f"<book>{context}</book>\nBased on the content of the book, Question: {self.retrieval_question}\nAnswer:"}, ]
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
            span_ids = set(token_span.tolist())
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
        context = self.generate_context(context_length, depth_percent)
        question = f"Based on the content of the book, Question: {self.retrieval_question}\nAnswer:"
        input_context = context + question
        input_ids = self.enc.tokenize(input_context)

        test_start_time = time.time()

        self.real_needle = "eat a sandwich and sit in Dolores Park on a sunny day"
        #self.prompt_ids = torch.concat([context_ids, question_ids], dim=1)[0, :]
        self.prompt_ids = input_ids

        self.needle_start, self.needle_end = self.find_needle_idx(self.real_needle)
        with torch.no_grad():
            output = self.model_to_test(prompt=input_ids, tokens_to_generate=50)
            response = self.enc.detokenize(output).strip()

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        score = scorer.score(self.real_needle, response)['rouge1'].recall * 100
        results = {
            'model': self.model_to_test_description,
            'context_length': int(context_length),
            'depth_percent': float(depth_percent),
            'version': self.results_version,
            'needle': self.needle,
            'model_response': response,
            'score': score,
            'test_duration_seconds': test_elapsed_time,
            'test_timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print(f"-- Test Summary -- ")
            print(f"Duration: {test_elapsed_time:.1f} seconds")
            print(f"Context: {context_length} tokens")
            print(f"Depth: {depth_percent}%")
            print(f"Score: {score}")
            print(f"Response: {response}\n")

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

    def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context

    def encode_text_to_tokens(self, text):
        return self.enc.tokenize(text)
    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            # import ipdb; ipdb.set_trace()

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            period_token = get_token_memoization(self.enc, '.')
            if period_token in [29889, 869]:
                period_tokens = [29889, 869]
            elif period_token in [88946, 13]:
                period_tokens = [88946, 13]
            elif period_token in [842, 28723]:
                period_tokens = [842, 28723]
            elif period_token in [918, 30930]:
                period_tokens = [918, 30930]
            else:
                period_tokens = [period_token]

            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            print("insertion at %d" % insertion_point)
            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        return len(self.encode_text_to_tokens(context))

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        return self.encode_text_to_tokens(context)

    def decode_tokens(self, tokens, context_length=None):
        return self.enc.detokenize(tokens)[:context_length]

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
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
        if self.print_ongoing_status:
            self.print_start_test_summary()
        #asyncio.run(self.run_test())
        self.run_test(args)


token_dict = {}
def get_token_memoization(enc, text):
    if text in token_dict:
        return token_dict[text]
    token_dict[text] = enc.tokenize(text)
    return token_dict[text]


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--s_len', metavar='N', type=int, help='a number')
    parser.add_argument('-e', '--e_len', metavar='N', type=int, help='a number')
    parser.add_argument('--model_path', type=str, default=None, help='path to model')
    parser.add_argument('--model_name', type=str, default=None, help='name of model')
    parser.add_argument('--model_name_suffix', type=str, default=None, help='name of model')
    parser.add_argument('--model_provider', type=str, default="Megatron", help='which model to use')
    parser.add_argument('--api_key', type=str, default="", help='OpenAI API Key')
    parser.add_argument('--mask_topk', type=int, default=0, help='mask topk heads, input a negative value to mask random heads')
    parser.add_argument('--num_intervals', type=int, default=40, help='number of intervals of the test')
    parser.add_argument('--device', type=str, default="auto", help="device")
    parser.add_argument('--service_url', type=str, default="localhost:5000", help="service url")
    # parser = add_args(parser)
    args = parser.parse_args()

    if (args.model_path is not None):
        assert (args.model_name is None)
        model_name = args.model_path
    else:
        assert (args.model_name is not None)
        model_name = args.model_name

    ht = LLMNeedleHaystackTester(model_name=model_name,
                                 model_name_suffix=args.model_name_suffix,
                                 model_provider=args.model_provider,
                                 save_contexts=True,
                                 save_results=True,
                                 mask_topk=args.mask_topk,
                                context_lengths_min=args.s_len,
                                context_lengths_max=args.e_len,
                                context_lengths_num_intervals=args.num_intervals,
                                device=args.device,
                                service_url=args.service_url
      )

    ht.start_test(args)
