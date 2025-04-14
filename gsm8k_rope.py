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
import concurrent
import copy
import math
#import tiktoken
import os
import glob
import json
import subprocess
import tempfile
import threading
from functools import partial
import sys
import random
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
from datasets import load_dataset
from rouge_score import rouge_scorer

import tiktoken
import openai
import litellm

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

import numpy as np
import argparse

from datetime import datetime, timezone
from collections import defaultdict
import time
import requests
from os import getpid
from os import getppid
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

number_re_pattern = r"-?(?:\d+,)*\d+\.?\d*"

output_level = "info"


def debug(*args, print_lock=None, **kwargs):
    if output_level == "debug":
        if print_lock:
            with print_lock:
                debug(*args, **kwargs)
        else:
            print("\033[90m", end="")
            print(*args, **kwargs)
            print("\033[0m", end="")


def info(*args, **kwargs):
    if output_level == "debug" or output_level == "info":
        print(*args, **kwargs)


def extract_numbers(target_str, remove_bullet_number=True):
    # split into lines
    target_str_lines = target_str.split("\n")
    # remove spaces
    target_str_lines = [target_str_line.strip() for target_str_line in target_str_lines]
    numbers = []
    for target_str_line in target_str_lines:
        if remove_bullet_number:
            bullet_number_match = re.match(r"^\s*\d+\.\s", target_str_line)
            if bullet_number_match:
                target_str_line = target_str_line[bullet_number_match.span()[1]:]
        numbers_str = re.findall(number_re_pattern, target_str_line)

        for number in numbers_str:
            num_str = (number.strip().replace(",", ''))
            numbers.append(float(num_str))
    return numbers


class GSM8KProblem:
    """
    .question: str
    .answer_num: float
    .cot_steps: (str, ) or (str, str)
    get_cot_step(idx): {"index": idx, "answer": str [, "question": str]}
    """

    def __init__(self, question_str, answer_str, idx=None):
        self._question_str = question_str
        self.problem_description = question_str
        self.question_str = question_str.split(". ")[-1]
        self._answer_str = answer_str
        cot_str, answer = answer_str.split("####")
        answer = answer.strip().replace(",", '')
        self.answer_str = answer
        self.answer_num = float(answer)
        self.cot_steps = [cot_step.split(" ** ") for cot_step in cot_str.strip().split("\n")]
        self.idx = idx

    def get_cot_step(self, idx):
        cot_step = self.cot_steps[idx]
        if len(cot_step) == 2:
            return {"index": idx, "question": cot_step[0], "answer": cot_step[1]}
        else:
            return {"index": idx, "question": "", "answer": cot_step[0]}

    def get_cot_steps_without_answer(self):
        last_step = [x for x in self.cot_steps[-1]]  # deep copy
        match = re.search(r"\d", last_step[-1])
        if match:
            first_digit_idx = match.start()
            last_step[-1] = last_step[-1][:first_digit_idx]
        return self.cot_steps[:-1] + [last_step]

    def format_problem_as_complete_paragraph(self, contain_last_step_answer=False):
        """
        returns: problem_description, prompt, answer
        """

        def cot_step_format(_step, with_answer=True):
            return "(?)" + _step["question"] + " (!)" + (_step["answer"] + ". " if with_answer else "")

        _problem_description = "[Problem Description]\n" + self.problem_description + "\n[Analysis]\n"
        _problem_description += ''.join(cot_step_format(self.get_cot_step(idx))
                                        for idx in range(len(self.cot_steps) - 1))
        last_step = self.get_cot_step(-1)
        return (_problem_description, cot_step_format(last_step, contain_last_step_answer),
                "[Answer] \n" + self.answer_str + '.')

    def format_problem(self, retrieval_format=False):
        """
        returns: problem_description, prompt, answer
        """

        def cot_step_format(_step, with_answer=True):
            return "(Q)" + _step["question"] + " (A)" + (_step["answer"] if with_answer else "")

        _problem_description = "# Problem Description\n" + self.problem_description + "\n# Analysis\n"
        _problem_description += ''.join(cot_step_format(self.get_cot_step(idx))
                                        for idx in range(len(self.cot_steps) - 1))
        _problem_description += "\n# Others\n"
        last_step = self.get_cot_step(-1)
        prompt_str = "\n# Question\n" + self.question_str + "\n# Answer\n"
        if retrieval_format:
            prompt_str += "Let's first recite ``# Problem Description'', ``# Analysis'' and ``# Question'' word by word, and then think and answer in the ``## Answer'' subsection. \n## Problem Description"
        return (_problem_description, prompt_str,
                "The answer is " + self.answer_str + '.')


class SimpleProblem:
    def __init__(self, question_str, answer_str, idx=None):
        self._question_str = question_str
        self.problem_description = question_str.rsplit(". ", maxsplit=1)[0]
        self.question_str = question_str.split(". ")[-1]
        self._answer_str = answer_str
        answer = answer_str
        answer = answer.strip().replace(",", '')
        self.answer_str = answer
        self.answer_num = int(answer)
        self.idx = idx

    def format_problem(self, retrieval_format=False):
        _problem_description = "# Problem Description\n" + self.problem_description
        prompt_str = "\n# Question\n" + self.question_str + "\n# Answer\n"
        answer_str = "The answer is " + self.answer_str + '.'
        if retrieval_format:
            prompt_str += "Let's first repeat the Problem Description and Question. ## Problem Description"
        else:
            prompt_str += "Let's think step by step. "
        return (_problem_description, prompt_str, answer_str)


class MMLUProblem:
    """
    .question: str
    .answer_num: float
    .cot_steps: (str, ) or (str, str)
    get_cot_step(idx): {"index": idx, "answer": str [, "question": str]}
    """

    def __init__(self, question_str, choices, answer_nbr, idx=None):
        self._question_str = question_str
        self.problem_description = question_str
        self.choices = choices
        self.question_str = "Choose the option that best satisfies the problem description. \n" + ' '.join(f"{a}: {b}" for a, b in zip(["0", "1", "2", "3"], self.choices))

        self.answer_str = str(answer_nbr)
        self.answer_num = answer_nbr
        self.idx = idx



    def format_problem(self, retrieval_format=False):
        """
        returns: problem_description, prompt, answer
        """


        _problem_description = "# Problem Description\n" + self.problem_description + "\n"
        prompt_str = "\n# Question\n" + self.question_str
        if retrieval_format:
            prompt_str += "\nFinish the answer with `The correct option is' and the number for that option. \n# Answer\n Let's first repeat the Problem Description and Analysis. ## Problem Description"
        else:
            prompt_str += "\nGive only the number for the correct option. # Answer\nThe correct option is "
        return (_problem_description, prompt_str,
                "The correct option is " + self.answer_str + '.')


class HumanEvalProblem:
    """
    .question: str
    .answer_num: float
    .cot_steps: (str, ) or (str, str)
    get_cot_step(idx): {"index": idx, "answer": str [, "question": str]}
    """

    def __init__(self, question_str, test_str, entry_point, idx=None):
        self._question_str = question_str
        self.problem_description = question_str
        self.test_str = test_str
        self.answer_str = "0"
        self.answer_num = 0
        lines = question_str.split("\n")
        def_line = ""
        for line in lines:
            if line.startswith("def "):
                def_line = line
                break
        self.question_str = def_line
        assert def_line, "No def line"

        self.idx = idx
        self.entry_point = entry_point



    def format_problem(self, retrieval_format=False):
        """
        returns: problem_description, prompt, answer
        """

        _problem_description = self.problem_description + "\n"
        _problem_description += '    """'
        prompt_str = '    """\n    # start answer here'  # wrap the filler inside a huge multiline

        return (_problem_description, prompt_str,
                None)

    def evaluate(self, full_response):
        try:
            # print("FFFF", full_response)
            response = full_response
            response_line = response.split("\n")
            response = ''
            start = False
            for line in response_line:
                if not line.startswith(" ") and len(line) > 0:
                    break
                response += line + "\n"
            if response.find("import") != -1:
                return 0
        except Exception as e:
            # print(e)
            return 0

        code = self.problem_description + "\n" + response + "\n" + self.test_str + f"\ncheck({self.entry_point})"
        # this is very dangerous and could lead to disastrous behavior. Consider safer evaluation

        # print()
        # print("EVAL")
        # print("CODE:")
        # print(code)

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_script = os.path.join(tmpdirname, "temp.py")
            with open(tmp_script, "w") as f:
                f.write(code)
            try:
                completed_process = subprocess.run(["python", tmp_script], capture_output=True, timeout=1)
            except subprocess.TimeoutExpired:
                return 0
            # print(completed_process.stdout.decode("utf-8"))
            # print(completed_process.stderr.decode("utf-8"))
            # print(f"return {completed_process.returncode} -----")
            if completed_process.returncode == 0:
                return 100
        return 0


class GSM8KProblemGenerator:
    """
    example_iter(num_few_shots=0): yields GSM8KProblem, List[GSM8KProblem]
    """

    def __init__(self, path="openai/gsm8k", subset="socratic", retrieval_format=False):
        # load dataset
        self.dataset = load_dataset(path, subset)
        self.train_set = self.dataset["train"]
        self.test_set = self.dataset["test"]
        self.retrieval_format = retrieval_format

        # build objects
        self.train_problems = [GSM8KProblem(item["question"], item["answer"]) for item in self.train_set]
        self.test_problems = [GSM8KProblem(item["question"], item["answer"], idx) for idx, item in
                              enumerate(self.test_set)]

    def __iter__(self, max_problems=None, num_few_shots=0):
        few_shot_examples = self.train_problems[:num_few_shots]
        few_shot_string = ''
        for example in few_shot_examples:
            few_shot_string += ' '.join(example.format_problem_as_complete_paragraph(True)) + '\n'
        numbers_in_few_shot = extract_numbers(few_shot_string)
        yielded_problems = 0
        for test_problem in self.test_problems:
            answer = test_problem.answer_num
            problem_description, _, _ = test_problem.format_problem(retrieval_format=self.retrieval_format)
            numbers_in_description = extract_numbers(problem_description)
            same_number_exists = False
            for number in numbers_in_description + numbers_in_few_shot:
                if abs(number - answer) < 1e-6:
                    same_number_exists = True  # filter
                    break
            if same_number_exists:
                continue
            yield test_problem, few_shot_examples
            yielded_problems += 1
            if yielded_problems == max_problems:
                return


class SimpleProblemGenerator:
    def __init__(self, retrieval_format=False):
        random.seed(1)
        self.retrieval_format = retrieval_format
        self.test_problems = []
        while len(self.test_problems) < 100:
            a = [random.randint(0, 100) for _ in range(20)]
            b = random.choices(list(range(20)), k=3)
            c = 0
            for i in b:
                c += a[i]
            qstr = "Let " + ", ".join(f"x_{i}={ai}" for i, ai in enumerate(a)) + ". "
            qstr += "If " + ", ".join(f"y_{i}=x_{bi}" for i, bi in enumerate(b)) + ", "
            qstr += "then what is " + "+".join(f"y_{i}" for i in range(len(b))) + "?"

            self.test_problems.append(SimpleProblem(qstr, f"{c}",
                                                    len(self.test_problems)))

    def __iter__(self, max_problems=None, num_few_shots=0):
        yielded_problems = 0
        for test_problem in self.test_problems:
            problem_description, _, _ = test_problem.format_problem(retrieval_format=self.retrieval_format)
            yield test_problem, []
            yielded_problems += 1
            if yielded_problems == max_problems:
                return


class SimplestProblem:
    def __init__(self, n1, n2, idx=None):
        self._question_str = f"Let x1={n1}, and let x2=x1. If x3=x2+{n2}, then what is x3?"
        self.problem_description = self._question_str
        self.question_str = self._question_str.split(". ")[-1]
        self.answer_str = f"{n1+n2}"
        self.answer_num = n1+n2
        self.idx = idx

    def format_problem(self, retrieval_format=False):
        _problem_description = "# Problem Description\n" + self.problem_description
        prompt_str = "\n# Question\n" + self.question_str + "\n# Answer\n"
        answer_str = "The answer is " + self.answer_str + '.'
        if retrieval_format:
            prompt_str += "Let's first repeat the Problem Description and Question. ## Problem Description"
        # else:
        #     prompt_str += "Let's think step by step. "
        return (_problem_description, prompt_str, answer_str)
class SimplestProblemGenerator:
    def __init__(self, retrieval_format=False):
        random.seed(1)
        self.retrieval_format = retrieval_format
        self.test_problems = []
        while len(self.test_problems) < 10000:
            a = random.randint(10, 100)
            b = random.randint(10, 100)
            self.test_problems.append(SimplestProblem(a, b, len(self.test_problems)))
    def __iter__(self, max_problems=None, num_few_shots=0):
        yielded_problems = 0
        for test_problem in self.test_problems:
            problem_description, _, _ = test_problem.format_problem(retrieval_format=self.retrieval_format)
            yield test_problem, []
            yielded_problems += 1
            if yielded_problems == max_problems:
                return


class MMLUProblemGenerator:
    """
    example_iter(num_few_shots=0): yields GSM8KProblem, List[GSM8KProblem]
    """

    def __init__(self, path="cais/mmlu", subset="all", retrieval_format=False):
        # load dataset
        self.dataset = load_dataset(path, subset)
        self.test_set = self.dataset["test"]
        self.retrieval_format = retrieval_format

        # build objects
        self.test_problems = [MMLUProblem(item["question"], item["choices"], item["answer"], idx) for idx, item in
                              enumerate(self.test_set)]

    def __iter__(self, max_problems=None, num_few_shots=0):
        yielded_problems = 0
        for test_problem in self.test_problems:
            yield test_problem, []
            yielded_problems += 1
            if yielded_problems == max_problems:
                return


class HumanEvalProblemGenerator:
    """
    example_iter(num_few_shots=0): yields GSM8KProblem, List[GSM8KProblem]
    """

    def __init__(self, path="openai/openai_humaneval", retrieval_format=False):
        # load dataset
        self.dataset = load_dataset(path)
        self.test_set = self.dataset["test"]
        self.retrieval_format = retrieval_format

        # build objects
        self.test_problems = [HumanEvalProblem(item["prompt"], item["test"], item["entry_point"], idx) for idx, item in
                              enumerate(self.test_set)]

    def __iter__(self, max_problems=None, num_few_shots=0):
        yielded_problems = 0
        for test_problem in self.test_problems:
            yield test_problem, []
            yielded_problems += 1
            if yielded_problems == max_problems:
                return



def megatron_client_generate(url, prompt_list, tokens_to_generate, window_size=None,
                             needle_positions=None, distance_between_positions=0):
    """
    masked_tokens: None or list[list[bool]]. Masked tokens for each example. True means not masked. False means masked.
    """
    if prompt_list is None:
        return None
    headers = {'Content-Type': 'application/json'}

    # format check
    assert isinstance(prompt_list, list)
    for prompt in prompt_list:
        assert isinstance(prompt, list) or isinstance(prompt, str)
    assert window_size is None
    assert needle_positions is None or isinstance(needle_positions, list)
    if needle_positions:
        for needle_position in needle_positions:
            assert isinstance(needle_position, list)
            for position in needle_position:
                assert isinstance(position, list)
                assert len(position) == 2
    assert isinstance(distance_between_positions,
                      int), f"{distance_between_positions} of Class {distance_between_positions.__class__}"

    data = {"prompts": prompt_list, "tokens_to_generate": tokens_to_generate,

            "ignore_special_tokens": True, "add_BOS": False, "random_seed": 0, "top_k": 1,
            "window_size": window_size, "stop_on_eol": False, "prevent_newline_after_colon": True,
            "distance_between_positions": distance_between_positions}  # for future implementation
    if needle_positions:
        data["oracle_positions"] = needle_positions

    # return [prompt_list[0] + f" This is some sample answer: Distance is {distance_between_positions}. "
    #                          f"Great work! The answer is: 1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12. 13. 14. 15. 16. 17. 18. 19. 20. -1. -2. -3. -4. -5. 10. "]

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

    data = {"texts": [text], "add_BOS": False, "ignore_special_tokens": True}
    # add kwargs to data
    for key, value in kwargs.items():
        data[key] = value
    # return [1, 2, 3, 4, 5]
    response = requests.put(url, data=json.dumps(data), headers=headers)

    if response.status_code != 200:
        raise ValueError(f"Error {response.status_code}: {response.json()}")
    else:
        return response.json()['token_ids'][0]


def megatron_client_detokenize(url, tokens, **kwargs):
    headers = {'Content-Type': 'application/json'}

    data = {"tokens": [tokens], "no_log": False, "ignore_special_tokens": True}
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
        raise ValueError(
            "Invalid request type. Must be 'generate', 'tokenize', or 'detokenize', or 'modify_window_size'.")


class APIModel:
    total_number = 0

    def __init__(self, static_model_info, dynamic_model_info, tokens_to_generate):
        """
        static_model_info: ANYTHING needed to define the model; if changed, the model output will change
        dynamic_model_info: ANYTHING needed to define the model; the model output will not change even if this changes
        """
        self.static_model_info = static_model_info
        self.dynamic_model_info = dynamic_model_info
        self.number = APIModel.total_number
        APIModel.total_number += 1
        self.tokens_to_generate = tokens_to_generate

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.static_model_info} {self.dynamic_model_info}>"

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__repr__() + str(self.number))

    def __eq__(self, other):
        return isinstance(other, APIModel) and self.number == other.number

    def tokenize(self, text, **kwargs):
        raise NotImplementedError

    def detokenize(self, tokens, **kwargs):
        raise NotImplementedError

    def __call__(self, prompt_list, tokens_to_generate=None, **kwargs):
        raise NotImplementedError

    def supports_rope_modification(self):
        raise NotImplementedError

    token_dict = {}

    def get_end_of_sentence_symbol_memoization(self):
        def memoize(enc):
            if '.' in self.token_dict:
                return self.token_dict['.']
            debug("get token by calling multiple cases")
            sentence1 = enc.tokenize("Good.", ignore_special_tokens=True)
            sentence2 = enc.tokenize("This is the last chance.", ignore_special_tokens=True)
            sentence3 = enc.tokenize("The answer is 2.", ignore_special_tokens=True)
            sentence4 = enc.tokenize("The answer is two!", ignore_special_tokens=True)
            sentence5 = enc.tokenize("Isn't that good?", ignore_special_tokens=True)

            self.token_dict['.'] = list(
                {sentence1[-1]} | {sentence2[-1]} | {sentence3[-1]} | {sentence4[-1]} | {sentence5[-1]})
            return self.token_dict['.']

        period_tokens = memoize(self)
        return period_tokens

    def get_space_memoization(self):
        if ' ' in self.token_dict:
            return self.token_dict[' ']
        possible_numbers_of_spaces = list(range(1, 32)) + list(2 ** k for k in range(5, 14))
        d = {}
        for i in possible_numbers_of_spaces:
            space_tokens = self.tokenize(" " * i, ignore_special_tokens=True)
            debug(space_tokens, f"Number of spaces: {i}, `{' ' * i}`")

            d[i] = space_tokens
            if len(space_tokens) >= 2 and space_tokens[0] == space_tokens[1]:
                break
            i += 1
        self.token_dict[' '] = space_tokens[0:1]
        debug(self.token_dict[" "], f"Number of spaces: {i}")
        return self.token_dict[' ']


class LiteLLMModel(APIModel):
    total_number = 0

    def __init__(self, model_name, api_key, tokens_to_generate=250):
        super().__init__(model_name, api_key, tokens_to_generate)
        self.api_key = api_key
        self.model_name = model_name

        # Import litellm

        self.litellm = litellm

        # Configure litellm with the API key and model settings
        if "azure" in model_name.lower():
            # For Azure models, expect api_key to be a dict with required Azure configs
            if isinstance(api_key, dict):
                for key, value in api_key.items():
                    setattr(self.litellm, key, value)
            else:
                raise ValueError("Azure models require api_key to be a dict with Azure configurations")
        elif "anthropic" in model_name.lower():
            # For Anthropic models

            os.environ["ANTHROPIC_API_KEY"] = api_key
            # Set model name to proper format (e.g., claude-2)
            self.model_name = model_name.replace("anthropic/", "")
        elif "openai" in model_name.lower():
            # For OpenAI models
            os.environ["OPENAI_API_KEY"] = api_key
            # Set model name to proper format (e.g., gpt-4)
            self.model_name = model_name.replace("openai/", "")
        elif "google" in model_name.lower():
            # For Google models
            os.environ["GOOGLE_API_KEY"] = api_key
            # Set model name to proper format (e.g., gemini-pro)
            self.model_name = model_name.replace("google/", "")
        elif "cohere" in model_name.lower():
            # For Cohere models
            os.environ["COHERE_API_KEY"] = api_key
            # Set model name to proper format (e.g., command)
            self.model_name = model_name.replace("cohere/", "")
        elif "together" in model_name.lower():
            # For Together AI models
            os.environ["TOGETHER_API_KEY"] = api_key
            # Set model name to proper format (e.g., togethercomputer/llama-2-70b)
            self.model_name = model_name.replace("together/", "")
            # Configure litellm to use Together AI
            self.litellm.api_base = "https://api.together.xyz"
        elif "sambanova" in model_name.lower():
            # For SambaNova models
            os.environ["SAMBANOVA_API_KEY"] = api_key
            # Set model name to proper format
            self.model_name = model_name.replace("sambanova/", "")
            # Configure litellm to use SambaNova endpoint
            if isinstance(api_key, dict) and "api_base" in api_key:
                self.litellm.api_base = api_key["api_base"]
            else:
                # Default SambaNova API endpoint
                self.litellm.api_base = "https://api.sambanova.ai"
        else:
            # Default to OpenAI if no provider specified
            os.environ["OPENAI_API_KEY"] = api_key
            self.model_name = model_name



    def tokenize(self, text, **kwargs):
        # Use litellm's tokenization
        return self.litellm.encode(model=self.model_name, text=text)

    def detokenize(self, tokens, **kwargs):
        # Use litellm's detokenization
        return self.litellm.decode(model=self.model_name, tokens=tokens)

    def __call__(self, prompt_list, tokens_to_generate=None, needle_positions=None, distance_between_positions=0):
        if tokens_to_generate is None:
            tokens_to_generate = self.tokens_to_generate

        # For litellm API, we just use the text directly
        if isinstance(prompt_list, list):
            if isinstance(prompt_list[0], list):
                # Handle tokenized input by converting back to text
                prompt = self.detokenize(prompt_list[0])
            else:
                # Handle string input
                prompt = prompt_list[0]
        else:
            prompt = prompt_list

        # Call litellm completion API
        try:
            response = self.litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=tokens_to_generate
            )

            return prompt + response.choices[0].message.content

        except Exception as e:
            print(f"Error calling LiteLLM API: {str(e)}")
            raise


class OpenAIModel(APIModel):

    def __init__(self, model_name, api_key, tokens_to_generate=250):
        super().__init__(model_name, api_key, tokens_to_generate)
        self.api_key = api_key
        self.model_name = model_name

        self.client = openai.OpenAI(api_key=self.api_key)

    def tokenize(self, text, **kwargs):
        # OpenAI uses tiktoken for tokenization

        encoding = tiktoken.encoding_for_model(self.model_name)
        return encoding.encode(text)

    def detokenize(self, tokens, **kwargs):
        # Convert tokens back to text
        encoding = tiktoken.encoding_for_model(self.model_name)
        return encoding.decode(tokens)

    def __str__(self):
        return f"OpenAIModel {self.model_name}"

    def __call__(self, prompt_list, tokens_to_generate=None, **_unused_kwargs):
        if tokens_to_generate is None:
            tokens_to_generate = self.tokens_to_generate

        # For OpenAI API, we just use the text directly
        if isinstance(prompt_list, list):
            if isinstance(prompt_list[0], list):
                # Handle tokenized input by converting back to text
                prompt = self.detokenize(prompt_list[0])
            else:
                # Handle string input
                prompt = prompt_list[0]
        else:
            prompt = prompt_list

        # Call OpenAI API
        print("CALL API")
        print(len(prompt))
        print(len(self.tokenize(prompt)))
        # print([{"role": "user", "content": prompt}],)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=tokens_to_generate
            )
        except Exception as e:
            print("Exception", e)
            print("!!!!")
            raise e
        print("RESPONSE", response.choices[0].message.content)
        return prompt + response.choices[0].message.content

    def supports_rope_modification(self):
        return False


class MegatronModel(APIModel):

    def __init__(self, _unused_static_info, url, tokens_to_generate=250):
        super().__init__(None, url, tokens_to_generate)
        self.url = url
        self.tokens_to_generate = tokens_to_generate

    def tokenize(self, text, **kwargs):
        return megatron_client_tokenize(get_url(self.url, "tokenize"), text, **kwargs)

    def detokenize(self, tokens, **kwargs):
        return megatron_client_detokenize(get_url(self.url, "detokenize"), tokens, **kwargs)

    def __call__(self, prompt_list, tokens_to_generate=None, needle_positions=None, distance_between_positions=0):
        if tokens_to_generate is None:
            tokens_to_generate = self.tokens_to_generate
        ret = megatron_client_generate(get_url(self.url, "generate"), prompt_list,
                                       tokens_to_generate, needle_positions=needle_positions,
                                       distance_between_positions=distance_between_positions)
        ret = ret[0]
        ret = ret.split("</s>", maxsplit=1)[0].rstrip()
        if ret.startswith("<s> "):
            ret = ret[4:]
        return ret

    def supports_rope_modification(self):
        return True


class Filler:
    def __init__(self, length, model):
        self.length = length
        self.model = model
        self.context_string = ""
        self.context_tokens = model.tokenize("", ignore_special_tokens=False, add_BOS=True)[0:1]  # get a <bos> token
        self.needle_positions = []
        self.distance_between_positions = 0

    def model_generate_kwargs(self):
        return {
            "prompt_list": [self.context_tokens],
            "needle_positions": [self.needle_positions],
            "distance_between_positions": self.distance_between_positions,
        }

    def model_generate(self):
        output = self.model(**self.model_generate_kwargs())
        debug("Output length:", len(output))
        debug("Input length:", len(self.context_string))
        return output[len(self.context_string):]

    def insert_needle(self, needle_tokens, insertion_point=None):
        pass

    def get_input_string(self):
        return self.context_string


class FillerGenerator:
    def __init__(self, filler_class, model=None, *args, **kwargs):
        self.filler_class = filler_class
        self.model = model
        self.args = args
        self.kwargs = kwargs

    def generate_filler(self, length):
        if self.model is None:
            raise ValueError("Model must be provided")
        return self.filler_class(length, self.model, *self.args, **self.kwargs)

    @staticmethod
    def create_filler_generator(filler_class, model, *args, **kwargs):
        filler_class_map = {
            "essay": PaulGrahamEssayFiller,
            "fake_distance": FakeDistanceFiller,
            "sequence": SequenceFiller,
            "hole": EssayHoleCreator
        }
        filler_class_cls = filler_class_map[filler_class]
        return FillerGenerator(filler_class_cls, model, *args, **kwargs)


class PaulGrahamEssayFiller(Filler):
    context_memo = {}
    context_tokens_memo = {}

    def __init__(self, length, model):
        length = max(length, 1)  # minimum length supported = 1  (bos token)
        super().__init__(length, model)
        self.files = glob.glob(f"PaulGrahamEssays/*.txt")
        self.files.sort()  # fix the order

        self.needle_positions = []
        self.model = model
        # collect tokens
        while len(self.context_tokens) < length:
            for file in self.files:

                # get string from file
                if file not in PaulGrahamEssayFiller.context_memo:
                    with open(file, 'r') as f:
                        new_context = f.read()
                    PaulGrahamEssayFiller.context_memo[file] = new_context
                else:
                    new_context = PaulGrahamEssayFiller.context_memo[file]

                # get tokens from tokenizer
                if model.number not in PaulGrahamEssayFiller.context_tokens_memo:
                    PaulGrahamEssayFiller.context_tokens_memo[model.number] = {}
                if file not in PaulGrahamEssayFiller.context_tokens_memo[model.number]:
                    new_context_tokens = model.tokenize(new_context, ignore_special_tokens=True)
                    PaulGrahamEssayFiller.context_tokens_memo[model.number][file] = new_context_tokens
                else:
                    new_context_tokens = PaulGrahamEssayFiller.context_tokens_memo[model.number][file]

                # save both string and tokens
                self.context_string += new_context
                self.context_tokens += new_context_tokens
                if len(self.context_tokens) >= length:
                    break

        # trim
        if len(self.context_tokens) > length:
            period_tokens = self.model.get_end_of_sentence_symbol_memoization()
            while length > 1 and self.context_tokens[length - 1] not in period_tokens:
                length -= 1
            self.context_tokens = self.context_tokens[:length]  # trim at complete sentence

            self.context_string = model.detokenize(self.context_tokens, ignore_special_tokens=True)

    def insert_needle(self, needle_tokens, insertion_point=None):
        if insertion_point is None:
            insertion_point = len(self.context_tokens)
        # we insert the needle next to the first period that comes after the position, or right after the bos token
        if insertion_point <= 1:
            insertion_point = 1
            self.context_tokens = self.context_tokens[:1] + needle_tokens + self.context_tokens[insertion_point:]
            self.needle_positions.append([1, len(needle_tokens) + 1])
        else:
            period_tokens = self.model.get_end_of_sentence_symbol_memoization()
            while insertion_point < len(self.context_tokens) and \
                    self.context_tokens[insertion_point] not in period_tokens:
                insertion_point += 1
            if insertion_point < len(self.context_tokens):
                insertion_point += 1
            if insertion_point >= len(self.context_tokens):
                insertion_point = len(self.context_tokens)
            self.context_tokens = (self.context_tokens[:insertion_point] + needle_tokens +
                                   self.context_tokens[insertion_point:])
            self.needle_positions.append([insertion_point, len(needle_tokens) + insertion_point])
            # insert the needles from front to back if there are multiple ones
        # debug(f"Needle: Position {self.needle_positions[-1]}, Length {len(self.context_tokens)}")
        self.context_string = self.model.detokenize(self.context_tokens, ignore_special_tokens=True)
        return insertion_point


class SequenceFiller(PaulGrahamEssayFiller):
    def __init__(self, length, model, filler_token_series):
        super().__init__(length, model)
        self.filler_token_series = filler_token_series
        self.substituted = False

    def model_generate_kwargs(self):
        if self.substituted:
            return super().model_generate_kwargs()
        last_end = 1
        token_counter = 0
        for start, end in self.needle_positions:
            while last_end < start:
                self.context_tokens[last_end] = self.filler_token_series[token_counter]
                token_counter = (token_counter + 1) % len(self.filler_token_series)
                last_end += 1
            last_end = end
            token_counter = 0
        self.context_string = self.model.detokenize(self.context_tokens, ignore_special_tokens=True)
        self.substituted = True
        return super().model_generate_kwargs()

class EssayHoleCreator(PaulGrahamEssayFiller):
    def __init__(self, length, model):
        super().__init__(length, model)
        self._real_needle_positions = []
    def create_holes(self, needle_length, hole_number=5):

        length = len(self.context_tokens) + needle_length * (hole_number - 1)
        hole_tokens = self.model.get_space_memoization() * needle_length
        for i in range(hole_number):
            self.insert_needle(hole_tokens, length // hole_number * i)

    def model_generate_kwargs(self):
        return {
            "prompt_list": [self.context_tokens],
            "needle_positions": [self._real_needle_positions],
            "distance_between_positions": self.distance_between_positions,
        }

    def fill_needle_at_idx(self, idx, needle_tokens):
        self._real_needle_positions = [self.needle_positions[idx]]
        insertion_point, end_point = self._real_needle_positions[0]
        self.context_tokens = (self.context_tokens[:insertion_point] + needle_tokens +
                               self.context_tokens[end_point:])
        self.context_string = self.model.detokenize(self.context_tokens, ignore_special_tokens=True)

    def insert_real_needle(self, needle_tokens, insertion_point=None):
        self.insert_needle(needle_tokens, insertion_point)
        self._real_needle_positions.append(self.needle_positions[-1])


class FakeDistanceFiller(Filler):
    def __init__(self, length, model):
        assert model.supports_rope_modification, f"Fake Distance Filler requires the model to support ROPE Modification, but model {model.__repr__()} does not"
        super().__init__(length, model)
        self.needle_positions = []
        self.needle_total_length = 1
        self.distance_between_positions = self.length

    def insert_needle(self, needle_tokens, insertion_point=None):
        assert insertion_point is None
        self.needle_positions.append([self.needle_total_length, self.needle_total_length + len(needle_tokens)])
        self.needle_total_length += len(needle_tokens)
        self.context_tokens = self.context_tokens + needle_tokens
        debug(f"Needle: Position {self.needle_positions[-1]}, Length {len(self.context_tokens)}")
        self.context_string = self.model.detokenize(self.context_tokens, ignore_special_tokens=True)
        return 0





class LazyTokenizedObject:
    def __init__(self, string_to_tokenize, default_tokenizer=None, ignore_special_tokens=True):
        self.tokenizer = default_tokenizer
        self.string_to_tokenize = string_to_tokenize
        self._tokens = None
        self.ignore_special_tokens = ignore_special_tokens

    def tokens(self, tokenizer=None, ) -> List[int]:
        """
        a tokenizer (MegatronModel.tokenize) is needed
        """
        if self._tokens is None:
            if tokenizer is None:
                tokenizer = self.tokenizer
            self._tokens = tokenizer(self.string_to_tokenize, ignore_special_tokens=self.ignore_special_tokens)
        return self._tokens

    def __str__(self):
        return self.string_to_tokenize

    def __repr__(self):
        return "LazyTokenizedObject(" + repr(self.string_to_tokenize) + ")"


class LazyTokenizedProblemObject:
    def __init__(self, problem_idx: int,
                 problem_description: LazyTokenizedObject,
                 prompt: LazyTokenizedObject,
                 answer_number,
                 question_str: str,
                 problem):
        self.problem_idx = problem_idx
        self.problem_description = problem_description
        self.prompt = prompt
        self.answer_number = answer_number
        self.question_str = question_str
        self.problem = problem


class Tester:
    count = 0

    @staticmethod
    def specs():
        raise NotImplementedError

    @staticmethod
    def running_specs():
        raise NotImplementedError

    @staticmethod
    def default_specs():
        raise NotImplementedError

    @staticmethod
    def default_running_specs():
        raise NotImplementedError

    def __init__(self, experiment_name, experiment_version, **test_specs):
        self.experiment_name = experiment_name
        self.experiment_version = experiment_version
        self.specs = test_specs
        self.specs_list = list((k, v) for k, v in self.specs.items())
        self.specs_list.sort(key=lambda x: x[0])
        self.savefile_root: Optional[str] = None
        self.tester_id = f"Tester No. {self.count} Created at {datetime.isoformat(datetime.now())}; Experiment " \
                         f"Name {experiment_name}; Version {experiment_version}; Specs {json.dumps(self.specs)}"
        self.count += 1

    def __hash__(self):
        return hash(self.tester_id)

    def get_status(self):
        return "unknown"

    def conventional_naming(self):
        raise NotImplementedError

    @classmethod
    def conventional_naming_st(cls, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def conventional_naming_template(experiment_name):
        return experiment_name + "*"

    def specs(self):
        return self.specs

    def is_running(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def save_name(self):
        # print("____CONV", self.conventional_naming())
        return self.conventional_naming()

    def save_path(self) -> str:
        if self.savefile_root is None:
            raise NotImplementedError
        # print("___________")
        # print(self.savefile_root)
        # print(self.save_name())
        # input()
        return os.path.join(self.savefile_root, self.save_name())

    def dump_specs(self):
        save_path = self.find_save_path()
        with open(os.path.join(save_path, "test_info.json"), "w") as f:
            json.dump({"type": "test",
                       "experiment_name": self.experiment_name,
                       "specs": self.specs,
                       "created": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z'), }, f)

    def find_save_path(self, create_if_not_exist=True):
        if os.path.exists(self.save_path()):
            return self.save_path()
        if create_if_not_exist:
            os.makedirs(self.save_path(), exist_ok=True)
            self.dump_specs()
            return self.save_path()
        return None

    def standard_start_test_adapter(self, **kwargs):
        raise NotImplementedError

    def standard_run_test_api(self, **kwargs):
        self.standard_start_test_adapter(**kwargs)

    def finished(self):
        raise NotImplementedError

    @staticmethod
    def parse_conventional_name(directory, experiment_name):
        """
        Return None if name is illegal; or specs
        """
        raise NotImplementedError


class FillerTester(Tester):
    """
    This class is used to test the LLM Needle Haystack.
    """

    @staticmethod
    def specs():
        return [
            "filler_type",
            "retrieval_format",
            "problem_set",
            "model_class_name",
            "model_static_info"
        ]

    @staticmethod
    def default_specs():
        return {
            "filler_type": "essay",
            "retrieval_format": False,
            "problem_set": "gsm8k",
            "model_class_name": "MegatronModel",
            "model_static_info": None
        }

    @staticmethod
    def running_specs():
        return [
            "context_lengths_min",
            "context_lengths_max",
            "context_lengths_num_intervals",
            "dynamic_model_info_list",
            "save_results",
            "skip_existing",
            "num_problems",
            "problem_index_start", "problem_index_end",
            "print_ongoing_status"
        ]

    @staticmethod
    def default_running_specs():
        return {
            "context_lengths_min": 0,
            "context_lengths_max": 30000,
            "context_lengths_num_intervals": 9,
            "save_results": True,
            "skip_existing": True,
            "num_problems": None,
            "problem_index_start": None, "problem_index_end": None,
            "print_ongoing_status": False
        }

    def __init__(self, experiment_name,
                 experiment_version=None,
                 savefile_root=None,
                 experiment_specs_str=None,
                 num_few_shots=0,
                 filler_type="fake_distance",
                 retrieval_format=False,
                 problem_set="gsm8k",
                 model_class_name="MegatronModel",
                 model_static_info=None
                 ):
        super().__init__(experiment_name,
                         experiment_version,
                         filler_type=filler_type,
                         retrieval_format=retrieval_format,
                         problem_set=problem_set,
                         model_class_name=model_class_name,
                         model_static_info=model_static_info)
        self.model_to_test_description = experiment_name
        self.problem_set = problem_set
        self.filler_type = filler_type
        if filler_type.startswith("sequence"):
            self.filler_type = "sequence"
        self.input_filler_type = filler_type
        self.retrieval_format = retrieval_format
        self.model_class_name = model_class_name
        self.model_static_info = model_static_info
        self.experiment_specs_str = experiment_specs_str or self.conventional_naming()
        self.savefile_root = savefile_root or os.path.join("results", "rope")
        self.results_version = experiment_version
        self.num_few_shots = num_few_shots


        self.dynamic_model_info_list = None
        self.print_ongoing_status = None
        self.save_results = None
        self.skip_existing = None
        self.num_problems = None
        self.context_lengths = None

        self.problem_index_range = None
        self.model_to_test_list = None  # [MegatronModel(url) for url in self.resource_list]
        self.first_model = None  # self.model_to_test_list[0]  # shorthand

        self.filler_generators = None

        if problem_set == "gsm8k":
            self.problem_generator = GSM8KProblemGenerator(retrieval_format=retrieval_format)
        elif problem_set == "simple":
            self.problem_generator = SimpleProblemGenerator(retrieval_format=retrieval_format)
        elif problem_set == "mmlu":
            self.problem_generator = MMLUProblemGenerator()
        elif problem_set == "human_eval":
            self.problem_generator = HumanEvalProblemGenerator()
        elif problem_set == "simplest":
            self.problem_generator = SimplestProblemGenerator(retrieval_format=retrieval_format)
        # prepare data
        (self.problem_descriptions, self.prompts, self.few_shots,
         self.answers_number, self.lazy_tokenized_problem_objects) = self.prepare_problems()
        self.results = None
        self.model_mapping = None
        self.temp_model_list = None
        self._finished = False

        self.stop_event = None
        self.running_status = {}
        self._status_text = "unknown"
        self.running_status_lock = threading.Lock()

    def get_status(self):
        with self.running_status_lock:
            return self._status_text

    def add_running_thread(self, things):
        with self.running_status_lock:
            self.running_status[threading.get_ident()] = things

    def pop_running_thread(self):
        with self.running_status_lock:
            if threading.get_ident() in self.running_status:
                self.running_status.pop(threading.get_ident())

    def get_thread_info(self):
        with self.running_status_lock:
            return copy.deepcopy(self.running_status)

    def is_running(self):
        with self.running_status_lock:
            return len(self.running_status) > 0

    def conventional_naming(self):
        # print("CONV")
        # print((f"rope_{self.problem_set}_{self.model_class_name}{('_' + self.model_static_info) if self.model_static_info else ''}_{self.experiment_name}_filler_"
        #         f"{self.input_filler_type}{'_retrieval' if self.retrieval_format else ''}"))
        return (f"rope_{self.problem_set}_{self.model_class_name}{('_' + self.model_static_info) if self.model_static_info else ''}_{self.experiment_name}_filler_"
                f"{self.input_filler_type}{'_retrieval' if self.retrieval_format else ''}")

    @staticmethod
    def conventional_naming_st(*args, **kwargs):
        keys = ["problem_set", "model_class_name", "model_static_info", "experiment_name", "input_filler_type", "retrieval_format"]
        problem_set, model_class_name, model_static_info, experiment_name, input_filler_type, retrieval_format = [kwargs.get(key, FillerTester.default_specs()[key]) for key in keys]
        return (
            f"rope_{problem_set}_{model_class_name}{('_' + model_static_info) if model_static_info else ''}_{experiment_name}_filler_"
            f"{input_filler_type}{'_retrieval' if retrieval_format else ''}")

    @staticmethod
    def conventional_naming_template(experiment_name):
        return f"rope_*_{experiment_name}_filler_*"



    @staticmethod
    def parse_conventional_name(directory, experiment_name):
        specs = FillerTester.default_specs()
        directory = Path(directory).name
        if not directory.startswith("rope"):
            return None
        parts = directory.split(experiment_name)
        if len(parts) != 2:
            return None
        first_part, second_part = parts
        first_part = first_part.strip("_")
        first_part = first_part.split("_")
        if len(first_part) < 2:
            return None
        specs["problem_set"] = first_part[1]
        if len(first_part) > 2:
            specs["model_class_name"] = first_part[2]
        if len(first_part) > 3:
            specs["model_static_info"] = first_part[3]

        second_part = second_part.strip("_")
        second_part = second_part.split("_")

        if second_part[-1] == "retrieval":
            specs["retrieval_format"] = True
            second_part = second_part[:-1]

        else:
            specs["retrieval_format"] = False
        if "filler" in second_part:
            filler_idx = second_part.index("filler")
            specs["filler_type"] = "_".join(second_part[filler_idx + 1:])
        else:
            return None
        return specs

    def generate_result_object(self, fake_context_length, index, needle, response, score, test_elapsed_time,
                               question, golden_number):
        return {
            'model': self.model_to_test_description,  # legacy
            'experiment_name': self.experiment_name,
            'specs': self.experiment_specs_str,
            'context_length': int(fake_context_length),
            'index': index,
            'version': self.results_version,
            'needle': needle,
            "question": question,
            'model_response': response,
            'score': score,
            'test_duration_seconds': test_elapsed_time,
            'test_timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z'),
            'golden_number': golden_number,
        }

    def prepare_problems(self):

        problem_descriptions = []
        prompts = []
        answers = []
        few_shots = None
        lazy_tokenized_problem_objects = []

        for problem, few_shot_problems in self.problem_generator.__iter__(self.num_problems, self.num_few_shots):
            problem_description, prompt, _answer = problem.format_problem(retrieval_format=self.retrieval_format)
            problem_description = LazyTokenizedObject(problem_description)
            prompt = LazyTokenizedObject(prompt)
            problem_descriptions.append(problem_description)
            prompts.append(prompt)
            answers.append(problem.answer_num)

            # organize
            question_str = problem.question_str
            problem_idx = problem.idx
            lazy_tokenized_problem_objects.append(
                LazyTokenizedProblemObject(problem_idx, problem_description, prompt, problem.answer_num, question_str, problem)
            )

        return problem_descriptions, prompts, few_shots, answers, lazy_tokenized_problem_objects

    def stop(self):
        if not self.is_running():
            return
        if self.stop_event is None:
            raise RuntimeError("Tester is running, but stop event is None")
        self.stop_event.set()
        t = 0
        while self.is_running():
            time.sleep(0.1)
            t += 1
            if t % 10 == 0:
                debug(f"Stop(): still running: {self.get_thread_info()}")

    def run_test(self, *, context_lengths_min,
                 context_lengths_max,
                 context_lengths_num_intervals,
                 dynamic_model_info_list,
                 save_results=True,
                 skip_existing=False,
                 num_problems=None,
                 problem_index_start=None, problem_index_end=None,
                 print_ongoing_status=False,
                 callback_function=None,
                 stop_event=None,
                 ):
        try:
            self.add_running_thread({"role": "main"})

            def callback_function_helper(thing):
                if callback_function:
                    callback_function(thing)

            # todo: implement stop signal for calling this function asynchronously
            self.context_lengths = np.round(
                np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals,
                            endpoint=True)).astype(int)

            self.dynamic_model_info_list = dynamic_model_info_list
            print("DMIL", dynamic_model_info_list)
            # input()
            self.save_results = save_results
            self.skip_existing = skip_existing
            self.num_problems = num_problems
            self.problem_index_range = [problem_index_start, problem_index_end]

            model_class = globals()[self.model_class_name]  # APIModel: OpenAIModel; MegatronModel
            self.model_to_test_list = [model_class(self.model_static_info, model_dynamic_info) for model_dynamic_info
                                       in self.dynamic_model_info_list]
            self.first_model = self.model_to_test_list[0]  # shorthand
            self.print_ongoing_status = print_ongoing_status

            if self.input_filler_type.startswith("sequence"):
                filler_type = self.input_filler_type.replace("sequence_", "")
                self.filler_type = "sequence"
                if filler_type == "space":
                    filler_args = [self.first_model.get_space_memoization()]
                else:
                    filler_args = [self.first_model.tokenize(filler_type.split("str")[1])]
            else:
                filler_args = []

            self.filler_generators = {
                model: FillerGenerator.create_filler_generator(self.filler_type, model, *filler_args)
                for model in self.model_to_test_list
            }

            self.results = []
            self._finished = False

            if self.print_ongoing_status:
                self.print_start_test_summary()

            if stop_event:
                self.stop_event = stop_event
            else:
                if not self.stop_event:
                    self.stop_event = threading.Event()
                self.stop_event.clear()
                stop_event = self.stop_event

            # Run through each iteration of context_lengths and depths
            job_input_list = []
            self.total_tasks = 0
            self.finished_tasks = 0
            debug("Collecting tests.")
            for context_length in self.context_lengths:
                if stop_event and stop_event.is_set():
                    self._finished = False
                    debug("Stop event triggered. Stopping the tester.")
                    return
                if context_length < context_lengths_min or context_length > context_lengths_max:
                    continue
                if self.skip_existing:
                    existing_idx = self.result_exists(context_length)
                else:
                    existing_idx = []

                for lazy_tokenized_problem_object in self.lazy_tokenized_problem_objects:
                    problem_idx = lazy_tokenized_problem_object.problem_idx
                    if self.problem_index_range:
                        if self.problem_index_range[0] is not None and problem_idx < self.problem_index_range[0]:
                            continue
                        if self.problem_index_range[1] is not None and problem_idx >= self.problem_index_range[1]:
                            continue
                    self.total_tasks += 1
                    if problem_idx in existing_idx:
                        if self.print_ongoing_status:
                            info(f"Existing: length {context_length} idx {problem_idx}")
                        self.finished_tasks += 1
                        continue

                    job_input_list.append((context_length, lazy_tokenized_problem_object))
            debug(f"There are {len(job_input_list)} tests queued. Submitting to executor.")
            callback_results = dict(total_tasks=self.total_tasks, finished_tasks=self.finished_tasks)
            callback_function_helper(callback_results)
            with ThreadPoolExecutor(len(self.model_to_test_list)) as executor:
                self.model_mapping = {}
                self.temp_model_list = [model for model in self.model_to_test_list]
                lock = Lock()
                lock2 = Lock()
                lock3 = Lock()
                self.results = []

                def evaluate_helper(*args, **kwargs):
                    try:
                        self.add_running_thread({"role": "evaluate"})
                        self.evaluate_and_log(*args, **kwargs)
                        self.pop_running_thread()
                    except Exception as e:
                        self.pop_running_thread()
                        raise e

                futures = [executor.submit(evaluate_helper, context_length, lazy_tokenized_problem_object, lock,
                                           lock2, lock3, callback_function_helper, stop_event)
                           for context_length, lazy_tokenized_problem_object in job_input_list]

                for future in as_completed(futures):
                    if stop_event and stop_event.is_set():
                        self._finished = False
                        info("Stop event triggered. Stopping the tester. The individual tests may take time to stop.")
                        self.pop_running_thread()
                        return
                    future.result()  # Wait for all to complete
            self._finished = True
            self.pop_running_thread()
            print("___FINISHED")
            return
        except Exception as e:
            self.pop_running_thread()
            raise e

    def finished(self):
        return self._finished

    def evaluate_and_log(self, context_length, lazy_tokenized_problem_object: LazyTokenizedProblemObject,
                         get_model_lock, gen_result_lock, print_lock, callback_function, stop_event):
        if stop_event and stop_event.is_set():
            return
        thread_id = threading.get_ident()

        with get_model_lock:
            if thread_id not in self.model_mapping:
                self.model_mapping[thread_id] = self.temp_model_list[0]
                self.temp_model_list = self.temp_model_list[1:]
            model = self.model_mapping[thread_id]
            # debug(f"Assigning thread_id {thread_id} to model {model.__str__()}, temp_model_list={self.temp_model_list}",
            #       )

        idx = lazy_tokenized_problem_object.problem_idx
        problem_description = lazy_tokenized_problem_object.problem_description
        prompt = lazy_tokenized_problem_object.prompt
        problem = lazy_tokenized_problem_object.problem
        golden_number = lazy_tokenized_problem_object.answer_number
        prompt_string = str(prompt)
        problem_description_string = str(problem_description)

        prompt_tokens = prompt.tokens(model.tokenize)
        problem_description_tokens = problem_description.tokens(model.tokenize)

        filler = self.filler_generators[model].generate_filler(int(context_length))
        filler.insert_needle(problem_description_tokens, 0)  # insert problem description at the beginning
        filler.insert_needle(prompt_tokens)  # insert prompt at the end

        # debug(f"Input kwargs: {filler.model_generate_kwargs()}")

        test_start_time = time.time()
        if self.retrieval_format:
            model.tokens_to_generate = len(problem_description_tokens) + 150
        else:
            if self.problem_set == "simple":
                model.tokens_to_generate = 600
            elif self.problem_set == "mmlu":
                model.tokens_to_generate = 1
            else:
                model.tokens_to_generate = 250
        full_response = filler.model_generate()

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        # problem description for comparison
        problem_description_for_comparison = problem_description_string
        problem_description_for_comparison = ''.join(re.split(r"\(Q\).*?\(A\)", problem_description_for_comparison))
        response = full_response.strip()

        if self.problem_set == "gsm8k":
            response = ''.join(re.split(r'<<.*?>>', response))
            problem_description_for_comparison = ''.join(re.split(r'<<.*?>>', problem_description_for_comparison))

            # find the question in section "problem description"
            x, y = problem_description_for_comparison.split("# Problem Description", maxsplit=1)
            x_without_question = x.replace(problem.question_str, "")
            problem_description_for_comparison = x_without_question + "# Problem Description" + y

        problem_description_for_comparison = problem_description_for_comparison.replace("# Problem Description\n", "").replace(
            "\n# Analysis\n", "")
        debug(f"Problem Description for Comparison: `{problem_description_for_comparison}`", print_lock=print_lock)

        rouge_score = scorer.score(problem_description_for_comparison, response)['rouge1'].recall * 100

        def compare(matches, golden_num, idx):
            matches_num = matches
            _scores = 0
            if idx is None:
                idx = slice(None)
            if isinstance(idx, slice):
                for num in matches_num[idx]:
                    if abs(num - golden_num) < 1e-6:
                        return 100
                return 0
            if isinstance(idx, int):
                try:
                    if abs(matches_num[idx] - golden_num) < 1e-6:
                        return 100
                    return 0
                except IndexError:
                    return 0
            return 0

        if self.problem_set == "human_eval":
            debug(f"Raw response: `{full_response}`")
            score = problem.evaluate(full_response)
            scores = {"loose": score, "strict": score, "flex": score, "rouge": 0}
            rouge_score = 0
        else:

            response_lines = response.split("\n", maxsplit=1)
            if len(response_lines) < 2:
                response_first_line, response_others = response, ""
            else:
                response_first_line, response_others = response_lines
            match_all = extract_numbers(response_first_line, False) + extract_numbers(response_others)
            # debug("Match all numbers:", match_all)
            debug(f"Raw response: `{response}`")

            if response.find("Answer") != -1:
                target_sentence = response.split("Answer")[-1]
                match = extract_numbers(target_sentence, False)
                scores = {"strict": compare(match, golden_number, -1), "flex": compare(match, golden_number, None)}
            else:
                # find the last number
                score = compare(match_all, golden_number, -1)
                scores = {"strict": score, "flex": score}
            scores["loose"] = compare(match_all, golden_number, None)
            scores["rouge"] = rouge_score
        results = self.generate_result_object(context_length, idx, problem_description_string, response, scores,
                                              test_elapsed_time, prompt_string, golden_number)
        with gen_result_lock:
            if self.print_ongoing_status:
                info()
                info(f"\033[32m---- Test Summary ----\033[0m ")
                info(f"\033[32mDuration:\033[0m {test_elapsed_time:.1f} seconds\033[0m ")
                info(f"\033[32mContext:\033[0m {context_length} tokens")
                info(f"\033[32mIndex:\033[0m {idx}")
                info(f"\033[32mNeedle:\033[0m {problem_description_string}")
                info(f"\033[32mQuestion:\033[0m {prompt_string}")
                info(f"\033[32mResponse:\033[0m `{response}`")
                info(f"\033[32mCorrect answer:\033[0m {golden_number}")
                info(f"\033[32mScore:\033[0m {scores['strict']}/{scores['flex']}/{scores['loose']}")
                info(f"\033[32mRetrieval Score:\033[0m {rouge_score}")
                debug(f"Input: `{filler.get_input_string()[:100].__repr__()}`...")
                debug(f"Input tokens: {filler.context_tokens[:8]}... of len {len(filler.context_tokens)}")
                debug(f"Positions: {filler.needle_positions}")
                debug(f"Tokens to generate: {model.tokens_to_generate}")

                info(f"\033[32m--- End Of Summary --- \033[0m")

            self.results.append(results)
            self.finished_tasks += 1
            if callback_function:
                callback_results = results.copy()
                callback_results.update(total_tasks=self.total_tasks, finished_tasks=self.finished_tasks)
                callback_function(callback_results)
            # input("Waiting for input.")

        if self.save_results:
            save_name = self.save_name()
            savefile_path = self.find_save_path()
            context_file_location = f'{save_name.replace(".", "_")}_len_{context_length}_problem_{idx}'

            # Save the result to file for retesting
            p = os.path.join(savefile_path, f"{context_file_location}_results.json")
            if self.print_ongoing_status:
                info(f"Writing at {p}")
            try:
                with open(p, 'w') as f:
                    json.dump(results, f)
            except Exception as e:
                print(e)
                raise e

    def save_name(self):
        # print("___sn", self.experiment_specs_str)
        return self.experiment_specs_str

    def summarize(self):
        score_per_length = {}
        retrieval_score_per_length = {}
        for result in self.results:
            # for now only check "loose"
            c_l = result["context_length"]
            l_s = result["score"]["loose"]
            l_r = result["score"]["rouge"]
            if not score_per_length.get(c_l, None):
                score_per_length[c_l] = []
                retrieval_score_per_length[c_l] = []
            score_per_length[c_l].append(l_s)
            retrieval_score_per_length[c_l].append(l_r)

        # calculate average

        full_score = 0
        full_retrieval = 0
        n_samples = 0
        for c_l in score_per_length:
            full_score += sum(score_per_length[c_l])
            full_retrieval += sum(retrieval_score_per_length[c_l])
            n_samples += len(score_per_length[c_l])
            score_per_length[c_l] = sum(score_per_length[c_l]) / max(1, len(score_per_length[c_l]))
            retrieval_score_per_length[c_l] = sum(retrieval_score_per_length[c_l]) / max(1, len(
                retrieval_score_per_length[c_l]))

        lengths = list(score_per_length.keys())
        lengths.sort()

        if self.print_ongoing_status:
            info()
            info("#### Final Summary ####")
            for length in lengths:
                info(f"{length:6}", end=" ")
            info()
            for length in lengths:
                info(f"{score_per_length[length] / 100:.4f}", end=" ")
            info("< reasoning")
            for length in lengths:
                info(f"{retrieval_score_per_length[length] / 100:.4f}", end=" ")
            info("< retrieval")
            info("Overall:", full_score / max(1, n_samples))
            info("Retrieval:", full_retrieval / max(1, n_samples))

    def result_exists(self, context_length):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = self.save_path()

        if self.print_ongoing_status:
            info("Searching existing results at %s ..." % results_dir, end="")
        if not os.path.exists(results_dir):
            if self.print_ongoing_status:
                info("Done")
            return []
        existing_idx = []
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                keys = filename.split("_")
                if "len" not in keys:
                    continue
                len_id = keys.index("len") + 1
                length = keys[len_id]
                problem_id = keys.index('problem') + 1
                pid = keys[problem_id]
                if length != str(context_length):
                    continue

                content_match = False
                if not content_match:
                    existing_idx.append(int(pid))
                    print(f"Exists: {context_length} {pid} {filename}")
                    continue

                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length

                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_to_test_description
                    # import ipdb; ipdb.set_trace()
                    if context_length_met and version_met and model_met:
                        existing_idx.append(result['index'])
                        if int(pid) != result['index']:
                            exit(1)
        if self.print_ongoing_status:
            info("Done")
        return existing_idx

    def print_start_test_summary(self):
        info("\n")
        info("Starting Test...")
        info(f"- Model: {self.model_to_test_description}")
        info(
            f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        info(
            f"- Number of problems: {len(self.problem_descriptions)}")
        info("\n\n")

    def start_test(self, *, context_lengths_min,
                   context_lengths_max,
                   context_lengths_num_intervals,
                   dynamic_model_info_list,
                   save_results=True,
                   skip_existing=False,
                   num_problems=None,
                   problem_index_start=None, problem_index_end=None,
                   print_ongoing_status=True,
                   callback_function=None,
                   stop_event=None
                   ):
        self.run_test(context_lengths_min=context_lengths_min, context_lengths_max=context_lengths_max,
                      context_lengths_num_intervals=context_lengths_num_intervals,
                      dynamic_model_info_list=dynamic_model_info_list,
                      save_results=save_results, skip_existing=skip_existing, num_problems=num_problems,
                      problem_index_start=problem_index_start, problem_index_end=problem_index_end,
                      print_ongoing_status=print_ongoing_status, callback_function=callback_function,
                      stop_event=stop_event)
        self.summarize()

    def standard_start_test_adapter(self, **kwargs):
        self.start_test(**kwargs)





class HoleFillingTester(FillerTester):
    @staticmethod
    def running_specs():
        return [
            "context_lengths_min",
            "context_lengths_max",
            "context_lengths_num_intervals",
            "dynamic_model_info_list",
            "save_results",
            "skip_existing",
            "num_problems",
            "problem_index_start", "problem_index_end",
            "hole_no_start", "hole_no_end",
            "print_ongoing_status"
        ]

    @staticmethod
    def default_running_specs():
        return {
            "context_lengths_min": 0,
            "context_lengths_max": 30000,
            "context_lengths_num_intervals": 9,
            "save_results": True,
            "skip_existing": True,
            "num_problems": None,
            "problem_index_start": None, "problem_index_end": None,
            "hole_no_start": 0, "hole_no_end": 5,
            "print_ongoing_status": False
        }

    def conventional_naming(self):
        # print("CONV")
        # print((f"rope_{self.problem_set}_{self.model_class_name}{('_' + self.model_static_info) if self.model_static_info else ''}_{self.experiment_name}_filler_"
        #         f"{self.input_filler_type}{'_retrieval' if self.retrieval_format else ''}"))
        return (f"hole_{self.problem_set}_{self.model_class_name}{('_' + self.model_static_info) if self.model_static_info else ''}_{self.experiment_name}_filler_"
                f"{self.input_filler_type}{'_retrieval' if self.retrieval_format else ''}")

    @staticmethod
    def conventional_naming_st(*args, **kwargs):
        keys = ["problem_set", "model_class_name", "model_static_info", "experiment_name", "input_filler_type", "retrieval_format"]
        problem_set, model_class_name, model_static_info, experiment_name, input_filler_type, retrieval_format = [kwargs.get(key, FillerTester.default_specs()[key]) for key in keys]
        return (
            f"hole_{problem_set}_{model_class_name}{('_' + model_static_info) if model_static_info else ''}_{experiment_name}_filler_"
            f"{input_filler_type}{'_retrieval' if retrieval_format else ''}")

    @staticmethod
    def conventional_naming_template(experiment_name):
        return f"hole_*_{experiment_name}_filler_*"



    @staticmethod
    def parse_conventional_name(directory, experiment_name):
        specs = FillerTester.default_specs()
        directory = Path(directory).name
        if not directory.startswith("hole"):
            return None
        parts = directory.split(experiment_name)
        if len(parts) != 2:
            return None
        first_part, second_part = parts
        first_part = first_part.strip("_")
        first_part = first_part.split("_")
        if len(first_part) < 2:
            return None
        specs["problem_set"] = first_part[1]
        if len(first_part) > 2:
            specs["model_class_name"] = first_part[2]
        if len(first_part) > 3:
            specs["model_static_info"] = first_part[3]

        second_part = second_part.strip("_")
        second_part = second_part.split("_")

        if second_part[-1] == "retrieval":
            specs["retrieval_format"] = True
            second_part = second_part[:-1]

        else:
            specs["retrieval_format"] = False
        if "filler" in second_part:
            filler_idx = second_part.index("filler")
            specs["filler_type"] = "_".join(second_part[filler_idx + 1:])
        else:
            return None
        return specs

    def generate_result_object(self, fake_context_length, hole_no, index, needle, response, score, test_elapsed_time,
                               question, golden_number):
        return {
            'model': self.model_to_test_description,  # legacy
            'experiment_name': self.experiment_name,
            'specs': self.experiment_specs_str,
            'context_length': int(fake_context_length),
            'hole_no': hole_no,
            'index': index,
            'version': self.results_version,
            'needle': needle,
            "question": question,
            'model_response': response,
            'score': score,
            'test_duration_seconds': test_elapsed_time,
            'test_timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z'),
            'golden_number': golden_number,
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_holes = 5

    def stop(self):
        if not self.is_running():
            return
        if self.stop_event is None:
            raise RuntimeError("Tester is running, but stop event is None")
        self.stop_event.set()
        t = 0
        while self.is_running():
            time.sleep(0.1)
            t += 1
            if t % 10 == 0:
                debug(f"Stop(): still running: {self.get_thread_info()}")

    def run_test(self, *, context_lengths_min,
                 context_lengths_max,
                 context_lengths_num_intervals,
                 dynamic_model_info_list,
                 save_results=True,
                 skip_existing=False,
                 num_problems=None,
                 problem_index_start=None, problem_index_end=None,
                 hole_no_start=None, hole_no_end=None,
                 print_ongoing_status=False,
                 callback_function=None,
                 stop_event=None,
                 ):
        try:
            self.add_running_thread({"role": "main"})

            def callback_function_helper(thing):
                if callback_function:
                    callback_function(thing)

            # todo: implement stop signal for calling this function asynchronously
            self.context_lengths = np.round(
                np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals,
                            endpoint=True)).astype(int)

            self.dynamic_model_info_list = dynamic_model_info_list
            print("DMIL", dynamic_model_info_list)
            # input()
            self.save_results = save_results
            self.skip_existing = skip_existing
            self.num_problems = num_problems
            self.problem_index_range = [problem_index_start, problem_index_end]

            model_class = globals()[self.model_class_name]  # APIModel: OpenAIModel; MegatronModel
            self.model_to_test_list = [model_class(self.model_static_info, model_dynamic_info) for model_dynamic_info
                                       in self.dynamic_model_info_list]
            self.first_model = self.model_to_test_list[0]  # shorthand
            self.print_ongoing_status = print_ongoing_status

            # if self.input_filler_type.startswith("sequence"):
            #     filler_type = self.input_filler_type.replace("sequence_", "")
            #     self.filler_type = "sequence"
            #     if filler_type == "space":
            #         filler_args = [self.first_model.get_space_memoization()]
            #     else:
            #         filler_args = [self.first_model.tokenize(filler_type.split("str")[1])]
            # else:
            filler_args = []

            self.filler_generators = {
                model: FillerGenerator.create_filler_generator("hole", model, *filler_args)
                for model in self.model_to_test_list
            }

            self.results = []
            self._finished = False

            if self.print_ongoing_status:
                self.print_start_test_summary()

            if stop_event:
                self.stop_event = stop_event
            else:
                if not self.stop_event:
                    self.stop_event = threading.Event()
                self.stop_event.clear()
                stop_event = self.stop_event

            # Run through each iteration of context_lengths and depths
            job_input_list = []
            self.total_tasks = 0
            self.finished_tasks = 0
            debug("Collecting tests.")
            if hole_no_end is None:
                hole_no_end = self.num_holes
            if hole_no_start is None:
                hole_no_start = 0
            hole_no_end = min(hole_no_end, self.num_holes)
            for context_length in self.context_lengths:
                for hole_no in range(hole_no_start, hole_no_end):
                    if stop_event and stop_event.is_set():
                        self._finished = False
                        debug("Stop event triggered. Stopping the tester.")
                        return
                    if context_length < context_lengths_min or context_length > context_lengths_max:
                        continue
                    if self.skip_existing:
                        existing_idx = self.result_exists(context_length, hole_no)
                    else:
                        existing_idx = []

                    for lazy_tokenized_problem_object in self.lazy_tokenized_problem_objects:
                        problem_idx = lazy_tokenized_problem_object.problem_idx
                        if self.problem_index_range:
                            if self.problem_index_range[0] is not None and problem_idx < self.problem_index_range[0]:
                                continue
                            if self.problem_index_range[1] is not None and problem_idx >= self.problem_index_range[1]:
                                continue
                        self.total_tasks += 1
                        if problem_idx in existing_idx:
                            if self.print_ongoing_status:
                                info(f"Existing: length {context_length} idx {problem_idx}")
                            self.finished_tasks += 1
                            continue

                        job_input_list.append((context_length, hole_no, lazy_tokenized_problem_object))
            debug(f"There are {len(job_input_list)} tests queued. Submitting to executor.")
            callback_results = dict(total_tasks=self.total_tasks, finished_tasks=self.finished_tasks)
            callback_function_helper(callback_results)
            with ThreadPoolExecutor(len(self.model_to_test_list)) as executor:
                self.model_mapping = {}
                self.temp_model_list = [model for model in self.model_to_test_list]
                lock = Lock()
                lock2 = Lock()
                lock3 = Lock()
                self.results = []

                def evaluate_helper(*args, **kwargs):
                    try:
                        self.add_running_thread({"role": "evaluate"})
                        self.evaluate_and_log(*args, **kwargs)
                        self.pop_running_thread()
                    except Exception as e:
                        self.pop_running_thread()
                        raise e

                futures = [executor.submit(evaluate_helper, context_length, hole_no, lazy_tokenized_problem_object, lock,
                                           lock2, lock3, callback_function_helper, stop_event)
                           for context_length, hole_no, lazy_tokenized_problem_object in job_input_list]

                for future in as_completed(futures):
                    if stop_event and stop_event.is_set():
                        self._finished = False
                        info("Stop event triggered. Stopping the tester. The individual tests may take time to stop.")
                        self.pop_running_thread()
                        return
                    future.result()  # Wait for all to complete
            self._finished = True
            self.pop_running_thread()
            print("___FINISHED")
            return
        except Exception as e:
            self.pop_running_thread()
            raise e

    def finished(self):
        return self._finished

    def evaluate_and_log(self, context_length, hole_no, lazy_tokenized_problem_object: LazyTokenizedProblemObject,
                         get_model_lock, gen_result_lock, print_lock, callback_function, stop_event):
        if stop_event and stop_event.is_set():
            return
        thread_id = threading.get_ident()

        with get_model_lock:
            if thread_id not in self.model_mapping:
                self.model_mapping[thread_id] = self.temp_model_list[0]
                self.temp_model_list = self.temp_model_list[1:]
            model = self.model_mapping[thread_id]
            # debug(f"Assigning thread_id {thread_id} to model {model.__str__()}, temp_model_list={self.temp_model_list}",
            #       )

        idx = lazy_tokenized_problem_object.problem_idx
        problem_description = lazy_tokenized_problem_object.problem_description
        prompt = lazy_tokenized_problem_object.prompt
        problem = lazy_tokenized_problem_object.problem
        golden_number = lazy_tokenized_problem_object.answer_number
        prompt_string = str(prompt)
        problem_description_string = str(problem_description)

        prompt_tokens = prompt.tokens(model.tokenize)
        problem_description_tokens = problem_description.tokens(model.tokenize)

        filler = self.filler_generators[model].generate_filler(int(context_length))
        filler.create_holes(len(problem_description_tokens), self.num_holes)
        filler.fill_needle_at_idx(hole_no, problem_description_tokens)  # insert problem description at the beginning
        filler.insert_real_needle(prompt_tokens)  # insert prompt at the end

        # debug(f"Input kwargs: {filler.model_generate_kwargs()}")

        test_start_time = time.time()
        if self.retrieval_format:
            model.tokens_to_generate = len(problem_description_tokens) + 150
        else:
            if self.problem_set == "simple":
                model.tokens_to_generate = 600
            elif self.problem_set == "mmlu":
                model.tokens_to_generate = 1
            else:
                model.tokens_to_generate = 250
        full_response = filler.model_generate()

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        # problem description for comparison
        problem_description_for_comparison = problem_description_string.replace("# Problem Description\n", "").replace(
            "\n# Analysis\n", "")
        problem_description_for_comparison = ''.join(re.split(r"\(Q\).*?\(A\)", problem_description_for_comparison))
        response = full_response.strip()

        if self.problem_set == "gsm8k":
            response = ''.join(re.split(r'<<.*?>>', response))
            problem_description_for_comparison = ''.join(re.split(r'<<.*?>>', problem_description_for_comparison))

        debug(f"Problem Description for Comparison: `{problem_description_for_comparison}`", print_lock=print_lock)

        rouge_score = scorer.score(problem_description_for_comparison, response)['rouge1'].recall * 100

        def compare(matches, golden_num, idx):
            matches_num = matches
            _scores = 0
            if idx is None:
                idx = slice(None)
            if isinstance(idx, slice):
                for num in matches_num[idx]:
                    if abs(num - golden_num) < 1e-6:
                        return 100
                return 0
            if isinstance(idx, int):
                try:
                    if abs(matches_num[idx] - golden_num) < 1e-6:
                        return 100
                    return 0
                except IndexError:
                    return 0
            return 0

        if self.problem_set == "human_eval":
            debug(f"Raw response: `{full_response}`")
            score = problem.evaluate(full_response)
            scores = {"loose": score, "strict": score, "flex": score, "rouge": 0}
            rouge_score = 0
        else:

            response_lines = response.split("\n", maxsplit=1)
            if len(response_lines) < 2:
                response_first_line, response_others = response, ""
            else:
                response_first_line, response_others = response_lines
            match_all = extract_numbers(response_first_line, False) + extract_numbers(response_others)
            # debug("Match all numbers:", match_all)
            debug(f"Raw response: `{response}`")

            if response.find("Answer") != -1:
                target_sentence = response.split("Answer")[-1]
                match = extract_numbers(target_sentence, False)
                scores = {"strict": compare(match, golden_number, -1), "flex": compare(match, golden_number, None)}
            else:
                # find the last number
                score = compare(match_all, golden_number, -1)
                scores = {"strict": score, "flex": score}
            scores["loose"] = compare(match_all, golden_number, None)
            scores["rouge"] = rouge_score
        results = self.generate_result_object(context_length, hole_no, idx, problem_description_string, response, scores,
                                              test_elapsed_time, prompt_string, golden_number)
        with gen_result_lock:
            if self.print_ongoing_status:
                info()
                info(f"\033[32m---- Test Summary ----\033[0m ")
                info(f"\033[32mDuration:\033[0m {test_elapsed_time:.1f} seconds\033[0m ")
                info(f"\033[32mContext:\033[0m {context_length} tokens")
                info(f"\033[32mHole No:\033[0m {hole_no}")
                info(f"\033[32mIndex:\033[0m {idx}")
                info(f"\033[32mNeedle:\033[0m {problem_description_string}")
                info(f"\033[32mQuestion:\033[0m {prompt_string}")
                info(f"\033[32mResponse:\033[0m `{response}`")
                info(f"\033[32mCorrect answer:\033[0m {golden_number}")
                info(f"\033[32mScore:\033[0m {scores['strict']}/{scores['flex']}/{scores['loose']}")
                info(f"\033[32mRetrieval Score:\033[0m {rouge_score}")
                debug(f"Input: `{filler.get_input_string()[:100].__repr__()}`...")
                debug(f"Input tokens: {filler.context_tokens[:8]}... of len {len(filler.context_tokens)}")
                debug(f"Positions: {filler.needle_positions}")
                debug(f"Tokens to generate: {model.tokens_to_generate}")

                info(f"\033[32m--- End Of Summary --- \033[0m")

            self.results.append(results)
            self.finished_tasks += 1
            if callback_function:
                callback_results = results.copy()
                callback_results.update(total_tasks=self.total_tasks, finished_tasks=self.finished_tasks)
                callback_function(callback_results)
            # input("Waiting for input.")

        if self.save_results:
            save_name = self.save_name()
            savefile_path = self.find_save_path()
            context_file_location = f'{save_name.replace(".", "_")}_len_{context_length}_holeno_{hole_no}_problem_{idx}'

            # Save the result to file for retesting
            p = os.path.join(savefile_path, f"{context_file_location}_results.json")
            if self.print_ongoing_status:
                info(f"Writing at {p}")
            try:
                with open(p, 'w') as f:
                    json.dump(results, f)
            except Exception as e:
                print(e)
                raise e

    def save_name(self):
        # print("___sn", self.experiment_specs_str)
        return self.experiment_specs_str

    def summarize(self):
        score_per_length = {}
        retrieval_score_per_length = {}
        for result in self.results:
            # for now only check "loose"
            c_l = result["context_length"]
            l_s = result["score"]["loose"]
            l_r = result["score"]["rouge"]
            if not score_per_length.get(c_l, None):
                score_per_length[c_l] = []
                retrieval_score_per_length[c_l] = []
            score_per_length[c_l].append(l_s)
            retrieval_score_per_length[c_l].append(l_r)

        # calculate average

        full_score = 0
        full_retrieval = 0
        n_samples = 0
        for c_l in score_per_length:
            full_score += sum(score_per_length[c_l])
            full_retrieval += sum(retrieval_score_per_length[c_l])
            n_samples += len(score_per_length[c_l])
            score_per_length[c_l] = sum(score_per_length[c_l]) / max(1, len(score_per_length[c_l]))
            retrieval_score_per_length[c_l] = sum(retrieval_score_per_length[c_l]) / max(1, len(
                retrieval_score_per_length[c_l]))

        lengths = list(score_per_length.keys())
        lengths.sort()

        if self.print_ongoing_status:
            info()
            info("#### Final Summary ####")
            for length in lengths:
                info(f"{length:6}", end=" ")
            info()
            for length in lengths:
                info(f"{score_per_length[length] / 100:.4f}", end=" ")
            info("< reasoning")
            for length in lengths:
                info(f"{retrieval_score_per_length[length] / 100:.4f}", end=" ")
            info("< retrieval")
            info("Overall:", full_score / max(1, n_samples))
            info("Retrieval:", full_retrieval / max(1, n_samples))

    def result_exists(self, context_length, hole_no):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = self.save_path()

        if self.print_ongoing_status:
            info("Searching existing results at %s ..." % results_dir, end="")
        if not os.path.exists(results_dir):
            if self.print_ongoing_status:
                info("Done")
            return []
        existing_idx = []
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                keys = filename.split("_")
                if "len" not in keys:
                    continue
                len_id = keys.index("len") + 1
                length = keys[len_id]
                problem_id = keys.index('problem') + 1
                pid = keys[problem_id]
                if length != str(context_length):
                    continue
                if "holeno" not in keys:
                    continue
                hole_id = keys.index("holeno") + 1
                hole_no_ = keys[hole_id]
                if hole_no_ != str(hole_no):
                    continue

                content_match = False
                if not content_match:
                    existing_idx.append(int(pid))
                    print(f"Exists: {context_length} {pid} {filename}")
                    continue

                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    hole_no_met = result["hole_no"] == hole_no
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_to_test_description
                    # import ipdb; ipdb.set_trace()
                    if context_length_met and version_met and model_met and hole_no_met:
                        existing_idx.append(result['index'])
                        if int(pid) != result['index']:
                            exit(1)
        if self.print_ongoing_status:
            info("Done")
        return existing_idx

    def print_start_test_summary(self):
        info("\n")
        info("Starting Test...")
        info(f"- Model: {self.model_to_test_description}")
        info(
            f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        info(
            f"- Number of problems: {len(self.problem_descriptions)}")
        info("\n\n")

    def start_test(self, *, context_lengths_min,
                   context_lengths_max,
                   context_lengths_num_intervals,
                   dynamic_model_info_list,
                   save_results=True,
                   skip_existing=False,
                   num_problems=None,
                   problem_index_start=None, problem_index_end=None,
                   hole_no_start=None, hole_no_end=None,
                   print_ongoing_status=True,
                   callback_function=None,
                   stop_event=None
                   ):
        self.run_test(context_lengths_min=context_lengths_min, context_lengths_max=context_lengths_max,
                      context_lengths_num_intervals=context_lengths_num_intervals,
                      dynamic_model_info_list=dynamic_model_info_list,
                      save_results=save_results, skip_existing=skip_existing, num_problems=num_problems,
                      problem_index_start=problem_index_start, problem_index_end=problem_index_end,
                      hole_no_start=hole_no_start, hole_no_end=hole_no_end,
                      print_ongoing_status=print_ongoing_status, callback_function=callback_function,
                      stop_event=stop_event)
        self.summarize()

    def standard_start_test_adapter(self, **kwargs):
        self.start_test(**kwargs)


class RepeatFillerTester(FillerTester):
    @staticmethod
    def specs():
        return [
            "filler_type",
            "problem_set",
            "model_class_name",
            "model_static_info"
        ]

    @staticmethod
    def default_specs():
        return {
            "filler_type": "essay",
            "problem_set": "gsm8k",
            "model_class_name": "MegatronModel",
            "model_static_info": None
        }

    @staticmethod
    def running_specs():
        return [
            "context_lengths_min",
            "context_lengths_max",
            "context_lengths_num_intervals",
            "dynamic_model_info_list",
            "save_results",
            "skip_existing",
            "num_problems",
            "problem_index_start", "problem_index_end",
            "print_ongoing_status"
        ]

    @staticmethod
    def default_running_specs():
        return {
            "context_lengths_min": 0,
            "context_lengths_max": 30000,
            "context_lengths_num_intervals": 9,
            "save_results": True,
            "skip_existing": True,
            "num_problems": None,
            "problem_index_start": None, "problem_index_end": None,
            "print_ongoing_status": False
        }

    def __init__(self, experiment_name,
                 experiment_version=None,
                 savefile_root=None,
                 experiment_specs_str=None,
                 num_few_shots=0,
                 filler_type="fake_distance",
                 retrieval_format=True,
                 problem_set="gsm8k",
                 model_class_name="MegatronModel",
                 model_static_info=None
                 ):
        super().__init__(experiment_name, experiment_version, savefile_root, experiment_specs_str, num_few_shots, filler_type,
                         True, problem_set, model_class_name, model_static_info)
    def evaluate_and_log(self, context_length, lazy_tokenized_problem_object: LazyTokenizedProblemObject,
                         get_model_lock, gen_result_lock, print_lock, callback_function, stop_event):
        if stop_event and stop_event.is_set():
            return
        thread_id = threading.get_ident()

        with get_model_lock:
            if thread_id not in self.model_mapping:
                self.model_mapping[thread_id] = self.temp_model_list[0]
                self.temp_model_list = self.temp_model_list[1:]
            model = self.model_mapping[thread_id]
            # debug(f"Assigning thread_id {thread_id} to model {model.__str__()}, temp_model_list={self.temp_model_list}",
            #       )

        idx = lazy_tokenized_problem_object.problem_idx
        problem_description = lazy_tokenized_problem_object.problem_description
        prompt = lazy_tokenized_problem_object.prompt
        problem = lazy_tokenized_problem_object.problem
        golden_number = lazy_tokenized_problem_object.answer_number
        prompt_string = str(prompt)
        problem_description_string = str(problem_description)

        prompt_tokens = prompt.tokens(model.tokenize)
        problem_description_tokens = problem_description.tokens(model.tokenize)

        filler = self.filler_generators[model].generate_filler(int(context_length))
        filler.insert_needle(problem_description_tokens, 0)  # insert problem description at the beginning
        filler.insert_needle(prompt_tokens)  # insert prompt at the end

        # debug(f"Input kwargs: {filler.model_generate_kwargs()}")

        test_start_time = time.time()
        model.tokens_to_generate = len(problem_description_tokens) + 50
        recite_response = filler.model_generate()
        _split = recite_response.split("## Answer", maxsplit=1)

        if len(_split) < 2:
            recite_part = recite_response
        else:
            recite_part = _split[0]

        _split = recite_part.split("## Others", maxsplit=1)
        if len(_split) < 2:
            recite_part = recite_part
        else:
            recite_part = _split[0]

        recite_part +=  "## Answer\n"

        if self.problem_set == "simple":
            model.tokens_to_generate = 600
        elif self.problem_set == "mmlu":
            model.tokens_to_generate = 1
        else:
            model.tokens_to_generate = 250
        full_response_1 = model(**{
            "prompt_list": [recite_part],
            "needle_positions": [],
            "distance_between_positions": 0,
        })

        full_response = full_response_1[len(recite_part):]

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time


        # problem description for comparison
        problem_description_for_comparison = problem_description_string
        problem_description_for_comparison = ''.join(re.split(r"\(Q\).*?\(A\)", problem_description_for_comparison))
        response = full_response.strip()
        recite_part_for_comparison = _split[0].strip()
        if self.problem_set == "gsm8k":
            recite_part_for_comparison = ''.join(re.split(r'<<.*?>>', recite_part_for_comparison))
            problem_description_for_comparison = ''.join(re.split(r'<<.*?>>', problem_description_for_comparison))

            # find the question in section "problem description"
            x, y = problem_description_for_comparison.split("# Analysis", maxsplit=1)
            x_without_question = x.replace(problem.question_str, "")
            problem_description_for_comparison = x_without_question + "# Analysis" + y
            __split = recite_part_for_comparison.split("## Analysis", maxsplit=1)
            if __split.__len__() == 2:
                _idx = __split[0].find(problem.question_str)
                if _idx != -1:
                    recite_part_for_comparison = recite_part_for_comparison[:_idx] + recite_part_for_comparison[_idx + len(problem.question_str):]

        problem_description_for_comparison = problem_description_for_comparison.replace("# Problem Description\n",
                                                                                        "").replace(
            "\n# Analysis\n", "")


        # debug(f"Problem Description for Comparison: `{problem_description_for_comparison}`", print_lock=print_lock)
        problem_description_for_comparison = problem_description_for_comparison.replace("\n", "")
        recite_part_for_comparison = recite_part_for_comparison.replace("\n", "")
        recite_part_for_comparison = recite_part_for_comparison.replace("## Problem Description", "")
        recite_part_for_comparison = recite_part_for_comparison.replace("## Analysis", "")
        recite_part_for_comparison = recite_part_for_comparison.replace("## Question", "")
        debug(f">>>\n{problem_description_for_comparison}\n<<<\n{recite_part_for_comparison}\n", print_lock=print_lock)
        rouge_score = scorer.score(problem_description_for_comparison, recite_part_for_comparison)['rouge1'].recall * 100

        def compare(matches, golden_num, idx):
            matches_num = matches
            _scores = 0
            if idx is None:
                idx = slice(None)
            if isinstance(idx, slice):
                for num in matches_num[idx]:
                    if abs(num - golden_num) < 1e-6:
                        return 100
                return 0
            if isinstance(idx, int):
                try:
                    if abs(matches_num[idx] - golden_num) < 1e-6:
                        return 100
                    return 0
                except IndexError:
                    return 0
            return 0

        if self.problem_set == "human_eval":
            debug(f"Raw response: `{full_response}`")
            score = problem.evaluate(full_response)
            scores = {"loose": score, "strict": score, "flex": score, "rouge": 0}
            rouge_score = 0
        else:

            response_lines = response.split("\n", maxsplit=1)
            if len(response_lines) < 2:
                response_first_line, response_others = response, ""
            else:
                response_first_line, response_others = response_lines
            match_all = extract_numbers(response_first_line, False) + extract_numbers(response_others)
            # debug("Match all numbers:", match_all)
            debug(f"Raw response: `{response}`")

            if response.find("Answer") != -1:
                target_sentence = response.split("Answer")[-1]
                match = extract_numbers(target_sentence, False)
                scores = {"strict": compare(match, golden_number, -1), "flex": compare(match, golden_number, None)}
            else:
                # find the last number
                score = compare(match_all, golden_number, -1)
                scores = {"strict": score, "flex": score}
            scores["loose"] = compare(match_all, golden_number, None)
            scores["rouge"] = rouge_score
        response = full_response_1
        results = self.generate_result_object(context_length, idx, problem_description_string, response, scores,
                                              test_elapsed_time, prompt_string, golden_number)
        with gen_result_lock:
            if self.print_ongoing_status:
                info()
                info(f"\033[32m---- Test Summary ----\033[0m ")
                info(f"\033[32mDuration:\033[0m {test_elapsed_time:.1f} seconds\033[0m ")
                info(f"\033[32mContext:\033[0m {context_length} tokens")
                info(f"\033[32mIndex:\033[0m {idx}")
                info(f"\033[32mNeedle:\033[0m {problem_description_string}")
                info(f"\033[32mQuestion:\033[0m {prompt_string}")
                info(f"\033[32mResponse:\033[0m `{response}`")
                info(f"\033[32mCorrect answer:\033[0m {golden_number}")
                info(f"\033[32mScore:\033[0m {scores['strict']}/{scores['flex']}/{scores['loose']}")
                info(f"\033[32mRetrieval Score:\033[0m {rouge_score}")
                debug(f"Input: `{filler.get_input_string()[:100].__repr__()}`...")
                debug(f"Input tokens: {filler.context_tokens[:8]}... of len {len(filler.context_tokens)}")
                debug(f"Positions: {filler.needle_positions}")
                debug(f"Tokens to generate: {model.tokens_to_generate}")

                info(f"\033[32m--- End Of Summary --- \033[0m")

            self.results.append(results)
            self.finished_tasks += 1
            if callback_function:
                callback_results = results.copy()
                callback_results.update(total_tasks=self.total_tasks, finished_tasks=self.finished_tasks)
                callback_function(callback_results)
            # input("Waiting for input.")

        if self.save_results:
            save_name = self.save_name()
            savefile_path = self.find_save_path()
            context_file_location = f'{save_name.replace(".", "_")}_len_{context_length}_problem_{idx}'

            # Save the result to file for retesting
            p = os.path.join(savefile_path, f"{context_file_location}_results.json")
            if self.print_ongoing_status:
                info(f"Writing at {p}")
            try:
                with open(p, 'w') as f:
                    json.dump(results, f)
            except Exception as e:
                print(e)
                raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tester', type=str, default="FillerTester", help='name of tester')
    parser.add_argument('--s', '--s_len', metavar='N', type=int, help='a number')
    parser.add_argument('--e', '--e_len', metavar='N', type=int, help='a number')

    parser.add_argument('--model_name', type=str, default=None, help='name of model')
    parser.add_argument('--model_name_suffix', type=str, default='', help='name of model')
    parser.add_argument('--num_intervals', type=int, default=40, help='number of intervals of the test')
    parser.add_argument('--dynamic_model_info_list', type=str, default="localhost:5000", help="dynamic_model_info_list (API etc)")
    parser.add_argument("--discard", action="store_true", help="discard the results")
    parser.add_argument("--skip_existing", action="store_true", help="skip existing")
    parser.add_argument("--num_problems", type=int, default=10, help="number of problems")
    parser.add_argument("--filler", type=str,
                        help='["essay", "fake_distance", "sequence_space", "sequence_<str>", "..."]')
    parser.add_argument("--output_level", type=str, choices=["info", "debug", "none"], default="info",
                        help="global output level")
    parser.add_argument("--print_ongoing_status", action="store_true",
                        help="output tester info (subordinated by global output level)")
    parser.add_argument("--retrieval", action="store_true", help="retrieval mode")
    parser.add_argument("--problem_index_start", type=int, default=None, help="start index of problem")
    parser.add_argument("--problem_index_end", type=int, default=None, help="end index of problem")
    parser.add_argument("--problem_set", type=str, default=None, help="ps")
    parser.add_argument("--model_class_name", type=str, default="MegatronModel", choices=["MegatronModel", "OpenAIModel"], help="")
    parser.add_argument("--model_static_info", type=str, default="gpt-4o", help="things like OpenAI model name (None for MegatronModel)")
    args = parser.parse_args()

    model_name = args.model_name

    if not hasattr(args, 's_len'):
        args.s_len = args.s
    if not hasattr(args, 'e_len'):
        args.e_len = args.e
    output_level = args.output_level
    ht = globals()[args.tester](experiment_name=model_name + args.model_name_suffix,
                      num_few_shots=0,
                      filler_type=args.filler,
                      retrieval_format=args.retrieval,
                      problem_set=args.problem_set,
                      model_class_name=args.model_class_name,
                      model_static_info=args.model_static_info
                      )

    ht.standard_run_test_api(
        save_results=not args.discard,
        context_lengths_min=args.s_len,
        context_lengths_max=args.e_len,
        context_lengths_num_intervals=args.num_intervals,
        dynamic_model_info_list=args.dynamic_model_info_list.split(','),
        skip_existing=args.skip_existing,
        num_problems=args.num_problems,
        problem_index_start=args.problem_index_start,
        problem_index_end=args.problem_index_end,
        print_ongoing_status=args.print_ongoing_status,
    )
