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
        last_step = self.get_cot_step(-1)
        prompt_str = "\n# Question\n" + self.question_str + "\n# Answer\n"
        if retrieval_format:
            prompt_str += "Let's first repeat the Problem Description and Analysis. ## Problem Description"
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
        return (_problem_description, prompt_str, answer_str)


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
            a = [random.randint(-100, 100) for _ in range(50)]
            b = random.choices(list(range(50)), k=3)
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


class MegatronModel:
    total_number = 0

    def __init__(self, url, tokens_to_generate=250):
        self.url = url
        self.number = MegatronModel.total_number
        MegatronModel.total_number += 1
        self.tokens_to_generate = tokens_to_generate

    def __repr__(self):
        return f"<Model {self.url}>"

    def __hash__(self):
        return hash(self.number)

    def __eq__(self, other):
        return isinstance(other, MegatronModel) and self.number == other.number

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
        return ret[0]


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
            period_tokens = get_end_of_sentence_symbol_memoization(self.model)
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
            period_tokens = get_end_of_sentence_symbol_memoization(self.model)
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
        debug(f"Needle: Position {self.needle_positions[-1]}, Length {len(self.context_tokens)}")
        self.context_string = self.model.detokenize(self.context_tokens, ignore_special_tokens=True)


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


class FakeDistanceFiller(Filler):
    def __init__(self, length, model):
        super().__init__(length, model)
        self.needle_positions = []
        self.needle_total_length = 1
        self.distance_between_positions = self.length

    def insert_needle(self, needle_tokens, insertion_point=None):
        self.needle_positions.append([self.needle_total_length, self.needle_total_length + len(needle_tokens)])
        self.needle_total_length += len(needle_tokens)
        self.context_tokens = self.context_tokens + needle_tokens
        debug(f"Needle: Position {self.needle_positions[-1]}, Length {len(self.context_tokens)}")
        self.context_string = self.model.detokenize(self.context_tokens, ignore_special_tokens=True)


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
                 question_str: str):
        self.problem_idx = problem_idx
        self.problem_description = problem_description
        self.prompt = prompt
        self.answer_number = answer_number
        self.question_str = question_str


class Tester:
    count = 0
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
        return self.experiment_name + ",".join(f"{k}={v}" for k, v in self.specs_list)

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
        return self.conventional_naming()

    def save_path(self) -> str:
        if self.savefile_root is None:
            raise NotImplementedError
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
        return {}

class FillerTester(Tester):
    """
    This class is used to test the LLM Needle Haystack.
    """

    def __init__(self, experiment_name,
                 experiment_version=None,
                 savefile_root=None,
                 experiment_specs_str=None,
                 num_few_shots=0,
                 filler_type="fake_distance",
                 retrieval_format=False,
                 problem_set="gsm8k",
                 ):
        super().__init__(experiment_name,
                         experiment_version,
                         num_few_shots=num_few_shots,
                         filler_type=filler_type,
                         retrieval_format=retrieval_format,
                         problem_set=problem_set)
        self.model_to_test_description = experiment_name
        self.problem_set = problem_set
        self.filler_type = filler_type
        if filler_type.startswith("sequence"):
            self.filler_type = "sequence"
        self.input_filler_type = filler_type
        self.retrieval_format = retrieval_format
        self.experiment_specs_str = experiment_specs_str or self.conventional_naming()
        self.savefile_root = savefile_root or os.path.join("results", "rope")
        self.results_version = experiment_version
        self.num_few_shots = num_few_shots

        self.service_url = None
        self.print_ongoing_status = None
        self.save_results = None
        self.skip_existing = None
        self.num_problems = None
        self.context_lengths = None

        self.problem_index_range = None
        self.model_to_test_list = None  # [MegatronModel(url) for url in self.service_url]
        self.first_model = None  # self.model_to_test_list[0]  # shorthand

        self.filler_generators = None

        if problem_set == "gsm8k":
            self.problem_generator = GSM8KProblemGenerator(retrieval_format=retrieval_format)
        elif problem_set == "simple":
            self.problem_generator = SimpleProblemGenerator(retrieval_format=retrieval_format)
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
        return (f"rope_{self.problem_set}_{self.experiment_name}_filler_"
                f"{self.input_filler_type}{'_retrieval' if self.retrieval_format else ''}")

    @staticmethod
    def conventional_naming_template(experiment_name):
        return f"rope_*_{experiment_name}_filler_*"

    @staticmethod
    def parse_conventional_name(directory, experiment_name):
        directory = Path(directory).name
        parts = directory.split(experiment_name)
        if len(parts) != 2:
            return None
        first_part, second_part = parts
        first_part = first_part.split("_")
        if len(first_part) < 2:
            return None
        problem_set = first_part[1]
        second_part = second_part.rstrip("_")
        second_part = second_part.split("_")


        if second_part[-1] == "retrieval":
            retrieval = True
            second_part = second_part[:-1]

        else:
            retrieval = False
        if "filler" in second_part:
            filler_idx = second_part.index("filler")
            filler_type = "_".join(second_part[filler_idx+1:])
        else:
            return None
        return {
            "problem_set": problem_set,
            "retrieval_format": retrieval,
            "filler_type": filler_type,
            # "num_few_shots": 0
        }


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
                LazyTokenizedProblemObject(problem_idx, problem_description, prompt, problem.answer_num, question_str)
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
                 url_list,
                 save_results=True,
                 skip_existing=False,
                 num_problems=None,
                 problem_index_start=None, problem_index_end=None,
                 print_ongoing_status=False,
                 callback_function=None,
                 stop_event=None
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

            self.service_url = url_list.split(",")
            if url_list.find("[") != -1:
                prefix = url_list[:url_list.find("[")]
                url_list = url_list[url_list.find("[") + 1:-1]
                url_list = url_list.split(",")
                self.service_url = [prefix + url + ":5000" for url in url_list]
            self.save_results = save_results
            self.skip_existing = skip_existing
            self.num_problems = num_problems
            self.problem_index_range = [problem_index_start, problem_index_end]
            self.model_to_test_list = [MegatronModel(url) for url in self.service_url]
            self.first_model = self.model_to_test_list[0]  # shorthand
            self.print_ongoing_status = print_ongoing_status

            if self.input_filler_type.startswith("sequence"):
                filler_type = self.input_filler_type.replace("sequence_", "")
                self.filler_type = "sequence"
                if filler_type == "space":
                    filler_args = [get_space_memoization(self.first_model)]
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
            callback_function(callback_results)
            with ThreadPoolExecutor(len(self.service_url)) as executor:
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
                                           lock2, lock3, callback_function, stop_event)
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
            debug(f"Assigning thread_id {thread_id} to model {model.url}, temp_model_list={self.temp_model_list}",
                  )

        idx = lazy_tokenized_problem_object.problem_idx
        problem_description = lazy_tokenized_problem_object.problem_description
        prompt = lazy_tokenized_problem_object.prompt
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
            else:
                model.tokens_to_generate = 250
        response = filler.model_generate().strip()

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        # problem description for comparison
        problem_description_for_comparison = problem_description_string.replace("# Problem Description\n", "").replace(
            "\n# Analysis\n", "")
        problem_description_for_comparison = ''.join(re.split(r"\(Q\).*?\(A\)", problem_description_for_comparison))
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

        response_lines = response.split("\n", maxsplit=1)
        if len(response_lines) < 2:
            response_first_line, response_others = response, ""
        else:
            response_first_line, response_others = response_lines
        match_all = extract_numbers(response_first_line, False) + extract_numbers(response_others)
        debug("Match all numbers:", match_all)
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
            with open(p, 'w') as f:
                json.dump(results, f)

    def save_name(self):
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
                 url_list,
                 save_results=True,
                 skip_existing=False,
                 num_problems=None,
                 problem_index_start=None, problem_index_end=None,
                 print_ongoing_status=False,
                   callback_function=None,
                   stop_event=None
                   ):
        self.run_test(context_lengths_min=context_lengths_min,
                      context_lengths_max=context_lengths_max,
                      context_lengths_num_intervals=context_lengths_num_intervals,
                      url_list=url_list,
                      save_results=save_results,
                      skip_existing=skip_existing,
                      num_problems=num_problems,
                      problem_index_start=problem_index_start,
                      problem_index_end=problem_index_end,
                      print_ongoing_status=print_ongoing_status,
                      callback_function=callback_function,
                      stop_event=stop_event)
        self.summarize()

    def standard_start_test_adapter(self, **kwargs):
        self.start_test(**kwargs)


token_dict = {}


def get_end_of_sentence_symbol_memoization(enc):
    def memoize(enc):
        if '.' in token_dict:
            return token_dict['.']
        debug("get token by calling multiple cases")
        sentence1 = enc.tokenize("Good.", ignore_special_tokens=True)
        sentence2 = enc.tokenize("This is the last chance.", ignore_special_tokens=True)
        sentence3 = enc.tokenize("The answer is 2.", ignore_special_tokens=True)
        sentence4 = enc.tokenize("The answer is two!", ignore_special_tokens=True)
        sentence5 = enc.tokenize("Isn't that good?", ignore_special_tokens=True)

        token_dict['.'] = list({sentence1[-1]} | {sentence2[-1]} | {sentence3[-1]} | {sentence4[-1]} | {sentence5[-1]})
        return token_dict['.']

    period_tokens = memoize(enc)
    return period_tokens


def get_space_memoization(enc):
    if ' ' in token_dict:
        return token_dict[' ']
    possible_numbers_of_spaces = list(range(1, 32)) + list(2 ** k for k in range(5, 14))
    d = {}
    for i in possible_numbers_of_spaces:
        space_tokens = enc.tokenize(" " * i, ignore_special_tokens=True)
        debug(space_tokens, f"Number of spaces: {i}, `{' ' * i}`")

        d[i] = space_tokens
        if len(space_tokens) >= 2 and space_tokens[0] == space_tokens[1]:
            break
        i += 1
    token_dict[' '] = space_tokens[0:1]
    debug(token_dict[" "], f"Number of spaces: {i}")
    return token_dict[' ']


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('--s', '--s_len', metavar='N', type=int, help='a number')
    parser.add_argument('--e', '--e_len', metavar='N', type=int, help='a number')
    parser.add_argument('--model_name', type=str, default=None, help='name of model')
    parser.add_argument('--model_name_suffix', type=str, default='', help='name of model')
    parser.add_argument('--num_intervals', type=int, default=40, help='number of intervals of the test')
    parser.add_argument('--url', type=str, default="localhost:5000", help="service url")
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
    # parser = add_args(parser)
    args = parser.parse_args()

    model_name = args.model_name

    if not hasattr(args, 's_len'):
        args.s_len = args.s
    if not hasattr(args, 'e_len'):
        args.e_len = args.e
    output_level = args.output_level
    ht = FillerTester(experiment_name=model_name + args.model_name_suffix,
                      num_few_shots=0,
                      filler_type=args.filler,
                      retrieval_format=args.retrieval,
                      problem_set=args.problem_set,
                      )

    ht.standard_run_test_api(
        save_results=not args.discard,
        context_lengths_min=args.s_len,
        context_lengths_max=args.e_len,
        context_lengths_num_intervals=args.num_intervals,
        url_list=args.url,
        skip_existing=args.skip_existing,
        num_problems=args.num_problems,
        problem_index_start=args.problem_index_start,
        problem_index_end=args.problem_index_end,
        print_ongoing_status=args.print_ongoing_status
    )
