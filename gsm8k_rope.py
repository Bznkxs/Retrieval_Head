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
from typing import List

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

number_re_pattern = r"-?(?:\d+,)*\d+\.?\d*"

_debug_flag = False
def debug(*args, **kwargs):
    if _debug_flag:
        print("\033[90m", end="")
        print(*args, **kwargs)
        print("\033[0m", end="")

def extract_numbers(target_str):
    numbers_str = re.findall(number_re_pattern, target_str)
    numbers = []
    for number in numbers_str:
        numbers.append(float(number.strip().replace(",", '')))
    return numbers


class GSM8KProblem:
    """
    .question: str
    .answer_num: float
    .cot_steps: (str, ) or (str, str)
    get_cot_step(idx): {"index": idx, "answer": str [, "question": str]}
    """

    def __init__(self, question_str, answer_str):
        self._question_str = question_str
        self.problem_description = question_str
        self.question_str = question_str.split(". ")[-1]
        self._answer_str = answer_str
        cot_str, answer = answer_str.split("####")
        answer = answer.strip().replace(",", '')
        self.answer_str = answer
        self.answer_num = float(answer)
        self.cot_steps = [cot_step.split(" ** ") for cot_step in cot_str.strip().split("\n")]

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


    def format_problem(self):
        """
        returns: problem_description, prompt, answer
        """

        def cot_step_format(_step, with_answer=True):
            return "(Q)" + _step["question"] + " (A)" + (_step["answer"] if with_answer else "")

        _problem_description = "# Problem Description\n" + self.problem_description + "\n# Analysis\n"
        _problem_description += ''.join(cot_step_format(self.get_cot_step(idx))
                                        for idx in range(len(self.cot_steps) - 1))
        last_step = self.get_cot_step(-1)
        return (_problem_description, "\n# Question\n" + self.question_str + "\n# Answer\nLet's first repeat the Problem Description and Analysis. ## Problem Description",
                "The answer is " + self.answer_str + '.')

class GSM8KProblemGenerator:
    """
    example_iter(num_few_shots=0): yields GSM8KProblem, List[GSM8KProblem]
    """

    def __init__(self, path="openai/gsm8k", subset="socratic"):
        # load dataset
        self.dataset = load_dataset(path, subset)
        self.train_set = self.dataset["train"]
        self.test_set = self.dataset["test"]

        # build objects
        self.train_problems = [GSM8KProblem(item["question"], item["answer"]) for item in self.train_set]
        self.test_problems = [GSM8KProblem(item["question"], item["answer"]) for item in self.test_set]

    def __iter__(self, max_problems=None, num_few_shots=0):
        few_shot_examples = self.train_problems[:num_few_shots]
        few_shot_string = ''
        for example in few_shot_examples:
            few_shot_string += ' '.join(example.format_problem_as_complete_paragraph(True)) + '\n'
        numbers_in_few_shot = extract_numbers(few_shot_string)
        yielded_problems = 0
        for test_problem in self.test_problems:
            answer = test_problem.answer_num
            problem_description, _, _ = test_problem.format_problem()
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
    assert isinstance(distance_between_positions, int), f"{distance_between_positions} of Class {distance_between_positions.__class__}"

    # print("Generate", len(prompt_list))
    data = {"prompts": prompt_list, "tokens_to_generate": tokens_to_generate,

            "ignore_special_tokens": True, "add_BOS": False, "random_seed": 0, "top_k": 1,
            "window_size": window_size, "stop_on_eol": False, "prevent_newline_after_colon": True,
            "distance_between_positions": distance_between_positions}  # for future implementation
    if needle_positions:
        data["oracle_positions"] = needle_positions
    # print(data)
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
    # print("Tokenize 1")
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
    # print("Detokenize 1")
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
    def __init__(self, filler_class, model, *args, **kwargs):
        self.filler_class = filler_class
        self.model = model
        self.args = args
        self.kwargs = kwargs

    def generate_filler(self, length):
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
        self.needle_total_length = 0
        self.distance_between_positions = self.length

    def insert_needle(self, needle_tokens, insertion_point=None):
        self.needle_positions.append([self.needle_total_length, self.needle_total_length + len(needle_tokens)])
        self.needle_total_length += len(needle_tokens)
        self.context_tokens = self.context_tokens + needle_tokens
        self.context_string = self.model.detokenize(self.context_tokens, ignore_special_tokens=True)




class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """

    def __init__(self, experiment_name, url, num_few_shots=0, context_lengths_min=0, context_lengths_max=4096,
                 context_lengths_num_intervals=1, save_results=True, skip_existing=False, num_problems=10,
                 filler_type="fake_distance"):
        self.model_to_test_description = experiment_name
        self.model_version = "rope_gsm8k_" + experiment_name + "_filler_" + filler_type
        self.service_url = url
        self.num_few_shots = num_few_shots
        self.print_ongoing_status = True
        self.save_results = save_results
        self.results_version = 1
        self.skip_existing = skip_existing
        self.num_problems = num_problems
        self.filler_type = filler_type

        self.model_to_test = MegatronModel(self.service_url)
        self.enc = self.model_to_test

        if filler_type.startswith("sequence"):
            filler_type = filler_type.replace("sequence_", "")
            self.filler_type = "sequence"
            if filler_type == "space":
                filler_args = [get_space_memoization(self.enc)]
            else:
                filler_args = [self.enc.tokenize(filler_type.split("str")[1])]
        else:
            filler_args = []

        self.filler_generator = FillerGenerator.create_filler_generator(self.filler_type, self.model_to_test, *filler_args)

        self.context_lengths = np.round(
            np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals,
                        endpoint=True)).astype(int)

        # (prompt_list, tokens_to_generate, needle_positions)
        self.problem_generator = GSM8KProblemGenerator()
        # prepare data
        self.problem_descriptions, self.prompts, self.few_shots, self.answers_number = self.prepare_problems()
        self.results = []
        # print(self.answers)

    def generate_result_object(self, fake_context_length, index, needle, response, score, test_elapsed_time, ):
        return {
            'model': self.model_to_test_description,
            'context_length': int(fake_context_length),
            'index': index,
            'version': self.results_version,
            'needle': needle,
            'model_response': response,
            'score': score,
            'test_duration_seconds': test_elapsed_time,
            'test_timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z'),
        }

    def prepare_problems(self):

        tokenizer = self.enc.tokenize

        class LazyTokenizedObject:
            def __init__(self, string_to_tokenize):
                self.tokenizer = tokenizer
                self.string_to_tokenize = string_to_tokenize
                self._tokens = None

            def tokens(self, ignore_special_tokens=True):
                if self._tokens is None:
                    self._tokens = self.tokenizer(self.string_to_tokenize, ignore_special_tokens=ignore_special_tokens)
                return self._tokens

            def __str__(self):
                return self.string_to_tokenize

            def __repr__(self):
                return "LazyTokenizedObject(" + repr(self.string_to_tokenize) + ")"

        problem_descriptions = []
        prompts = []
        answers = []
        few_shots = None

        for problem, few_shot_problems in self.problem_generator.__iter__(self.num_problems, self.num_few_shots):
            problem_description, prompt, _answer = problem.format_problem()
            problem_descriptions.append(LazyTokenizedObject(problem_description))
            prompts.append(LazyTokenizedObject(prompt))
            answers.append(problem.answer_num)

        return problem_descriptions, prompts, few_shots, answers

    def run_test(self, args):

        # Run through each iteration of context_lengths and depths
        for context_length in self.context_lengths:
            if context_length < args.s_len or context_length > args.e_len:
                continue
            self.evaluate_and_log(context_length)

    def decode(self, q_outputs, inp, decode_len):
        return q_outputs, None

    def generate_problems_iter(self):

        for idx in range(len(self.problem_descriptions)):
            problem_description = self.problem_descriptions[idx]
            question_str = self.problem_generator.test_problems[idx].question_str
            prompt = self.prompts[idx]
            problem_description_tokens = problem_description.tokens()
            prompt_tokens = prompt.tokens()
            yield (problem_description_tokens, prompt_tokens,
                   str(problem_description), str(prompt), self.answers_number[idx], question_str)

    def evaluate_and_log(self, context_length):
        save_name = self.model_version
        for idx, (problem_description_tokens, prompt_tokens,
                  problem_description_string, prompt_string,
                  golden_number, question_string) in enumerate(
                self.generate_problems_iter()):

            filler = self.filler_generator.generate_filler(int(context_length))
            filler.insert_needle(problem_description_tokens, 0)   # insert problem description at the beginning
            filler.insert_needle(prompt_tokens)   # insert prompt at the end

            # debug(f"Input kwargs: {filler.model_generate_kwargs()}")

            test_start_time = time.time()
            response = filler.model_generate()

            test_end_time = time.time()
            test_elapsed_time = test_end_time - test_start_time

            # problem description for comparison
            problem_description_for_comparison = problem_description_string.replace("[Problem Description]\n", "").replace("[\nAnalysis]\n", "")
            problem_description_for_comparison = ''.join(re.split(r"\(Q\).*?\(A\)", problem_description_for_comparison))
            response = ''.join(re.split(r'<<.*?>>', response))
            problem_description_for_comparison = ''.join(re.split(r'<<.*?>>', problem_description_for_comparison))
            debug(f"Problem Description for Comparison: `{problem_description_for_comparison}`")

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

            match_all = extract_numbers(response)
            debug("Match all numbers:", match_all)
            debug(f"Raw response: `{response}`")
            if response.find("Question") != -1:
                response = response.split("Question")[0]
            if response.find("he answer is") != -1:
                target_sentence = response.split("he answer is")[-1]
                match = extract_numbers(target_sentence)
                scores = {"strict": compare(match, golden_number, -1), "flex": compare(match, golden_number, None)}
            else:
                # find the last number
                score = compare(match_all, golden_number, -1)
                scores = {"strict": score, "flex": score}
            scores["loose"] = compare(match_all, golden_number, None)
            scores["rouge"] = rouge_score
            results = self.generate_result_object(context_length, idx, filler.get_input_string(), response, scores,
                                                  test_elapsed_time, )

            if self.print_ongoing_status:
                print()
                print(f"---- Test Summary ---- ")
                print(f"Duration: {test_elapsed_time:.1f} seconds")
                print(f"Context: {context_length} tokens")
                print(f"Index: {idx}")
                print(f"Needle: {problem_description_string}")
                print(f"Question: {question_string}")
                print(f"Response: `{response}`")
                print(f"Correct answer: {golden_number}")
                print(f"Score: {scores['strict']}/{scores['flex']}/{scores['loose']}")
                print(f"Retrieval Score: {rouge_score}")
                debug(f"Input: `{filler.get_input_string()[:100].__repr__()}`...")
                debug(f"Input tokens: {filler.context_tokens[:8]}... of len {len(filler.context_tokens)}")
                debug(f"Positions: {filler.needle_positions}")
                debug(f"Tokens to generate: {self.model_to_test.tokens_to_generate}")

                print(f"--- End Of Summary --- ")

            self.results.append(results)
            # input("Waiting for input.")
            context_file_location = f'{self.model_version.replace(".", "_")}_len_{context_length}_problem_{idx}'

            if self.save_results:
                # Save the context to file for retesting
                if not os.path.exists(f'results/rope/{save_name}'):
                    os.makedirs(f'results/rope/{save_name}')

                # Save the result to file for retesting
                p = f'results/rope/{save_name}/{context_file_location}_results.json'
                print("Writing at %s" % p)
                with open(p, 'w') as f:
                    json.dump(results, f)

    def summarize(self):
        score_per_length = {}
        for result in self.results:
            # for now only check "loose"
            c_l = result["context_length"]
            l_s = result["score"]["loose"]
            if not score_per_length.get(c_l, None):
                score_per_length[c_l] = []
            score_per_length[c_l].append(l_s)

        # calculate average
        full_score = 0
        n_samples = 0
        for c_l in score_per_length:
            full_score += sum(score_per_length[c_l])
            n_samples += len(score_per_length[c_l])
            score_per_length[c_l] = sum(score_per_length[c_l]) / max(1, len(score_per_length[c_l]))

        lengths = list(score_per_length.keys())
        lengths.sort()

        for length in lengths:
            print(f"{length:6}", end=" ")
        print()
        for length in lengths:
            print(f"{score_per_length[length] / 100:.4f}", end=" ")
        print()
        print("Overall:", full_score / max(1, n_samples))

    def result_exists(self, context_length, idx):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = 'results/rope/' + self.model_version
        print("Searching existing results at %s" % results_dir)
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['index'] == idx
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_to_test_description
                    # import ipdb; ipdb.set_trace()
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    def print_start_test_summary(self):
        print("\n")
        print("Starting Needle In A Haystack Testing...")
        print(f"- Model: {self.model_version}")
        print(
            f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print(
            f"- Number of problems: {len(self.problem_descriptions)}")
        print("\n\n")

    def start_test(self, args):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        self.run_test(args)
        self.summarize()


token_dict = {}


def get_end_of_sentence_symbol_memoization(enc):
    def memoize(enc):
        if '.' in token_dict:
            return token_dict['.']
        print("get token by calling multiple cases")
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
    token_dict[' '] = enc.tokenize("  ", ignore_special_tokens=True)
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
    parser.add_argument("--filler", type=str, help='["essay", "fake_distance", "sequence_space", "sequence_<str>", "..."]')
    parser.add_argument("--debug", action="store_true", help="debug mode")

    # parser = add_args(parser)
    args = parser.parse_args()

    model_name = args.model_name

    if not hasattr(args, 's_len'):
        args.s_len = args.s
    if not hasattr(args, 'e_len'):
        args.e_len = args.e
    _debug_flag = args.debug
    ht = LLMNeedleHaystackTester(experiment_name=model_name + args.model_name_suffix,
                                 save_results=not args.discard,
                                 context_lengths_min=args.s_len,
                                 context_lengths_max=args.e_len,
                                 context_lengths_num_intervals=args.num_intervals,
                                 url=args.url,
                                 skip_existing=args.skip_existing,
                                 num_few_shots=0,
                                 num_problems=args.num_problems,
                                 filler_type=args.filler
                                 )

    ht.start_test(args)
