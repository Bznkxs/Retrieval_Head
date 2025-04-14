import re

results_dir = 'results/rope/rope_gsm8k_v0312_filler_sequence_space'
import sys
if len(sys.argv) > 1:
    results_dir = sys.argv[1]
if len(sys.argv) > 2:
    remove_invalid = True if (sys.argv[2] == "True") else False
else:
    remove_invalid = False
import os
import json
import tqdm
from datasets import load_dataset

from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)


number_re_pattern = r"(?:(?<![a-zA-Z0-9])-)?(?:\d+,)*\d+(?:\.\d+)?"

def number_str_list_to_num(numbers_str):
    try:
        numbers = []
        for number in numbers_str:
            numbers.append(float(number.strip().replace(",", '')))
    except ValueError:
        print(numbers_str)
        exit(1)
    return numbers
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
            numbers.append(num_str)
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
        if answer.find(".") == -1:
            self.is_int_answer = True
        else:
            self.is_int_answer = False
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
        self.test_problems = [GSM8KProblem(item["question"], item["answer"], idx) for idx, item in enumerate(self.test_set)]

    def __iter__(self, max_problems=None, num_few_shots=0):
        few_shot_examples = self.train_problems[:num_few_shots]
        few_shot_string = ''
        for example in few_shot_examples:
            few_shot_string += ' '.join(example.format_problem_as_complete_paragraph(True)) + '\n'
        numbers_in_few_shot = extract_numbers(few_shot_string)
        numbers_in_few_shot = number_str_list_to_num(numbers_in_few_shot)

        yielded_problems = 0
        for test_problem in self.test_problems:
            answer = test_problem.answer_num
            problem_description, _, _ = test_problem.format_problem(retrieval_format=self.retrieval_format)
            numbers_in_description = extract_numbers(problem_description)
            numbers_in_description = number_str_list_to_num(numbers_in_description)
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

generator = GSM8KProblemGenerator()
problems = generator.test_problems #list(i for i, _ in generator.__iter__())


def good(name):
    return True
    w1 = name.split('_')
    w2 = w1.index("problem")
    w3 = w1[w2+1]
    if int(w3) < 100:
        return True
    return False

def read_files(results_dir, filename_list, _remove_invalid=False):
    results = []
    for filename in tqdm.tqdm(filename_list):
        with open(os.path.join(results_dir, filename), 'r') as f:
            try:
                result = json.load(f)
            except Exception as e:
                print("EXCEPTION", e, )
                print(f"When reading {os.path.join(results_dir, filename)}")
                print("----")
                os.remove(os.path.join(results_dir, filename))
                continue
            if result.get("context_length") is None:
                print(filename)
                print(result)
                print()
                continue
            if result["score"]["loose"] == 0 and result["score"]["rouge"] > 95 and result["context_length"] == 3750:
                beautify_result(result)
                continue
            results.append(result)
    return results


def get_result_at_point(results, length, idx):
    for result in results:
        if result["context_length"] == length and result["index"] == idx:
            return result

def beautify_result(result):
    for key in result:
        print_str = f"\033[32m{key}:\033[0m`"
        print_str_result = result[key] if isinstance(result[key], str) else str(result[key])
        psr = print_str_result.replace('\n', '\n' + ' ' * (len(key) + 1) + '`')
        print_str += psr + "`"
        print(print_str)

def calculate_retrieval_score(result):
    problem_description_string = result['needle']
    response = result["model_response"]
    problem_description_for_comparison = problem_description_string.replace("# Problem Description\n", "").replace(
        "\n# Analysis\n", "")
    problem_description_steps = re.split(r"\(Q\).*?\(A\)", problem_description_for_comparison)
    problem_description_for_comparison = ''.join(problem_description_steps)
    response_without_calculation = ''.join(re.split(r'<<.*?>>', response))
    problem_description_for_comparison = ''.join(re.split(r'<<.*?>>', problem_description_for_comparison))
    response_numbers = extract_numbers(response)
    problem_description_numbers = extract_numbers(problem_description_for_comparison, False)
    response_numbers_num = number_str_list_to_num(response_numbers)
    problem_description_numbers_num = number_str_list_to_num(problem_description_numbers)
    final_numbers_for_each_step = []
    for step in problem_description_steps:
        final_numbers_for_each_step.extend(extract_numbers(step, False)[-1:])
    final_numbers_for_each_step_num = number_str_list_to_num(final_numbers_for_each_step)
    numbers_in_problem_description = extract_numbers(problem_description_steps[0], False)
    numbers_in_problem_description_num = number_str_list_to_num(numbers_in_problem_description)
    rouge_score = scorer.score(problem_description_for_comparison, response_without_calculation)['rouge1'].recall * 100
    def match_bag(set1, set2):
        return len(set1.intersection(set2)) / max(1, len(set1)) * 100
    number_match_score_bag = match_bag(set(problem_description_numbers_num), set(response_numbers_num))
    substep_number_match = match_bag(set(final_numbers_for_each_step_num).union(set(numbers_in_problem_description_num)),
                                     set(response_numbers_num))
    if substep_number_match > 99.9 and rouge_score > 85:
        retrieval_binary_score = 100
    else:
        retrieval_binary_score = 0

    real_loose_score = 0
    problem = problems[result['index']]

    if problem.is_int_answer:
        if problem.answer_num in response_numbers_num:
            real_loose_score = 100
        else:
            real_loose_score = 0
    else:
        if problem.answer_num in response_numbers_num:
            real_loose_score = 100
        else:
            real_loose_score = 0
    if abs(problem.answer_num) < 10:
        small_answer = 1
    else:
        small_answer = 0

    def print_outlier(condition):
        if eval(condition):
            print("\033[31mOutlier\033[0m")
            print(f"\033[31mCondition: {condition}\033[0m")
            beautify_result(result)
            print_result = {}
            print_result["Numbers in problem description only"] = set(numbers_in_problem_description_num)
            print_result["Final Numbers for Each Step"] = set(final_numbers_for_each_step_num)
            print_result["Golden Number Set"] = set(final_numbers_for_each_step_num).union(
                set(numbers_in_problem_description_num))
            print_result["Response Number Set"] = set(response_numbers_num)
            print_result["Lacking in Response Number"] = set(print_result["Golden Number Set"]).difference(
                response_numbers_num)
            print_result["retrieval_binary_score"] = retrieval_binary_score
            print_result["real_loose_score"] = real_loose_score
            print_result["Format Problem"] = ''.join(problem.format_problem())
            beautify_result(print_result)
            input()

    print_outlier( "result['score']['loose'] != real_loose_score")  # < 100 and retrieval_binary_score == 100:
    # if substep_number_match < 100 and rouge_score > 85 and result["context_length"] != 30000:

        # input()
    return {"rouge": rouge_score, "number_match": number_match_score_bag, "substep_number_match": substep_number_match,
            "retrieval_binary": retrieval_binary_score, "real_loose_score": real_loose_score, "small_answer": small_answer}



class Summary:
    def __init__(self, results_dir):
        print("Searching existing results at %s" % results_dir)
        filenames = []
        for filename in os.listdir(results_dir):
            if filename.endswith('.json') and good(filename):
                filenames.append(filename)
        # print(filenames)
        results = read_files(results_dir, filenames, remove_invalid)
        results.sort(key=lambda x: (int(x['context_length']), int(x['index'])))
        self.results = results
        # print(get_result_at_point(results, 0, 0))
        self.scores_per_length = {}
        self.full_scores = {}
        self.avg_scores_per_length = {}
        self.n_samples = {}

    def add_score_record(self, length, scores):
        if scores.get("small_answer"):
            return
        if scores.get("small_answer") is not None:
            scores.pop("small_answer")
        if not self.scores_per_length.get(length, None):
            self.scores_per_length[length] = {key: [] for key in scores.keys()}  # {len: {key: [number]}}

        for key, value in scores.items():
            self.scores_per_length[length][key].append(value)

    def calculate_average_scores(self):
        lengths = list(self.scores_per_length.keys())
        lengths.sort()
        score_keys = set()
        for length in lengths:
            score_keys.update(set(self.scores_per_length[length].keys()))

        full_scores = {key: 0. for key in score_keys}  # {key: number}
        avg_scores_per_length = {length: {} for length in lengths}   # {len: {key: number}}
        avg_scores_up_to_length = {length: {} for length in lengths}  # {len: {key: number}}
        n_samples = {key: 0 for key in score_keys}  # {key: number}
        n_samples_per_length = {length: {} for length in lengths}
        for length in lengths:
            for key in self.scores_per_length[length].keys():
                score_records = self.scores_per_length[length].get(key, [])
                sum_scores = sum(score_records)
                n_samples_l_k = len(score_records)
                avg_scores_per_length[length][key] = sum_scores / max(1., n_samples_l_k)
                n_samples_per_length[length][key] = n_samples_l_k
                avg_scores_per_length[length]["_n_samples"] = n_samples_l_k
                full_scores[key] += sum_scores
                n_samples[key] += n_samples_l_k
                avg_scores_up_to_length[length][key] = full_scores[key] / max(1., n_samples[key])

        for key in score_keys:
            full_scores[key] = full_scores[key] / max(1, n_samples[key])
        self.full_scores = full_scores
        self.avg_scores_per_length = avg_scores_per_length
        self.avg_scores_up_to_length = avg_scores_up_to_length
        self.n_samples = n_samples
        self.score_keys = score_keys
        # self.score_keys.add("_n_samples")
        # self.score_keys = ["rouge", "rouge1", 'number_match', 'substep_number_match', 'retrieval_binary',]
        self.lengths = lengths



    def summarize(self):
        results = self.results
        for result in results:
            # for now only check "loose"
            c_l = result["context_length"]
            if result.get("hole_no") is not None:
                c_l = (c_l, result["hole_no"])
            l_s = result["score"]["loose"]
            l_r = result["score"].get("rouge", 0)
            # l_c = calculate_retrieval_score(result)
            score = {"reasoning": l_s, "rouge1": l_r,}
            # score.update(**l_c)
            self.add_score_record(c_l, score)

        # calculate average
        self.calculate_average_scores()
        scores_per_length = self.avg_scores_per_length
        full_scores = self.full_scores

        print()
        print("---------- Summary ----------")
        for length in self.lengths:
            if isinstance(length, int):
                print(f"{length:8}", end="  ")
            else:
                print(f"{length[0]:6}, {length[1]:1}", end=" ")
        print()
        for key in list(self.score_keys) + ["_n_samples"]:
            for length in self.lengths:
                if key != "_n_samples":
                    print(f"{scores_per_length[length][key] / 100:.4f}", end="    ")
                else:
                    print(f"{scores_per_length[length][key]:8}", end="  ")
            print(f" < {key}")
        print("Overall:")
        for key in self.score_keys:
            print(f"{key}:", full_scores[key])

        # summarize()

summary = Summary(results_dir)
summary.summarize()

# summary = Summary('results/rope/rope_gsm8k_v0312_filler_sequence_space')
# summary2 = Summary('results/rope/rope_gsm8k_v0312_filler_fake_distance')
# for result1, result2 in zip(summary.results, summary2.results):
#     if result1["context_length"] != result2["context_length"]:
#         print("1")
#     if result1["model_response"] != result2["model_response"]:
#         print("_____")
#         print(result1, result2)
#         print("_____")