import json
import gzip
import os
import sys

from human_eval.data import read_problems


def extract_prompt(path):
    problems = read_problems(path)
    for task_id in problems:
        prompt = problems[task_id]["prompt"]
        id = str(task_id[10:])
        problem_dir = './HumanEval/test/' + ((4 - len(id)) * '0') + id
        if not os.path.exists(problem_dir):
            os.makedirs(problem_dir)
        with open(problem_dir + '/question.txt', 'w') as f:
            f.write(prompt)


if __name__ == '__main__':
    path = sys.argv[1]  # path to HumanEval.json.gz
    extract_prompt()
