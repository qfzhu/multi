import json
import gzip
import os


def read_problems(evalset_file):
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}


def stream_jsonl(filename):
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def extract_prompt():
    problems = read_problems('HumanEval.jsonl.gz')
    for task_id in problems:
        prompt = problems[task_id]["prompt"]
        id = str(task_id[10:])
        problem_dir = './HumanEval/test/' + ((4 - len(id)) * '0') + id
        if not os.path.exists(problem_dir):
            os.makedirs(problem_dir)
        with open(problem_dir + '/question.txt', 'w') as f:
            f.write(prompt)


if __name__ == '__main__':
    extract_prompt()
