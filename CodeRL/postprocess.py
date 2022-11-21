import json
import sys
from human_eval.data import write_jsonl


def to_human_eval_format(path):
    samples = []
    for i in range(164):
        with open(path + str(i) + '.json') as fr:
            content = json.load(fr)
            solutions = content[str(i)]['code']
            for sol in solutions:
                samples.append({'task_id': 'HumanEval/' + str(i), 'completion': sol})

    write_jsonl("samples.jsonl", samples)


if __name__ == '__main__':
    path = sys.argv[1]  # path to coderl ouptuts/codes dir
    to_human_eval_format(path)