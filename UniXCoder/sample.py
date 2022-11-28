import torch
from unixcoder import UniXcoder
from human_eval.data import write_jsonl, read_problems

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniXcoder("microsoft/unixcoder-base")
model.to(device)

problems = read_problems()

num_samples_per_task = 100

with print_time('sampling'):
    samples = []
    for task_id in problems:
        sols = generate_for_problem(problems[task_id]["prompt"], num_samples_per_task)
        for sol in sols:
            samples.append(dict(task_id=task_id, completion=sol))

write_jsonl('unixcoder.jsonl', samples)


def generate_for_problem(context, beam_size):
    context = '"""' + 'Create a python function. ' + context + '"""'
    tokens_ids = model.tokenize([context],max_length=512,mode="<encoder-decoder>")
    source_ids = torch.tensor(tokens_ids).to(device)
    prediction_ids = model.generate(source_ids, decoder_only=False, beam_size=beam_size, max_length=512)
    predictions = model.decode(prediction_ids)
    return([x.replace("<mask0>","").strip() for x in predictions[0]])
