'''Make predictions on the dataset using the model.'''
import os
import sys
from solver import Model
from utils import load_data, MODEL_CONFIG
import jsonlines
import time
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    config = MODEL_CONFIG

    dataset_frn = config['dataset']
    dataset = load_data(dataset_frn)

    model = Model(config)

    output_fwn = config['output_path']

    line_id = 0
    if os.path.isfile(output_fwn):
        with open(output_fwn, "r") as fr:
            reader = jsonlines.Reader(fr)
            for line_id, line in enumerate(reader):
                example_id = line["id"]
    if line_id > 0:
        start_id = line_id+1
    else:
        start_id = 0

    print(f"Making predictions on dataset {config['dataset']} using model {config['LM']},\nstarting from the {start_id}th example...")

    with open(output_fwn, 'a') as fw:
        writer = jsonlines.Writer(fw, flush=True)
        t0 = time.time()
        for i, example in tqdm(enumerate(dataset), file=sys.stdout):
            if i < start_id:
                continue
            question = example["question"]
            question_id = int(example["id"])
            try:
                output = model.predict(example)
                answer = output["answer"]
                completion = output["completion"]
            except Exception as e:
                answer, completion = "[error]", str(e)
                print(f"Error at example {i}: {str(e)}", file=sys.stderr)

            row = {
                "id": question_id,
                "answer": answer,
                "completion": completion
            }
            writer.write(row)

        if i % 50 == 0:
            print(f"Finished {i} examples in {time.time() - t0} seconds.", flush=True)
