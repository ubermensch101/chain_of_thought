import os
from utils import load_data, extract_gold_answer, extract_pred_answer, MODEL_CONFIG
import argparse
import jsonlines
import regex as re

def is_correct(gold_answers, pred_answer):
    gold_answer = gold_answers
    return pred_answer == gold_answer

def evaluate_acc(dataset, predictions, non_empty_only=False, valid_only=False, debug=False):
	(correct_count, total_count) = (0, 0)
	for example, prediction in zip(dataset, predictions):
		gold_id = int(example["id"])
		if prediction == {}:
			continue
		pred_id = int(prediction["id"])

		try:
			assert gold_id == pred_id
		except:
			raise AssertionError(f"Gold id {gold_id} doesn't match pred id {pred_id}.")

		try:
			gold_answer = extract_gold_answer(example["answer"])
		except SyntaxError as e:
			print("Error: ", e)
			print(gold_id)
			exit(-1)
		pred_answer = extract_pred_answer(prediction["answer"])

		if non_empty_only and pred_answer == "":
			continue

		if valid_only:
			if type(pred_answer)==str and ("invalid" in pred_answer or "error" in pred_answer):
				continue

		total_count += 1

		try:
			correct = is_correct(gold_answer, pred_answer)
		except Exception as e:
			print("Error: ", e)
			print("Example: ", gold_id)
			print("Question: ", example["question"])
			print("Gold answer: ", gold_answer, type(gold_answer))
			print("Pred answer: ", pred_answer, type(pred_answer))
			print("Completion: ", prediction["completion"])
			print("\n")
			exit(-1)

		if correct:
			correct_count += 1

	acc = round(correct_count / total_count * 100, 1)
	return acc

if __name__ == "__main__":
    config = MODEL_CONFIG

    dataset_frn = config['dataset']
    dataset = load_data(dataset_frn)
	
    pred_frn = config['output_path']

    with open(pred_frn) as fr:
        reader = jsonlines.Reader(fr)
        predictions = [line for line in reader]

    acc = evaluate_acc(
		dataset=dataset,
		predictions=predictions
	)

    print(f"Dataset: {config['dataset']}\nModel: {config['LM']}")
    print(f"Answer accuracy: {acc}")
