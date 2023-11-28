'''Dataset utilities.'''

import json
import re
import csv
from collections import Counter
from fractions import Fraction
import math


MODEL_CONFIG = {
	"api_key": "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",
	"LM": "gpt-3.5-turbo",
	"temperature": 0.5,
	"dataset": "data/GSM8K_test.jsonl",
	"output_path": "data/predictions.jsonl",
	"prompt_path": "text/prompt.txt",
	"template_path": "text/template.txt",
	"prefix_path": "text/prefix.txt",
}


INVALID_ANS = "[invalid]"

CODE_STOP_TOKEN = "# Q:"

CODE_MAX_TOKEN = 1000


def load_data(frn):
	'''Load data from a file.
	:param frn (str): The dataset file name.

	:return: The dataset (a list of examples, each as a dictionary).
	'''
	if frn.endswith(".jsonl"):
		with open(frn, 'r') as fr:
			lines = []
			for i, line in enumerate(fr):
				if line.strip() == "":
					continue
				try:
					lines.append(json.loads(line))
				except json.decoder.JSONDecodeError as e:
					print(f"Error in line {i}: {line}\n {e}")
					exit(-1)
		return lines
	elif frn.endswith(".csv"):
		with open(frn) as fr:
			reader = csv.DictReader(fr)
			return [line for line in reader]

def str2num(answer_str, rounding="int", abs_val=True):
	'''Convert a string to a number.
	@:param answer_str (str): The string to convert.
	@:param rounding (str): The rounding method for the answer. Can be "int", "ceil", or "floor".
	@:param abs_val (bool): Whether to take the absolute value of the answer.

	@:return The converted number.
	'''
	if "/" in answer_str:
		answer_str =  float(sum(Fraction(s) for s in answer_str.split()))
	answer_str = float(answer_str)

	if rounding == "int":
		answer_str = int(answer_str)
	elif rounding == "ceil":
		answer_str = math.ceil(answer_str)
	elif rounding == "floor":
		answer_str = math.floor(answer_str)

	if abs_val:
		answer_str = abs(answer_str)

	return answer_str

def extract_gold_answer(gold_completion):
	ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
	match = ANS_RE.search(gold_completion)
	if match:
		match_str = match.group(1).strip()
		match_str = match_str.replace(",", "")
		return int(match_str)
	else:
		return INVALID_ANS

def extract_pred_answer(pred_completion, rounding="int", abs_val=True):
	if INVALID_ANS in str(pred_completion):
		return INVALID_ANS
	
	if type(pred_completion) == int:
		pred_answer = pred_completion
	elif type(pred_completion) == str:
		ANS_RE = re.compile(r"(\-?[0-9\.\,]+)")
		match = ANS_RE.search(pred_completion)
		if match:
			match_str = match.group(1).strip()
			match_str = match_str.replace(",", "")
			try:
				pred_answer = str2num(match_str, rounding, abs_val)
			except:
				pred_answer = INVALID_ANS
		else:
			pred_answer = INVALID_ANS
	return pred_answer
