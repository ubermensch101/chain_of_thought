import re
import os
import time
from io import StringIO
import openai
from utils import CODE_STOP_TOKEN, CODE_MAX_TOKEN, MODEL_CONFIG

class Model():
    def __init__(self, config):
        with open(config["prompt_path"], 'r', encoding='utf-8') as f:
            self.prompt = f.read()
        with open(config["template_path"], 'r', encoding='utf-8') as f:
            self.template = f.read()
        
        self.api_key = config["api_key"]
        openai.api_key = self.api_key

        self.LM = config["LM"]
        self.temperature = config["temperature"]
    
    def apply_template(self, template: str, example: dict):
        example_in_template = template
        for field in re.findall(r"\[.*?\]", template):
            field_name = field[1:-1]
            field_name = field_name.lower()
            if field_name in example:
                example_in_template = example_in_template.replace(field, str(example[field_name]))
        return example_in_template
    
    def query(self, prompt, stop, temperature=0.0, max_tokens=1024):
        response = openai.ChatCompletion.create(
            model=self.LM,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop
        )
        choices = response["choices"]
        completion_objs = [choice.message for choice in choices]
        completions = [completion.content for completion in completion_objs]
        return completions[0]
    
    def derive_answer_from_completions(self, completion):
        try:
            answer = solve_mwp(completion)
        except Exception as e:
            print(f"Error executing completion: {completion}.\n Error: {e}")
            return "[invalid]", None

        if type(answer) == str and "invalid" in answer:
            return "[invalid]", None

        answer = self.postprocess_answer(answer)

        final_completion = completion
        answer = answer

        return answer, final_completion

    def postprocess_answer(self, answer):
        answer = str(answer).strip()
        answer = answer.split("\n")[-1]  # only get the last output
        return answer
    
    def predict(self, example_dict: dict):
        question = example_dict["question"]
        templated_example = self.apply_template(template=self.template, example=example_dict)
        prompt_and_example = f"{self.prompt}\n\n{templated_example}"
        stop_token = CODE_STOP_TOKEN

        counter = 0
        while True:
            try:
                completion = self.query(prompt=prompt_and_example,
                    stop=[stop_token],
                    max_tokens=CODE_MAX_TOKEN,
                    temperature=self.temperature
                )
                break
            except:
                if counter > 3:
                    exit()
                counter+=1
                print("Sleeping for 30 seconds")
                time.sleep(30)

        answer, final_completion = self.derive_answer_from_completions(completion=completion)

        output = {
            "answer": answer,
            "completion": final_completion
        }

        return output

def solve_mwp(completion):
	prefix_frn = MODEL_CONFIG['prefix_path']
	with open(prefix_frn, "r") as fr:
		prefix = fr.read()

	completion = completion.rstrip("#")

	code = f"{prefix}\n{completion}"

	try:
		locs = {}
		exec(code, locs, locs)
		answer = locs["answer"]
	except Exception as e:
		answer = "[invalid]"

	return answer
