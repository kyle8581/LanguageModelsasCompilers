import sys, os
import argparse
import random

sys.path.append(os.environ.get('PROJECTPATH'))
from src.utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=LIST_OF_TASKS)
    parser.add_argument("--task_version", type=str)
    parser.add_argument("--planner_llm", type=str, default="gpt")
    parser.add_argument("--use_free_format", action="store_true", default=False, help="Depends on the result of the task classification")
    parser.add_argument("--reproduce", action="store_true", default=True, help="If you want to reproduce our results, use this argument.")
    parser.add_argument("--shot", type=int, default=3)
    parser.add_argument("--port", type=str, default="8000")
    return parser.parse_args()

def load_prompt(args):
    with open(f"prompts/nl_prompt_generation_from_scratch_shot{'_2' if args.use_free_format else ''}.txt", "r") as f:
        prompt = f.read()
    return prompt

def load_demonstration(task_name):
    with open(f"tasks/{task_name}/data.json", "r") as f:
        data = json.load(f)['examples']
    exemplars = "\n\n".join([sample['input'] for sample in random.sample(data, 5)])
    return exemplars

def load_example_instances_and_code_prompt(task_name, args):
    example_questions = load_demonstration(task_name)
    with open(f"tasks/{task_name}/prompts/explanation{'_2' if args.use_free_format else ''}.txt", "r") as f:
            code_prompt = f.read()
    return example_questions, code_prompt

def parsing(text):
    text = text.strip()
    if "\n\n" in text:
        text = text.split("\n\n")[0].strip()
    text = "\n".join(text.split("\n")[:10])
    return text

async def main(args):
    model_name_dict = {
        "codellama": "codellama/CodeLlama-7b-Instruct-hf",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "gpt": "gpt-3.5-turbo",
        "gpt4": "gpt-4",
        "codellama-13b": "codellama/CodeLlama-13b-Instruct-hf",
        "codellama-34b": "codellama/CodeLlama-34b-Instruct-hf",
    }
    model_name = model_name_dict[args.planner_llm]
    if 'gpt' in args.planner_llm:
        scr_llm = ChatOpenAI(
                            model_name=model_name,
                            # temperature=args.temperature,
                            temperature = 0.0,
                            max_retries=100,
                            stop=["[Example"],
                            max_tokens=3000
                        )
    else:
        scr_llm = OpenAI(
                model_name=model_name,
                temperature=0.0,
                max_retries=100,
                openai_api_key='EMPTY',
                openai_api_base=f"http://localhost:{args.port}/v1",
                stop=["[Example"],
                max_tokens= 2000
            ) 
   
    prompt = load_prompt(args)
    
    output_formats = {
        'object_counting': "A positive integer",
        'temporal_sequences': "'(A)', '(B)', '(C)', ...",
        'reasoning_about_colored_objects': "'(A)', '(B)', '(C)', ...",
        'tracking_shuffled_objectives': "'(A)', '(B)', '(C)', ...",
        'dyck_languages': "A string of closing brakets seperated with a space.",
        'web_of_lies': "'Yes' or 'No'",
        'navigate': "'Yes' or 'No'",
        'geometric_shapes': "'(A)', '(B)', '(C)', ...",
        "penguins_in_a_table": "'(A)', '(B)', '(C)', ...",
    }
    if args.use_free_format:
        example_tasks = ["penguins_in_a_table", "reasoning_about_colored_objects", "geometric_shapes", "navigate"]
    else:
        example_tasks = list(output_formats.keys())
    
    helper = helper_dict[args.task](args)
    # check if directory exists. If not, make directory.
    if not os.path.exists(f'tasks/{args.task}/generated_prompts'):
        os.makedirs(f'tasks/{args.task}/generated_prompts')
    
    if args.task in example_tasks:
        example_tasks.remove(args.task)
    sampled_example_tasks = random.sample(example_tasks, args.shot)

    exemplar = "[Example 1]\nExample task instances:\n"
    for i, task in enumerate(sampled_example_tasks):
        task_instance, code_prompt = load_example_instances_and_code_prompt(task,args)
        exemplar += task_instance + "\n\nOutput Format:\n" + output_formats[task] + "\n\nExplanation:\n" + code_prompt
        exemplar += f"\n\n[Example {i+2}]\nExample task instances:\n"
    task_instance = load_demonstration(args.task)
    exemplar += task_instance + "\n\nOutput Format:\n" + output_formats[args.task] + "\n\nExplanation:"

    # paths
    save_path = f"tasks/{args.task}/generated_prompts/a_explanation_{args.shot}shot_by_{args.planner_llm}.json"
    if args.use_free_format:
        save_path = save_path.replace(".json", "_use_free_format.json")
    
    if args.reproduce:
        with open(f"tasks/{args.task}/prompts/nl_prompt_generation_example_prompt.txt", "r") as f:
            list_of_model_inputs = [f.read()]
    else:
        list_of_model_inputs = [
        prompt.format(exemplars=exemplar, function_name=helper.function_name)
    ]
    outputs = await generate_concurrently(scr_llm, list_of_model_inputs, None)
    outputs = [o.strip() for o in outputs]
    print(outputs)


    with open(save_path, "w") as f:
        json.dump({"input": list_of_model_inputs[0].split("\n"),"output": [o for o in outputs]}, f, indent=4)
    
    with open(save_path.replace(".json", ".txt"), "w") as f:
        f.write(outputs[0])

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
