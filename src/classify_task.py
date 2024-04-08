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
    parser.add_argument("--shot", type=int, default=3)
    parser.add_argument("--port", type=str, default="8001")
    parser.add_argument("--num_sample", type=int, default=-1)
    return parser.parse_args()

def load_prompt():
    with open("prompts/task_classification_prompt.txt", "r") as f:
        prompt = f.read()
    return prompt

def load_target_exmplar(args):
    helper = helper_dict[args.task](args)
    _, data = helper.load_data()
    data = helper.load_and_prepare_data('test')
    template = "[Example 5]\n"
    template += "Example task instances:\n"
    template += "\n\n".join([sample['input'] for sample in random.sample(data, args.shot)])
    template += "\n\nReason for the decision & answer:"
    return template

def load_exemplars(args):
    free_format_tasks = ["dyck_languages", "navigate", "tracking_shuffled_objectives"]
    variables_tracking_tasks = ["penguins_in_a_table", "reasoning_about_colored_objects", "geometric_shapes"]
    if args.task in free_format_tasks:
        free_format_tasks.remove(args.task)
    elif args.task in variables_tracking_tasks:
        variables_tracking_tasks.remove(args.task)
    
    tasks = [sample for sample in random.sample(free_format_tasks, 2)] + [sample for sample in random.sample(variables_tracking_tasks, 2)]
    random.shuffle(tasks)
    exemplars = ""
    for i, task in enumerate(tasks):
        with open(f"tasks/{task}/prompts/example_classification.txt", "r") as f:
            exemplar = f.read()
        exemplars += f"[Example {i+1}]\n" + exemplar + "\n\n"
    return exemplars

def load_final_prompt(args):
    template = load_prompt()
    exemplars = load_exemplars(args)
    target_exemplar = load_target_exmplar(args)
    return template.format(exemplar=exemplars+target_exemplar)

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
   
    prompt = load_final_prompt(args)
    
    # check if directory exists. If not, make directory.
    if not os.path.exists(f'tasks/{args.task}/generated_prompts'):
        os.makedirs(f'tasks/{args.task}/generated_prompts')

    # paths
    save_path = f"tasks/{args.task}/generated_prompts/classification_result_{args.planner_llm}_{args.shot}shot.json"
        
    list_of_model_inputs = [
        prompt
    ]
    outputs = await generate_concurrently(scr_llm, list_of_model_inputs, None)
    outputs = [output.strip() for output in outputs]
    print("\n".join(outputs))

    with open(save_path, "w") as f:
        json.dump({"input": list_of_model_inputs[0].split("\n"),"output": [o.split("\n") for o in outputs]}, f, indent=4)
    
    with open(save_path.replace(".json", ".txt"), "w") as f:
        f.write(outputs[0])

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))