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
    parser.add_argument("--port", type=str, default="8000")
    parser.add_argument("--use_free_format", action="store_true", default=False)
    parser.add_argument("--reproduce", action="store_true", default=True, help="If you want to reproduce our results, use this argument.")
    parser.add_argument("--num_sample", type=int, default=-1)
    return parser.parse_args()

def load_prompt(args):
    with open(f"prompts/code_prompt_generation_from_explanation{'_2' if args.use_free_format else ''}.txt", "r") as f:
        prompt = f.read()
    return prompt

def load_description(task):
    with open(f"../BIG-Bench-Hard/cot-prompts/{task}.txt", "r") as f:
        prompt = f.read().split("\n")[2]
    return prompt

def load_plan(task_name, args):
    with open(f"tasks/{task_name}/prompts/explanation{'_2' if args.use_free_format else ''}.txt", "r") as f:
        plan = f.read()
    return plan
    
def load_target_plan(args):
    with open(f"tasks/{args.task}/generated_prompts/a_explanation_{args.shot}shot_by_{args.planner_llm}{'_use_free_format' if args.use_free_format else ''}.txt", "r") as f:
        plan = f.read()
    return plan

def load_demonstration(task, args):
    helper = helper_dict[task](args)
    _, data = helper.load_data()
    data = helper.load_and_prepare_data('test')
    template = "input_text = \"{input_text}\"\n"
    template += "final_answer = {function_name}(input_text)\n"
    template += "print(\"Final answer:\" + final_answer)"
    exemplars = "\n\n".join([template.format(input_text=sample['input_text'], function_name=helper.function_name) for sample in random.sample(data, args.shot)])
    return exemplars

def load_example_instances_and_code_prompt(task, args):
    example_questions = load_demonstration(task, args)
    with open(f"tasks/{task}/prompts/code_prompt_cot_edited{'_2' if args.use_free_format else ''}.txt", "r") as f:
        code_prompt = f.read()
    return example_questions, code_prompt

def remove_example_usage(code_text):
    if "[Example 5]" in code_text:
        code_text = code_text.split("[Example 5]")[0].strip()
    last_def_index = code_text.rfind("def ")
    if last_def_index == -1:
        return code_text
        
    last_return_index = code_text.rfind("  return ", last_def_index)
    if last_return_index == -1:
        last_return_index = code_text.rfind("  pass", last_def_index)
    cut_index = code_text.find("\n", last_return_index)
    if cut_index == -1:
        return code_text
    return code_text[:cut_index+1].strip()


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
    output_format_dict = {
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
    helper = helper_dict[args.task](args)
    
    # check if directory exists. If not, make directory.
    if not os.path.exists(f'tasks/{args.task}/generated_prompts'):
        os.makedirs(f'tasks/{args.task}/generated_prompts')
    
    if not args.reproduce:
        example_tasks_list = list(output_format_dict.keys())
        if args.task in example_tasks_list:
            example_tasks_list.remove(args.task)
        sampled_example_tasks = random.sample(example_tasks_list, args.shot)

        exemplar = "[Example 1]\nTask description:\n"
        for i, task in enumerate(sampled_example_tasks):
            description = load_description(task)
            task_instance, code_prompt = load_example_instances_and_code_prompt(task, args)
            plan = load_plan(task, args)
            exemplar += description + "\n\nExample task instances and the code usage:\n" + task_instance + "\n\nFormat of the Final answer:\n" + output_format_dict[task] + f"\n\nExplanation:\n{plan}"+"\n\nCode prompt:\n" + code_prompt
            exemplar += f"\n\n[Example {i+2}]\nTask description:\n"
        description = load_description(args.task)
        task_instance = load_demonstration(args.task, args)
        target_plan = load_target_plan(args)
        exemplar += description + "\n\nExample task instances and the code usage:\n" + task_instance + "\n\nFormat of the Final answer:\n" + output_format_dict[task] + f"\n\nExplanation:\n{target_plan}"+"\n\nCode prompt:"

    # paths
    save_path = f"tasks/{args.task}/generated_prompts/a_code_prompt_from_explanation_planner_{args.planner_llm}_{args.shot}shot.json"
    if args.use_free_format:
        save_path = save_path.replace(".json", "_use_free_format.json")
    if args.reproduce:
        save_path = save_path.replace(".json", "_reproduce.json")

    if args.reproduce:
        with open(f"tasks/{args.task}/prompts/code_prompt_generation_example_prompt.txt", "r") as f:
            list_of_model_inputs = [f.read()]
    else:
        list_of_model_inputs = [
        prompt.format(exemplars=exemplar, function_name=helper.function_name)
    ]
    
    outputs = await generate_concurrently(scr_llm, list_of_model_inputs, None)
    outputs = [remove_example_usage(output) for output in outputs]
    print("\n".join(outputs))


    with open(save_path, "w") as f:
        json.dump({"input": list_of_model_inputs[0].split("\n"),"output": [o.split("\n") for o in outputs]}, f, indent=4)
    
    with open(save_path.replace(".json", ".txt"), "w") as f:
        f.write(outputs[0])

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
