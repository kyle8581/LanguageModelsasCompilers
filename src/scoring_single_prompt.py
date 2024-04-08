import argparse
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from tqdm import tqdm
import asyncio
import json
import random
import copy
import sys
import os

sys.path.append(os.environ.get('PROJECTPATH'))
from src.utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scr_model_name", type=str, default="codellama")
    parser.add_argument("--score_type", type=str, default="pass_rate")
    parser.add_argument("--code_prompt", action="store_true", help="Enable code prompt if flag is present")
    parser.add_argument("--num_sample", type=int, default=-1)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt_path", type=str)
    parser.add_argument("--compositional_inference", action="store_true")
    parser.add_argument("--inst_token", action="store_true")
    parser.add_argument("--use_extraction", action="store_true")
    parser.add_argument("--nl_type", type=str, default="direct", choices=['direct', 'cot', 'pns'])
    parser.add_argument("-r", "--retry_threshold", type=int, default=10 )
    parser.add_argument("--task", type=str, required=True)
    return parser.parse_args()

def load_prompt(args):
    with open(args.prompt_path, "r") as f:
        prompt = f.read() 
    return prompt

def load_templates(args):
    prompt_type = "code" if args.code_prompt else "nl"
    if args.code_prompt:
        with open(f"tasks/{args.task}/scoring_prompt_template_{prompt_type}.txt", "r") as f:
            scoring_prompt = f.read()
    else:
        with open(f"tasks/{args.task}/scoring_prompt_template_{prompt_type}_{args.nl_type}.txt", "r") as f:
            scoring_prompt = f.read()
    return scoring_prompt


async def calculate_score_for_optimized_prompt(llm, data, scoring_prompt, optimized_prompt, helper):
    ''' evaluate an optimized instruction using scorer model '''
    # construct model inputs using instances in evaluation set
    list_of_model_inputs = [scoring_prompt.format(input_text=d['input_text'], prompt=optimized_prompt, function_name=helper.function_name) for d in data]
    
    outputs = await generate_concurrently(llm, list_of_model_inputs, args)
    if helper.args.use_extraction:
        list_of_model_inputs = [mi + "\n" + output[:output.rfind("Final answer")].strip() + "\n" + "Therefore, the answer is " if not args.code_prompt else mi + "\n" + output[:output.rfind("Final answer")].strip() + "\n" + "Final answer:" for mi, output in zip(list_of_model_inputs, outputs)]
        progress_tracker = {'completed': 0, 'dynamic_threshold': int(len(list_of_model_inputs) * 0.8)}
        tasks = [
            async_generate(llm, i, mi, args, progress_tracker) for i, mi in enumerate(list_of_model_inputs)
        ]
        results = []
        for f in tqdm_async(asyncio.as_completed(tasks), total=len(tasks)):
            result = await f
            results.append(result)

        sorted_results = sorted(results, key=lambda x: x[0])
        outputs = [output for index, output in sorted_results]
    result_score, individual_score = helper.evaluate_prediction(outputs)

    return result_score, individual_score, outputs, list_of_model_inputs

################################################################
# MAIN FUNCTION
################################################################
async def main(args):

    model_name_dict = {
        "codellama": "codellama/CodeLlama-7b-Instruct-hf",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "gpt": "gpt-3.5-turbo",
        "gpt4": "gpt-4",
        "codellama-13b": "codellama/CodeLlama-13b-Instruct-hf",
        "llama-13b": "meta-llama/Llama-2-13b-hf",
        "codellama-34b": "codellama/CodeLlama-34b-Instruct-hf",
    }
    model_name = model_name_dict[args.scr_model_name]
    if 'gpt' in args.scr_model_name:
        scr_llm = ChatOpenAI(
                            model_name=model_name,
                            # temperature=args.temperature,
                            temperature = 0.0,
                            max_retries=100,
                            max_tokens=1500,
                        )
    else:
        if args.compositional_inference:
            num_tokens = 200
        elif args.use_extraction:
            num_tokens = 500
        else:
            num_tokens = 1500
        scr_llm = OpenAI(
                model_name=model_name,
                temperature=0.0,
                max_retries=100,
                openai_api_key='EMPTY',
                openai_api_base=f"http://localhost:{args.port}/v1",
                max_tokens= num_tokens,
                stop = ["\n"] if args.compositional_inference else None
            ) 

    task_helper = helper_dict[args.task](args)
    # load data and templates
    _, test_data = task_helper.load_data()
    _, test_data = task_helper.load_and_prepare_data("train"), task_helper.load_and_prepare_data("test")
    best_prompt = load_prompt(args)

    # load template for meta prompt
    scoring_prompt = load_templates(args)
    if "cp_wo_comment_semantics" in args.prompt_path:
        scoring_prompt = scoring_prompt.replace(task_helper.function_name, "function1")
    if args.inst_token:
        scoring_prompt = "[INST]\n" + scoring_prompt + "\n[/INST]"

    # check if directory exists. If not, make directory.
    if not os.path.exists(f'tasks/{args.task}/results'):
        os.makedirs(f'tasks/{args.task}/results')

    # paths
    # save_path = f"{args.task}/results/"+f"{args.scr_model_name.replace('/', '-')}_"+args.prompt_path.split("/")[-1].replace(".txt","")+f"_sample{args.num_sample}.json"
    save_path = f"tasks/{args.task}/results/{args.prompt_path.split('/')[-1].split('.')[0]}_{args.scr_model_name}{'_compositional' if args.compositional_inference else ''}{'_inst_token' if args.inst_token else ''}{'_extraction' if args.use_extraction else ''}" + ".json"
    if not args.code_prompt:
        save_path = save_path.replace(".json", f"_{args.nl_type}.json")
    print(save_path)
    print(args.compositional_inference)
    # evaluate newly generated instructions using the scorer model
    score_dic, individual_score, raw_prediction, list_of_model_inputs = await calculate_score_for_optimized_prompt(scr_llm, test_data, scoring_prompt, best_prompt, task_helper)
    output = dict()
    output['prompt'] = best_prompt.split("\n")
    output['score'] = score_dic
    output['inference'] = [{"input": list_of_model_inputs[i].split("\n"), "output": raw_prediction[i].strip().split("\n"), "score": individual_score[i]} for i in range(len(raw_prediction))] 
    with open(save_path, "w") as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
