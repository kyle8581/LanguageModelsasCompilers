import argparse
from datasets import load_dataset
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from tqdm import tqdm
import asyncio
import json
import numpy as np
import pandas as pd
import sys
import os
from tqdm.asyncio import tqdm as tqdm_async
sys.path.append(os.environ.get('PROJECTPATH'))
from src.tasks.geometric_shapes.helper import GeometricShapesHelper
from src.tasks.temporal_sequences.helper import TemporalSequencesHelper
from src.tasks.navigate.helper import NavigateHelper
from src.tasks.web_of_lies.helper import WebOfLiesHelper
from src.tasks.tracking_shuffled_objectives.helper import TrackingHelper
from src.tasks.dyck_languages.helper import DyckLanguagesHelper
from src.tasks.reasoning_about_colored_objects.helper import ColoredObjectsHelper


async def async_generate(llm, index, model_input, args, progress_tracker):
    while True:
        try:
            if type(llm) == OpenAI:
                response = await llm.agenerate([model_input])
                output = response.generations[0][0].text
            else:
                response_future = asyncio.ensure_future(llm.agenerate([[HumanMessage(content=model_input)]]))
                call_success = False
                time_count = 0
                while not call_success:
                    await asyncio.sleep(1)  # Check every second
                    time_count += 1
                    if response_future.done():
                        response = response_future.result()
                        call_success = True
                    elif (time_count > 30) and (progress_tracker['completed'] >= progress_tracker['dynamic_threshold']):
                        time_count = 0
                        response_future.cancel()
                        response_future = asyncio.ensure_future(llm.agenerate([[HumanMessage(content=model_input)]]))
                
                output = response.generations[0][0].text
            progress_tracker['completed'] += 1
            break
        except Exception as e:
            print(f"Exception occurred: {e}")
            response = None

    return index, output 

async def async_generate_compositional(llm, index, model_input, args, progress_tracker):
    cur_output = []
    max_iteration = 15
    iter_count = 0
    while True:
        if "Final answer" in "\n".join(cur_output):
            break
        cur_input = model_input+"\n".join(cur_output)
        while True:
            try:
                if type(llm) == OpenAI:
                    response = await llm.agenerate([cur_input])
                    output = response.generations[0][0].text
                else:
                    response_future = asyncio.ensure_future(llm.agenerate([[HumanMessage(content=cur_input)]]))
                    call_success = False
                    time_count = 0
                    while not call_success:
                        await asyncio.sleep(1)  # Check every second
                        time_count += 1
                        if response_future.done():
                            response = response_future.result()
                            call_success = True
                        elif (time_count > 30) and (progress_tracker['completed'] >= progress_tracker['dynamic_threshold']):
                            time_count = 0
                            response_future.cancel()
                            response_future = asyncio.ensure_future(llm.agenerate([[HumanMessage(content=cur_input)]]))
                    
                    output = response.generations[0][0].text
                progress_tracker['completed'] += 1
                break
            except Exception as e:
                print(f"Exception occurred: {e}")
                response = None
        iter_count += 1
        if iter_count == max_iteration:
            cur_output.append("Final answer:")
        cur_output.append(output)

    return index, "\n".join(cur_output)

async def generate_concurrently(llm, list_of_model_inputs, args):
    progress_tracker = {'completed': 0, 'dynamic_threshold': int(len(list_of_model_inputs) * 0.8)}
    if hasattr(args, 'compositional_inference') and args.compositional_inference:
        tasks = [
            async_generate_compositional(llm, i, mi, args, progress_tracker) for i, mi in enumerate(list_of_model_inputs)
        ]
    else:
        tasks = [
            async_generate(llm, i, mi, args, progress_tracker) for i, mi in enumerate(list_of_model_inputs)
        ]
    
    results = []
    for f in tqdm_async(asyncio.as_completed(tasks), total=len(tasks)):
        result = await f
        results.append(result)

    sorted_results = sorted(results, key=lambda x: x[0])
    outputs = [output for index, output in sorted_results]

    return outputs

def select_best_k_prompts(
    optimization_history,
    max_num_prompts, # number of instruction-score pairs to be used in the optimization process
    min_score_threshold,
    args
):
    """Generate the string that includes instruction-score pairs."""
    # old_instructions_and_scores_str = ""
    counted_propmts = []
    unique_prompt_score_pair = []
    for ps in optimization_history:
        if (ps['prompt'] not in counted_propmts) and (ps['score'][args.score_type]>min_score_threshold):
            unique_prompt_score_pair.append(ps)
            counted_propmts.append(ps['prompt'])

    selected_prompts = sorted(
        unique_prompt_score_pair, key=lambda x: x['score'][args.score_type]
    )[-max_num_prompts:]


    return selected_prompts


async def calculate_score_for_optimized_prompt(llm, data, scoring_prompt, optimized_prompt, helper):
    ''' evaluate an optimized instruction using scorer model '''
    # construct model inputs using instances in evaluation set
    list_of_model_inputs = [scoring_prompt.format(input_text=d['input_text'], prompt=optimized_prompt, function_name=helper.function_name) for d in data]
    
    outputs = await generate_concurrently(llm, list_of_model_inputs, helper.args)
    result_score, individual_score = helper.evaluate_prediction(outputs)

    return result_score, individual_score, outputs, list_of_model_inputs

def parse_instruction(raw_instruction):
    start_token, end_token = "<PROMPT>", "</PROMPT>"
    start_index = raw_instruction.find(start_token) + len(start_token) if start_token in raw_instruction else 0
    end_index = raw_instruction.find(end_token) if end_token in raw_instruction else len(raw_instruction)
    return raw_instruction[start_index:end_index].strip()



def save_avg_step_scores(path, save_path, args):
    with open(path, 'r') as f:
        f1 = json.load(f)
    df = pd.DataFrame(f1)
    df[args.score_type] = [score[args.score_type] for score in df['score']]
    avg_scores = df.groupby('step')[args.score_type].mean().to_dict()
    print(avg_scores)
    # avg_scores.pop(0)
    with open(save_path, 'w') as f:
        json.dump(avg_scores, f, indent=4)

LIST_OF_TASKS = [
    'temporal_sequences',
    'reasoning_about_colored_objects',
    'dyck_languages',
    'web_of_lies',
    'geometric_shapes',
    'navigate',
    'tracking_shuffled_objectives',
]

helper_dict = {
    "navigate": NavigateHelper,
    "web_of_lies": WebOfLiesHelper,
    "tracking_shuffled_objectives": TrackingHelper,
    "dyck_languages": DyckLanguagesHelper,
    "reasoning_about_colored_objects": ColoredObjectsHelper,
    "temporal_sequences": TemporalSequencesHelper,
    "geometric_shapes": GeometricShapesHelper,
}
