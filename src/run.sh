#!/bin/bash
export PROJECTPATH="" # your path
export OPENAI_API_KEY="" # your API key

tasks=("dyck_languages" "navigate" "geometric_shapes" "reasoning_about_colored_objects" "temporal_sequences" "tracking_shuffled_objectives" "web_of_lies")

for task in "${tasks[@]}"; do
    python generate_analysis.py --task $task 
    python generate_code_prompt.py --task $task 
    python scoring_single_prompt.py --task $task --scr_model_name gpt --prompt_path tasks/$task/generated_prompts/a_code_prompt_from_explanation_planner_gpt_3shot_reproduce.txt --code_prompt
done