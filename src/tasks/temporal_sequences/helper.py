import sys
import numpy as np
import os
from datasets import load_dataset
from tqdm import tqdm
import json
import re
sys.path.append(os.environ.get('PROJECTPATH'))
from src.abstract_helper import AbstractHelper


class TemporalSequencesHelper(AbstractHelper):
    def __init__(self, args):
        super(TemporalSequencesHelper, self).__init__(args)
        self.function_name = "solve_temporal_sequences_quiz"
    
    def evaluate_prediction(self, outputs):
        agg_pass = []
        agg_task_accuracy = []
        predicted_answer = []
       
        for oi, o in enumerate(outputs):
            o = o.replace("'", "").replace('"', "").replace(".", "").replace(",", "")
            if "Final answer:" in o:
                model_output = o.split("Final answer:")[-1].strip()
            else:
                model_output = o.strip()
            predicted_answer.append(model_output)
            label = self.test_data[oi]['answer'].strip()
            dic = {option.split()[0]:option for option in self.test_data[oi]['options']}
            is_pass = True
            for option in self.test_data[oi]['options']:
                if model_output in option and model_output:
                    is_pass = True
                    break
            agg_pass.append(is_pass)
            if model_output in dic[label] and model_output:
                agg_task_accuracy.append(True)
            else:
                agg_task_accuracy.append(False)
        
        task_accuracy = np.mean(agg_task_accuracy).item()
        pass_rate = sum(agg_pass)/len(agg_pass)
        individual_score = [{"pass_rate": agg_pass[i], "task_accuracy": agg_task_accuracy[i], "answer": self.test_data[i]['answer'], "prediction": predicted_answer[i]} for i in range(len(agg_task_accuracy))]
        result_score = {
            "pass_rate": pass_rate,
            "task_accuracy": task_accuracy
        }
        return result_score, individual_score
    
    def load_data(self):
        data_name = "tasks/temporal_sequences/data.json"
        with open(data_name, "r") as f:
            data = json.load(f)['examples']
        train_data = [d for d in data]
        test_data = [d for d in data]
        if self.args.num_sample != -1:
            test_data = [test_data[i] for i in range(self.args.num_sample)]

        self.train_data = train_data
        self.test_data = test_data
        return train_data, test_data
    
    def load_and_prepare_data(self, dataset_split):
        dataset = self.train_data if dataset_split == "train" else self.test_data
        all_processed_data = []
        for data in tqdm(dataset):
            cur_data = {k:v for k,v in data.items()}
            cur_data['input_text'] = cur_data['input'].split("We know that:")[-1].strip()
            cur_data['answer'] = cur_data['target']
            cur_data['label'] = cur_data['target']
            cur_data['options'] = cur_data['input'].split("Options:")[-1].strip().split("\n")
            all_processed_data.append(cur_data)
        
        if dataset_split == "train":
            self.train_data = all_processed_data
        else:
            self.test_data = all_processed_data
            
        return all_processed_data
        