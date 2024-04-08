import sys
import numpy as np
import os
from datasets import load_dataset
from tqdm import tqdm
import json
import re
sys.path.append(os.environ.get('PROJECTPATH'))
from src.abstract_helper import AbstractHelper


class DyckLanguagesHelper(AbstractHelper):
    def __init__(self, args):
        super(DyckLanguagesHelper, self).__init__(args)
        self.function_name = "complete_dyck_languages"
    
    def evaluate_prediction(self, outputs):
        agg_pass = []
        agg_task_accuracy = []
       
        for oi, o in enumerate(outputs):
            o = o.lower()
            if "final answer:" in o:
                model_output = o.split("final answer:")[-1].strip()
            else:
                model_output = o.strip()
            label = self.test_data[oi]['answer'].strip()
            is_pass = True
            for c in model_output.replace(" ", ""):
                if c not in ["(", ")", "{", "}", "[", "]", "<", ">"]:
                    is_pass = False
            agg_pass.append(is_pass)
            if label.replace(" ", "") == model_output.replace(" ", ""):
                agg_task_accuracy.append(True)
            else:
                agg_task_accuracy.append(False)
        
        task_accuracy = np.mean(agg_task_accuracy).item()
        pass_rate = sum(agg_pass)/len(agg_pass)
        individual_score = [{"pass_rate": agg_pass[i], "task_accuracy": agg_task_accuracy[i]} for i in range(len(agg_task_accuracy))]
        result_score = {
            "pass_rate": pass_rate,
            "task_accuracy": task_accuracy
        }
        return result_score, individual_score
    
    def load_data(self):
        data_name = "tasks/dyck_languages/data.json"
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
            cur_data['input_text'] = cur_data['input'].split("Input: ")[-1]
            cur_data['answer'] = cur_data['target']
            cur_data['label'] = cur_data['target']
            all_processed_data.append(cur_data)
        
        if dataset_split == "train":
            self.train_data = all_processed_data
        else:
            self.test_data = all_processed_data
            
        return all_processed_data
        