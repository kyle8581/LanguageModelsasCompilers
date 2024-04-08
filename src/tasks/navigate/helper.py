import sys
import numpy as np
import os
from datasets import load_dataset
from tqdm import tqdm
import json
import re
sys.path.append(os.environ.get('PROJECTPATH'))
from src.abstract_helper import AbstractHelper


class NavigateHelper(AbstractHelper):
    def __init__(self, args):
        super(NavigateHelper, self).__init__(args)
        self.function_name = "ends_up_at_start"
    
    def evaluate_prediction(self, outputs):
        agg_pass = []
        agg_task_accuracy = []
        model_outputs = []

        for oi, o in enumerate(outputs):
            o = o.lower()
            if "answer:" in o:
                model_output = o.split("answer:")[-1].strip()
            else:
                model_output = o.strip()
            label = self.test_data[oi]['answer'].lower()
            is_pass = False
            for ans in ['yes', 'no']:
                if ans in model_output[:5]:
                    is_pass = True
                    break
            agg_pass.append(is_pass)
            if label in model_output[:5]:
                agg_task_accuracy.append(True)
            else:
                agg_task_accuracy.append(False)
            model_outputs.append(model_output)
        
        task_accuracy = np.mean(agg_task_accuracy).item()
        pass_rate = sum(agg_pass)/len(agg_pass)
        individual_score = [{"pass_rate": agg_pass[i], "task_accuracy": agg_task_accuracy[i], "prediction": model_outputs[i], "answer": self.test_data[i]['answer']} for i in range(len(agg_task_accuracy))]
        result_score = {
            "pass_rate": pass_rate,
            "task_accuracy": task_accuracy
        }
        return result_score, individual_score
    
    def load_data(self):
        data_name = "tasks/navigate/data.json"
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
            cur_data['input_text'] = cur_data['input'][92:-20] if "Always face forward. " in cur_data['input'] else cur_data['input'][71:-20]
            cur_data['answer'] = cur_data['target']
            cur_data['label'] = cur_data['target']
            all_processed_data.append(cur_data)
        
        if dataset_split == "train":
            self.train_data = all_processed_data
        else:
            self.test_data = all_processed_data
            
        return all_processed_data
        