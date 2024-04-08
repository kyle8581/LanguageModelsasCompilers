from datasets import load_dataset

class AbstractHelper:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    
    def evaluate_prediction(self, output, data):
        return
    
    
    def load_data(self):
        train_data = [d for d in load_dataset(self.args.dataset_name)["train"]]
        test_data = [d for d in load_dataset(self.args.dataset_name)['test']]
        if self.args.num_sample > 0:
            test_data = [test_data[i] for i in range(self.args.num_sample)]

        self.train_data = train_data
        self.test_data = test_data
        return train_data, test_data
    
    def load_and_prepare_data(self, dataset_split):
        return
    