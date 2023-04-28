from datasets import load_dataset
import tqdm
import os
import json
import collections
import torch

class Data:
    def __init__(self, path, type_of_file = "json", name_of_data='wiki_snippets', sub_name_of_data='wiki40b_en_100_0'):
        self.dataset = []
        if path is not None:
            if not os.path.exists(path):
                raise Exception('File does not exist')
            # read the file line by line
            try:
                print(f"Loaded the dataset from {path}")
                self.dataset = load_dataset(type_of_file, data_files=path)['train']
            except Exception as e:
                print(e)
        else:
            # load the dataset
            try:
                print(f"Downloaded the dataset {sub_name_of_data}")
                self.dataset = load_dataset(name_of_data, sub_name_of_data)['train']
                print(f"Saved the dataset to {path}")
                self.write_down(os.join.path('data', f'{sub_name_of_data}.json'))
            except Exception as e:
                print(e)
    
    def _get_len(self):
        """Returns the length of the dataset
        Returns:
            int: length of the dataset
        """
        return len(self.dataset)
    
    def _get_data(self):
        """Returns the dataset
        Returns:
            list: dataset
        """
        return self.dataset
    
    def write_down(self, path):
        """Writes down the dataset to a file
        Args:
            path (str): path to the file
        """
        with open(path, 'w') as f:
            for data in self.dataset:
                json.dump(data, f)
                f.write('\n')
        print('Done writing down the dataset')
    
    def print_sample(self, index):
        """Prints the sample at the given index
        Args:
            index (int): index of the sample
        """
        print(self.dataset[index])


    def combine_passage_same_id(self, key_id, key_text):
        """Combines the passages with the same wiki_id
        Returns:
            dict: dictionary containing the combined passages
        """

        dataset_dict = collections.defaultdict(list)
        for i in tqdm.tqdm(range(len(self.dataset))):
            dataset_dict[self.dataset[i][key_id]].append(self.dataset[i][key_text])

        for key in tqdm.tqdm(dataset_dict.keys()):
            dataset_dict[key] = ' '.join(dataset_dict[key])

        print('Done combining passages with the same id')

        return dataset_dict
    


