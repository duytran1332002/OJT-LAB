from typing import List, Tuple
from rouge import Rouge
import json
import torch
from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer):
        """Constructor for Dataset class
        Args:
            hf_dataset (datasets.arrow_dataset.Dataset): HuggingFace Dataset
            tokenizer: HuggingFace Tokenizer
        Raises:
            Exception: if two between questions, answers and contexts have different length it will raise an exception
        """
        self.tokenizer = tokenizer
        self.questions: List[str] = []
        self.answers: List[str] = []
        self.contexts: List[str] = []

        with open(data_path, 'r') as f:
            for line in f:
                example = json.loads(line)
                self.contexts.append(example['ctxs'])
                self.questions.append(example['question'])
                self.answers.append(example['answers'])

        if len(self.questions) != len(self.answers) or len(self.questions) != len(self.contexts):
            raise Exception(
                "something wrong while building the dataset: questions, contexts and answers result in different dimensions")

        self.item_count: int = len(self.questions)

    def __len__(self):
        """Magic method over-ride for class lenght evaluation
        Returns:
            int: lenght of the object 
        """
        return self.item_count

    def __getitem__(self, index: int):
        """Magic method over-ride for class getitem method
        Args:
            index (int): index for identify question-context and answer example
        Returns:
            Tuple(str,str,str): (Context, Question, Answer)
        """
        return self.contexts[index], self.questions[index], self.answers[index]

    def pack_minibatch(self, data: List[Tuple[str, str]]):
        """Pack mini-batch function
        Args:
            data (Tuple[List[str],List[str],List[str]]): (Contexts, Questions, Answers)
        Returns:
            Tuple[List[str],List[str],List[str]]: (Contexts, Questions, Answers)
        """
        return zip(*data)

    def __exact_match_score(self, prediction, ground_truth):
        """_summary_
        Args:
            prediction (_type_): _description_
            ground_truth (_type_): _description_
        Returns:
            _type_: _description_
        """
        if len(ground_truth) == len(prediction):
            if ground_truth == prediction:
                return 1
        return 0

    def __f1_score(self, prediction_tokens, ground_truth_tokens):
        """_summary_
        Args:
            prediction (_type_): _description_
            ground_truth (_type_): _description_
        Returns:
            _type_: _description_
        """
        
        # ROUGE-L score
        rouge = Rouge()
        f1_rouge_l = rouge.get_scores(prediction_tokens, ground_truth_tokens)[0]['rouge-l']["f"]
        return f1_rouge_l

    def evaluate(self, predictions, gold_answers):
        """_summary_
        Args:
            predictions (_type_): _description_
            gold_answers (_type_): _description_
        Returns:
            _type_: _description_
        """
        f1 = exact_match = 0

        for ground_truths, prediction in tqdm(zip(gold_answers, predictions)):
            # get the f1 score for each prediction
            f1 += self.__f1_score(prediction, ground_truths)
            exact_match += self.__exact_match_score(prediction, ground_truths)
        return f1/len(predictions), exact_match/len(predictions)