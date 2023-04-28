from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import Dataset


# Define a custom dataset class
class WikiDataset(Dataset):
    def __init__(self, wiki_dict):
        self.wiki_dict = wiki_dict
        self.wiki_ids = list(self.wiki_dict.keys())

    def __getitem__(self, index):
        wiki_id = self.wiki_ids[index]
        passage_text = self.wiki_dict[wiki_id]
        return wiki_id, passage_text

    def __len__(self):
        return len(self.wiki_dict)

class RetrievalModel:
    def __init__(self, model_name = 'sentence-transformers/all-mpnet-base-v1'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to("cuda") if torch.cuda.is_available() else self.model.to("cpu")
        print(f"Load model from {model_name} successfully")

    #Mean Pooling - Take attention mask into account for correct averaging
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self, text, padding = True, truncation = True, return_tensors = 'pt'):
        """Returns the embedding of the text
        Args:
            text (str): text to be embedded
        Returns:
            torch.tensor: embedding of the text
        """

        encoded_input = self.tokenizer(text, padding=padding, 
                                       truncation=truncation, 
                                       return_tensors=return_tensors)
        # Move to GPU
        encoded_input.to("cuda") if torch.cuda.is_available() else encoded_input.to("cpu")

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, mean pooling
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])

        return sentence_embeddings
