from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import faiss
import torch

class DPRModel:
    def __init__(self, model_name, device="cpu"):
        self.tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
        self.model = DPRContextEncoder.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)
        self.device = device

    def encode_passages(self, passages):
        """Encodes the passages into embeddings
        Args:
            passages (list): list of passages
        Returns:
            passage_embeddings (list): list of passage embeddings
        """
        passage_encodings = self.tokenizer(passages, truncation=True, padding=True, return_tensors="pt")
        passage_encodings.to(self.device)
        passage_embeddings = self.model(**passage_encodings).pooler_output.detach().cpu().numpy()
        return passage_embeddings

    def search(self, query, passages, k=3):
        """Searches for the most similar passages to the query
        Args:
            query (str): query
            passages (list): list of passages
            k (int): number of passages to return
        Returns:
            results (list): list of the most similar passages
        """

        query_encoding = self.tokenizer(query, truncation=True, padding=True, return_tensors="pt")
        query_encoding.to(self.device)
        passage_embeddings = self.encode_passages(passages)

        # Create an index for the passage embeddings
        index = faiss.IndexFlatIP(passage_embeddings.shape[1])
        index.add(passage_embeddings)

        # Search for the most similar passages to the query
        query_embedding = self.model(**query_encoding).pooler_output.detach().cpu().numpy()
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = index.search(query_embedding, k)

        # Return the most similar passages
        results = []
        for i in range(k):
            result = {'passage': passages[indices[0][i]], 'score': distances[0][i]}
            results.append(result)
        return results

        
