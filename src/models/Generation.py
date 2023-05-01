from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
class GenerationModel:
    def __init__(self, model_pretrain_path = "", tokenizer_pretrain_path = "", model_name = 't5-base', device = 'cpu'):
        self.model_name = model_name
        if os.path.exists(model_pretrain_path) and os.path.exists(tokenizer_pretrain_path):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_pretrain_path)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_pretrain_path, model_max_length=512)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)
        self.device = device

    def preprocess_text_train(self, text):
        # replace \n with space
        text = text.replace("\n", " ")
        # remove unicode characters
        text = text.encode('ascii', 'ignore').decode()
        # replace \t with space
        text = text.replace("\t", " ")
        #remove extra spaces
        text = " ".join(text.split())

        return text
    
    def generate(self, question, context, max_input_length=512,
                 min_output_length = 64, max_output_length=256, 
                 temperature=0.8, do_sample=True, top_k=70, top_p=0.94, 
                 repetition_penalty=1.2, no_repeat_ngram_size=10, 
                 early_stopping=True, remove_invalid_values=True):

        inputs = f"question: {question}  context: {context}"
        
        # preprocess inputs
        inputs = self.preprocess_text_train(inputs)

        encoded_inputs = self.tokenizer(
            inputs,
            padding="longest",
            max_length=max_input_length,
            truncation=True,
            return_tensors="pt",
        )
    
        encoded_inputs, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask

        encoded_inputs = encoded_inputs.to(self.device)
        attention_mask = attention_mask.to(self.device)

        model_predictions = self.model.generate(
            input_ids=encoded_inputs, 
            attention_mask=attention_mask,
            do_sample=do_sample,
            min_length=min_output_length,
            max_length=max_output_length,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            early_stopping=early_stopping,
            remove_invalid_values=remove_invalid_values)
        
        prediction = self.tokenizer.decode(model_predictions[0], skip_special_tokens=True)
        return prediction
