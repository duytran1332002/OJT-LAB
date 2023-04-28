
from data.Data import Data
from data.Database import Database
from models.Retrieval import RetrievalModel, WikiDataset
from torch.utils.data import DataLoader
import torch
import argparse
import tqdm
import json


def parse_command_line_arguments():

    parser = argparse.ArgumentParser(
        description='Make embeddings for the database')
    
    # Load the and preprocess the dataset
    parser.add_argument('--data_path', type=str, default="D:\OJT Training\wiki40b_en_100_0.jsonl",
                        help="Path to the dataset")
    
    parser.add_argument('--preprocess', type=bool, default=True,
                        help="Preprocess the dataset or not")
    
    # retrieval model
    parser.add_argument('--model_name', type=str, default="sentence-transformers/all-mpnet-base-v1",
                        help="name of model retrieved from huggingface")
    
    parser.add_argument('--batch_size', type=int, default=5,
                        help="batch size of the model")
    # database
    parser.add_argument('--local_host', type=str, default="localhost",
                        help="local host of the database")
    
    parser.add_argument('--name_database', type=str, default="OJT_LAB_Q2",
                        help="name of the database")
    
    parser.add_argument('--user_name', type=str, default="postgres",
                        help="user name of the database")
    
    parser.add_argument('--password', type=str, default="adidaphat",
                        help="password of the database")
    
    
    

    parsed_arguments = parser.parse_args()

    return parsed_arguments

def preprocess_data(sample):
        """Preprocesses the data by removing newlines, tabs, and multiple spaces
        Args:
            examples (dict): dictionary containing the data
            key (str, optional): key for the data. Defaults to 'passage_text'.
        Returns:
            dict: dictionary containing the preprocessed data
            """

        # lowercase the text
        sample = sample.lower()
        # remove newlines
        sample = sample.replace('\n', '')
        # remove tabs
        sample = sample.replace('\t', '')
        # remove multiple spaces
        sample = sample.replace('  ', ' ')
        return sample

if __name__ == '__main__':

    args = parse_command_line_arguments()

    dataset = Data(args.data_path)


    # combine passage with the same id
    data_dict = dataset.combine_passage_same_id(key_id="wiki_id", key_text="passage_text")
    print(f"Length of the data_dict: {len(data_dict)}")

    # preprocess data
    if args.preprocess:
        for key in tqdm.tqdm(data_dict.keys()):
            data_dict[key] = preprocess_data(data_dict[key])

    # write data_dict to json file
    with open("data_dict.json", "w") as f:
        for key in data_dict.keys():
            json.dump({key: data_dict[key]}, f)
            f.write("\n")

    del dataset
    #print 10 samples
    for i in range(5):
        print(data_dict[list(data_dict.keys())[i]])
        print("--------------------------------------------------------------------------------------")

    # create dataloader
    wiki_dataset = WikiDataset(data_dict)
    data_loader = DataLoader(wiki_dataset, batch_size=args.batch_size, shuffle=False)

    del data_dict
    del wiki_dataset

    # clear cuda
    torch.cuda.empty_cache()

    # create model
    model = RetrievalModel(args.model_name)

    print("save embedding to database")
    # create database
    database = Database(args.local_host, args.name_database, args.user_name, args.password)

    # insert data to database
    table_name = "public.wiki_snippsets"
    attribute = "wiki_id, passage_text, embedding_vector"

    for batch in list(tqdm.tqdm(data_loader)):
        batch_embedding = model.get_embedding(batch[1])
        for wiki_id, passage_text, embedding in zip(batch[0], batch[1], batch_embedding):
            database.insert_values(table_name = table_name, attribute = attribute, 
                                   values = (wiki_id, passage_text, embedding.reshape(-1).tolist()))
            

    # clear memory
    del model
    del data_loader
    del database
    torch.cuda.empty_cache()
    print("Done")
