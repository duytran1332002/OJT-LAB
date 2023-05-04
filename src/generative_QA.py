from models.Retrieval import RetrievalModel
from models.DPR import DPRModel
from models.Generation import GenerationModel
from data.Database import Database
import torch
import time 

if __name__ == "__main__":

    # set up all models and database
    # Load the retrieval model
    retrieval_model = RetrievalModel(device="cuda")
    # Load the DPR model
    dpr_model = DPRModel("facebook/dpr-ctx_encoder-multiset-base")

    # Load the generation model
    generation_model = GenerationModel(model_pretrain_path="result\2023-04-16 10_39_15\model\best-f1", 
                                       tokenizer_pretrain_path="result\2023-04-16 10_39_15\tokenizer\best-f1")
    # load the database
    try:
        database = Database("localhost", "OJT_LAB_Q2", "postgres", "adidaphat")
    except:
        print("Cannot connect to the database")
        exit(0)

    while True:
        print("=========================================================")
        # User input the question 
        question = input("Please input your question: ")
        while not question:
            print("Please input your question")
            question = input("Please input your question: ")

        start = time.time()
        try:
            question_embedding = retrieval_model.get_embedding(question)
        except:
            print("Cannot get the embedding of the question")
            break
        
        question_embedding = question_embedding.reshape(-1)
        
        # get all the questions and answers from the database
        db_result = database.implement_find_similar_passage(question_embedding, k = 2)

        # Get the most similar passage
        
        passages = [result[0] for result in db_result]
        # results = dpr_model.search(question, passages, k=2)
        # context = " ".join(result['passage'] for result in results)
        context = " ".join(passages)
        answer = generation_model.generate(question, context)
        end = time.time()
        print(f"==> time: {end - start}")
        print(f"==> answer: {answer}")
        print("=========================================================")
        time.sleep(1)
        # check if the user want to continue
        continue_flag = input("Do you want to continue? (y/n): ")
        if continue_flag == "n":
            break
        else:
            continue


    







    

