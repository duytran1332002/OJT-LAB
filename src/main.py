from __future__ import print_function
from typing import List, Tuple
from tqdm import tqdm
import torch
import os
import wandb
import json
import time

from transformers import PreTrainedTokenizer, T5ForConditionalGeneration, T5Tokenizer, AdamW, set_seed
from torch.utils.data import DataLoader
import argparse
from accelerate import Accelerator
from data.MyDataset import Dataset


def parse_command_line_arguments():

    parser = argparse.ArgumentParser(
        description='CLI for training T5 T2T model')
    
    parser.add_argument('--train_data_path', type=str, default="data/ELI5.jsonl",
                        help="training data path")
    
    parser.add_argument('--val_data_path', type=str, default="data/ELI5_val.jsonl",
                        help="validation data path")
    
    parser.add_argument('--pretrain_model_path', type=str, default=None,
                        help="pretrained model path")
    
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help="tokenizer path")

    parser.add_argument('--t5_model', type=str, default="t5-base",
                        help="What type of T5 model do you want use?")
    
    parser.add_argument('--test_only', type=bool, default=False,
                        help="Do you want to test only?")

    parser.add_argument('--batch_size', type=int, default=16,
                        help='mini-batch size (default: 16)')

    parser.add_argument('--epochs', type=int, default=40,
                        help='number of training epochs (default: 40)')
    
    parser.add_argument('--optimizer', type=str, default="AdamW",
                        help='type of optimizer (3 types: AdamW, Adam, SGD)')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (Adam) (default: 1e-4)')

    parser.add_argument('--workers', type=int, default=10,
                        help='number of working units used to load the data (default: 10)')

    parser.add_argument('--device', default='cuda', type=str,
                        help='device to be used for computations (in {cpu, cuda:0, cuda:1, ...}, default: cpu)')

    parser.add_argument('--max_input_length', type=int, default=512,
                        help='Maximum lenght of input text, (default: 512, maximum admitted: 512)')

    parser.add_argument('--seed', type=int, default=7,
                        help='Seed for random initialization (default: 7)')

    parsed_arguments = parser.parse_args()

    return parsed_arguments


def evaluate(model: T5ForConditionalGeneration,  
            tokenizer: PreTrainedTokenizer,   validation_set: Dataset,  device: str, 
            max_input_length: int = 512):
    """_summary_
    Args:
        model (T5ForConditionalGeneration): _description_
        tokenizer (PreTrainedTokenizer): _description_
        validation_set (Dataset): _description_
        device (str): _description_
        batch_size (int): _description_
    """
    my_validation_dataloader = DataLoader(validation_set, batch_size=args.batch_size,
                                          num_workers=args.workers, collate_fn=lambda data: validation_set.pack_minibatch(data))

    model.eval()
    model.to(device)


    with torch.no_grad():
        model_predictions_encoded = []
        target_encoded = []
        for contexts, questions, answers in tqdm(my_validation_dataloader):
            inputs = list(map(lambda tuple: f"question:{tuple[0]}  context:{tuple[1]}", zip(
                questions, contexts)))
            answers = list(map(lambda answer: f"answer:{answer[0]}", answers))
            encoded_inputs = tokenizer(
                inputs,
                padding="longest",
                max_length=max_input_length,
                truncation=True,
                return_tensors="pt",
            )
            encoded_targets = tokenizer(
                answers,
                padding="longest",
                max_length=max_input_length,
                truncation=True,
                return_tensors="pt",
            )
            encoded_inputs, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
            encoded_targets = encoded_targets.input_ids

            encoded_inputs = encoded_inputs.to(device)
            encoded_targets = encoded_targets.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=encoded_inputs, attention_mask=attention_mask, labels=encoded_targets)
            val_loss = outputs.loss
            model_predictions = model.generate(
                input_ids=encoded_inputs, 
                attention_mask=attention_mask,
                do_sample=True,
                max_length=64,
                top_k=70,
                top_p=0.96,
                early_stopping=True)

            model_predictions_encoded.append(tokenizer.decode(model_predictions[0], skip_special_tokens=True))
            target_encoded.append(tokenizer.decode(encoded_targets[0], skip_special_tokens=True))
    f1, exact_match = validation_set.evaluate(model_predictions_encoded, target_encoded)
    print(f"\t Validation Loss = {val_loss:.2f}, Validation F1 = {f1:.2f}, EM = {exact_match:.2f}")
    return val_loss, f1, exact_match
    

def train(model: T5ForConditionalGeneration, scheduler, 
          tokenizer: PreTrainedTokenizer, optimizer: AdamW, 
          train_set: Dataset, validation_set: Dataset, 
          num_train_epochs: int, device: str, 
          batch_size: int, max_input_length: int = 512):
    """_summary_
    Args:
        model (T5ForConditionalGeneration): _description_
        tokenizer (PreTrainedTokenizer): _description_
        optimizer (AdamW): _description_
        train_set (Dataset): _description_
        validation_set (Dataset): _description_
        num_train_epochs (int): _description_
        device (str): _description_
        batch_size (int): _description_
    """
    time_start = time.time()
    # change format of time_start to be more readable
    time_start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_start))

    my_trainset_dataloader = DataLoader(train_set, batch_size=args.batch_size,
                                        num_workers=args.workers, collate_fn=lambda data: train_set.pack_minibatch(data))
    

    # set training mode on the model
    model.train()

    # model to device
    model.to(device)


    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="OJT_LAB",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        "architecture": args.t5_model,
        "dataset": args.train_data_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_input_length": args.max_input_length,
        "optimizer": args.optimizer,
        "pretrain": args.pretrain_model_path if args.pretrain_model_path else "None",
        }
    )
    # accelerate training
    accelerator = Accelerator(gradient_accumulation_steps=2)
    model, optimizer, my_trainset_dataloader = accelerator.prepare(model, optimizer, my_trainset_dataloader)

    histories = []
    f1_old: int = 0
    
    for epoch in range(num_train_epochs):
        epoch_train_loss = 0.
        for contexts,questions,answers in tqdm(my_trainset_dataloader):
            with accelerator.accumulate(model):
                inputs = list(map(lambda tuple: f"question:{tuple[0]}  context:{tuple[1]}", zip(questions,contexts)))
                answers = list(map(lambda answer: f"{answer[0]}", answers))

                encoded_inputs = tokenizer(
                                        inputs,
                                        padding="longest",
                                        max_length=max_input_length,
                                        truncation=True,
                                        return_tensors="pt",
                                    )
                encoded_targets = tokenizer(
                                        answers,
                                        padding="longest",
                                        max_length=max_input_length,
                                        truncation=True,
                                        return_tensors="pt",
                                    )

                input_ids, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
                encoded_targets = encoded_targets.input_ids

                # replace padding target token id's of the labels by -100, crossEntropy skip target label == -100
                encoded_targets[encoded_targets == tokenizer.pad_token_id] = -100

                input_ids = input_ids.to(device)
                encoded_targets = encoded_targets.to(device)
                attention_mask = attention_mask.to(device)


                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=encoded_targets)
                loss = outputs.loss
                accelerator.backward(loss)
                #loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_train_loss += loss.item() * batch_size        
        scheduler.step()
        print(f"epoch={epoch + 1}/{num_train_epochs}")
        print(f"\t Train loss = {epoch_train_loss/len(train_set):.4f}")

        # evaluate on validation set
        val_loss, f1, exact_match = evaluate(model, tokenizer, validation_set, 
                                   device, max_input_length)
        
        if f1 > f1_old :
            model.save_pretrained(f'results/{model.name_or_path}/{time_start}/model/best-f1')
            tokenizer.save_pretrained(f'results/{model.name_or_path}/{time_start}/tokenizer/best-f1')
            f1_old = f1
        if epoch+1 % 10 == 0:
            model.save_pretrained(f'results/{model.name_or_path}/{time_start}/model/checkpoint-{epoch+1}')
            tokenizer.save_pretrained(f'results/{model.name_or_path}/{time_start}/tokenizer/checkpoint-{epoch+1}')

        # log metrics to wandb
        wandb.log({"train_loss": epoch_train_loss/len(train_set), "val_loss": val_loss, "val_f1": f1, "val_em": exact_match, "learning_rate": scheduler.get_lr()[0]})
        
        # save the history of the training
        history = {"epoch": [], "train_loss": [], "val_f1": [], "val_em": [], "lr": []}
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_train_loss/len(train_set))
        history["val_f1"].append(f1)
        history["val_em"].append(exact_match)
        history["lr"].append(scheduler.get_lr()[0])
        histories.append(history)
        model.train()

    model.save_pretrained(
        f'results/{model.name_or_path}/{time_start}/model/checkpoint-{epoch+1}')
    tokenizer.save_pretrained(
        f'results/{model.name_or_path}/{time_start}/tokenizer/checkpoint-{epoch+1}')
    
    # save the history of the training
    with open(f'results/{model.name_or_path}/{time_start}/history.json', 'w') as f:
        json.dump(histories, f)
    

if __name__ == '__main__':
    args = parse_command_line_arguments()

    for k, v in args.__dict__.items():
        print(k + '=' + str(v))

    # Set seed
    set_seed(args.seed)


    # creating the model
    if args.pretrain_model_path is not None and args.tokenizer_path is not None:
        if not os.path.exists(args.pretrain_model_path):
            raise ValueError("Pretrain model path is not valid")
        if not os.path.exists(args.tokenizer_path):
            raise ValueError("Tokenizer path is not valid")
        model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_path)
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.t5_model)
        tokenizer = T5Tokenizer.from_pretrained(args.t5_model)

    
    #check validation data path is valid
    if not os.path.exists(args.val_data_path):
        raise ValueError("Validation data path is not valid")
    else:
        validation_set = Dataset(args.val_data_path, tokenizer)

    if args.test_only:
        evaluate(model = model, 
                 tokenizer = tokenizer, 
                 validation_set = validation_set, 
                 device = args.device, 
                 max_input_length = args.max_input_length)
        exit()
    
    #check training data path is valid
    if not os.path.exists(args.train_data_path):
        raise ValueError("Training data path is not valid")
    else:
        train_set = Dataset(args.train_data_path, tokenizer)


    # creating the optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Optimizer not supported")
    
    # schedule learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)    

    train(model=model,
          tokenizer=tokenizer,
          optimizer=optimizer,
          train_set=train_set,
          validation_set=validation_set,
          num_train_epochs=args.epochs, scheduler=scheduler,
          device=args.device, batch_size=args.batch_size)