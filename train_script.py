import argparse
import logging
import json
from transformers import Seq2SeqTrainer, AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments
import sys
from datetime import datetime
import torch
import os
from shutil import copyfile

import json

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="google/mt5-base")
parser.add_argument("--epochs", default=4, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--max_source_length", default=320, type=int)
parser.add_argument("--max_target_length", default=64, type=int)
parser.add_argument("--eval_size", default=100, type=int)
#parser.add_argument("--fp16", default=False, action='store_true')
args = parser.parse_args()



def main():
    ############ Load dataset
    
    # Opening JSON file
    f = open('data\squad2-slo-mt-dev-short.json', encoding="utf8")
    
    # returns JSON object as 
    # a dictionary
    data = json.load(f)

    train_pairs = []
    eval_pairs = []

    query_data = data["data"]
    for record in query_data:

        q = record["question"]
        answers = record["answers"]["text"]
        if (len(answers) > 0):
            pair = (q, answers[0]) #provide the first answer
            if len(eval_pairs) < args.eval_size:
                eval_pairs.append(pair)
            else:
                train_pairs.append(pair)

    print(len(train_pairs))




    ############ Model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    save_steps = 100

    output_dir = 'output/'+'test-SLV'+'-'+args.model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print("Output dir:", output_dir)

    # Write self to path
    os.makedirs(output_dir, exist_ok=True)

    train_script_path = os.path.join(output_dir, 'train_script.py')
    copyfile(__file__, train_script_path)
    with open(train_script_path, 'a') as fOut:
        fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))

    ####

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        #bf16=True,
        per_device_train_batch_size=args.batch_size,
        evaluation_strategy="steps",
        save_steps=save_steps,
        logging_steps=10,
        eval_steps=save_steps, #logging_steps,
        warmup_steps=100,
        save_total_limit=1,
        num_train_epochs=args.epochs,
    )

    ############ Arguments

    ############ Load datasets


    print("Input:", train_pairs[0][1])
    print("Target:", train_pairs[0][0])

    print("Input:", eval_pairs[0][1])
    print("Target:", eval_pairs[0][0])


    def data_collator(examples):
        targets = [row[0] for row in examples]
        inputs = [row[1] for row in examples]
        label_pad_token_id = -100

        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=True, truncation=True, return_tensors='pt', pad_to_multiple_of=8 if training_args.fp16 else None)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=args.max_target_length, padding=True, truncation=True, pad_to_multiple_of=8 if training_args.fp16 else None)

        # replace all tokenizer.pad_token_id in the labels by -100 to ignore padding in the loss.
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else label_pad_token_id) for l in label] for label in labels["input_ids"]
        ]


        model_inputs["labels"] = torch.tensor(labels["input_ids"])
        return model_inputs

    ## Define the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_pairs,
        eval_dataset=eval_pairs,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    ### Save the model
    train_result = trainer.train()
    trainer.save_model()
    
    
if __name__ == "__main__":
    main()

# Script was called via:
#python train_hf_trainer_multilingual.py --lang russian