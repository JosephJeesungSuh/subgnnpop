# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import ast
import json
from multiprocessing import Lock

import datasets
import pandas as pd

def get_preprocessed_opinionqa_ce_or_wd_loss(
    dataset_config, tokenizer, split, chat_template, save = True,
):

    def tokenize_add_label(sample, chat_template=False):

        if not chat_template: # using pretrained base model
            prompt = tokenizer.encode(
                tokenizer.bos_token + sample["prompt"],
                add_special_tokens=False
            )
            answer: int = tokenizer.encode(
                "Answer: " + sample["label"],
                add_special_tokens=False
            )[-1]
        else:
            print("Something is wrong!!!!!!!!!!!!!!!!!!")
            raise NotImplementedError()
                     
        sample = {
            "input_ids": prompt,
            "attention_mask" : [1] * len(prompt),
            "target_token_id": answer,
            }
        return sample
        
    preprocessed_file_dir = (
        split.split(".jsonl")[0]
        + "_" + tokenizer.name_or_path.split("/")[-1]
        + "_preprocessed.json"
    ) # detail: preprocessing file is dependent on the tokenizer used

    if os.path.exists(preprocessed_file_dir): # if preprocessed file exists
        with open(preprocessed_file_dir, 'r', encoding="utf-8") as f:
            print("preprocessed file exists.")
            content = f.read().strip()
            dataset_dict = json.loads(content)
            dataset = datasets.Dataset.from_dict(dataset_dict)
    else: # if preprocessed file does not exist, preprocess the dataset
        dataset = datasets.load_dataset(
            'json', 
            data_files = split
        )['train']  # detail: not sure why, 
                    # but getting DatasetDict with 'train' key every time
        dataset = dataset.map(
            tokenize_add_label,
            remove_columns=list(dataset.features),
            num_proc=32
        )

        if save:
            # save dataset to json format
            dataset_dict = dataset.to_dict()
            with Lock():
                with open(preprocessed_file_dir, 'w', encoding='utf-8') as f:
                    json.dump(dataset_dict, f, indent=4)
                    f.flush()

    return dataset