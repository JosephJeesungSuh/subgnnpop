# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import ast
import json
from multiprocessing import Lock
import re
from typing import List, Dict

import datasets
import pandas as pd

def prompt_chat_formatter(prompt: str) -> List[Dict[str, str]]:
    """
    Reformat the QA steering prompt into the chat conversation.
    QA prompt is in Question:-Ansewer: format. Convert to user-assistant.
    """
    messages = []
    messages.append( # subpopulation steering question
        {"role": "system", "content": "Respond to the following question by choosing one of the available options, and strictly answering with the option letter (e.g., 'A', 'B', etc.). Do not provide any additional text or explanation."}
    )
    prompt = prompt.strip()
    parts = re.split(r'(Question:|Answer:)', prompt)
    user_prompts =  [parts[0] + parts[1] + parts[2] + parts[3]]
    assistant_prompts = []
    for _idx in range(4, len(parts)-1, 4):
        assistant_prompts.append(parts[_idx].strip())
        user_prompts.append(parts[_idx+1] + parts[_idx+2] + parts[_idx+3])
    assert len(user_prompts) == len(assistant_prompts) + 1
    for idx in range(len(assistant_prompts)):
        messages.append(
            {"role": "user", "content": user_prompts[idx].strip()}
        )
        messages.append(
            {"role": "assistant", "content": assistant_prompts[idx].strip()}
        )
    messages.append(
        {"role": "user", "content": user_prompts[-1].strip()}
    )
    return messages

def get_preprocessed_opinionqa_ce_or_wd_loss(
    dataset_config, tokenizer, split, chat_template, save = True,
):
    
    tokenizer.padding_side = "left"

    def _is_gpt_oss(tok):
        # Robust detection: either model path says gpt-oss, or the template includes Harmony tags.
        name = (getattr(tok, "name_or_path", "") or "").lower()
        tmpl = getattr(tok, "chat_template", "") or ""
        return ("gpt-oss" in name) or ("<|start|>" in tmpl and "<|channel|>" in tmpl and "<|message|>" in tmpl)
    _uses_harmony = _is_gpt_oss(tokenizer)
    
    chat_template = True if chat_template.lower() == 'true' else False
    print(f"--> is_chat_template: {chat_template}")

    def tokenize_add_label(sample):

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
            messages = prompt_chat_formatter(sample["prompt"])
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=True,
                add_generation_prompt=True,
            )
            if _uses_harmony:
                assistant_header = tokenizer.encode("<|channel|>final<|message|>", add_special_tokens=False)
                prompt = prompt + assistant_header
                answer: int = tokenizer.encode(
                    sample["label"], add_special_tokens=False
                )[0]
            else:
                answer: int = tokenizer.encode(
                    "Answer: " + sample["label"],
                    add_special_tokens=False
                )[-1]

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