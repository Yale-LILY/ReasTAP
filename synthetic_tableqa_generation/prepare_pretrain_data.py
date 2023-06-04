import os
import json
import random
from tqdm import tqdm
import copy
import argparse

import datasets
from transformers import (
    AutoConfig,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    BartTokenizer,
    set_seed,
)

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")


def prepare_sythetic_qa_data(json_file):
    total_data = []
    error_count = 0
    empty_answer_count = 0
    
    src_data = json.load(open(json_file, "r"))
    for example in tqdm(src_data):
        header = copy.deepcopy(example["header"])
        rows = copy.deepcopy(example["rows"])
        # random shulle the order of the rows and columns
        if random.random() < 0.5:
            col_order = list(range(len(header)))
            random.shuffle(col_order)
            
            header = [header[i] for i in col_order]
            rows = [[row[i] for i in col_order] for row in rows]
            random.shuffle(rows)
        
        table_content = {
            "header": header,
            "rows": rows
        }
        
        for qa in example["qas"]:
            question = qa["question"].replace("\n", " ")
            answer = [i.strip() for i in qa["answers"] if i.strip()]

            if len(answer) == 0:
                empty_answer_count += 1
                continue
            total_data.append({
                "source": "synthetic_qa",
                "reasoning_type": qa["reasoning_type"],
                "question": question,
                "table": table_content,
                "answers": answer,
            })

    print("Error count for ReasTAP Synthetic QA data: ", error_count)
    print("Total examples for ReasTAP Synthetic QA data: ", len(total_data))
    print("Empty answer count for ReasTAP Synthetic QA data: ", empty_answer_count)
    return total_data

def prepare_tapex_sql_data(json_file):
    data = json.load(open(json_file, "r"))
    print("Total examples for TAPEX SQL data: ", len(data))
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output/pretrain_data")
    parser.add_argument("--qa_file", type=str, default="output/synthetic_qa_output.json")
    parser.add_argument("--sql_file", type=str, default="output/tapex_pretrain_data.json")
    
    args = parser.parse_args()
    
    synthetic_qa_data = prepare_sythetic_qa_data(args.qa_file)
    sql_data = prepare_tapex_sql_data(args.sql_file)
    total_data = synthetic_qa_data + sql_data
    
    filtered_total_data = []
    for example in tqdm(total_data):
        answer_str = ", ".join(example["answers"])
        answers = tokenizer(answer_str, return_tensors="pt")
        if answers["input_ids"].shape[1] < 128:
            filtered_total_data.append(example)
    
    os.makedirs(args.output_dir, exist_ok=True)
    random.shuffle(filtered_total_data)
    
    train_data = filtered_total_data[:3980000]
    dev_data = filtered_total_data[-20000:]
    
    # save to jsonl
    for split, data in zip(["train", "dev"], [train_data, dev_data]):
        output_file = os.path.join(args.output_dir, f"{split}.jsonl")
        with open(output_file, "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")
    
    print("Total examples for pretrain data: ", len(train_data + dev_data))
    
    
    