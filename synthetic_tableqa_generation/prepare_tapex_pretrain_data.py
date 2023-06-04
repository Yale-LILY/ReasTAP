import json
import re
from tqdm import tqdm
import random


def process_one_line(line):
    question = line.split("col : ")[0].strip()
    if not question:
        return None, None
    
    header = line.split("col : ")[1].split("row 1 : ")[0].strip()
    header = [i.strip() for i in header.split("|")]
    num_col = len(header)
    
    
    rows = re.split("row \d+ : ", line.split("col : ")[1].strip())
    output_rows = []
    for row in rows:
        row = row.strip()
        if row == "":
            continue
        row = row.split("|")
        row = [x.strip() for x in row]
        output_rows.append(row)
        if len(row) != num_col:
            return None, None
    
    table = {
        "header": header,
        "rows": output_rows,
    }
    
    return question, table

def process_tapex_data(src_file, tgt_file):
    src_lines = open(src_file).readlines()
    tgt_lines = open(tgt_file).readlines()
    outputs = []
    err_count = 0
    
    for i, line in enumerate(tqdm(src_lines)):
        question, table = process_one_line(line)
        if not question:
            err_count += 1
            continue
        answer = tgt_lines[i].strip().split(", ")
        
        cur_example = {
            "source": "tapex",
            "question": question,
            "table": table,
            "answers": answer,
        }
        
        outputs.append(cur_example)
    
    print("Error count: ", err_count)
    print("Total examples: ", len(outputs))
    return outputs

if __name__ == "__main__":
    src_file = "tapex_data/sql_executor.src"
    tgt_file = "tapex_data/sql_executor.tgt"
    output_file = "output/tapex_pretrain_data.json"
    output_sample_file = "output/tapex_pretrain_data_sample.json"
    
    outputs = process_tapex_data(src_file, tgt_file)
    json.dump(outputs, open(output_file, "w"), indent=4)
    json.dump(random.sample(outputs, 1000), open(output_sample_file, "w"), indent=4)