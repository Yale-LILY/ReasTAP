import json, os
from tqdm import tqdm
import re

def read_json_file(path):
    return [json.loads(line) for line in open(path, 'r')]

def process_cell(text):
    text = text.replace("\n", " ")
    text = text.replace("[[", " ")
    text = text.replace("]]", " ")
    
    pattern = r"http[s]?://\S+"
    cur_text = re.sub(pattern, "", text)
    if text != cur_text:
        text = cur_text
        text.replace("[", "")
        text.replace("]", "")
    
    return text.strip()
    
    
def process_single_table_data(table_data):
    output_table_data = {
        "id": table_data["table_id"],
        "title": table_data["table_name"],
        "url": table_data["table_url"],
        "header": [],
        "rows": [],
        "column_types": [],
        "key_column": -1,
        "numeric_columns": [],
        "date_columns": {},
    }
    
    for header in table_data["header"]:
        # skip tables that might contain complex structures
        if "&nbsp" in header["column_name"]:
            return {}
        if header["column_name"].strip() == "":
            return {}
        output_table_data["header"].append(process_cell(header["column_name"]))
    
    for row in table_data["table_rows"]:
        output_table_data["rows"].append([process_cell(cell["text"]) for cell in row])

        
    for i, header in enumerate(table_data["header"]):
        orig_metadata = header["metadata"]
        if "type" in orig_metadata:
            if orig_metadata["type"] == "float":
                output_table_data["numeric_columns"].append(i)
                output_table_data["column_types"].append("numeric")
            elif orig_metadata["type"] == "Datetime":
                if orig_metadata["parsed_values"]:
                    output_table_data["date_columns"][i] = orig_metadata["parsed_values"]
                    output_table_data["column_types"].append("datetime")
                    
        else:
            output_table_data["column_types"].append("string")
        
        if "is_key_column" in orig_metadata:
            output_table_data["key_column"] = i
    
    
    return output_table_data

def process_json_file(json_path):
    data = read_json_file(json_path)
    output_data = []
    for table_data in data:
        processed_table_data = process_single_table_data(table_data)
        if not processed_table_data:
            continue
        if processed_table_data["key_column"] == -1:
            continue
        if len(processed_table_data["rows"]) < 8 and len(processed_table_data["rows"]) > 30 and len(processed_table_data["header"]) > 10 and len(processed_table_data["header"]) < 3:
            continue
        if len(processed_table_data["numeric_columns"]) == 0 and len(processed_table_data["date_columns"]) == 0:
            continue
        
        output_data.append(processed_table_data)
    return output_data
    
        
if __name__ == "__main__":
    output_dir = "table_data"
    table_data_dir = "raw_table_data"
    os.makedirs(output_dir, exist_ok=True)
    i = 0
    num_table = 0
    for dir in tqdm(os.listdir(table_data_dir)):
        if not os.path.isdir(os.path.join(table_data_dir, dir)):
            continue
        orig_file_path = os.path.join(table_data_dir, dir, "ClassifyTableColumnsFiltered.jsonl")

        output_path = os.path.join(output_dir, f"output_{i}.json")
        output_data = process_json_file(orig_file_path)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent = 4)
        i += 1
        num_table += len(output_data)
        
    print(f"Collect {num_table} tables")