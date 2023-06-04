from utils.generate_condition import *
from utils.table_wrapper import WikiTable


def generate_counting_question(table_data, question_template, type, source_col_count=2):
    """
    "question_template": "How many [target_col] had [source_col_1] [cond_1]?"
    """
    count_map = {
        "counting_1": 1,
        "counting_2": 2
    }
    table = WikiTable(table_data)
    
    source_cols = sample_source_cols(table, count_map[type])
    if source_cols is None:
        return None, None
    target_col_idx = table.key_column_idx
    target_col_name = table.header[target_col_idx]
    
    if type == "counting_1":
        cond, ans_ids = generate_condition(source_cols[0]["column_data"], source_cols[0]["type"])
        # reduce some bias
        if not cond or (len(ans_ids) in [0, 1] and random.random() < 0.9) or (len(ans_ids) == 2 and random.random() < 0.6) or (len(ans_ids) == 3 and random.random() < 0.4):
            return None, None
        
    elif type == "counting_2":
        cond, cond2, ans_ids = generate_joint_condition(source_cols[0]["column_data"], source_cols[1]["column_data"], source_cols[0]["type"], source_cols[1]["type"])
        if not cond or (len(ans_ids) in [0, 1] and random.random() < 0.9) or (len(ans_ids) == 2 and random.random() < 0.6) or (len(ans_ids) == 3 and random.random() < 0.4):
            return None, None
        
    ans = [str(len(ans_ids))]
    question = question_template
    question = question.replace("[cond_1]", cond)
    question = question.replace("[source_col_1]", source_cols[0]["column_name"])
    question = question.replace("[target_col]", target_col_name)
    if type == "counting_2":
        question = question.replace("[cond_2]", cond2)
        question = question.replace("[source_col_2]", source_cols[1]["column_name"])
        
    return question, ans
    
        
