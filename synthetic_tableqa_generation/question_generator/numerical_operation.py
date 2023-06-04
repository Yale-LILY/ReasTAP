from utils.generate_condition import *
from utils.table_wrapper import WikiTable
from utils.parse_numeric_datetime import *

BADCASE = None, None


def generate_numerical_operation_question(table_data, question_template, question_type, source_col_count=2):
    """
    "question_template": "What was the [operator] of [source_col_1] when the [source_col_2] [cond_2]?"
    """
    table = WikiTable(table_data)
    tmp = sample_source_cols_of_type(table, "numeric", 1)
    if not tmp:
        return BADCASE
    source_col_1 = tmp[0]
    source_col_1_idx = source_col_1["col_idx"]

    source_col_2 = sample_source_cols(table, 1)
    if not source_col_2 or source_col_2[0]["col_idx"] == source_col_1_idx:
        return BADCASE
    source_col_2 = source_col_2[0]
    
    target_col_idx = table.key_column_idx
    target_col_name = table.header[target_col_idx]
    cond, ans_ids = generate_condition(source_col_2["column_data"], source_col_2["type"])
    
    if cond is None or len(ans_ids) < 2:
        return BADCASE
    
    selected_vals = [source_col_1["column_data"][i] for i in ans_ids]

    if question_type == "average":
        ans = sum(selected_vals) / len(selected_vals)
        ans = [str(round(ans, 2))]
    elif question_type == "sum":
        ans = sum(selected_vals)
        ans = [str(ans)]
    else:
        return BADCASE
    question = question_template
    question = question.replace("[cond_2]", cond)
    question = question.replace("[operator]", question_type)
    question = question.replace("[target_col]", target_col_name)
    question = question.replace("[source_col_1]", source_col_1["column_name"])
    question = question.replace("[source_col_2]", source_col_2["column_name"])
    return question, ans
