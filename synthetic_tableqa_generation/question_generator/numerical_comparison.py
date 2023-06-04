from utils.generate_condition import *
from utils.table_wrapper import WikiTable
BADCASE = None, None

def generate_numerical_comparison_question(table_data, question_template, type, source_col_count=2):
    """
    "question_template": "Which [target_col], with [source_col_1] [cond_1], has the [ordinal] [source_col_2]?"
    """
    table = WikiTable(table_data)
    source_cols = sample_source_cols(table, source_col_count)
    if not source_cols:
        return BADCASE
    
    target_col_idx = table.key_column_idx
    target_col_name = table.header[target_col_idx]

    tmp = sample_source_cols_of_type(table, "numeric", 1)
    if tmp is None:
        return BADCASE
    else:
        source_col_2 = tmp[0]
    source_col_2_idx = source_col_2["col_idx"]
    source_col_2_data = source_col_2["column_data"]

    tmp = sample_source_cols(table, 1)
    if tmp is None or tmp[0]["col_idx"] == source_col_2_idx:
        return BADCASE
    else:
        source_col_1 = tmp[0]

    cond, ans_ids = generate_condition(source_col_1["column_data"], source_col_1["type"])
    # at least has three candidate rows
    if cond is None or len(ans_ids) < 2:
        return BADCASE
    
    col2 = []
    for idx in ans_ids:
        col2.append(
            (idx, source_col_2_data[idx])
        )
    col2.sort(key=lambda x: x[1], reverse=True)
    
    
    ordinal_idx = random.randint(1, len(col2))
    ordinal = str(ordinal_idx) + {1: 'st', 2: 'nd', 3: 'rd'}.get(ordinal_idx if ordinal_idx < 10 else ordinal_idx % 10, "th")
    
    # multiple same value
    res_ids = [row[0] for row in col2 if row[1] == col2[ordinal_idx-1][1]]
    answer_list = [table.rows[i][target_col_idx] for i in res_ids]
    answers = unify_answers(answer_list)
    
    question = question_template
    question = question.replace("[cond_1]", cond)
    question = question.replace("[target_col]", target_col_name)
    question = question.replace("[source_col_1]", source_col_1["column_name"])
    question = question.replace("[source_col_2]", source_col_2["column_name"])
    question = question.replace("[ordinal]", ordinal)
    return question, answers
