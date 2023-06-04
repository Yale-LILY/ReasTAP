from utils.generate_condition import *
from utils.table_wrapper import WikiTable
from utils.parse_numeric_datetime import *
import random

default_date = datetime.combine(date.today(), datetime.min.time()).replace(day=1)

def generate_temporal_comparison_question(table_data, question_template, type, source_col_count=1):
    """
    question_template: "Which [target_col], with [source_col_1] [cond_1], happened the [ordinal] according to [source_col_2]?"
    """
    table = WikiTable(table_data)
    source_cols = sample_source_cols(table, source_col_count)
    if source_cols is None:
        return None, None
    
    target_col_idx = table.key_column_idx
    target_col_name = table.header[target_col_idx]

    cond, answer_row_ids = generate_condition(source_cols[0]["column_data"], source_cols[0]["type"])
    # at least has three candidate rows
    if not answer_row_ids or len(answer_row_ids) < 3:
        return None, None
    
    # process ordinal
    datetime_col_ids = list([int(i) for i in table.datetime_col_ids.keys()])
    if target_col_idx in datetime_col_ids:
        datetime_col_ids.remove(target_col_idx)
    
    if not datetime_col_ids:
        return None, None
    
    datetime_col_idx = random.sample(datetime_col_ids, 1)[0]
    dates = []
    for index in answer_row_ids:
        dates.append(
            (index, parse_datetime_cell(table.cols[datetime_col_idx][index]))
        )
    dates.sort(key=lambda x: x[1])
    
    ordinal_idx = random.randint(1, len(dates))
    ordinal = str(ordinal_idx) + {1: 'st', 2: 'nd', 3: 'rd'}.get(ordinal_idx if ordinal_idx < 10 else ordinal_idx % 10, "th")
    
    # consider multiple answers
    ordinal_date = dates[ordinal_idx - 1][1]
    ordinal_row_ids = []
    for date in dates:
        if date[1] == ordinal_date:
            ordinal_row_ids.append(date[0])
    
    ordinal_row_ids = sorted(ordinal_row_ids)

    answers = [table.cols[table.key_column_idx][i] for i in ordinal_row_ids]
    if len(answers) > 3 and random.random() > 0.2:
        return None, None

    answers = unify_answers(answers)
    
    question = question_template
    question = question.replace("[cond_1]", cond)
    question = question.replace("[target_col]", target_col_name)
    question = question.replace("[source_col_1]", source_cols[0]["column_name"])
    question = question.replace("[source_col_2]", table.header[datetime_col_idx])
    question = question.replace("[ordinal]", ordinal)
    return question, answers
