from utils.generate_condition import *
from utils.table_wrapper import WikiTable
BADCASE = None, None

def parse_date(date_str):
    if type(date_str) == str:
        if date_str.count('T'):
            time = date_str[:date_str.index('T')].split('-')
            return tuple(map(int, time))
    # only year
    else:
        return int(date_str), None, None


def generate_date_difference_question(table_data, question_template, question_type, source_col_count=1):
    """
    "question_template": "How many years had passed between when the [source_col_1] was [val_1] and when the [source_col_1] was [val_2]?"
    """
    table = WikiTable(table_data)
    if not table.datetime_col_ids.keys():
        return BADCASE
    source_cols = sample_source_cols(table, source_col_count)
    if source_cols is None:
        return BADCASE
    
    row_ids = random.sample(range(table.row_num), 2)
    
    time_col_ids = list([int(i) for i in table.datetime_col_ids.keys()])
    time_col_id = random.sample(time_col_ids, 1)[0]
    time_col = table.datetime_col_ids[str(time_col_id)]
    if time_col_id == source_cols[0]["col_idx"]:
        return BADCASE
    
    source_col = source_cols[0]["column_data"]
    val1, val2 = source_col[row_ids[0]], source_col[row_ids[1]]
    
    # avoid duplicate values
    if source_col.count(val1) > 1 or source_col.count(val2) > 1:
        return BADCASE
    
    datetime1, datetime2 = time_col[row_ids[0]], time_col[row_ids[1]]
    difference = None
    y1, m1, d1 = parse_date(datetime1)
    y2, m2, d2 = parse_date(datetime2)
    
    if not y1 or not y2:
        return BADCASE
    if not m1 or not m2:
        difference = abs(y1 - y2)
        if difference == 0 and random.random() < 0.7:
            return BADCASE
        date_str = "years" 
    elif m1 and m2:
        date_str = "months"
        if y1 < y2 or (y1 == y2 and m1 < m2):
            y1, m1, val1, y2, m2, val2 = y2, m2, val2, y1, m1, val1 
        year_diff = y1- y2
        month_diff = m1 - m2
        difference = year_diff * 12 + month_diff
        if difference == 0 and random.random() < 0.7:
            return BADCASE
    else:
        return BADCASE

    question = question_template
    question = question.replace("[source_col_1]", source_cols[0]["column_name"])
    question = question.replace("[val_1]", val1)
    question = question.replace("[val_2]", val2)
    question = question.replace("[date]", date_str)
    answer = [str(difference)]
    return question, answer
