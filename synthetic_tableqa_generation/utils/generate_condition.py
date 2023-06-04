import random
from dateutil import parser
from datetime import datetime, date
from utils.parse_numeric_datetime import *
from nltk.corpus import wordnet
default_date = datetime.combine(date.today(), datetime.min.time()).replace(day=1)

def unify_answers(answer_list):
    answers = []
    for ans in answer_list:
        if ans not in answers:
            answers.append(ans)
    return answers
    
def generate_numeric_condition(column):
    templates = ["was equal to", "was greater than", "was less than", "was greater than or equal to", "was less than or equal to"]
    template = random.sample(templates, 1)[0]
    
    _, parsed_column = try_parse_column_to_numeric_column(column)
    selected_cell_idx = random.sample(range(len(column)), 1)[0]
    
    selected_cell_value = parsed_column[selected_cell_idx]
    value = column[selected_cell_idx]
    
    if template == "was equal to":
        row_ids = [i for i, x in enumerate(parsed_column) if x == selected_cell_value]
    elif template == "was greater than":
        row_ids = [i for i, x in enumerate(parsed_column) if x > selected_cell_value]
    elif template == "was less than":
        row_ids = [i for i, x in enumerate(parsed_column) if x < selected_cell_value]
    elif template == "was greater than or equal to":
        row_ids = [i for i, x in enumerate(parsed_column) if x >= selected_cell_value]
    elif template == "was less than or equal to":
        row_ids = [i for i, x in enumerate(parsed_column) if x <= selected_cell_value]
        
    return f"{template} {value}", set(row_ids)

def generate_datetime_condition(column):
    templates = ["was before", "was after", "was on"]
    template = random.sample(templates, 1)[0]
    
    try:
        parsed_column = [parser.parse(x, default=default_date) for x in column]
        selected_cell_idx = random.sample(range(len(column)), 1)[0]
        selected_cell_value = parsed_column[selected_cell_idx]
        
        if template == "was before":
            row_ids = [i for i, x in enumerate(parsed_column) if x < selected_cell_value]
        elif template == "was after":
            row_ids = [i for i, x in enumerate(parsed_column) if x > selected_cell_value]
        elif template == "was on":
            row_ids = [i for i, x in enumerate(parsed_column) if x == selected_cell_value]
        
        orig_selected_cell_value = column[selected_cell_idx]
        return f"{template} {orig_selected_cell_value}", set(row_ids)
    
    except:
        return "", set([])

def generate_string_condition(column, max_trail = 50):
    templates = ["starts with", "ends with", "contains"]
    template = random.choices(templates, weights=[0.5, 0.1, 0.4], k=1)[0]
    
    selected_cell_value = ""
    time = 0 
    while selected_cell_value.strip() == "" and time < max_trail:
        selected_cell_value = random.sample(column, 1)[0]
        time += 1
        
    if selected_cell_value.strip() == "":
        return "", set([])
    
    if template == "starts with":
        value = selected_cell_value.strip().split()[0]
        row_ids = [i for i, x in enumerate(column) if x.startswith(value)]
    elif template == "ends with":
        value = selected_cell_value.strip().split()[-1]
        row_ids = [i for i, x in enumerate(column) if x.endswith(value)]
    elif template == "contains":
        value = random.sample(selected_cell_value.strip().split(), 1)[0]
        row_ids = [i for i, x in enumerate(column) if value in x]
    
    # check if the value is an English word
    if not wordnet.synsets(value):
        return "", set([])
    
    if len(selected_cell_value.strip().split()) == 1:
        row_ids = [i for i, x in enumerate(column) if x.strip() == selected_cell_value.strip()]
        return f"is {value}", set(row_ids)
    else:
        return f"string {template} {value}", set(row_ids)

def generate_condition(column, column_type):
    if column_type == "numeric":
        return generate_numeric_condition(column)
    elif column_type == "datetime":
        return generate_datetime_condition(column)
    elif column_type == "string":
        return generate_string_condition(column)
    
def generate_joint_condition(column1, column2, type1, type2, max_trail = 50):
    '''
    return row ids that satisfy both conditions
    '''
    row_ids1, row_ids2 = set([]), set([])
    time = 0
    while time < max_trail and not row_ids1.intersection(row_ids2):
        cond1, row_ids1 = generate_condition(column1, type1)
        cond2, row_ids2 = generate_condition(column2, type2)
        time += 1
    
    if row_ids1.intersection(row_ids2):
        return cond1, cond2, sorted(list(row_ids1.intersection(row_ids2)))
    else:
        return None, None, None
    
def generate_quantify_condition(column1, column2, type1, type2, quantify_type, max_trail = 50):
    '''
    return row ids that satisfy both conditions
    '''
    row_ids1, row_ids2 = set([]), set([])
    time = 0
    while time < max_trail and not row_ids1.intersection(row_ids2):
        cond1, row_ids1 = generate_condition(column1, type1)
        cond2, row_ids2 = generate_condition(column2, type2)
        time += 1
    
    if row_ids1.intersection(row_ids2):
        if quantify_type == "only":
            return cond1, cond2, ["Yes"] if row_ids1.intersection(row_ids2) == row_ids2 else ["No"]
        elif quantify_type == "every":
            return cond1, cond2, ["Yes"] if row_ids1.intersection(row_ids2) == row_ids1 else ["No"]
        else:
            raise ValueError("quantify_type should be either 'only' or 'every'")
    else:
        return None, None, None
            
def sample_source_cols(table, source_col_count):
    '''
    sample #source_col_count source columns from table 
    '''
    target_col_idx = table.key_column_idx
    
    source_cols = []
    
    # prepare candidate source cols, exclude 1) target col, 2) column name is empty
    cand_col_ids = [i for i, col_name in enumerate(table.header) if col_name.strip() and i != target_col_idx]
    nonstring_col_ids = table.numeric_col_ids + list([int(i) for i in table.datetime_col_ids.keys()])
    nonstring_col_ids = [i for i in nonstring_col_ids if table.header[i].strip() and i in cand_col_ids]
    
    if len(cand_col_ids) < source_col_count:
        return None
    
    for _ in range(source_col_count):
        # grant higher priority to non-string columns
        if len(nonstring_col_ids) > 0 and random.random() > 0.7:
            source_col_idx = random.sample(nonstring_col_ids, 1)[0]
        else:
            source_col_idx = random.sample(cand_col_ids, 1)[0]
            
        if source_col_idx in nonstring_col_ids:
            nonstring_col_ids.remove(source_col_idx)
        cand_col_ids.remove(source_col_idx)
        source_col = table.cols[source_col_idx]
        source_col_name = table.header[source_col_idx]
        source_cols.append({
            "col_idx": source_col_idx,
            "column_name": source_col_name,
            "column_data": source_col,
            "type": table.col_type[source_col_idx],
        })
    
    return source_cols


def sample_source_cols_of_type(table, col_type, source_col_count):
    """
    sample #source_col_count source columns of type #col_type from table
    """
    target_col_idx = table.key_column_idx
    source_cols = []

    # prepare candidate source cols, exclude 1) target col, 2) column name is empty
    cand_col_ids = [i for i, col_name in enumerate(table.header) if col_name.strip() and i != target_col_idx and table.col_type[i] == col_type]

    if len(cand_col_ids) < source_col_count:
        return None

    for _ in range(source_col_count):
        source_col_idx = random.sample(cand_col_ids, 1)[0]
        cand_col_ids.remove(source_col_idx)
        flag, source_col = try_parse_column_to_numeric_column(table.cols[source_col_idx])
        if not flag:
            return None
        source_col_name = table.header[source_col_idx]
        source_cols.append({
            "col_idx": source_col_idx,
            "column_name": source_col_name,
            "column_data": source_col,
            "type": table.col_type[source_col_idx],
        })

    return source_cols