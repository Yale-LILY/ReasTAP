import random, json
import os

from utils.generate_condition import *
from utils.table_wrapper import WikiTable
from tqdm import tqdm
import warnings
import multiprocessing as mp
warnings.filterwarnings("ignore")

    
def generate_quantifier_question(table_data, question_template, operator, source_col_count = 2):
    '''
    "question_template": "Does [operator] [target_col], with [source_col_1] [cond_1], have [source_col_2] [cond_2]?"
    '''
    table = WikiTable(table_data)
    
    source_cols = sample_source_cols(table, source_col_count)
    if source_cols == None:
        return None, None
    
    target_col_idx = table.key_column_idx
    target_col_name = table.header[target_col_idx]
    
    cond1, cond2, answer = generate_quantify_condition(source_cols[0]["column_data"], source_cols[1]["column_data"], source_cols[0]["type"], source_cols[1]["type"], operator)
    
    
    question = question_template
    if cond1 != None:
        question = question.replace("[cond_1]", cond1)
        question = question.replace("[cond_2]", cond2)
        question = question.replace("[operator]", operator)
        question = question.replace("[target_col]", target_col_name)
        question = question.replace("[source_col_1]", source_cols[0]["column_name"])
        question = question.replace("[source_col_2]", source_cols[1]["column_name"])
        return question, answer
    else:
        return None, None