import random, json
import os

from utils.generate_condition import *
from utils.table_wrapper import WikiTable
from tqdm import tqdm
import warnings
import multiprocessing as mp
warnings.filterwarnings("ignore")

def generate_conjunction_question(table_data, question_template, type, source_col_count = 2):
    '''
    "question_template": "what was the [target_col] when the [source_col_1] was [cond_1] and the [source_col_2] was [cond_2]?"
    '''
    table = WikiTable(table_data)
    
    source_cols = sample_source_cols(table, source_col_count)
    if source_cols == None:
        return None, None
    
    target_col_idx = table.key_column_idx
    target_col_name = table.header[target_col_idx]
    
    cond1, cond2, answer_row_ids = generate_joint_condition(source_cols[0]["column_data"], source_cols[1]["column_data"], source_cols[0]["type"], source_cols[1]["type"])
    
    question = question_template
    if cond1 != None:
        question = question.replace("[cond_1]", cond1)
        question = question.replace("[cond_2]", cond2)
        question = question.replace("[target_col]", target_col_name)
        question = question.replace("[source_col_1]", source_cols[0]["column_name"])
        question = question.replace("[source_col_2]", source_cols[1]["column_name"])
        
        
        answers = [table.rows[i][target_col_idx] for i in answer_row_ids]
        answers = unify_answers(answers)
        return question, answers
    else:
        return None, None
    
    
    
    