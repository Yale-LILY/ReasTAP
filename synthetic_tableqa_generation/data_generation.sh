# collect pretrain data from tapex after filtering
python prepare_tapex_pretrain_data.py;

# process table data
python prepare_table_data.py;

# generate synthetic table QA data
python generate_synthetic_tableqa_data.py;

# collect pretrain data
python prepare_pretrain_data.py;