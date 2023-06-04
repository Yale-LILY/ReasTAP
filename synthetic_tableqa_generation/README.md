# ReasTAP Pretraining Corpus
## Run Pretraining Data Generation
After Downloading the intermidiate files from [Google Drive](https://drive.google.com/drive/folders/1YRmRibz_fVZbrb2W1ynFWS6h-uwJw0oN?usp=sharing), you can directly run the following command to generate the pretraining data.
```
bash data_generation.sh
```
We describe the details of the pretraining corpus generation pipeline as follows.

## Synthetic Table QA Generation
### Collect and Process Wikipedia Tables
Following [Turning Tables](https://github.com/oriyor/turning_tables), we collect and process Wikipedia tables from the [wiki dump](https://archive.org/details/enwiki-20220220), and classify each table column as `numeric`, `date`, or `string`. The processed tables are stored in the `raw_table_data` folder. Then we use `prepare_table_data.py` to further process the tables and store them in the `table_data` folder.

### Generate Synthetic Table QA Examples
We define 7 types of table reasoning skills, with each type corresponding to question generator in the `question_generator` folder. Following `question_template.json`, the question generator produce synthetic table QA questions for each table. 

## Pretraining Data Format
Each example contains 3 fields `source`, `reasoning_type`, `question`, `table` and `answers`, where
- `source` indicates whether the example is from the synthetic table QA generation pipeline or the SQL execution data generation pipeline (from tapex).
- `reasoning_type` indicates the type of table reasoning skills required to answer the question, if the example is from the synthetic table QA generation pipeline.
- `table` is a 2-dimensional table with a `header` and one or multiple `rows`.
- `question` is the natural language question or SQL query.
- `answers` is a list of answers or executed results.

