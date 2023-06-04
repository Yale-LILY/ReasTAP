import re
import json
class WikiTable():
    def __init__(self, table):
        self._table = table
        self.header = table["header"]
        self.rows = table["rows"]
        self.cols = [list(col) for col in zip(*self.rows)]
        
        self.col_num = len(self._table["rows"][0])
        self.row_num = len(self._table["rows"])
        self.key_column_idx = self._table["key_column"]
        self.numeric_col_ids = self._table["numeric_columns"]
        self.datetime_col_ids = self._table["date_columns"]
        
        self.col_type = self._table["column_types"]

