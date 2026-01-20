import csv
import sqlite3
import os
import numpy as np
import pandas as pd
import ast

DB_PATH = r'C:\Users\ers334\Documents\databases\DAS_Annotations\A25.db'
OUT_CSV = r'C:\Users\ers334\Documents\gitRepos\cv4e_dataPrep\tx_labels_Bp.csv'

tx_table = 'tx_labels'

QUERY = (
    "SELECT * "
    "FROM tx_labels "
    "WHERE label_name IN ('Bp_B', 'Bp_A');"
)

def export_query_to_csv(db_path: str, query: str, out_csv: str):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute(query)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []

        x_col = cols.index('x_m') if 'x_m' in cols else -1
        t_col = cols.index('t_s') if 't_s' in cols else -1

        cols_out = cols.copy()
        cols_out.remove('x_m')
        cols_out.remove('t_s')
        cols_out.extend(['x_min_m', 'x_max_m', 't_start_s', 't_end_s'])
    
        new_rows = []
        for row in rows:
            row = list(row)
            x_list = ast.literal_eval(row[x_col])
            t_list = ast.literal_eval(row[t_col])

            x_min = min(x_list) if x_list else None
            x_max = max(x_list) if x_list else None
            t_start = min(t_list) if t_list else None
            t_end = max(t_list) if t_list else None

            new_row = row[:]
            del new_row[x_col:t_col+1]
            new_row.extend([x_min, x_max, t_start, t_end])
            new_rows.append(new_row)        
        # actually I want to use a pandas dataframe, so make the 
        # pd.dataframe
        df = pd.DataFrame(new_rows, columns=cols_out)
        df.to_csv(out_csv, index=False)
        # with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        #     writer = csv.writer(f)
        #     if cols_out:
        #         writer.writerow(cols_out)
        #     writer.writerows(new_rows)

        return len(new_rows)
    finally:
        conn.close()

if __name__ == '__main__':
    try:
        count = export_query_to_csv(DB_PATH, QUERY, OUT_CSV)
        print(f"Wrote {count} rows to {OUT_CSV}")
    except Exception as e:
        print(f"Error: {e}")
        raise