import csv
import sqlite3
import os

DB_PATH = r'C:\Users\ers334\Documents\databases\DAS_Annotations\A25.db'
# Join fx_labels to tx_labels to include the source_file field from tx_labels


QUERY = (
	"SELECT fx.*, tx.source_file "
	"FROM fx_labels AS fx "
	"LEFT JOIN tx_labels AS tx ON fx.tx_id = tx.id "
	"WHERE fx.label_name IN ('Bp_B', 'Bp_A');"
)
OUT_CSV = 'fx_labels_Bp.csv'

def export_query_to_csv(db_path: str, query: str, out_csv: str):
	if not os.path.exists(db_path):
		raise FileNotFoundError(f"Database not found: {db_path}")

	conn = sqlite3.connect(db_path)
	cur = conn.cursor()
	try:
		cur.execute(query)
		rows = cur.fetchall()
		cols = [d[0] for d in cur.description] if cur.description else []

		with open(out_csv, 'w', newline='', encoding='utf-8') as f:
			writer = csv.writer(f)
			if cols:
				writer.writerow(cols)
			writer.writerows(rows)

		return len(rows)
	finally:
		conn.close()


if __name__ == '__main__':
	try:
		count = export_query_to_csv(DB_PATH, QUERY, OUT_CSV)
		print(f"Wrote {count} rows to {OUT_CSV}")
	except Exception as e:
		print(f"Error: {e}")
		raise