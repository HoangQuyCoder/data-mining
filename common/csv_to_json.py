import pandas as pd
import json
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_csv = os.path.join(base, "data/raw/tiki_all_2026-01-07_12-26.csv")
output_json = os.path.join(base, "data/raw/tiki_all_2026-01-07_12-26.json")

# đọc csv
df = pd.read_csv(input_csv, encoding="utf-8-sig")

# chuyển DataFrame -> list dict
records = df.to_dict(orient="records")

# ghi json
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Done! {len(records)} records")
print(f"Saved to {output_json}")
