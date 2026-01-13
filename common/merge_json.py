import json
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = os.path.join(base, "data")
raw = os.path.join(data, "raw")

output = os.path.join(base, "data/raw/merged_raw_data.json")

merged = []

for fname in os.listdir(raw):
    if fname.endswith(".json") and fname != output:
        with open(os.path.join(raw, fname), "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                merged.extend(data)

with open(output, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print("DONE")
print("Tổng records:", len(merged))
