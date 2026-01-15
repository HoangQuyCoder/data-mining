import json
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = os.path.join(base, "data")
raw = os.path.join(data, "preliminary")

output = os.path.join(base, "data/preliminary/merged_preliminary_data.json")

merged = []

output_name = os.path.basename(output)

for fname in os.listdir(raw):
    if fname.endswith(".json") and fname != output_name:
        print(fname)
        with open(os.path.join(raw, fname), "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                merged.extend(data)


with open(output, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print("Tổng records:", len(merged))
