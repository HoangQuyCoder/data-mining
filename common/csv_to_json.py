import os
from utils import csv_to_json, get_base_dir

# Get base directory
base = get_base_dir()

# Define input and output paths
input_csv = os.path.join(base, "data/raw/tiki_all_2026-01-07_12-26.csv")
output_json = os.path.join(base, "data/raw/tiki_all_2026-01-07_12-26.json")

# Convert CSV to JSON using utility function
csv_to_json(input_csv, output_json)
