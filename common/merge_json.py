import os
from utils import merge_json_files, get_base_dir

# Get base directory
base = get_base_dir()

# Define input directory and output file
input_dir = os.path.join(base, "data/raw")
output_file = os.path.join(base, "data/raw/merged_raw_data.json")

# Merge all JSON files in the directory
merge_json_files(input_dir, output_file)
