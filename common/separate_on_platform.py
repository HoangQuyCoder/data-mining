import os
from utils import separate_by_field, get_base_dir

# Get base directory
base = get_base_dir()

# Define input file and output directory
input_file = os.path.join(base, "merged_all_data.json")
output_dir = os.path.join(base, "output_by_platform")

# Separate data by platform field
separate_by_field(input_file, output_dir, field_name="platform")
