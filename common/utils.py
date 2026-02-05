"""
Common utility functions for data processing tasks.
This module contains reusable functions for CSV/JSON conversion, merging, and data separation.
"""

import pandas as pd
import json
import os
from typing import List, Dict, Any, Optional


def get_base_dir() -> str:
    """
    Get the base directory of the project (parent of common folder).
    
    Returns:
        str: Absolute path to the base directory
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def csv_to_json(input_csv: str, output_json: str, encoding: str = "utf-8-sig") -> int:
    """
    Convert a CSV file to JSON format.
    
    Args:
        input_csv: Path to input CSV file
        output_json: Path to output JSON file
        encoding: Encoding format for reading CSV (default: "utf-8-sig")
    
    Returns:
        int: Number of records converted
    """
    # Read CSV
    df = pd.read_csv(input_csv, encoding=encoding)
    
    # Convert DataFrame to list of dictionaries
    records = df.to_dict(orient="records")
    
    # Write JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    
    print(f"Done! {len(records)} records")
    print(f"Saved to {output_json}")
    
    return len(records)


def merge_json_files(
    input_dir: str,
    output_file: str,
    exclude_files: Optional[List[str]] = None
) -> int:
    """
    Merge multiple JSON files from a directory into a single JSON file.
    
    Args:
        input_dir: Directory containing JSON files to merge
        output_file: Path to output merged JSON file
        exclude_files: List of filenames to exclude from merging (default: output filename)
    
    Returns:
        int: Total number of records merged
    """
    merged = []
    
    # Default exclude list includes the output filename
    if exclude_files is None:
        exclude_files = [os.path.basename(output_file)]
    elif os.path.basename(output_file) not in exclude_files:
        exclude_files.append(os.path.basename(output_file))
    
    # Iterate through all JSON files in the directory
    for fname in os.listdir(input_dir):
        if fname.endswith(".json") and fname not in exclude_files:
            print(f"Processing: {fname}")
            file_path = os.path.join(input_dir, fname)
            
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Extend merged list if data is a list
                if isinstance(data, list):
                    merged.extend(data)
                else:
                    print(f"Warning: {fname} does not contain a list, skipping...")
    
    # Write merged data to output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    print(f"Tổng records: {len(merged)}")
    print(f"Saved to {output_file}")
    
    return len(merged)


def separate_by_field(
    input_file: str,
    output_dir: str,
    field_name: str = "platform",
    default_value: str = "UNKNOWN"
) -> Dict[str, int]:
    """
    Separate a JSON file into multiple files based on a field value.
    
    Args:
        input_file: Path to input JSON file
        output_dir: Directory to save separated files
        field_name: Field name to use for separation (default: "platform")
        default_value: Default value if field is missing (default: "UNKNOWN")
    
    Returns:
        Dict[str, int]: Dictionary mapping field values to record counts
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load input data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Group data by field value
    field_groups = {}
    
    for item in data:
        field_value = item.get(field_name, default_value)
        field_groups.setdefault(field_value, []).append(item)
    
    # Write each group to a separate file
    result_counts = {}
    
    for field_value, items in field_groups.items():
        # Create filename from field value (lowercase, replace spaces with underscores)
        filename = f"{field_value.lower().replace(' ', '_')}.json"
        file_path = os.path.join(output_dir, filename)
        
        # Write to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        
        result_counts[field_value] = len(items)
        print(f"{field_value}: {len(items)} records → {file_path}")
    
    return result_counts


def load_json(file_path: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Any: Loaded JSON data
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to output JSON file
        indent: Indentation level for pretty printing (default: 2)
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    
    print(f"Saved to {file_path}")


def ensure_dir(directory: str) -> None:
    """
    Ensure a directory exists, create it if it doesn't.
    
    Args:
        directory: Path to directory
    """
    os.makedirs(directory, exist_ok=True)
