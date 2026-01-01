#!/usr/bin/env python3
"""
Simple Task 1: EDA and Preprocessing
"""

import pandas as pd
import os

def main():
    print("Starting Task 1...")
    
    # Check if data exists
    data_path = "data/raw/complaints.csv"
    
    if not os.path.exists(data_path):
        print("ERROR: Data file not found!")
        print(f"Please download the dataset and save it to: {data_path}")
        print("Download from: https://drive.google.com/file/d/1MMmioXFFOVMIc7GTrXNefgXM6UiHuCZ8/view?usp=sharing")
        return
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Data loaded. Shape: {df.shape}")
    
    # Basic info
    print("\n=== Basic Information ===")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Columns: {list(df.columns)[:10]}...")  # First 10 columns
    
    # Save to processed folder
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/filtered_complaints.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Task 1 completed!")
    print(f"Data saved to: {output_path}")

if __name__ == "__main__":
    main()
