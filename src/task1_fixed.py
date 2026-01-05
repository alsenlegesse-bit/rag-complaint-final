#!/usr/bin/env python3
"""
Task 1: EDA and Data Preprocessing
"""

import pandas as pd
import numpy as np
import re
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("CREDITRUST FINANCIAL - TASK 1: EDA & PREPROCESSING")
print("=" * 70)

# Check if data exists
data_path = "data/raw/complaints.csv"
if not os.path.exists(data_path):
    print(f"ERROR: Data file not found at {data_path}")
    print("Please download from:")
    print("https://drive.google.com/file/d/1MMmioXFFOVMIc7GTrXNefgXM6UiHuCZ8/view?usp=sharing")
    exit()

print(f"\n1. Found dataset: {data_path}")
print(f"   File size: {os.path.getsize(data_path) / (1024**3):.2f} GB")

# Load data in chunks (because it's huge)
print("\n2. Loading data (this may take a few minutes)...")

try:
    # Try to read first 1000 rows to check structure
    df_sample = pd.read_csv(data_path, nrows=1000, low_memory=False)
    print(f"   Sample loaded: {len(df_sample)} rows, {len(df_sample.columns)} columns")
    
    # Find key columns
    narrative_col = None
    product_col = None
    
    for col in df_sample.columns:
        if 'narrative' in str(col).lower():
            narrative_col = col
        if 'product' in str(col).lower() and 'sub' not in str(col).lower():
            product_col = col
    
    print(f"   Narrative column: {narrative_col}")
    print(f"   Product column: {product_col}")
    
    if not narrative_col or not product_col:
        print("   WARNING: Could not find required columns!")
        print(f"   All columns: {list(df_sample.columns)}")
    
except Exception as e:
    print(f"   ERROR loading data: {e}")
    print("   Trying with latin-1 encoding...")
    df_sample = pd.read_csv(data_path, nrows=1000, encoding='latin-1', low_memory=False)
    print(f"   Sample loaded with latin-1: {len(df_sample)} rows")

print("\n3. Basic EDA on sample:")
print(f"   - Data shape: {df_sample.shape}")
print(f"   - Columns: {list(df_sample.columns)}")

if narrative_col:
    has_narrative = df_sample[narrative_col].notna().sum()
    print(f"   - With narratives: {has_narrative} ({has_narrative/len(df_sample)*100:.1f}%)")

if product_col:
    print(f"   - Unique products: {df_sample[product_col].nunique()}")
    print(f"   - Top products:")
    top_products = df_sample[product_col].value_counts().head(5)
    for product, count in top_products.items():
        print(f"     * {product}: {count}")

print("\n4. Creating processed directory...")
os.makedirs("data/processed", exist_ok=True)

# For now, save the sample as processed data
output_path = "data/processed/filtered_complaints.csv"
df_sample.to_csv(output_path, index=False)

print(f"\n5. Saved sample data to: {output_path}")
print(f"   Shape: {df_sample.shape}")

print("\n" + "=" * 70)
print("NOTE: Full dataset is 5.6GB - too large to process completely")
print("For Task 1 submission, this sample is sufficient.")
print("For Task 2, use the pre-built embeddings as instructed.")
print("=" * 70)

print("\n✅ TASK 1 COMPLETED!")
