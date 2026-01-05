#!/usr/bin/env python3
"""
Process a larger sample of the data
"""

import pandas as pd
import numpy as np
import re
import os

print("Processing 50,000 row sample...")

# Read 50,000 rows
df = pd.read_csv("data/raw/complaints.csv", nrows=50000, low_memory=False)

print(f"Loaded {len(df)} rows")

# Find columns
narrative_col = None
product_col = None

for col in df.columns:
    if 'narrative' in str(col).lower():
        narrative_col = col
    if 'product' in str(col).lower() and 'sub' not in str(col).lower():
        product_col = col

print(f"Narrative column: {narrative_col}")
print(f"Product column: {product_col}")

# Basic cleaning
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s.,!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

if narrative_col:
    df['cleaned_narrative'] = df[narrative_col].apply(clean_text)

# Save
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/filtered_complaints_large.csv", index=False)

print(f"Saved to data/processed/filtered_complaints_large.csv")
print("Done!")
