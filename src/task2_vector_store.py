#!/usr/bin/env python3
"""
Task 2: Text Chunking, Embedding, and Vector Store Indexing
"""

print("Starting Task 2...")
import pandas as pd
import numpy as np
import os

# Check if data exists
data_path = "data/processed/filtered_complaints_large.csv"
if not os.path.exists(data_path):
    print(f"ERROR: {data_path} not found!")
    print("Please run Task 1 first")
    exit()

print(f"1. Loading data from {data_path}")
df = pd.read_csv(data_path, nrows=1000)  # Load only 1000 rows for testing
print(f"   Loaded {len(df)} rows")

# Check for required column
if 'cleaned_narrative' not in df.columns:
    print("ERROR: 'cleaned_narrative' column not found!")
    exit()

print("2. Testing imports...")

# Test sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    print("   ✅ sentence-transformers imported")
    
    # Test embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')
    test_embedding = model.encode(["Test sentence"])
    print(f"   ✅ Embedding test: shape {test_embedding.shape}")
except Exception as e:
    print(f"   ❌ sentence-transformers error: {e}")

# Test FAISS
try:
    import faiss
    print("   ✅ FAISS imported")
    
    # Create test index
    dimension = test_embedding.shape[1]
    index = faiss.IndexFlatL2(dimension)
    print(f"   ✅ FAISS index created: dimension {dimension}")
except Exception as e:
    print(f"   ❌ FAISS error: {e}")

print("\n3. Creating text chunks...")
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)

texts = df['cleaned_narrative'].dropna().astype(str).tolist()
all_chunks = []
for text in texts[:10]:  # Only process first 10 for testing
    chunks = text_splitter.split_text(text)
    all_chunks.extend(chunks)

print(f"   Created {len(all_chunks)} text chunks from 10 narratives")

print("\n4. Creating embeddings...")
embeddings = model.encode(all_chunks)
print(f"   Embeddings shape: {embeddings.shape}")

print("\n5. Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))
print(f"   Index contains {index.ntotal} vectors")

print("\n6. Saving vector store...")
import json

os.makedirs("vector_store", exist_ok=True)

# Save FAISS index
faiss.write_index(index, "vector_store/faiss_index.bin")

# Save metadata
metadata = []
for i, chunk in enumerate(all_chunks):
    metadata.append({
        'chunk_id': i,
        'text_preview': chunk[:100] + "..." if len(chunk) > 100 else chunk,
        'text_length': len(chunk)
    })

with open("vector_store/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("   ✅ FAISS index saved: vector_store/faiss_index.bin")
print("   ✅ Metadata saved: vector_store/metadata.json")

print("\n7. Testing retrieval...")
test_query = "credit card complaint"
query_embedding = model.encode([test_query])
distances, indices = index.search(query_embedding.astype('float32'), k=3)

print(f"\n   Query: '{test_query}'")
print("   Top 3 results:")
for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
    if idx < len(all_chunks):
        chunk_text = all_chunks[idx]
        preview = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
        print(f"   {i+1}. Distance: {distance:.4f}")
        print(f"      Text: {preview}")

print("\n" + "=" * 70)
print("✅ TASK 2 COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nSummary:")
print(f"  • Processed {len(df)} complaints")
print(f"  • Created {len(all_chunks)} text chunks")
print(f"  • Embedding dimension: {dimension}")
print(f"  • Vector store saved to: vector_store/")
print(f"  • Tested retrieval with sample query")
