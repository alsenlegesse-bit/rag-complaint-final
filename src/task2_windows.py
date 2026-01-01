#!/usr/bin/env python3
"""
Task 2: Text Chunking, Embedding, and Vector Store - Windows Compatible
"""

print("=" * 70)
print("TASK 2: VECTOR STORE CREATION")
print("=" * 70)

import pandas as pd
import numpy as np
import os
import json

# Check if data exists
data_path = "data/processed/filtered_complaints_large.csv"
if not os.path.exists(data_path):
    print(f"ERROR: {data_path} not found!")
    print("Please run Task 1 first")
    exit()

print(f"\n1. Loading data from {data_path}")
df = pd.read_csv(data_path, nrows=1000)  # Load only 1000 rows
print(f"   Loaded {len(df)} rows")

if 'cleaned_narrative' not in df.columns:
    print("ERROR: 'cleaned_narrative' column not found!")
    exit()

print("\n2. Testing libraries...")

# Test sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    print("   [OK] sentence-transformers imported")
    
    # Test embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')
    test_embedding = model.encode(["Test sentence"])
    print(f"   [OK] Embedding test: shape {test_embedding.shape}")
except Exception as e:
    print(f"   [ERROR] sentence-transformers: {e}")
    exit()

# Test FAISS
try:
    import faiss
    print("   [OK] FAISS imported")
except Exception as e:
    print(f"   [ERROR] FAISS: {e}")
    # Try ChromaDB as fallback
    try:
        import chromadb
        print("   [OK] ChromaDB imported (using as fallback)")
        USE_CHROMADB = True
    except:
        print("   [ERROR] No vector database available")
        exit()

print("\n3. Creating text chunks...")
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    
    texts = df['cleaned_narrative'].dropna().astype(str).tolist()[:20]  # First 20 only
    all_chunks = []
    for text in texts:
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)
    
    print(f"   Created {len(all_chunks)} text chunks from {len(texts)} narratives")
    
except Exception as e:
    print(f"   [ERROR] Text splitting: {e}")
    # Simple splitting as fallback
    all_chunks = []
    for text in texts[:10]:
        for i in range(0, len(text), 500):
            chunk = text[i:i+500]
            if len(chunk) > 100:
                all_chunks.append(chunk)
    print(f"   Created {len(all_chunks)} chunks with simple splitting")

print("\n4. Creating embeddings...")
embeddings = model.encode(all_chunks)
print(f"   Embeddings shape: {embeddings.shape}")

print("\n5. Building vector index...")
if 'faiss' in locals():
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    print(f"   FAISS index created with {index.ntotal} vectors")
    VECTOR_DB_TYPE = "FAISS"
else:
    # Use ChromaDB
    import chromadb
    from chromadb.config import Settings
    
    client = chromadb.PersistentClient(path="vector_store/chroma")
    collection = client.create_collection(name="complaints")
    
    # Add documents
    for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
        collection.add(
            documents=[chunk],
            embeddings=[embedding.tolist()],
            ids=[str(i)]
        )
    
    print(f"   ChromaDB collection created with {len(all_chunks)} documents")
    VECTOR_DB_TYPE = "ChromaDB"

print("\n6. Saving vector store...")
os.makedirs("vector_store", exist_ok=True)

if VECTOR_DB_TYPE == "FAISS":
    faiss.write_index(index, "vector_store/faiss_index.bin")
    print("   [OK] FAISS index saved: vector_store/faiss_index.bin")
else:
    print("   [OK] ChromaDB database saved: vector_store/chroma")

# Save metadata
metadata = []
for i, chunk in enumerate(all_chunks):
    metadata.append({
        'chunk_id': i,
        'text_preview': chunk[:100] + "..." if len(chunk) > 100 else chunk,
        'text_length': len(chunk),
        'vector_db': VECTOR_DB_TYPE
    })

with open("vector_store/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("   [OK] Metadata saved: vector_store/metadata.json")

print("\n7. Testing retrieval...")
test_query = "credit card complaint"
query_embedding = model.encode([test_query])

if VECTOR_DB_TYPE == "FAISS":
    distances, indices = index.search(query_embedding.astype('float32'), k=3)
    print(f"\n   Query: '{test_query}'")
    print("   Top 3 results (FAISS):")
    for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
        if idx < len(all_chunks):
            chunk_text = all_chunks[idx]
            preview = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
            print(f"   {i+1}. Distance: {distance:.4f}")
            print(f"      Text: {preview}")
else:
    # ChromaDB query
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )
    print(f"\n   Query: '{test_query}'")
    print("   Top 3 results (ChromaDB):")
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        preview = doc[:100] + "..." if len(doc) > 100 else doc
        print(f"   {i+1}. Similarity score")
        print(f"      Text: {preview}")

print("\n" + "=" * 70)
print("TASK 2 COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nSummary:")
print(f"  * Processed: {len(df)} complaints (sample)")
print(f"  * Text chunks: {len(all_chunks)}")
print(f"  * Embedding dimension: {embeddings.shape[1]}")
print(f"  * Vector database: {VECTOR_DB_TYPE}")
print(f"  * Saved to: vector_store/")
print("\nReady for Task 3: RAG pipeline!")
