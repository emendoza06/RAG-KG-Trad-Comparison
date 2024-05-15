import numpy as np
import faiss

# Load embeddings from file
embeddings_array = np.load('processed-data/embeddings.npy').astype('float32')

# Dimension of embeddings
dimension = embeddings_array.shape[1]

# Initialize the FAISS index
index = faiss.IndexFlatL2(dimension)  # Standard L2 distance index

# Add vectors to the index
index.add(embeddings_array)

# Save the index
index_path = "paragraph_embeddings.faiss"
faiss.write_index(index, index_path)
print(f"FAISS index saved at {index_path}")
