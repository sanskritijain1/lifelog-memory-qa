import numpy as np
import faiss

print("Loading embeddings...")

embeddings = np.load("data/frame_embeddings.npy")

print("Shape:", embeddings.shape)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

faiss.normalize_L2(embeddings)

index.add(embeddings)

faiss.write_index(
    index,
    "data/frame_index.faiss"
)

print("Done.")
print("Indexed vectors:", index.ntotal)