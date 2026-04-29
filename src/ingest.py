import pymupdf
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json


def get_overlapped_chunks(textin, chunksize, overlapsize):
    return [" ".join(textin[a:a+chunksize]) for a in range(0, len(textin), chunksize-overlapsize)]

with pymupdf.open("data/thesis_report.pdf") as doc: 
    text = " ".join([page.get_text() for page in doc])

words = text.split()
chunks = get_overlapped_chunks(words, 200, 30)

model = SentenceTransformer("intfloat/multilingual-e5-large")
embeddings = model.encode(["passage: " + c for c in chunks])
print(embeddings.shape)

embeddings_np = np.array(embeddings).astype("float32")

dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(embeddings_np)

print(f"vectors in index: {index.ntotal}")

faiss.write_index(index, "data/faiss_index.bin")

with open("data/chunks.json", "w") as file:
    json.dump(chunks, file)