# Encodez "Tesla revenue" avec les deux et comparez
import numpy as np
from fastembed import TextEmbedding
from sentence_transformers import SentenceTransformer

fe = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")
st = SentenceTransformer("BAAI/bge-large-en-v1.5")

v1 = list(fe.embed(["Tesla revenue"]))[0]
v2 = st.encode("Tesla revenue", normalize_embeddings=True)

# Similarité entre les deux
sim = np.dot(v1, v2)
print(f"Similarité : {sim:.4f}")  # < 0.99 = problème