import os
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2", cache_dir="embed_cache"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.model = SentenceTransformer(self.model_name)

    def _hash_text(self, text):
        """Genera un hash único para cada fragmento de texto."""
        return hashlib.sha1(text.encode('utf-8')).hexdigest()

    def embed(self, text):
        """Obtiene el embedding de un fragmento de texto, usando caché en disco."""
        text_hash = self._hash_text(text)
        cache_file = os.path.join(self.cache_dir, f"{text_hash}_{self.model_name}.npy")
        if os.path.exists(cache_file):
            # Cargar el embedding desde el caché
            return np.load(cache_file)
        # Calcular el embedding
        emb = self.model.encode([text])[0]
        np.save(cache_file, emb)
        return emb

    def embed_batch(self, texts):
        """Embeddings en batch con caché por fragmento."""
        embs = []
        for text in texts:
            embs.append(self.embed(text))
        return embs
