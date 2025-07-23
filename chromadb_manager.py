import chromadb

class ChromaDBManager:
    def __init__(self, db_path="chroma_db", collection_name="docs_llm"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_documents(self, fragments, metadatas, embeddings, ids):
        """Agrega fragmentos con embeddings y metadatos a la colección."""
        self.collection.add(
            documents=fragments,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids
        )

    def get_all_documents(self):
        """Recupera todos los documentos de la colección."""
        return self.collection.get(ids=None)

    def query_by_embedding(self, embedding, n_results=3):
        """Busca documentos similares a un embedding dado."""
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )
        return results["documents"][0], results["metadatas"][0]

    def get_existing_ids(self):
        """Recupera todos los IDs almacenados."""
        data = self.collection.get(ids=None)
        return set(data["ids"])

    def reset_collection(self):
        """Elimina y recrea la colección."""
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name)

    def indexar_batch(self, fragmentos, metadatos, embedding_manager, existing_ids):
        nuevos = 0
        to_index = []
        to_meta = []
        to_ids = []
        seen_ids = set()
        for i, fragmento in enumerate(fragmentos):
            idx = embedding_manager._hash_text(fragmento)
            if idx not in existing_ids and idx not in seen_ids:
                to_index.append(fragmento)
                to_meta.append(metadatos[i])
                to_ids.append(idx)
                seen_ids.add(idx)
        for start in range(0, len(to_index), 32):
            batch_frags = to_index[start:start + 32]
            batch_meta = to_meta[start:start + 32]
            batch_ids = to_ids[start:start + 32]
            if batch_frags:
                batch_embs = embedding_manager.embed_batch(batch_frags)
                self.collection.add(
                    embeddings=list(batch_embs),
                    documents=batch_frags,
                    metadatas=batch_meta,
                    ids=batch_ids
                )
                nuevos += len(batch_frags)
        return nuevos

    def get_all_ids(self):
        """Devuelve un set de todos los IDs almacenados en la colección."""
        all_items = self.collection.get(ids=None)
        return set(all_items["ids"]) if "ids" in all_items else set()
