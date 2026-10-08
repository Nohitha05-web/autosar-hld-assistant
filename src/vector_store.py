"""
Phase 4: Embeddings and Vector Database Engine
Stores numerical representations of AUTOSAR chunks in ChromaDB
and enables fast semantic search with metadata filtering.
"""

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

DEFAULT_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vector_db_store")
DEFAULT_COLLECTION = "autosar_hld_knowledge"
MODEL_NAME = "all-MiniLM-L6-v2"

class AUTOSARVectorStore:
    """
    Manages persistent ChromaDB vector storage and semantic retrieval
    for AUTOSAR High-Level Design artifacts.
    """

    def __init__(self, persist_dir: str = DEFAULT_DB_DIR, collection_name: str = DEFAULT_COLLECTION):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        os.makedirs(self.persist_dir, exist_ok=True)

        # Initialize local SentenceTransformer model
        self.encoder = SentenceTransformer(MODEL_NAME)

        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Computes embeddings and stores chunks with their metadata into ChromaDB.
        """
        if not chunks:
            return 0

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # Generate embeddings
        embeddings = self.encoder.encode(documents, normalize_embeddings=True).tolist()

        # Chroma upsert allows idempotent updates
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        return len(ids)

    def query(
        self,
        query_text: str,
        n_results: int = 4,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic vector search against the indexed AUTOSAR knowledge base.
        Returns top results with similarity confidence scores.
        """
        query_emb = self.encoder.encode([query_text], normalize_embeddings=True).tolist()

        query_params = {
            "query_embeddings": query_emb,
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }
        if where_filter:
            query_params["where"] = where_filter

        results = self.collection.query(**query_params)

        formatted_results = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            for idx in range(len(results["ids"][0])):
                doc_id = results["ids"][0][idx]
                content = results["documents"][0][idx]
                meta = results["metadatas"][0][idx]
                dist = results["distances"][0][idx]
                
                # Cosine distance to similarity percentage
                similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))

                formatted_results.append({
                    "id": doc_id,
                    "content": content,
                    "metadata": meta,
                    "distance": dist,
                    "similarity": round(similarity, 4),
                    "confidence": "HIGH" if similarity > 0.75 else ("MEDIUM" if similarity > 0.55 else "LOW")
                })

        return formatted_results

    def get_document_count(self) -> int:
        """Returns total chunks stored in the collection."""
        return self.collection.count()

    def is_document_indexed(self, doc_name: str) -> bool:
        """Checks if chunks from doc_name are already indexed."""
        results = self.collection.get(
            where={"source": doc_name},
            limit=1
        )
        return len(results["ids"]) > 0

    def delete_document(self, doc_name: str) -> None:
        """Removes all indexed chunks associated with doc_name."""
        self.collection.delete(where={"source": doc_name})

    def reset_collection(self) -> None:
        """Wipes the current collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )


if __name__ == "__main__":
    from src.pdf_parser import AUTOSARDocumentParser
    from src.chunker import AUTOSARChunker

    pdf_file = "data/AUTOSAR_BodyControlModule_HLD.pdf"
    if os.path.exists(pdf_file):
        print(f"Indexing {pdf_file}...")
        parser = AUTOSARDocumentParser(pdf_file)
        doc_data = parser.extract_document()
        chunker = AUTOSARChunker()
        chunks = chunker.chunk_document(doc_data)

        store = AUTOSARVectorStore()
        indexed_count = store.index_chunks(chunks)
        print(f"Successfully indexed {indexed_count} chunks into ChromaDB!")

        query = "What is the periodicity and ASIL level of SWC_ExteriorLighting?"
        print(f"\nTesting Query: '{query}'")
        res = store.query(query, n_results=2)
        for r in res:
            print(f"- [Score: {r['similarity']} | Page {r['metadata']['page']}] {r['content'][:140]}...")
