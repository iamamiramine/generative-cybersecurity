import os
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from time import sleep

EMBED_DELAY = 0.02  # 20 milliseconds


# This is to get the Streamlit app to use less CPU while embedding documents into Chromadb.
class EmbeddingProxy:
    def __init__(self, embedding):
        self.embedding = embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_query(text)



# This happens all at once, not ideal for large datasets.
def create_vector_db(texts, embeddings=None, collection_name="chroma", force_reload=False):
    try:
        # Select embeddings
        if not embeddings:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        proxy_embeddings = EmbeddingProxy(embeddings)
        persist_directory = os.path.join("./data/store/", collection_name)
        
        # Try to load existing database
        if os.path.exists(persist_directory) and not force_reload:
            print(f"Loading existing vector database from {persist_directory}", flush=True)
            db = Chroma(
                collection_name=collection_name,
                embedding_function=proxy_embeddings,
                persist_directory=persist_directory
            )
        else:
            # Create new database if it doesn't exist or force reload
            if not texts:
                print("Empty texts passed in to create vector database", flush=True)
                texts = []
            
            # Delete existing directory if force_reload
            if force_reload and os.path.exists(persist_directory):
                import shutil
                shutil.rmtree(persist_directory)
            
            print(f"Creating new vector database in {persist_directory}", flush=True)
            db = Chroma(
                collection_name=collection_name,
                embedding_function=proxy_embeddings,
                persist_directory=persist_directory
            )
            if texts:
                # Log document statistics
                print(f"Processing {len(texts)} documents", flush=True)
                # Validate documents before adding
                valid_texts = [
                    text for text in texts 
                    if hasattr(text, 'page_content') and text.page_content.strip()
                ]
                print(f"Found {len(valid_texts)} valid documents", flush=True)
                db.add_documents(valid_texts)

        return db
    except Exception as e:
        print(f"Error creating vector database: {e}", flush=True)
        raise


def find_similar(vs, query):
    docs = vs.similarity_search(query)
    return docs
