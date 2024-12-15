import os
from pathlib import Path
import pickle
import hashlib
from typing import Dict

from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_core.messages.base import BaseMessage


def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False)

    contents = docs
    if docs and isinstance(docs[0], Document):
        contents = [doc.page_content for doc in docs]

    texts = text_splitter.create_documents(contents)
    n_chunks = len(texts)
    print(f"Split into {n_chunks} chunks")
    return texts


def get_files_hash(data_dir: str = "./data") -> str:
    """Calculate a hash of all files in the data directory to detect changes."""
    files_info = []
    
    # Get info for txt files
    for path in Path(data_dir).glob('**/*.txt'):
        stat = path.stat()
        files_info.append((str(path), stat.st_size, stat.st_mtime))
    
    # Get info for csv files
    for path in Path(data_dir).glob('**/*.csv'):
        stat = path.stat()
        files_info.append((str(path), stat.st_size, stat.st_mtime))

    # Get info for pdf files
    for path in Path(data_dir).glob('**/*.pdf'):
        stat = path.stat()
        files_info.append((str(path), stat.st_size, stat.st_mtime))
    
    # Sort for consistency
    files_info.sort()
    
    # Create a hash of the files information
    hasher = hashlib.md5()
    for file_info in files_info:
        hasher.update(str(file_info).encode())
    
    return hasher.hexdigest()


def save_dataset_metadata(hash_value: str, file_path: str = "./data/store/dataset_metadata.pkl"):
    """Save dataset metadata including hash."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    metadata = {
        "hash": hash_value,
        "timestamp": os.path.getmtime(file_path) if os.path.exists(file_path) else None
    }
    with open(file_path, 'wb') as f:
        pickle.dump(metadata, f)


def load_dataset_metadata(file_path: str = "./data/store/dataset_metadata.pkl") -> Dict:
    """Load dataset metadata."""
    if not os.path.exists(file_path):
        return {"hash": None, "timestamp": None}
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def has_dataset_changed(data_dir: str = "./data") -> bool:
    """Check if the dataset has changed since last processing."""
    current_hash = get_files_hash(data_dir)
    stored_metadata = load_dataset_metadata()
    
    return current_hash != stored_metadata["hash"]


def save_text_chunks(texts, file_path: str = "./data/store/chunks.pkl"):
    """Save text chunks and update dataset metadata."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(texts, f)
    
    # Save current dataset state
    current_hash = get_files_hash("./data")
    save_dataset_metadata(current_hash)


def load_txt_files(data_dir: str = "./data", force_reload: bool = False):
    """Load text files and check for changes in the dataset."""
    if not force_reload and not has_dataset_changed(data_dir):
        # Try to load existing chunks if dataset hasn't changed
        chunks_path = "./data/store/chunks.pkl"
        if os.path.exists(chunks_path):
            print("Loading existing chunks (dataset unchanged)")
            return load_text_chunks(chunks_path)
    
    # Load and process files if dataset changed or force_reload is True
    print("Processing files (dataset changed or force reload)")
    docs = []
    paths = list_txt_files(data_dir)
    for path in paths:
        print(f"Loading {path}")
        loader = TextLoader(path)
        docs.extend(loader.load())
    
    # Process and save new chunks
    texts = split_documents(docs)
    save_text_chunks(texts)
    return texts


def load_text_chunks(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def get_question(input):
    if not input:
        return None
    elif isinstance(input,str):
        return input
    elif isinstance(input,dict) and 'question' in input:
        return input['question']
    elif isinstance(input,BaseMessage):
        return input.content
    else:
        raise Exception("string or dict with 'question' key expected as RAG chain input.")
    

def format_docs(docs): # docs is a doc loader from langchain
    return "\n\n".join(doc.page_content for doc in docs)


def list_txt_files(data_dir="./data"):
    paths = Path(data_dir).glob('**/*.txt')
    for path in paths:
        yield str(path)


def load_csv_files(data_dir="./data"):
    docs = []
    paths = Path(data_dir).glob('**/*.csv')
    for path in paths:
        loader = CSVLoader(file_path=str(path))
        docs.extend(loader.load())
    return docs


# Use with result of file_to_summarize = st.file_uploader("Choose a file") or a string.
# or a file like object.
def get_document_text(uploaded_file, title=None):
    docs = []
    fname = uploaded_file.name
    if not title:
        title = os.path.basename(fname)
    if fname.lower().endswith('pdf'):
        pdf_reader = PdfReader(uploaded_file)
        for num, page in enumerate(pdf_reader.pages):
            page = page.extract_text()
            doc = Document(page_content=page, metadata={'title': title, 'page': (num + 1)})
            docs.append(doc)

    else:
        # assume text
        doc_text = uploaded_file.read().decode()
        docs.append(doc_text)

    return docs