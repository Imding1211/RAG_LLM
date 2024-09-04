
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
import argparse
import shutil
import os

#=============================================================================#

CHROMA_PATH     = "chroma"
DATA_PATH       = "data"

#=============================================================================#

def populate_database(embedding_model: str):
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks, embedding_model)

#=============================================================================#

def clear_database():
    # 清空資料庫目錄
    print("Clearing Database")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

#=============================================================================#

def load_documents():
    # 載入指定資料夾中的PDF文件
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

#=============================================================================#

def split_documents(documents: list[Document]):
    # 使用遞歸字符分割器分割文件
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size         = 800,   # 每塊的大小
        chunk_overlap      = 80,    # 每塊之間的重疊部分
        length_function    = len,   # 用於計算塊長度的函數
        is_separator_regex = False, # 是否使用正則表達式作為分隔符
    )
    return text_splitter.split_documents(documents)

#=============================================================================#

def add_to_chroma(chunks: list[Document], embedding_model: str):
    # 初始化Chroma向量存儲
    db = Chroma(
        persist_directory  = CHROMA_PATH,  # 持久化存儲目錄
        embedding_function = OllamaEmbeddings(model=embedding_model)  # 嵌入函數
    )

    # 計算每個塊的ID
    chunks_with_ids = calculate_chunk_ids(chunks)

    # 獲取現有文件的ID
    existing_items = db.get(include=[])
    existing_ids   = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)

    else:
        print("No new documents to add")

#=============================================================================#

def calculate_chunk_ids(chunks):
    # 計算每個塊的唯一ID
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page   = chunk.metadata.get("page")

        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"

        last_page_id = current_page_id

        # 將ID添加到頁面的元數據中
        chunk.metadata["id"] = chunk_id

    return chunks

#=============================================================================#

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--reset", action="store_true", help="Reset the database.")

    args = parser.parse_args()

    if args.reset:
        clear_database()

    documents = load_documents()

    chunks = split_documents(documents)
    
    add_to_chroma(chunks, "all-minilm")
