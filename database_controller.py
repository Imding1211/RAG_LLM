
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
import shutil
import os

#=============================================================================#

def populate_database(embedding_model, data_path, chroma_path):

    documents = load_documents(data_path)

    chunks = split_documents(documents)

    existing_ids, new_chunks = add_to_chroma(chunks, embedding_model, chroma_path)

    return existing_ids, new_chunks

#=============================================================================#

def clear_database(chroma_path):

    # 清空資料庫目錄
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

#=============================================================================#

def load_documents(data_path):

    # 載入指定資料夾中的PDF文件
    document_loader = PyPDFDirectoryLoader(data_path)

    return document_loader.load()

#=============================================================================#

def split_documents(documents):

    # 使用遞歸字符分割器分割文件
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size         = 800,   # 每塊的大小
        chunk_overlap      = 80,    # 每塊之間的重疊部分
        length_function    = len,   # 用於計算塊長度的函數
        is_separator_regex = False, # 是否使用正則表達式作為分隔符
    )

    return text_splitter.split_documents(documents)

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

def calculate_existing_ids(db):

    # 獲取現有文件的ID
    existing_items = db.get(include=[])
    existing_ids   = set(existing_items["ids"])

    return existing_ids

#=============================================================================#

def add_to_chroma(chunks, embedding_model, chroma_path):

    # 初始化Chroma向量存儲
    db = Chroma(
        persist_directory  = chroma_path,  # 持久化存儲目錄
        embedding_function = OllamaEmbeddings(model=embedding_model)  # 嵌入函數
    )

    # 計算每個塊的ID
    chunks_with_ids = calculate_chunk_ids(chunks)

    # 獲取現有文件的ID
    existing_ids = calculate_existing_ids(db)
    

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)

    return existing_ids, new_chunks
