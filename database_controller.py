
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

#=============================================================================#

def populate_database(embedding_model, data_path, database):

    documents = load_documents(data_path)

    chunks = split_documents(documents)

    new_chunks = add_to_chroma(chunks, database)

    return new_chunks

#=============================================================================#

def clear_database(delete_ids, database):

    if list(delete_ids):
        database.delete(ids=list(delete_ids))

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

def calculate_existing_ids(database):

    # 獲取現有文件的ID
    existing_items = database.get(include=[])
    existing_ids   = set(existing_items["ids"])

    return existing_ids

#=============================================================================#

def add_to_chroma(chunks, database):

    # 計算每個塊的ID
    chunks_with_ids = calculate_chunk_ids(chunks)

    # 獲取現有文件的ID
    existing_ids = calculate_existing_ids(database)
    
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        database.add_documents(new_chunks, ids=new_chunk_ids)

    return new_chunks
