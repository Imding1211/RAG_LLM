
from database_controller import populate_database, clear_database, calculate_existing_ids
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from query_controller import query_rag
from langchain_chroma import Chroma
import argparse

#=============================================================================#

LLM_MODEL_NAME       = "gemma2:2b"
EMBEDDING_MODEL_NAME = "all-minilm"

QUERY_NUM       = 5
DATA_PATH       = "data"
CHROMA_PATH     = "chroma"

LLM_MODEL       = Ollama(model=LLM_MODEL_NAME)
EMBEDDING_MODEL = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

# 初始化Chroma向量存儲
DATABASE = Chroma(
    persist_directory  = CHROMA_PATH, 
    embedding_function = EMBEDDING_MODEL
    )

#=============================================================================#

PROMPT_TEMPLATE = """

{context}

---

根據以上資料用繁體中文回答問題: {question}
"""

#=============================================================================#

def run():

    while True:
        query_text = input("Enter your question or enter exit to stop:\n")

        if query_text == "exit":
            break

        response = query_rag(query_text, QUERY_NUM, LLM_MODEL, PROMPT_TEMPLATE, DATABASE)
        print(response)
        print("\n")

#=============================================================================#

def populate(reset):

    if reset:
        clear()

    existing_ids = calculate_existing_ids(DATABASE)

    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = populate_database(EMBEDDING_MODEL, DATA_PATH, DATABASE)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")

    else:
        print("No new documents to add")

#=============================================================================#

def clear():

    delete_ids = calculate_existing_ids(DATABASE)
    clear_database(delete_ids, DATABASE)

    print("Clearing Database")

#=============================================================================#

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="程式描述")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # 定義 run 子命令
    parser_run = subparsers.add_parser('run', help='執行程式')
    parser_run.set_defaults(func=lambda args: run())

    # 定義 populate 子命令
    parser_populate = subparsers.add_parser('populate', help='更新資料庫')
    parser_populate.add_argument('--reset', action='store_true', help='重置資料庫')
    parser_populate.set_defaults(func=lambda args: populate(args.reset))

    # 定義 clear 子命令
    parser_clear = subparsers.add_parser('clear', help='清空資料庫')
    parser_clear.set_defaults(func=lambda args: clear())

    args = parser.parse_args()

    # 根據子命令呼叫對應的函式
    if args.command:
        args.func(args)
    else:
        parser.print_help()
