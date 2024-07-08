from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.llms.ollama import Ollama
import argparse


CHROMA_PATH = "chroma"

# 定義提示模板
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # 解析命令行參數
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # 獲取嵌入函數
    embedding_function = OllamaEmbeddings(model="mxbai-embed-large")

    # 初始化Chroma向量存儲
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # 進行相似度搜索
    results = db.similarity_search_with_score(query_text, k=5)

    # 構建上下文文本
    context_text    = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt          = prompt_template.format(context=context_text, question=query_text)

    # 初始化Ollama模型
    model         = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    # 格式化並輸出回應
    sources            = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}"

    # 如需要顯示sources將以下註解刪除
    #formatted_response = f"Response: {response_text}\nSources: {sources}"
    
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
