
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_chroma import Chroma
import argparse

#=============================================================================#

# 定義提示模板
PROMPT_TEMPLATE = """

{context}

---

根據以上資料用繁體中文回答問題: {question}
"""

#=============================================================================#

def generate_results(query_text, query_num, chroma_path, embedding_model):
    
    # 初始化Chroma向量存儲
    db = Chroma(
        persist_directory  = chroma_path, 
        embedding_function = OllamaEmbeddings(model=embedding_model)
        )

    # 進行相似度搜索
    query_results = db.similarity_search_with_score(query_text, k=query_num)

    return query_results

#=============================================================================#

def generate_prompt(query_text, query_results):
    
    # 構建上下文文本
    context_text    = "\n\n---\n\n".join([doc.page_content for doc, _score in query_results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt          = prompt_template.format(context=context_text, question=query_text)

    return prompt

#=============================================================================#

def generate_response(prompt, query_results, llm_model, show_sources=False):

    # 初始化Ollama模型
    model = Ollama(model=llm_model)

    # 生成回覆
    response_text = model.invoke(prompt)

    # 格式化並輸出回應
    sources  = [doc.metadata.get("id", None) for doc, _score in query_results]
    response = f"Response: {response_text}"

    if show_sources:
        # 如需要顯示sources將以下註解刪除
        response = f"Response: {response_text}\nSources: {sources}"
    
    return response

#=============================================================================#

def query_rag(query_text, query_num, chroma_path, llm_model, embedding_model):
    
    results  = generate_results(query_text, query_num, chroma_path, embedding_model)
    
    prompt   = generate_prompt(query_text, results)
    
    response = generate_response(prompt, results, llm_model, show_sources=False)
    
    print(response)

