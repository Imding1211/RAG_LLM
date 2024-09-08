
from langchain_core.prompts import ChatPromptTemplate

#=============================================================================#

def query_rag(query_text, query_num, llm_model, prompt_template, database):
    
    results  = generate_results(query_text, query_num, database)
    
    prompt   = generate_prompt(query_text, results, prompt_template)
    
    response = generate_response(prompt, results, llm_model)
    
    return response

#=============================================================================#

def generate_results(query_text, query_num, database):
    
    # 進行相似度搜索
    query_results = database.similarity_search_with_score(query_text, k=query_num)

    return query_results

#=============================================================================#

def generate_prompt(query_text, query_results, prompt_template):
    
    # 構建上下文文本
    context_text    = "\n\n---\n\n".join([doc.page_content for doc, _score in query_results])
    prompt_template = ChatPromptTemplate.from_template(prompt_template)
    prompt          = prompt_template.format(context=context_text, question=query_text)

    return prompt

#=============================================================================#

def generate_response(prompt, query_results, llm_model, show_sources=False):

    # 生成回覆
    response_text = llm_model.invoke(prompt)

    # 格式化並輸出回應
    sources  = [doc.metadata.get("id", None) for doc, _score in query_results]
    response = f"Response: {response_text}"

    if show_sources:
        # 如需要顯示sources將以下註解刪除
        response = f"Response: {response_text}\nSources: {sources}"
    
    return response



