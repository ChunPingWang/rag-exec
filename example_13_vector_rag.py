"""
範例 13：使用向量搜尋的 RAG 系統
這個範例使用向量嵌入（Embedding）來搜尋最相關的文件

使用方式：python example_13_vector_rag.py
需要：LM Studio 運行中，已載入模型和嵌入模型，Local Server 已啟動
需要安裝：pip install numpy openai
"""

import numpy as np
from openai import OpenAI

# 初始化客戶端
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"
)

# 知識庫文件
DOCUMENTS = [
    "Python 是一種簡單易學的程式語言，適合初學者入門。",
    "JavaScript 是網頁開發的核心語言，可以讓網頁產生互動效果。",
    "機器學習讓電腦能從資料中學習，不需要明確的程式指令。",
    "深度學習是機器學習的一個分支，使用神經網路來處理複雜問題。",
    "自然語言處理（NLP）讓電腦能理解和生成人類語言。",
    "RAG 技術結合了資訊檢索和文字生成，提高 AI 回答的準確性。",
]

# 儲存文件的向量表示
document_embeddings = []


def get_embedding(text):
    """
    取得文字的向量表示（Embedding）

    向量嵌入是什麼？
    - 把文字轉換成一串數字（向量）
    - 意思相近的文字，向量也會相近
    - 這樣電腦就能「理解」文字的意義
    """
    response = client.embeddings.create(
        model="text-embedding-nomic-embed-text-v1.5",  # 嵌入模型
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(vec1, vec2):
    """
    計算兩個向量的餘弦相似度

    餘弦相似度是什麼？
    - 衡量兩個向量方向的相似程度
    - 值介於 -1 到 1 之間
    - 1 表示完全相同，0 表示無關，-1 表示完全相反
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def initialize_knowledge_base():
    """
    初始化知識庫：為所有文件計算向量
    """
    global document_embeddings
    print("正在初始化知識庫...")

    for doc in DOCUMENTS:
        embedding = get_embedding(doc)
        document_embeddings.append(embedding)

    print(f"已載入 {len(DOCUMENTS)} 份文件")


def vector_search(query, top_k=2):
    """
    向量搜尋：找出與問題最相關的文件

    參數：
        query: 使用者的問題
        top_k: 要返回的文件數量

    回傳：
        最相關的文件列表
    """
    # 取得問題的向量
    query_embedding = get_embedding(query)

    # 計算與每份文件的相似度
    similarities = []
    for i, doc_embedding in enumerate(document_embeddings):
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((similarity, DOCUMENTS[i]))

    # 按相似度排序，取前 k 個
    similarities.sort(reverse=True)
    return [doc for _, doc in similarities[:top_k]]


def rag_with_vector_search(question):
    """
    使用向量搜尋的 RAG 聊天
    """
    # 搜尋相關文件
    relevant_docs = vector_search(question, top_k=2)

    # 建立提示詞
    context = "\n".join([f"- {doc}" for doc in relevant_docs])

    prompt = f"""請根據以下參考資料回答問題：

參考資料：
{context}

問題：{question}

請用繁體中文簡潔回答："""

    # 呼叫 AI
    response = client.chat.completions.create(
        model="gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content, relevant_docs


# 主程式
if __name__ == "__main__":
    # 初始化知識庫
    initialize_knowledge_base()

    print("\n=== 向量搜尋 RAG 系統 ===")
    print("輸入 'quit' 結束")
    print("-" * 50)

    while True:
        question = input("\n你的問題：")

        if question.lower() == "quit":
            break

        answer, sources = rag_with_vector_search(question)

        print(f"\n找到的相關文件：")
        for i, doc in enumerate(sources, 1):
            print(f"  {i}. {doc[:50]}...")

        print(f"\nAI 回答：{answer}")
