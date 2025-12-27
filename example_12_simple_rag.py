"""
範例 12：簡易 RAG 系統
這個範例展示 RAG 的基本概念：先搜尋文件，再讓 AI 根據文件回答

使用方式：python example_12_simple_rag.py
需要：LM Studio 運行中，已載入模型，Local Server 已啟動
"""

from openai import OpenAI

# 模擬的知識庫（實際應用中可能是資料庫或文件系統）
KNOWLEDGE_BASE = {
    "python": """
    Python 是一種高階程式語言，由 Guido van Rossum 於 1991 年創建。
    Python 的特點：
    - 語法簡潔易讀
    - 支援多種程式設計範式
    - 擁有豐富的第三方套件
    - 廣泛用於網頁開發、資料科學、人工智慧等領域
    """,
    "javascript": """
    JavaScript 是一種腳本語言，主要用於網頁開發。
    JavaScript 的特點：
    - 可在瀏覽器中直接執行
    - 支援事件驅動程式設計
    - 可用於前端和後端（Node.js）開發
    - 是網頁互動功能的核心技術
    """,
    "機器學習": """
    機器學習是人工智慧的一個分支，讓電腦從資料中學習規律。
    機器學習的類型：
    - 監督式學習：使用有標籤的資料訓練
    - 非監督式學習：從無標籤資料中發現模式
    - 強化學習：透過獎勵機制學習最佳策略
    常見應用：圖像辨識、語音辨識、推薦系統等
    """
}


def simple_search(query):
    """
    簡單的關鍵字搜尋
    在實際應用中，這裡會使用向量搜尋或全文搜尋引擎

    參數：
        query: 搜尋關鍵字

    回傳：
        找到的相關文件內容
    """
    results = []
    query_lower = query.lower()

    for keyword, content in KNOWLEDGE_BASE.items():
        if keyword.lower() in query_lower or query_lower in keyword.lower():
            results.append(content)

    return results


def rag_chat(question):
    """
    RAG 聊天函數
    先搜尋相關文件，再讓 AI 根據文件回答

    參數：
        question: 使用者的問題

    回傳：
        AI 的回答
    """

    # 步驟 1：搜尋相關文件
    retrieved_docs = simple_search(question)

    # 步驟 2：建立提示詞
    if retrieved_docs:
        context = "\n\n".join(retrieved_docs)
        prompt = f"""請根據以下參考資料回答問題。如果參考資料中沒有相關資訊，請說明你不確定。

參考資料：
{context}

問題：{question}

請用繁體中文回答："""
    else:
        prompt = f"""問題：{question}

注意：我找不到相關的參考資料，請根據你的知識回答，但要說明這是你的一般知識，不是來自特定文件。

請用繁體中文回答："""

    # 步驟 3：呼叫 AI
    client = OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="not-needed"
    )

    response = client.chat.completions.create(
        model="gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# 主程式
if __name__ == "__main__":
    print("=== 簡易 RAG 系統 ===")
    print("可以問關於 Python、JavaScript、機器學習的問題")
    print("輸入 'quit' 結束")
    print("-" * 50)

    while True:
        question = input("\n你的問題：")

        if question.lower() == "quit":
            break

        answer = rag_chat(question)
        print(f"\nAI 回答：{answer}")
