"""
範例 14：文件問答系統
讀取文字檔案，讓 AI 回答關於文件內容的問題

使用方式：python example_14_document_qa.py
需要：LM Studio 運行中，已載入模型，Local Server 已啟動
"""

import os
from openai import OpenAI


def read_document(file_path):
    """
    讀取文件內容

    參數：
        file_path: 文件路徑

    回傳：
        文件內容
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_document(text, chunk_size=500, overlap=50):
    """
    將長文件切割成小段落

    為什麼要切割？
    - AI 有輸入長度限制
    - 小段落更容易精確搜尋
    - 可以只傳送相關的部分給 AI

    參數：
        text: 文件內容
        chunk_size: 每段的字數
        overlap: 段落之間重疊的字數（避免資訊被切斷）
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # 重疊部分

    return chunks


class DocumentQA:
    """
    文件問答系統類別
    """

    def __init__(self):
        self.client = OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="not-needed"
        )
        self.chunks = []

    def load_document(self, file_path):
        """載入文件"""
        text = read_document(file_path)
        self.chunks = chunk_document(text)
        print(f"已載入文件，共 {len(self.chunks)} 個段落")

    def load_text(self, text):
        """直接載入文字"""
        self.chunks = chunk_document(text)
        print(f"已載入文字，共 {len(self.chunks)} 個段落")

    def find_relevant_chunks(self, question, top_k=3):
        """
        找出與問題相關的段落（簡單的關鍵字匹配）
        """
        scored_chunks = []

        # 將問題拆成關鍵字
        keywords = question.lower().split()

        for chunk in self.chunks:
            score = 0
            chunk_lower = chunk.lower()

            # 計算每個關鍵字出現的次數
            for keyword in keywords:
                if keyword in chunk_lower:
                    score += chunk_lower.count(keyword)

            scored_chunks.append((score, chunk))

        # 排序並返回最相關的段落
        scored_chunks.sort(reverse=True)
        return [chunk for score, chunk in scored_chunks[:top_k] if score > 0]

    def ask(self, question):
        """
        提問並獲得回答
        """
        # 找出相關段落
        relevant_chunks = self.find_relevant_chunks(question)

        if not relevant_chunks:
            return "抱歉，我在文件中找不到相關資訊。"

        # 建立上下文
        context = "\n---\n".join(relevant_chunks)

        prompt = f"""你是一個文件問答助手。請根據以下文件內容回答問題。
如果文件中沒有相關資訊，請誠實說明。

文件內容：
{context}

問題：{question}

請用繁體中文回答："""

        response = self.client.chat.completions.create(
            model="gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content


# 主程式
if __name__ == "__main__":
    qa = DocumentQA()

    # 範例：載入一段說明文字
    sample_text = """
    人工智慧（Artificial Intelligence，簡稱 AI）是電腦科學的一個分支，
    致力於創造能夠執行通常需要人類智慧的任務的機器。這些任務包括學習、
    推理、問題解決、感知和語言理解。

    機器學習是人工智慧的一個子領域，專注於開發能夠從資料中學習的演算法。
    深度學習是機器學習的一個分支，使用多層神經網路來處理複雜的資料模式。

    自然語言處理（NLP）是 AI 的另一個重要領域，讓電腦能夠理解、解釋和
    生成人類語言。ChatGPT 就是一個著名的 NLP 應用。

    AI 的應用非常廣泛，包括：
    - 語音助手（如 Siri、Alexa）
    - 自動駕駛汽車
    - 醫療診斷輔助
    - 推薦系統（如 Netflix、YouTube）
    - 遊戲 AI
    """

    qa.load_text(sample_text)

    print("\n=== 文件問答系統 ===")
    print("輸入 'quit' 結束")
    print("-" * 50)

    while True:
        question = input("\n你的問題：")

        if question.lower() == "quit":
            break

        answer = qa.ask(question)
        print(f"\n回答：{answer}")
