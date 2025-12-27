"""
範例 6：使用 LM Studio 進行基本對話
LM Studio 使用 OpenAI 相容的 API 格式

使用方式：python example_06_lmstudio_basic.py
需要：LM Studio 運行中，已載入模型，Local Server 已啟動
"""

import requests


def chat_with_lmstudio(message):
    """
    發送訊息給 LM Studio 並獲得回應

    參數：
        message: 你想問 AI 的問題

    回傳：
        AI 的回應
    """

    # LM Studio 的 API 網址（預設埠號 1234）
    url = "http://localhost:1234/v1/chat/completions"

    # 準備要發送的資料（OpenAI 格式）
    data = {
        "model": "gpt-oss-120b",  # 模型名稱（在 LM Studio 中載入的模型）
        "messages": [
            {
                "role": "user",
                "content": message
            }
        ]
    }

    # 發送請求
    response = requests.post(url, json=data)
    result = response.json()

    # 從回應中取得 AI 的訊息
    return result["choices"][0]["message"]["content"]


# 主程式
if __name__ == "__main__":
    question = "什麼是人工智慧？請用簡單的方式解釋。"

    print(f"問題：{question}")
    print("-" * 50)

    answer = chat_with_lmstudio(question)
    print(f"AI 回答：{answer}")
