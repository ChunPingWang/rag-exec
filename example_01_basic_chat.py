"""
範例 1：與 AI 進行簡單對話
這是最基本的使用方式，就像跟 AI 聊天一樣

使用方式：python example_01_basic_chat.py
需要：Ollama 運行中，已下載 gpt-oss:120b 模型
"""

import requests
import json

def chat_with_ai(prompt):
    """
    發送訊息給 AI 並獲得回應

    參數：
        prompt: 你想問 AI 的問題（字串）

    回傳：
        AI 的回應（字串）
    """

    # Ollama 的 API 網址（在你的電腦上運行）
    url = "http://localhost:11434/api/generate"

    # 準備要發送的資料
    data = {
        "model": "gpt-oss:120b",  # 使用的模型名稱
        "prompt": prompt,          # 你的問題
        "stream": False            # 不使用串流（一次返回完整回應）
    }

    # 發送請求給 Ollama
    response = requests.post(url, json=data)

    # 解析回應
    result = response.json()

    return result["response"]


# 主程式
if __name__ == "__main__":
    # 問 AI 一個問題
    question = "什麼是人工智慧？請用簡單的方式解釋。"

    print(f"問題：{question}")
    print("-" * 50)

    answer = chat_with_ai(question)
    print(f"AI 回答：{answer}")
