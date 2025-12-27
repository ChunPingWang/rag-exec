"""
範例 4：設定系統提示詞
可以讓 AI 扮演特定角色，例如：老師、翻譯官、程式專家等

使用方式：python example_04_system_prompt.py
需要：Ollama 運行中，已下載 gpt-oss:120b 模型
"""

import requests
import json

def chat_with_role(system_prompt, user_message):
    """
    使用特定角色與 AI 對話

    參數：
        system_prompt: 系統提示詞，定義 AI 的角色和行為
        user_message: 使用者的訊息
    """

    url = "http://localhost:11434/api/chat"

    data = {
        "model": "gpt-oss:120b",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        "stream": False
    }

    response = requests.post(url, json=data)
    result = response.json()

    return result["message"]["content"]


# 主程式
if __name__ == "__main__":
    # 範例：讓 AI 扮演高中數學老師
    system = """你是一位親切的高中數學老師。
    - 用簡單易懂的方式解釋數學概念
    - 多舉生活中的例子
    - 鼓勵學生，保持正向態度
    - 使用繁體中文回答"""

    question = "什麼是微積分？為什麼要學它？"

    print("角色：高中數學老師")
    print(f"問題：{question}")
    print("-" * 50)

    answer = chat_with_role(system, question)
    print(f"老師：{answer}")
