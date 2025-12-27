"""
範例 3：串流輸出
像 ChatGPT 一樣，一個字一個字地顯示回應

使用方式：python example_03_streaming.py
需要：Ollama 運行中，已下載 gpt-oss:120b 模型
"""

import requests
import json

def stream_chat(prompt):
    """
    使用串流方式獲得 AI 回應
    回應會一個字一個字地顯示出來
    """

    url = "http://localhost:11434/api/generate"

    data = {
        "model": "gpt-oss:120b",
        "prompt": prompt,
        "stream": True  # 啟用串流模式
    }

    # 使用串流方式發送請求
    response = requests.post(url, json=data, stream=True)

    print("AI：", end="", flush=True)

    # 逐行讀取回應
    for line in response.iter_lines():
        if line:
            # 解析每一行 JSON
            chunk = json.loads(line)

            # 印出這一小段文字（不換行）
            print(chunk["response"], end="", flush=True)

            # 如果完成了，就跳出迴圈
            if chunk.get("done", False):
                break

    print()  # 最後換行


# 主程式
if __name__ == "__main__":
    question = "請寫一首關於程式設計的短詩。"
    print(f"問題：{question}")
    print("-" * 50)
    stream_chat(question)
