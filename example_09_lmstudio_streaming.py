"""
範例 9：LM Studio 串流輸出
即時顯示 AI 的回應，像 ChatGPT 一樣一個字一個字出現

使用方式：python example_09_lmstudio_streaming.py
需要：LM Studio 運行中，已載入模型，Local Server 已啟動
需要安裝：pip install openai
"""

from openai import OpenAI


def stream_chat_lmstudio(message):
    """
    使用串流方式獲得 LM Studio 回應
    """

    client = OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="not-needed"
    )

    # 發送串流請求
    stream = client.chat.completions.create(
        model="gpt-oss-120b",
        messages=[{"role": "user", "content": message}],
        stream=True  # 啟用串流模式
    )

    print("AI：", end="", flush=True)

    # 逐步接收並顯示回應
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print()  # 最後換行


# 主程式
if __name__ == "__main__":
    question = "請用三句話介紹台灣。"
    print(f"問題：{question}")
    print("-" * 50)
    stream_chat_lmstudio(question)
