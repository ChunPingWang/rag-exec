"""
範例 7：使用 OpenAI 套件連接 LM Studio
這種方式的好處是：如果你之前用過 OpenAI API，程式碼幾乎不用改！

使用方式：python example_07_lmstudio_openai.py
需要：LM Studio 運行中，已載入模型，Local Server 已啟動
需要安裝：pip install openai
"""

from openai import OpenAI


def chat_with_openai_sdk(message):
    """
    使用 OpenAI SDK 連接 LM Studio

    參數：
        message: 你想問 AI 的問題

    回傳：
        AI 的回應
    """

    # 建立 OpenAI 客戶端，指向 LM Studio
    client = OpenAI(
        base_url="http://localhost:1234/v1",  # LM Studio 的網址
        api_key="not-needed"                   # LM Studio 不需要 API 金鑰
    )

    # 發送聊天請求
    response = client.chat.completions.create(
        model="gpt-oss-120b",  # 模型名稱
        messages=[
            {"role": "user", "content": message}
        ]
    )

    # 取得回應
    return response.choices[0].message.content


# 主程式
if __name__ == "__main__":
    question = "請解釋什麼是迴圈，並給我一個 Python 範例。"

    print(f"問題：{question}")
    print("-" * 50)

    answer = chat_with_openai_sdk(question)
    print(f"AI 回答：{answer}")
