"""
範例 11：通用聊天程式
可以在 Ollama 和 LM Studio 之間切換

使用方式：python example_11_universal_chatbot.py
需要：Ollama 或 LM Studio 運行中
"""

import requests
from openai import OpenAI


class UniversalChatBot:
    """
    通用聊天機器人
    支援 Ollama 和 LM Studio 兩種後端
    """

    def __init__(self, backend="lmstudio", model=None):
        """
        初始化聊天機器人

        參數：
            backend: "ollama" 或 "lmstudio"
            model: 模型名稱（可選）
        """
        self.backend = backend
        self.messages = []

        if backend == "lmstudio":
            self.client = OpenAI(
                base_url="http://localhost:1234/v1",
                api_key="not-needed"
            )
            self.model = model or "gpt-oss-120b"
        elif backend == "ollama":
            self.url = "http://localhost:11434/api/chat"
            self.model = model or "gpt-oss:120b"
        else:
            raise ValueError("backend 必須是 'ollama' 或 'lmstudio'")

    def chat(self, user_message):
        """發送訊息並獲得回應"""

        self.messages.append({"role": "user", "content": user_message})

        if self.backend == "lmstudio":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages
            )
            ai_message = response.choices[0].message.content

        else:  # ollama
            data = {
                "model": self.model,
                "messages": self.messages,
                "stream": False
            }
            response = requests.post(self.url, json=data)
            result = response.json()
            ai_message = result["message"]["content"]

        self.messages.append({"role": "assistant", "content": ai_message})
        return ai_message


# 主程式
if __name__ == "__main__":
    # 選擇後端：改成 "ollama" 可切換到 Ollama
    bot = UniversalChatBot(backend="lmstudio")

    print(f"=== 使用 {bot.backend.upper()} 後端 ===")
    print("輸入 'quit' 結束對話")
    print("-" * 50)

    while True:
        user_input = input("\n你：")

        if user_input.lower() == "quit":
            print("再見！")
            break

        response = bot.chat(user_input)
        print(f"\nAI：{response}")
