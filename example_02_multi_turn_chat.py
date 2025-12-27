"""
範例 2：多輪對話
AI 會記住之前的對話內容，就像真正的聊天一樣

使用方式：python example_02_multi_turn_chat.py
需要：Ollama 運行中，已下載 gpt-oss:120b 模型
"""

import requests
import json

class ChatBot:
    """
    聊天機器人類別
    可以進行多輪對話，AI 會記住對話歷史
    """

    def __init__(self):
        """初始化聊天機器人"""
        self.url = "http://localhost:11434/api/chat"
        self.model = "gpt-oss:120b"
        self.messages = []  # 儲存對話歷史

    def chat(self, user_message):
        """
        發送訊息並獲得回應

        參數：
            user_message: 使用者的訊息

        回傳：
            AI 的回應
        """

        # 將使用者訊息加入歷史
        self.messages.append({
            "role": "user",
            "content": user_message
        })

        # 準備請求資料
        data = {
            "model": self.model,
            "messages": self.messages,
            "stream": False
        }

        # 發送請求
        response = requests.post(self.url, json=data)
        result = response.json()

        # 取得 AI 回應
        ai_message = result["message"]["content"]

        # 將 AI 回應加入歷史（這樣 AI 就能記住）
        self.messages.append({
            "role": "assistant",
            "content": ai_message
        })

        return ai_message

    def clear_history(self):
        """清除對話歷史，開始新對話"""
        self.messages = []
        print("對話歷史已清除！")


# 主程式
if __name__ == "__main__":
    bot = ChatBot()

    print("=== 多輪對話示範 ===")
    print("輸入 'quit' 結束對話")
    print("輸入 'clear' 清除對話歷史")
    print("-" * 50)

    while True:
        user_input = input("\n你：")

        if user_input.lower() == "quit":
            print("再見！")
            break
        elif user_input.lower() == "clear":
            bot.clear_history()
            continue

        response = bot.chat(user_input)
        print(f"\nAI：{response}")
