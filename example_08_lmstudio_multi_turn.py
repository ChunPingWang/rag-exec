"""
範例 8：LM Studio 多輪對話
使用 OpenAI SDK 實現有記憶的對話

使用方式：python example_08_lmstudio_multi_turn.py
需要：LM Studio 運行中，已載入模型，Local Server 已啟動
需要安裝：pip install openai
"""

from openai import OpenAI


class LMStudioChatBot:
    """
    LM Studio 聊天機器人
    支援多輪對話，AI 會記住對話歷史
    """

    def __init__(self, model_name="gpt-oss-120b"):
        """初始化聊天機器人"""
        self.client = OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="not-needed"
        )
        self.model = model_name
        self.messages = []

    def set_system_prompt(self, system_prompt):
        """
        設定系統提示詞（AI 的角色）

        參數：
            system_prompt: 描述 AI 角色的文字
        """
        self.messages = [{"role": "system", "content": system_prompt}]

    def chat(self, user_message):
        """
        發送訊息並獲得回應

        參數：
            user_message: 使用者的訊息

        回傳：
            AI 的回應
        """
        # 加入使用者訊息
        self.messages.append({"role": "user", "content": user_message})

        # 發送請求
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages
        )

        # 取得 AI 回應
        ai_message = response.choices[0].message.content

        # 加入 AI 回應到歷史
        self.messages.append({"role": "assistant", "content": ai_message})

        return ai_message

    def clear_history(self):
        """清除對話歷史"""
        self.messages = []
        print("對話歷史已清除！")


# 主程式
if __name__ == "__main__":
    bot = LMStudioChatBot()

    # 設定 AI 角色為英文老師
    bot.set_system_prompt("""你是一位友善的英文老師。
    - 用繁體中文解釋英文文法和單字
    - 提供實用的例句
    - 鼓勵學生多練習""")

    print("=== 英文老師聊天室 ===")
    print("輸入 'quit' 結束對話")
    print("-" * 50)

    while True:
        user_input = input("\n你：")

        if user_input.lower() == "quit":
            print("再見！Keep learning!")
            break

        response = bot.chat(user_input)
        print(f"\n老師：{response}")
