"""
範例 5：程式碼助手
讓 AI 幫你解釋程式碼、找錯誤、或寫程式

使用方式：python example_05_code_assistant.py
需要：Ollama 運行中，已下載 gpt-oss:120b 模型
"""

import requests
import json

def code_assistant(code, question):
    """
    程式碼助手：分析程式碼並回答問題

    參數：
        code: 要分析的程式碼
        question: 關於程式碼的問題
    """

    url = "http://localhost:11434/api/chat"

    system_prompt = """你是一位程式設計專家和教師。
    - 用清楚易懂的方式解釋程式碼
    - 如果發現錯誤，指出錯誤並提供修正建議
    - 使用繁體中文回答
    - 解釋時要考慮到學習者可能是初學者"""

    user_prompt = f"""請分析以下程式碼並回答問題。

程式碼：
```
{code}
```

問題：{question}"""

    data = {
        "model": "gpt-oss:120b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }

    response = requests.post(url, json=data)
    result = response.json()

    return result["message"]["content"]


# 主程式
if __name__ == "__main__":
    # 要分析的程式碼
    my_code = '''
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

result = calculate_average([])
print(result)
'''

    question = "這段程式碼有什麼問題？如何修正？"

    print("=== 程式碼助手 ===")
    print(f"問題：{question}")
    print("-" * 50)

    answer = code_assistant(my_code, question)
    print(f"AI 分析：\n{answer}")
