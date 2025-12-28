# 使用 Ollama 與 LM Studio 運行本地 AI 模型教學

> 適合高中程度學習者的本地 AI 模型完整教學，涵蓋基礎到進階應用。

---

## 📓 分支說明 (Branch Information)

| 分支 | 說明 | 適用對象 |
|------|------|----------|
| **`main`** (目前分支) | Python 腳本格式 (.py)，可直接執行 | 開發者、進階使用者 |
| `jupyter-notebook` | Jupyter Notebook 格式，互動式學習體驗 | 學生、初學者、教學用途 |

### 💻 main 分支特色

此分支包含所有範例的 **Python 腳本 (.py)** 格式：

- **直接執行**：`python example_01_basic_chat.py`
- **適合開發**：可整合到其他專案中
- **命令列操作**：適合熟悉終端機的使用者

### 📁 Python 檔案列表

```
example_01_basic_chat.py        - 基本對話
example_02_multi_turn_chat.py   - 多輪對話（有記憶）
example_03_streaming.py         - 串流輸出
example_04_system_prompt.py     - 系統提示詞與角色扮演
example_05_code_assistant.py    - 程式碼助手
example_06_lmstudio_basic.py    - LM Studio 基本對話
example_07_lmstudio_openai.py   - 使用 OpenAI SDK
example_08_lmstudio_multi_turn.py - LM Studio 多輪對話
example_09_lmstudio_streaming.py  - LM Studio 串流輸出
example_10_list_models.py       - 列出可用模型
example_11_universal_chatbot.py - 通用聊天程式
example_12_simple_rag.py        - 簡易 RAG 系統
example_13_vector_rag.py        - 向量搜尋 RAG
example_14_document_qa.py       - 文件問答系統
example_15_prepare_dataset.py   - 準備 Fine-Tuning 資料
example_16_ollama_modelfile.py  - Ollama 自訂模型
example_17_data_augmentation.py - 資料增強
example_18_evaluate_finetune.py - 評估模型效果
example_19_evaluation_system.py - 完整評估系統
```

### 🎓 想要互動式學習？

切換到 `jupyter-notebook` 分支獲得 Jupyter Notebook 版本：
```bash
git checkout jupyter-notebook
```

---

## 目錄

### 基礎概念
- [什麼是大型語言模型 (LLM)？](#什麼是大型語言模型-llm)
- [什麼是 Ollama？](#什麼是-ollama)
- [什麼是 LM Studio？](#什麼是-lm-studio)
- [環境準備](#環境準備)

### Ollama 範例程式碼
- [範例 1：基本對話](#範例-1基本對話)
- [範例 2：多輪對話（有記憶的聊天）](#範例-2多輪對話有記憶的聊天)
- [範例 3：串流輸出（即時顯示）](#範例-3串流輸出即時顯示)
- [範例 4：設定系統提示詞（角色扮演）](#範例-4設定系統提示詞角色扮演)
- [範例 5：程式碼助手](#範例-5程式碼助手)

### LM Studio 範例程式碼
- [範例 6：LM Studio 基本對話](#範例-6lm-studio-基本對話使用-requests)
- [範例 7：LM Studio 使用 OpenAI 套件](#範例-7lm-studio-使用-openai-套件)
- [範例 8：LM Studio 多輪對話](#範例-8lm-studio-多輪對話)
- [範例 9：LM Studio 串流輸出](#範例-9lm-studio-串流輸出)
- [範例 10：列出可用模型](#範例-10列出-lm-studio-可用模型)
- [範例 11：通用聊天程式](#範例-11通用聊天程式支援-ollama-和-lm-studio)

### 重要概念
- [API 是什麼？](#api-是什麼)
- [JSON 是什麼？](#json-是什麼)
- [HTTP 請求是什麼？](#http-請求是什麼)
- [常見問題](#常見問題)

### 進階主題：RAG（檢索增強生成）
- [什麼是 RAG？](#什麼是-rag)
- [範例 12：簡易 RAG 系統](#範例-12簡易-rag-系統)
- [範例 13：向量搜尋 RAG](#範例-13使用向量搜尋的-rag)
- [範例 14：文件問答系統](#範例-14文件問答系統)

### 進階主題：Fine-Tuning（微調）
- [什麼是 Fine-Tuning？](#什麼是-fine-tuning)
- [範例 15：準備訓練資料集](#範例-15準備-fine-tuning-資料集)
- [範例 16：使用 Ollama 建立自訂模型](#範例-16使用-ollama-進行-fine-tuning)
- [範例 17：資料增強](#範例-17fine-tuning-資料增強)
- [範例 18：評估模型效果](#範例-18評估-fine-tuning-效果)

### Fine-Tuning 成效評估
- [為什麼要評估？](#為什麼要評估)
- [評估方法一：定量指標](#評估方法一定量指標用數字衡量)
- [評估方法二：定性評估](#評估方法二定性評估人工判斷)
- [評估方法三：A/B 測試](#評估方法三ab-測試)
- [範例 19：完整評估系統](#範例-19完整的模型評估系統)
- [監控訓練過程](#監控訓練過程)
- [評估檢查清單](#評估檢查清單)

### 附錄
- [延伸學習](#延伸學習)
- [授權](#授權)

---

## 什麼是大型語言模型 (LLM)？

大型語言模型就像一個讀過大量書籍的「超級學生」。它透過學習網路上的文章、書籍、程式碼等資料，學會了如何理解和生成人類語言。

**簡單比喻：**
- 想像你讀了 1000 本書後，能夠回答各種問題、寫文章、甚至幫人解決問題
- LLM 就是讀了「幾乎整個網路」的資料後，學會這些能力的程式

## 什麼是 Ollama？

Ollama 是一個讓你在自己電腦上運行 AI 模型的工具。就像你在電腦上安裝遊戲一樣，Ollama 讓你「安裝」和「運行」AI 模型。

**優點：**
- 免費使用
- 資料不會上傳到網路（隱私安全）
- 不需要網路也能使用
- 命令列操作，適合開發者

## 什麼是 LM Studio？

LM Studio 是另一個本地運行 AI 模型的工具，提供圖形化介面（GUI），更適合初學者使用。

**優點：**
- 圖形化介面，操作簡單直覺
- 支援 OpenAI 相容 API（可直接使用現有的 OpenAI 程式碼）
- 可以輕鬆切換不同模型
- 內建模型下載管理器

**Ollama vs LM Studio 比較：**

| 特點 | Ollama | LM Studio |
|------|--------|-----------|
| 介面 | 命令列 | 圖形化 |
| API 格式 | Ollama 專用 | OpenAI 相容 |
| 預設埠號 | 11434 | 1234 |
| 適合對象 | 開發者 | 初學者/一般使用者 |

---

## 環境準備

### 方法一：安裝 Ollama

1. 前往 [Ollama 官網](https://ollama.com) 下載並安裝
2. 開啟終端機，下載模型：
```bash
ollama pull gpt-oss:120b
```

### 方法二：安裝 LM Studio

1. 前往 [LM Studio 官網](https://lmstudio.ai) 下載並安裝
2. 開啟 LM Studio，在搜尋欄搜尋並下載想要的模型
3. 點擊左側「Local Server」圖示，啟動本地伺服器

### 安裝 Python 套件

```bash
pip install requests openai
```

---

## Ollama Python 程式碼範例

### 範例 1：基本對話

```python
"""
範例 1：與 AI 進行簡單對話
這是最基本的使用方式，就像跟 AI 聊天一樣
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
```

---

### 範例 2：多輪對話（有記憶的聊天）

```python
"""
範例 2：多輪對話
AI 會記住之前的對話內容，就像真正的聊天一樣
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
```

---

### 範例 3：串流輸出（即時顯示）

```python
"""
範例 3：串流輸出
像 ChatGPT 一樣，一個字一個字地顯示回應
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
```

---

### 範例 4：設定系統提示詞（角色扮演）

```python
"""
範例 4：設定系統提示詞
可以讓 AI 扮演特定角色，例如：老師、翻譯官、程式專家等
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
```

---

### 範例 5：程式碼助手

```python
"""
範例 5：程式碼助手
讓 AI 幫你解釋程式碼、找錯誤、或寫程式
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
```

---

## LM Studio Python 程式碼範例

LM Studio 使用 OpenAI 相容的 API，所以可以使用 `openai` 套件或 `requests` 直接呼叫。

### 範例 6：LM Studio 基本對話（使用 requests）

```python
"""
範例 6：使用 LM Studio 進行基本對話
LM Studio 使用 OpenAI 相容的 API 格式
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
```

---

### 範例 7：LM Studio 使用 OpenAI 套件

```python
"""
範例 7：使用 OpenAI 套件連接 LM Studio
這種方式的好處是：如果你之前用過 OpenAI API，程式碼幾乎不用改！
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
```

---

### 範例 8：LM Studio 多輪對話

```python
"""
範例 8：LM Studio 多輪對話
使用 OpenAI SDK 實現有記憶的對話
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
```

---

### 範例 9：LM Studio 串流輸出

```python
"""
範例 9：LM Studio 串流輸出
即時顯示 AI 的回應，像 ChatGPT 一樣一個字一個字出現
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
```

---

### 範例 10：列出 LM Studio 可用模型

```python
"""
範例 10：查看 LM Studio 中可用的模型
這個範例展示如何取得 LM Studio 目前載入的模型清單
"""

import requests

def list_models():
    """
    取得 LM Studio 中可用的模型清單
    """

    url = "http://localhost:1234/v1/models"

    response = requests.get(url)
    result = response.json()

    print("=== LM Studio 可用模型 ===")
    print("-" * 40)

    for model in result["data"]:
        print(f"• {model['id']}")

    return result["data"]


# 主程式
if __name__ == "__main__":
    models = list_models()
    print(f"\n共 {len(models)} 個模型可用")
```

---

### 範例 11：通用聊天程式（支援 Ollama 和 LM Studio）

```python
"""
範例 11：通用聊天程式
可以在 Ollama 和 LM Studio 之間切換
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
```

---

## 重要概念解釋

### API 是什麼？

API（Application Programming Interface）就像餐廳的菜單和點餐系統：

1. **你（程式）** = 顧客
2. **API** = 服務生 + 菜單
3. **Ollama** = 廚房

流程：
```
你的程式 → 透過 API 發送請求 → Ollama 處理 → 透過 API 返回結果 → 你的程式收到回應
```

### JSON 是什麼？

JSON 是一種資料格式，就像填寫表格一樣，有固定的格式：

```python
# 這是一個 JSON 格式的資料
{
    "name": "小明",      # 鍵: 值
    "age": 16,           # 可以是數字
    "hobbies": ["程式", "音樂"]  # 可以是清單
}
```

### HTTP 請求是什麼？

就像寄信一樣：
- **POST 請求**：寄出一封信，裡面有內容（你的問題）
- **回應**：收到回信（AI 的答案）

---

## 常見問題

### Q1: 為什麼 AI 回應很慢？

**原因：** 120B 模型非常大（1200 億參數），需要大量計算。

**解決方法：**
- 使用較小的模型（如 7B、13B）
- 確保電腦有足夠的記憶體和 GPU

### Q2: 出現「連線錯誤」怎麼辦？

**Ollama 解決步驟：**
1. 確認 Ollama 正在運行：在終端機輸入 `ollama list`
2. 確認模型已下載：`ollama pull gpt-oss:120b`
3. 重啟 Ollama 服務

**LM Studio 解決步驟：**
1. 確認 LM Studio 已開啟
2. 確認已啟動 Local Server（左側面板）
3. 確認有載入模型（模型名稱會顯示在上方）

### Q3: 如何讓 AI 回答更準確？

**技巧：**
1. 問題要具體明確
2. 提供足夠的背景資訊
3. 使用系統提示詞設定 AI 的角色

---

## 進階主題：RAG（檢索增強生成）

### 什麼是 RAG？

RAG（Retrieval-Augmented Generation，檢索增強生成）是一種讓 AI「查資料後再回答」的技術。

**生活比喻：**
- **沒有 RAG 的 AI**：像一個只靠記憶回答問題的學生，可能會記錯或不知道最新資訊
- **有 RAG 的 AI**：像一個可以翻課本、查筆記後再回答的學生，答案更準確

**RAG 的運作流程：**
```
使用者問問題 → 搜尋相關文件 → 把文件內容給 AI 參考 → AI 根據文件回答
```

**為什麼需要 RAG？**
1. AI 的知識有截止日期，無法知道最新資訊
2. AI 可能會「幻覺」（編造不存在的資訊）
3. 企業需要 AI 回答公司內部的專有知識

---

### 範例 12：簡易 RAG 系統

```python
"""
範例 12：簡易 RAG 系統
這個範例展示 RAG 的基本概念：先搜尋文件，再讓 AI 根據文件回答
"""

from openai import OpenAI

# 模擬的知識庫（實際應用中可能是資料庫或文件系統）
KNOWLEDGE_BASE = {
    "python": """
    Python 是一種高階程式語言，由 Guido van Rossum 於 1991 年創建。
    Python 的特點：
    - 語法簡潔易讀
    - 支援多種程式設計範式
    - 擁有豐富的第三方套件
    - 廣泛用於網頁開發、資料科學、人工智慧等領域
    """,
    "javascript": """
    JavaScript 是一種腳本語言，主要用於網頁開發。
    JavaScript 的特點：
    - 可在瀏覽器中直接執行
    - 支援事件驅動程式設計
    - 可用於前端和後端（Node.js）開發
    - 是網頁互動功能的核心技術
    """,
    "機器學習": """
    機器學習是人工智慧的一個分支，讓電腦從資料中學習規律。
    機器學習的類型：
    - 監督式學習：使用有標籤的資料訓練
    - 非監督式學習：從無標籤資料中發現模式
    - 強化學習：透過獎勵機制學習最佳策略
    常見應用：圖像辨識、語音辨識、推薦系統等
    """
}


def simple_search(query):
    """
    簡單的關鍵字搜尋
    在實際應用中，這裡會使用向量搜尋或全文搜尋引擎

    參數：
        query: 搜尋關鍵字

    回傳：
        找到的相關文件內容
    """
    results = []
    query_lower = query.lower()

    for keyword, content in KNOWLEDGE_BASE.items():
        if keyword.lower() in query_lower or query_lower in keyword.lower():
            results.append(content)

    return results


def rag_chat(question):
    """
    RAG 聊天函數
    先搜尋相關文件，再讓 AI 根據文件回答

    參數：
        question: 使用者的問題

    回傳：
        AI 的回答
    """

    # 步驟 1：搜尋相關文件
    retrieved_docs = simple_search(question)

    # 步驟 2：建立提示詞
    if retrieved_docs:
        context = "\n\n".join(retrieved_docs)
        prompt = f"""請根據以下參考資料回答問題。如果參考資料中沒有相關資訊，請說明你不確定。

參考資料：
{context}

問題：{question}

請用繁體中文回答："""
    else:
        prompt = f"""問題：{question}

注意：我找不到相關的參考資料，請根據你的知識回答，但要說明這是你的一般知識，不是來自特定文件。

請用繁體中文回答："""

    # 步驟 3：呼叫 AI
    client = OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="not-needed"
    )

    response = client.chat.completions.create(
        model="gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# 主程式
if __name__ == "__main__":
    print("=== 簡易 RAG 系統 ===")
    print("可以問關於 Python、JavaScript、機器學習的問題")
    print("輸入 'quit' 結束")
    print("-" * 50)

    while True:
        question = input("\n你的問題：")

        if question.lower() == "quit":
            break

        answer = rag_chat(question)
        print(f"\nAI 回答：{answer}")
```

---

### 範例 13：使用向量搜尋的 RAG

```python
"""
範例 13：使用向量搜尋的 RAG 系統
這個範例使用向量嵌入（Embedding）來搜尋最相關的文件

需要安裝：pip install numpy
"""

import numpy as np
from openai import OpenAI

# 初始化客戶端
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"
)

# 知識庫文件
DOCUMENTS = [
    "Python 是一種簡單易學的程式語言，適合初學者入門。",
    "JavaScript 是網頁開發的核心語言，可以讓網頁產生互動效果。",
    "機器學習讓電腦能從資料中學習，不需要明確的程式指令。",
    "深度學習是機器學習的一個分支，使用神經網路來處理複雜問題。",
    "自然語言處理（NLP）讓電腦能理解和生成人類語言。",
    "RAG 技術結合了資訊檢索和文字生成，提高 AI 回答的準確性。",
]

# 儲存文件的向量表示
document_embeddings = []


def get_embedding(text):
    """
    取得文字的向量表示（Embedding）

    向量嵌入是什麼？
    - 把文字轉換成一串數字（向量）
    - 意思相近的文字，向量也會相近
    - 這樣電腦就能「理解」文字的意義
    """
    response = client.embeddings.create(
        model="text-embedding-nomic-embed-text-v1.5",  # 嵌入模型
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(vec1, vec2):
    """
    計算兩個向量的餘弦相似度

    餘弦相似度是什麼？
    - 衡量兩個向量方向的相似程度
    - 值介於 -1 到 1 之間
    - 1 表示完全相同，0 表示無關，-1 表示完全相反
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def initialize_knowledge_base():
    """
    初始化知識庫：為所有文件計算向量
    """
    global document_embeddings
    print("正在初始化知識庫...")

    for doc in DOCUMENTS:
        embedding = get_embedding(doc)
        document_embeddings.append(embedding)

    print(f"已載入 {len(DOCUMENTS)} 份文件")


def vector_search(query, top_k=2):
    """
    向量搜尋：找出與問題最相關的文件

    參數：
        query: 使用者的問題
        top_k: 要返回的文件數量

    回傳：
        最相關的文件列表
    """
    # 取得問題的向量
    query_embedding = get_embedding(query)

    # 計算與每份文件的相似度
    similarities = []
    for i, doc_embedding in enumerate(document_embeddings):
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((similarity, DOCUMENTS[i]))

    # 按相似度排序，取前 k 個
    similarities.sort(reverse=True)
    return [doc for _, doc in similarities[:top_k]]


def rag_with_vector_search(question):
    """
    使用向量搜尋的 RAG 聊天
    """
    # 搜尋相關文件
    relevant_docs = vector_search(question, top_k=2)

    # 建立提示詞
    context = "\n".join([f"- {doc}" for doc in relevant_docs])

    prompt = f"""請根據以下參考資料回答問題：

參考資料：
{context}

問題：{question}

請用繁體中文簡潔回答："""

    # 呼叫 AI
    response = client.chat.completions.create(
        model="gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content, relevant_docs


# 主程式
if __name__ == "__main__":
    # 初始化知識庫
    initialize_knowledge_base()

    print("\n=== 向量搜尋 RAG 系統 ===")
    print("輸入 'quit' 結束")
    print("-" * 50)

    while True:
        question = input("\n你的問題：")

        if question.lower() == "quit":
            break

        answer, sources = rag_with_vector_search(question)

        print(f"\n找到的相關文件：")
        for i, doc in enumerate(sources, 1):
            print(f"  {i}. {doc[:50]}...")

        print(f"\nAI 回答：{answer}")
```

---

### 範例 14：文件問答系統

```python
"""
範例 14：文件問答系統
讀取文字檔案，讓 AI 回答關於文件內容的問題
"""

import os
from openai import OpenAI


def read_document(file_path):
    """
    讀取文件內容

    參數：
        file_path: 文件路徑

    回傳：
        文件內容
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_document(text, chunk_size=500, overlap=50):
    """
    將長文件切割成小段落

    為什麼要切割？
    - AI 有輸入長度限制
    - 小段落更容易精確搜尋
    - 可以只傳送相關的部分給 AI

    參數：
        text: 文件內容
        chunk_size: 每段的字數
        overlap: 段落之間重疊的字數（避免資訊被切斷）
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # 重疊部分

    return chunks


class DocumentQA:
    """
    文件問答系統類別
    """

    def __init__(self):
        self.client = OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="not-needed"
        )
        self.chunks = []

    def load_document(self, file_path):
        """載入文件"""
        text = read_document(file_path)
        self.chunks = chunk_document(text)
        print(f"已載入文件，共 {len(self.chunks)} 個段落")

    def load_text(self, text):
        """直接載入文字"""
        self.chunks = chunk_document(text)
        print(f"已載入文字，共 {len(self.chunks)} 個段落")

    def find_relevant_chunks(self, question, top_k=3):
        """
        找出與問題相關的段落（簡單的關鍵字匹配）
        """
        scored_chunks = []

        # 將問題拆成關鍵字
        keywords = question.lower().split()

        for chunk in self.chunks:
            score = 0
            chunk_lower = chunk.lower()

            # 計算每個關鍵字出現的次數
            for keyword in keywords:
                if keyword in chunk_lower:
                    score += chunk_lower.count(keyword)

            scored_chunks.append((score, chunk))

        # 排序並返回最相關的段落
        scored_chunks.sort(reverse=True)
        return [chunk for score, chunk in scored_chunks[:top_k] if score > 0]

    def ask(self, question):
        """
        提問並獲得回答
        """
        # 找出相關段落
        relevant_chunks = self.find_relevant_chunks(question)

        if not relevant_chunks:
            return "抱歉，我在文件中找不到相關資訊。"

        # 建立上下文
        context = "\n---\n".join(relevant_chunks)

        prompt = f"""你是一個文件問答助手。請根據以下文件內容回答問題。
如果文件中沒有相關資訊，請誠實說明。

文件內容：
{context}

問題：{question}

請用繁體中文回答："""

        response = self.client.chat.completions.create(
            model="gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content


# 主程式
if __name__ == "__main__":
    qa = DocumentQA()

    # 範例：載入一段說明文字
    sample_text = """
    人工智慧（Artificial Intelligence，簡稱 AI）是電腦科學的一個分支，
    致力於創造能夠執行通常需要人類智慧的任務的機器。這些任務包括學習、
    推理、問題解決、感知和語言理解。

    機器學習是人工智慧的一個子領域，專注於開發能夠從資料中學習的演算法。
    深度學習是機器學習的一個分支，使用多層神經網路來處理複雜的資料模式。

    自然語言處理（NLP）是 AI 的另一個重要領域，讓電腦能夠理解、解釋和
    生成人類語言。ChatGPT 就是一個著名的 NLP 應用。

    AI 的應用非常廣泛，包括：
    - 語音助手（如 Siri、Alexa）
    - 自動駕駛汽車
    - 醫療診斷輔助
    - 推薦系統（如 Netflix、YouTube）
    - 遊戲 AI
    """

    qa.load_text(sample_text)

    print("\n=== 文件問答系統 ===")
    print("輸入 'quit' 結束")
    print("-" * 50)

    while True:
        question = input("\n你的問題：")

        if question.lower() == "quit":
            break

        answer = qa.ask(question)
        print(f"\n回答：{answer}")
```

---

## 進階主題：Fine-Tuning（微調）

### 什麼是 Fine-Tuning？

Fine-Tuning（微調）是在已訓練好的模型基礎上，用特定資料進行額外訓練，讓模型更適合特定任務。

**生活比喻：**
- **原始模型**：像一個受過通識教育的學生，什麼都知道一點
- **Fine-Tuning**：像讓這個學生專攻某個領域（如法律、醫學），變成專家

**Fine-Tuning vs RAG 比較：**

| 特點 | Fine-Tuning | RAG |
|------|-------------|-----|
| 知識儲存 | 存在模型參數中 | 存在外部資料庫 |
| 更新知識 | 需要重新訓練 | 只需更新資料庫 |
| 計算資源 | 需要 GPU 訓練 | 只需推理資源 |
| 適用場景 | 改變模型行為風格 | 需要最新/專有知識 |
| 成本 | 較高 | 較低 |

**什麼時候用 Fine-Tuning？**
- 需要模型用特定風格/語氣回答
- 需要模型執行特定格式的任務
- 需要提高特定領域的準確性

**什麼時候用 RAG？**
- 需要最新資訊
- 需要引用來源
- 知識會頻繁更新

---

### 範例 15：準備 Fine-Tuning 資料集

```python
"""
範例 15：準備 Fine-Tuning 資料集
這個範例展示如何準備用於微調的訓練資料

Fine-Tuning 資料格式通常是「對話」形式：
- 輸入（問題/指令）
- 輸出（期望的回答）
"""

import json


def create_training_example(instruction, input_text, output):
    """
    建立一筆訓練資料

    參數：
        instruction: 任務指令
        input_text: 輸入內容（可為空）
        output: 期望的輸出

    回傳：
        格式化的訓練資料
    """
    if input_text:
        text = f"### 指令：\n{instruction}\n\n### 輸入：\n{input_text}\n\n### 回答：\n{output}"
    else:
        text = f"### 指令：\n{instruction}\n\n### 回答：\n{output}"

    return {"text": text}


def create_chat_example(system, user, assistant):
    """
    建立對話格式的訓練資料（OpenAI 格式）

    參數：
        system: 系統提示詞
        user: 使用者訊息
        assistant: AI 回答
    """
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ]
    }


# 範例：建立客服機器人的訓練資料
def create_customer_service_dataset():
    """
    建立客服機器人的訓練資料集
    """

    system_prompt = "你是一個友善的線上商店客服助手，專門回答關於訂單和產品的問題。"

    training_data = [
        create_chat_example(
            system_prompt,
            "我的訂單什麼時候會到？",
            "您好！一般訂單會在 3-5 個工作天內送達。如果您能提供訂單編號，我可以幫您查詢更準確的配送狀態。請問您的訂單編號是多少呢？"
        ),
        create_chat_example(
            system_prompt,
            "我想退貨",
            "沒問題！我們提供 7 天內無條件退貨服務。請確保商品保持原包裝且未使用。您可以透過以下步驟申請退貨：\n1. 登入會員中心\n2. 點選「我的訂單」\n3. 選擇要退貨的商品\n4. 填寫退貨原因\n\n請問需要我協助您其他事項嗎？"
        ),
        create_chat_example(
            system_prompt,
            "這個產品有保固嗎？",
            "是的，我們所有產品都提供原廠保固！保固期限依產品類型而定：\n- 電子產品：1 年保固\n- 家電產品：2 年保固\n- 配件類：6 個月保固\n\n請問您想了解哪個產品的保固資訊呢？"
        ),
        create_chat_example(
            system_prompt,
            "可以改寄送地址嗎？",
            "可以的！如果訂單尚未出貨，您可以修改寄送地址。請提供：\n1. 您的訂單編號\n2. 新的寄送地址\n\n我會盡快為您處理。不過如果訂單已經出貨，就需要聯繫物流公司進行轉寄囉。"
        ),
        create_chat_example(
            system_prompt,
            "付款失敗怎麼辦？",
            "別擔心！付款失敗可能有幾個原因：\n1. 信用卡額度不足\n2. 銀行拒絕交易\n3. 網路連線問題\n\n建議您可以：\n- 確認信用卡狀態正常\n- 嘗試使用其他付款方式\n- 稍後重新下單\n\n如果問題持續發生，請提供錯誤訊息，我可以進一步協助您！"
        ),
    ]

    return training_data


def save_dataset(data, filename):
    """
    儲存資料集為 JSONL 格式
    （每一行是一個 JSON 物件）
    """
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"已儲存 {len(data)} 筆資料到 {filename}")


# 主程式
if __name__ == "__main__":
    # 建立客服資料集
    dataset = create_customer_service_dataset()

    # 顯示資料集內容
    print("=== 訓練資料集預覽 ===\n")
    for i, item in enumerate(dataset[:2], 1):
        print(f"--- 範例 {i} ---")
        for msg in item["messages"]:
            print(f"{msg['role'].upper()}: {msg['content'][:100]}...")
        print()

    # 儲存資料集
    save_dataset(dataset, "customer_service_training.jsonl")
```

---

### 範例 16：使用 Ollama 進行 Fine-Tuning

```python
"""
範例 16：使用 Ollama 建立自訂模型
Ollama 支援透過 Modelfile 建立自訂模型

注意：這不是真正的 Fine-Tuning，而是透過系統提示詞來「定制」模型行為
真正的 Fine-Tuning 需要使用專門的訓練框架（如 Hugging Face、Axolotl 等）
"""

import subprocess
import os


def create_modelfile(base_model, system_prompt, model_name):
    """
    建立 Ollama Modelfile

    參數：
        base_model: 基礎模型名稱
        system_prompt: 系統提示詞
        model_name: 新模型名稱
    """

    modelfile_content = f'''FROM {base_model}

SYSTEM """
{system_prompt}
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
'''

    # 儲存 Modelfile
    modelfile_path = f"Modelfile_{model_name}"
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    print(f"已建立 Modelfile: {modelfile_path}")
    return modelfile_path


def create_ollama_model(modelfile_path, model_name):
    """
    使用 Ollama 建立模型

    參數：
        modelfile_path: Modelfile 路徑
        model_name: 新模型名稱
    """
    print(f"正在建立模型 {model_name}...")

    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", modelfile_path],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"模型 {model_name} 建立成功！")
            print("使用方式：ollama run " + model_name)
        else:
            print(f"建立失敗：{result.stderr}")

    except FileNotFoundError:
        print("找不到 ollama 指令，請確認 Ollama 已安裝")


# 範例：建立一個程式教學助手
if __name__ == "__main__":
    # 定義系統提示詞
    system_prompt = """你是一位親切的程式設計教師，專門教導初學者學習程式。

你的特點：
- 使用簡單易懂的語言解釋概念
- 提供大量的程式碼範例
- 用生活中的例子來比喻抽象概念
- 鼓勵學生，保持正向態度
- 如果學生犯錯，耐心解釋錯誤原因
- 使用繁體中文回答

回答格式：
1. 先簡單解釋概念
2. 提供程式碼範例
3. 解釋程式碼的每個部分
4. 給予練習建議"""

    # 建立 Modelfile
    modelfile_path = create_modelfile(
        base_model="gpt-oss:120b",
        system_prompt=system_prompt,
        model_name="programming-teacher"
    )

    print("\n" + "=" * 50)
    print("Modelfile 內容預覽：")
    print("=" * 50)
    with open(modelfile_path, "r", encoding="utf-8") as f:
        print(f.read())

    print("\n要建立模型，請執行以下指令：")
    print(f"  ollama create programming-teacher -f {modelfile_path}")
    print("\n建立完成後，使用以下指令執行：")
    print("  ollama run programming-teacher")
```

---

### 範例 17：Fine-Tuning 資料增強

```python
"""
範例 17：Fine-Tuning 資料增強
使用 AI 來幫助生成更多訓練資料

資料增強是什麼？
- 用少量的種子資料，生成更多類似的訓練資料
- 可以增加訓練資料的多樣性
- 提高模型的泛化能力
"""

from openai import OpenAI
import json


client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"
)


def augment_qa_pair(question, answer, num_variations=3):
    """
    生成問答對的變體

    參數：
        question: 原始問題
        answer: 原始回答
        num_variations: 要生成的變體數量

    回傳：
        問答對變體列表
    """

    prompt = f"""請根據以下問答對，生成 {num_variations} 個類似但不同的問答變體。
保持回答的核心資訊相同，但改變問法和表達方式。

原始問題：{question}
原始回答：{answer}

請用以下 JSON 格式輸出：
[
    {{"question": "變體問題1", "answer": "變體回答1"}},
    {{"question": "變體問題2", "answer": "變體回答2"}},
    ...
]

只輸出 JSON，不要其他文字："""

    response = client.chat.completions.create(
        model="gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        variations = json.loads(response.choices[0].message.content)
        return variations
    except json.JSONDecodeError:
        print("解析回應失敗")
        return []


def augment_dataset(seed_data, variations_per_item=2):
    """
    增強整個資料集

    參數：
        seed_data: 種子資料（問答對列表）
        variations_per_item: 每筆資料生成的變體數

    回傳：
        增強後的資料集
    """
    augmented_data = []

    for item in seed_data:
        # 加入原始資料
        augmented_data.append(item)

        # 生成變體
        variations = augment_qa_pair(
            item["question"],
            item["answer"],
            variations_per_item
        )

        augmented_data.extend(variations)

    return augmented_data


# 主程式
if __name__ == "__main__":
    # 種子資料
    seed_data = [
        {
            "question": "Python 的 list 和 tuple 有什麼差別？",
            "answer": "list 是可變的（mutable），可以新增、修改、刪除元素；tuple 是不可變的（immutable），建立後就不能改變。tuple 的效能比 list 好，適合用於不需要修改的資料。"
        },
        {
            "question": "什麼是迴圈？",
            "answer": "迴圈是讓程式重複執行某段程式碼的結構。Python 有兩種迴圈：for 迴圈用於遍歷序列，while 迴圈用於條件判斷。迴圈可以減少重複的程式碼，提高效率。"
        }
    ]

    print("=== 資料增強示範 ===\n")
    print(f"原始資料：{len(seed_data)} 筆")
    print("-" * 50)

    # 增強資料
    augmented = augment_dataset(seed_data, variations_per_item=2)

    print(f"\n增強後資料：{len(augmented)} 筆")
    print("-" * 50)

    # 顯示結果
    for i, item in enumerate(augmented, 1):
        print(f"\n[{i}] Q: {item.get('question', 'N/A')[:50]}...")
        print(f"    A: {item.get('answer', 'N/A')[:50]}...")
```

---

### 範例 18：評估 Fine-Tuning 效果

```python
"""
範例 18：評估模型效果
比較原始模型和微調後模型的回答品質
"""

from openai import OpenAI
import json


client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"
)


def get_response(model, question, system_prompt=None):
    """
    取得模型回應
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    return response.choices[0].message.content


def evaluate_response(question, response, criteria):
    """
    使用 AI 評估回答品質

    參數：
        question: 原始問題
        response: 模型回答
        criteria: 評估標準

    回傳：
        評估結果（1-5 分）
    """

    prompt = f"""請評估以下回答的品質，給予 1-5 分的評分。

問題：{question}

回答：{response}

評估標準：
{criteria}

請用以下 JSON 格式回答：
{{"score": <1-5的數字>, "reason": "評分原因"}}

只輸出 JSON："""

    eval_response = client.chat.completions.create(
        model="gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        result = json.loads(eval_response.choices[0].message.content)
        return result
    except:
        return {"score": 0, "reason": "評估失敗"}


def compare_models(test_questions, model1, model2, system_prompt=None):
    """
    比較兩個模型的表現
    """
    criteria = """
    - 準確性：回答是否正確
    - 完整性：是否涵蓋重要資訊
    - 清晰度：是否容易理解
    - 實用性：是否提供有用的範例或建議
    """

    results = []

    for question in test_questions:
        print(f"\n問題：{question}")
        print("-" * 40)

        # 取得兩個模型的回答
        response1 = get_response(model1, question, system_prompt)
        response2 = get_response(model2, question, system_prompt)

        # 評估兩個回答
        eval1 = evaluate_response(question, response1, criteria)
        eval2 = evaluate_response(question, response2, criteria)

        print(f"\n模型 1 ({model1})：")
        print(f"  回答：{response1[:100]}...")
        print(f"  評分：{eval1.get('score', 'N/A')}/5")

        print(f"\n模型 2 ({model2})：")
        print(f"  回答：{response2[:100]}...")
        print(f"  評分：{eval2.get('score', 'N/A')}/5")

        results.append({
            "question": question,
            "model1_score": eval1.get("score", 0),
            "model2_score": eval2.get("score", 0)
        })

    return results


# 主程式
if __name__ == "__main__":
    # 測試問題
    test_questions = [
        "如何在 Python 中讀取 CSV 檔案？",
        "解釋什麼是 API",
        "for 迴圈和 while 迴圈有什麼差別？"
    ]

    print("=== 模型比較評估 ===")

    # 比較有無系統提示詞的差異
    results = compare_models(
        test_questions,
        model1="gpt-oss-120b",
        model2="gpt-oss-120b",
        system_prompt="你是一位專業的程式設計教師，用簡單易懂的方式回答問題。"
    )

    # 統計結果
    print("\n" + "=" * 50)
    print("評估總結")
    print("=" * 50)

    avg1 = sum(r["model1_score"] for r in results) / len(results)
    avg2 = sum(r["model2_score"] for r in results) / len(results)

    print(f"模型 1 平均分數：{avg1:.2f}")
    print(f"模型 2 平均分數：{avg2:.2f}")
```

---

## Fine-Tuning 成效評估方法

### 為什麼要評估？

Fine-Tuning 後，你需要知道：
- 模型有沒有變好？
- 好了多少？
- 有沒有副作用（例如其他能力變差）？

**生活比喻：** 就像學生補習後要考試，看看成績有沒有進步。

---

### 評估方法一：定量指標（用數字衡量）

| 指標 | 說明 | 數值意義 |
|------|------|----------|
| **Loss（損失值）** | 模型預測與正確答案的差距 | 越低越好 |
| **Perplexity（困惑度）** | 模型對文字的預測信心 | 越低越好 |
| **Accuracy（準確率）** | 正確回答的比例 | 越高越好 |
| **F1 Score** | 精確率與召回率的平衡 | 越高越好（0-1）|
| **BLEU Score** | 生成文字與標準答案的相似度 | 越高越好（0-100）|

---

### 評估方法二：定性評估（人工判斷）

```
┌─────────────────────────────────────────────────┐
│  人工評估檢查清單                                  │
├─────────────────────────────────────────────────┤
│  □ 回答準確性 - 資訊是否正確無誤？                 │
│  □ 風格一致性 - 語氣和格式是否符合期望？           │
│  □ 任務完成度 - 是否完整回答問題？                 │
│  □ 流暢度     - 文字是否自然通順？                 │
│  □ 安全性     - 是否避免不當或有害內容？           │
│  □ 創造性     - 回答是否有見解而非死板？           │
└─────────────────────────────────────────────────┘
```

---

### 評估方法三：A/B 測試

比較原始模型和 Fine-Tuned 模型的表現：

```
        相同的問題
            │
    ┌───────┴───────┐
    ▼               ▼
┌────────┐    ┌────────┐
│ 原始   │    │ 微調後 │
│ 模型   │    │ 模型   │
└────────┘    └────────┘
    │               │
    ▼               ▼
  回答 A          回答 B
    │               │
    └───────┬───────┘
            ▼
        比較評分
```

---

### 範例 19：完整的模型評估系統

```python
"""
範例 19：完整的 Fine-Tuning 評估系統
包含多種評估指標和視覺化結果
"""

from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"
)


class ModelEvaluator:
    """
    模型評估器
    用於評估 Fine-Tuning 的效果
    """

    def __init__(self, base_model, finetuned_model=None):
        """
        初始化評估器

        參數：
            base_model: 原始模型名稱
            finetuned_model: 微調後模型名稱（可選）
        """
        self.base_model = base_model
        self.finetuned_model = finetuned_model
        self.results = []

    def get_response(self, model, question, system_prompt=None):
        """取得模型回應"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content

    def score_response(self, question, response, criteria):
        """
        使用 AI 評分回答品質（1-5 分）
        """
        prompt = f"""請評估以下回答的品質，針對每個標準給予 1-5 分。

問題：{question}

回答：{response}

評估標準：
{criteria}

請用以下 JSON 格式回答（只輸出 JSON）：
{{
    "accuracy": <1-5>,
    "completeness": <1-5>,
    "clarity": <1-5>,
    "usefulness": <1-5>,
    "overall": <1-5>,
    "comment": "簡短評語"
}}"""

        eval_response = client.chat.completions.create(
            model=self.base_model,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            return json.loads(eval_response.choices[0].message.content)
        except:
            return {"overall": 0, "comment": "評估失敗"}

    def evaluate_single(self, question, expected_answer=None, system_prompt=None):
        """
        評估單一問題
        """
        criteria = """
        - accuracy（準確性）：資訊是否正確
        - completeness（完整性）：是否涵蓋所有重點
        - clarity（清晰度）：是否容易理解
        - usefulness（實用性）：是否有幫助
        - overall（整體）：綜合評分
        """

        result = {"question": question}

        # 評估原始模型
        base_response = self.get_response(
            self.base_model, question, system_prompt
        )
        base_score = self.score_response(question, base_response, criteria)
        result["base"] = {
            "response": base_response,
            "scores": base_score
        }

        # 如果有微調模型，也進行評估
        if self.finetuned_model:
            ft_response = self.get_response(
                self.finetuned_model, question, system_prompt
            )
            ft_score = self.score_response(question, ft_response, criteria)
            result["finetuned"] = {
                "response": ft_response,
                "scores": ft_score
            }

        self.results.append(result)
        return result

    def evaluate_batch(self, questions, system_prompt=None):
        """
        批次評估多個問題
        """
        print(f"開始評估 {len(questions)} 個問題...\n")

        for i, q in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] 評估中: {q[:30]}...")
            self.evaluate_single(q, system_prompt=system_prompt)

        return self.get_summary()

    def get_summary(self):
        """
        取得評估摘要
        """
        if not self.results:
            return "尚無評估結果"

        summary = {
            "total_questions": len(self.results),
            "base_model": {
                "avg_overall": 0,
                "avg_accuracy": 0,
                "avg_clarity": 0
            }
        }

        # 計算原始模型平均分數
        base_scores = [r["base"]["scores"] for r in self.results]
        summary["base_model"]["avg_overall"] = sum(
            s.get("overall", 0) for s in base_scores
        ) / len(base_scores)
        summary["base_model"]["avg_accuracy"] = sum(
            s.get("accuracy", 0) for s in base_scores
        ) / len(base_scores)
        summary["base_model"]["avg_clarity"] = sum(
            s.get("clarity", 0) for s in base_scores
        ) / len(base_scores)

        # 如果有微調模型的結果
        if self.finetuned_model and "finetuned" in self.results[0]:
            summary["finetuned_model"] = {"avg_overall": 0, "avg_accuracy": 0, "avg_clarity": 0}
            ft_scores = [r["finetuned"]["scores"] for r in self.results]
            summary["finetuned_model"]["avg_overall"] = sum(
                s.get("overall", 0) for s in ft_scores
            ) / len(ft_scores)
            summary["finetuned_model"]["avg_accuracy"] = sum(
                s.get("accuracy", 0) for s in ft_scores
            ) / len(ft_scores)
            summary["finetuned_model"]["avg_clarity"] = sum(
                s.get("clarity", 0) for s in ft_scores
            ) / len(ft_scores)

            # 計算改善幅度
            summary["improvement"] = {
                "overall": summary["finetuned_model"]["avg_overall"] - summary["base_model"]["avg_overall"],
                "accuracy": summary["finetuned_model"]["avg_accuracy"] - summary["base_model"]["avg_accuracy"],
            }

        return summary

    def print_report(self):
        """
        印出評估報告
        """
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("📊 Fine-Tuning 評估報告")
        print("=" * 60)

        print(f"\n📝 評估問題數：{summary['total_questions']}")

        print(f"\n🔵 原始模型 ({self.base_model})：")
        print(f"   整體評分：{summary['base_model']['avg_overall']:.2f}/5")
        print(f"   準確性：  {summary['base_model']['avg_accuracy']:.2f}/5")
        print(f"   清晰度：  {summary['base_model']['avg_clarity']:.2f}/5")

        if "finetuned_model" in summary:
            print(f"\n🟢 微調模型 ({self.finetuned_model})：")
            print(f"   整體評分：{summary['finetuned_model']['avg_overall']:.2f}/5")
            print(f"   準確性：  {summary['finetuned_model']['avg_accuracy']:.2f}/5")
            print(f"   清晰度：  {summary['finetuned_model']['avg_clarity']:.2f}/5")

            print(f"\n📈 改善幅度：")
            imp = summary["improvement"]
            overall_pct = (imp["overall"] / summary["base_model"]["avg_overall"]) * 100 if summary["base_model"]["avg_overall"] > 0 else 0
            print(f"   整體：{imp['overall']:+.2f} ({overall_pct:+.1f}%)")
            print(f"   準確性：{imp['accuracy']:+.2f}")

        print("\n" + "=" * 60)


# 主程式
if __name__ == "__main__":
    # 建立評估器
    evaluator = ModelEvaluator(
        base_model="gpt-oss-120b",
        # finetuned_model="my-finetuned-model"  # 如果有微調模型
    )

    # 測試問題集
    test_questions = [
        "什麼是變數？請用簡單的方式解釋。",
        "Python 中 list 和 dictionary 有什麼差別？",
        "如何處理程式中的錯誤？",
        "什麼是遞迴？請舉例說明。",
        "解釋什麼是 API，以及為什麼要用它。"
    ]

    # 執行評估
    evaluator.evaluate_batch(
        test_questions,
        system_prompt="你是一位程式設計教師，用簡單的方式回答問題。"
    )

    # 印出報告
    evaluator.print_report()
```

---

### 監控訓練過程

訓練 Fine-Tuning 時，要注意這些警告信號：

```
✅ 正常情況：
   Training Loss:    ████████░░ → 逐漸下降
   Validation Loss:  ████████░░ → 跟著下降

⚠️ 過擬合 (Overfitting)：
   Training Loss:    ██████████ → 持續下降
   Validation Loss:  ████░░░░░░ → 開始上升 ❌

   症狀：訓練資料表現好，新資料表現差
   解法：Early Stopping、增加資料、Dropout

⚠️ 欠擬合 (Underfitting)：
   Training Loss:    ████░░░░░░ → 下降緩慢
   Validation Loss:  ████░░░░░░ → 都很高 ❌

   症狀：訓練和測試都表現不好
   解法：增加訓練時間、調整學習率
```

---

### 評估檢查清單

在宣布 Fine-Tuning 成功前，確認以下項目：

```
□ 1. 測試集獨立
     └── 測試資料沒有參與訓練

□ 2. 基準線比較
     └── 與原始模型比較，確認有改善

□ 3. 多維度評估
     └── 不只看一個指標，綜合評估

□ 4. 邊界案例測試
     └── 測試困難、特殊的問題

□ 5. 人工抽查
     └── 隨機抽 50-100 個回答人工審核

□ 6. 副作用檢查
     └── 確認其他能力沒有退步

□ 7. 實際場景測試
     └── 在真實使用情境中測試
```

---

### 常見評估錯誤

| 錯誤 | 問題 | 正確做法 |
|------|------|----------|
| 用訓練資料測試 | 會得到虛假的好成績 | 保留 20% 資料作測試 |
| 只看單一指標 | 評估片面不完整 | 多指標綜合評估 |
| 忽略邊界案例 | 漏掉極端情況 | 加入困難測試案例 |
| 不設基準線 | 無法量化改善程度 | 先測原始模型表現 |
| 樣本太少 | 結果不可靠 | 至少 50-100 個測試樣本 |

---

## 延伸學習

1. **Ollama 官方文件**：https://ollama.com
2. **LM Studio 官方網站**：https://lmstudio.ai
3. **OpenAI Python SDK**：https://github.com/openai/openai-python
4. **Python requests 教學**：https://docs.python-requests.org
5. **JSON 格式介紹**：https://www.json.org

---

## 授權

本教學內容採用 MIT 授權，歡迎自由使用與修改。
