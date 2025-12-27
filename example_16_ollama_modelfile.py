"""
範例 16：使用 Ollama 建立自訂模型
Ollama 支援透過 Modelfile 建立自訂模型

注意：這不是真正的 Fine-Tuning，而是透過系統提示詞來「定制」模型行為
真正的 Fine-Tuning 需要使用專門的訓練框架（如 Hugging Face、Axolotl 等）

使用方式：python example_16_ollama_modelfile.py
需要：已安裝 Ollama
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
