"""
範例 10：查看 LM Studio 中可用的模型
這個範例展示如何取得 LM Studio 目前載入的模型清單

使用方式：python example_10_list_models.py
需要：LM Studio 運行中，Local Server 已啟動
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
