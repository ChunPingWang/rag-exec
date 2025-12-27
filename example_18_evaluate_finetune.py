"""
範例 18：評估模型效果
比較原始模型和微調後模型的回答品質

使用方式：python example_18_evaluate_finetune.py
需要：LM Studio 運行中，已載入模型，Local Server 已啟動
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
