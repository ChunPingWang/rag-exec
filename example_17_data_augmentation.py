"""
範例 17：Fine-Tuning 資料增強
使用 AI 來幫助生成更多訓練資料

資料增強是什麼？
- 用少量的種子資料，生成更多類似的訓練資料
- 可以增加訓練資料的多樣性
- 提高模型的泛化能力

使用方式：python example_17_data_augmentation.py
需要：LM Studio 運行中，已載入模型，Local Server 已啟動
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
