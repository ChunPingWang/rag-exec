"""
範例 15：準備 Fine-Tuning 資料集
這個範例展示如何準備用於微調的訓練資料

Fine-Tuning 資料格式通常是「對話」形式：
- 輸入（問題/指令）
- 輸出（期望的回答）

使用方式：python example_15_prepare_dataset.py
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
