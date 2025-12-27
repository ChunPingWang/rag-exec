"""
範例 19：完整的 Fine-Tuning 評估系統
包含多種評估指標和視覺化結果

使用方式：python example_19_evaluation_system.py
需要：LM Studio 運行中，已載入模型，Local Server 已啟動
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
        print("Fine-Tuning 評估報告")
        print("=" * 60)

        print(f"\n評估問題數：{summary['total_questions']}")

        print(f"\n原始模型 ({self.base_model})：")
        print(f"   整體評分：{summary['base_model']['avg_overall']:.2f}/5")
        print(f"   準確性：  {summary['base_model']['avg_accuracy']:.2f}/5")
        print(f"   清晰度：  {summary['base_model']['avg_clarity']:.2f}/5")

        if "finetuned_model" in summary:
            print(f"\n微調模型 ({self.finetuned_model})：")
            print(f"   整體評分：{summary['finetuned_model']['avg_overall']:.2f}/5")
            print(f"   準確性：  {summary['finetuned_model']['avg_accuracy']:.2f}/5")
            print(f"   清晰度：  {summary['finetuned_model']['avg_clarity']:.2f}/5")

            print(f"\n改善幅度：")
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
