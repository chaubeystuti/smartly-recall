
import json
import csv
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent

FAQ_FILE = BASE_DIR / "faqs.json"
EVAL_FILE = BASE_DIR / "evaluation_data.json"
OUTPUT_FILE = BASE_DIR / "comparison_results.csv"

THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55,
              0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

# Load data
with open(FAQ_FILE, encoding="utf-8") as f:
    faqs = json.load(f)

with open(EVAL_FILE, encoding="utf-8") as f:
    evaluation_data = json.load(f)

faq_questions = [faq["question"] for faq in faqs]
faq_ids = [faq["id"] for faq in faqs]

eval_questions = [item["question"] for item in evaluation_data]
expected_ids = [item["expectedFaqId"] for item in evaluation_data]

positive_count = sum(x is not None for x in expected_ids)
negative_count = len(expected_ids) - positive_count


def evaluate_model(model_name, model_path):

    print(f"\nLoading {model_name}...")

    model = SentenceTransformer(model_path)

    faq_embeddings = model.encode(
        faq_questions,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=16
    )

    query_embeddings = model.encode(
        eval_questions,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=16
    )

    # Cosine similarity because embeddings are normalized
    similarities = query_embeddings @ faq_embeddings.T

    predictions = np.argmax(similarities, axis=1)
    scores = np.max(similarities, axis=1)

    results = []

    for threshold in THRESHOLDS:

        correct = 0
        wrong_mapping = 0
        false_positive = 0
        misses = 0

        for i, expected_id in enumerate(expected_ids):

            predicted_id = faq_ids[predictions[i]]
            score = float(scores[i])

            if expected_id is None:
                if score >= threshold:
                    false_positive += 1

            elif score >= threshold:
                if predicted_id == expected_id:
                    correct += 1
                else:
                    wrong_mapping += 1

            else:
                misses += 1

        accepted = correct + wrong_mapping + false_positive

        precision = (
            correct / accepted * 100 if accepted else 0
        )

        recall = (
            correct / positive_count * 100
            if positive_count else 0
        )

        fpr = (
            false_positive / negative_count * 100
            if negative_count else 0
        )

        results.append({
            "Model": model_name,
            "Threshold": threshold,
            "Correct": correct,
            "Wrong Mapping": wrong_mapping,
            "False Positives": false_positive,
            "Misses": misses,
            "Precision": round(precision, 2),
            "Recall": round(recall, 2),
            "FPR": round(fpr, 2)
        })

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


print("========== MODEL COMPARISON ==========")
print("Evaluation questions:", len(evaluation_data))
print("FAQ anchors:", len(faqs))
print("Positive questions:", positive_count)
print("Negative questions:", negative_count)

original_results = evaluate_model(
    "Original",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

finetuned_results = evaluate_model(
    "Fine-tuned",
    str(BASE_DIR / "fine_tuned_model")
)

all_results = original_results + finetuned_results

# Display results
print("\n========== COMPARISON RESULTS ==========")

for model_name in ["Original", "Fine-tuned"]:

    print(f"\n--- {model_name} Model ---")
    print(
        "Threshold | Correct | Wrong | False Pos | Misses | "
        "Precision | Recall | FPR"
    )

    for row in all_results:

        if row["Model"] == model_name:
            print(
                f'{row["Threshold"]:.2f}      | '
                f'{row["Correct"]:7} | '
                f'{row["Wrong Mapping"]:5} | '
                f'{row["False Positives"]:9} | '
                f'{row["Misses"]:6} | '
                f'{row["Precision"]:9.2f}% | '
                f'{row["Recall"]:6.2f}% | '
                f'{row["FPR"]:5.2f}%'
            )

# Save results
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=all_results[0].keys()
    )

    writer.writeheader()
    writer.writerows(all_results)

print("\nResults saved to:", OUTPUT_FILE)
print("\n========== EVALUATION COMPLETE ==========")
