
import random
import csv
from pathlib import Path

# Reproducible dataset generation
random.seed(42)

# Each FAQ has its original question and two paraphrases.
# All paraphrases should express exactly the same intent.

faq_data = [
    (1, "How can I track my order?", [
        "Where is my order right now?",
        "How do I check my package tracking status?"
    ]),
    (2, "How long does a refund take?", [
        "When will I get my money back?",
        "How many days does it take to receive a refund?"
    ]),
    (3, "Can I cancel my order?", [
        "How can I cancel a purchase?",
        "Is it possible to cancel my order?"
    ]),
    (4, "What is your return policy?", [
        "How can I return a product?",
        "What are the rules for returning an item?"
    ]),
    (5, "How can I change my delivery address?", [
        "Can I update the address for my order?",
        "How do I correct my shipping address?"
    ]),
    (6, "What payment methods are accepted?", [
        "Which payment options can I use?",
        "Can I pay using UPI or a card?"
    ]),
    (7, "What should I do if my order is delayed?", [
        "Why hasn't my order arrived on time?",
        "What can I do about a late delivery?"
    ]),
    (8, "Can I exchange a product?", [
        "How do I replace an item with another one?",
        "Is product exchange available?"
    ]),
    (9, "What if I receive a damaged product?", [
        "What should I do if my item arrives broken?",
        "How can I report a damaged delivery?"
    ]),
    (10, "How do I contact customer support?", [
        "How can I reach your support team?",
        "Where can I get help from customer service?"
    ]),
    (11, "Can I pay using cash on delivery?", [
        "Do you offer cash on delivery?",
        "Can I pay in cash when my package arrives?"
    ]),
    (12, "Why was my payment declined?", [
        "Why did my payment fail?",
        "What caused my transaction to be rejected?"
    ]),
    (13, "Will I receive a refund for a cancelled order?", [
        "Do I get my money back after cancelling an order?",
        "Is a cancelled prepaid order refundable?"
    ]),
    (14, "How can I download my invoice?", [
        "Where can I get my purchase invoice?",
        "How do I save my order bill?"
    ]),
    (15, "Is free shipping available?", [
        "Do you charge delivery fees?",
        "Can I get my order delivered without shipping charges?"
    ]),
    (16, "Can I return a product after opening it?", [
        "Am I allowed to return an item once it is opened?",
        "Can opened products still be returned?"
    ]),
    (17, "What happens if I miss my delivery?", [
        "What if I am not home when my order arrives?",
        "Will the delivery person try again if I miss them?"
    ]),
    (18, "How do I reset my password?", [
        "I forgot my password. How can I change it?",
        "How can I create a new account password?"
    ]),
    (19, "Can I place an order without creating an account?", [
        "Can I buy something as a guest?",
        "Is account registration compulsory to place an order?"
    ]),
    (20, "How can I update my phone number?", [
        "Where can I change my registered mobile number?",
        "How do I edit my contact number?"
    ]),
    (21, "Why is my refund delayed?", [
        "Why haven't I received my refund yet?",
        "What is causing my refund to take longer?"
    ]),
    (22, "Can I change the size of my product?", [
        "How can I request a different size?",
        "Is it possible to exchange my item for another size?"
    ]),
    (23, "Do you deliver on weekends?", [
        "Can my order arrive on Saturday or Sunday?",
        "Are weekend deliveries available?"
    ]),
    (24, "How can I report a missing item?", [
        "What should I do if something is missing from my package?",
        "How do I complain about an item missing from my order?"
    ]),
    (25, "Can I use multiple discount coupons?", [
        "Can I apply two coupons to one order?",
        "Is it possible to combine discount codes?"
    ]),
    (26, "How do I delete my account?", [
        "I want to permanently remove my account.",
        "How can I request account deletion?"
    ]),
    (27, "Are online payments secure?", [
        "Is it safe to pay on your website?",
        "How secure are your online transactions?"
    ]),
    (28, "Can I modify my order after placing it?", [
        "Can I make changes to an order I already placed?",
        "Is it possible to edit my order after checkout?"
    ]),
    (29, "What if I receive the wrong product?", [
        "What should I do if the delivered item is incorrect?",
        "How can I report receiving a different product?"
    ]),
    (30, "How can I check my refund status?", [
        "Where can I see the progress of my refund?",
        "How do I find out whether my refund has been processed?"
    ])
]

# Store original FAQ questions by ID
faq_questions = {
    faq_id: question
    for faq_id, question, variations in faq_data
}

positive_pairs = []
negative_pairs = []

# Generate positive and negative pairs
for faq_id, original_question, variations in faq_data:

    for variation in variations:

        # Positive: same intent
        positive_pairs.append(
            (original_question, variation, 1)
        )

        # Negative: different FAQ intent
        other_ids = [
            other_id
            for other_id in faq_questions
            if other_id != faq_id
        ]

        selected_negative_ids = random.sample(other_ids, 2)

        for negative_id in selected_negative_ids:
            negative_question = faq_questions[negative_id]

            negative_pairs.append(
                (variation, negative_question, 0)
            )

# Combine and shuffle the dataset
training_pairs = positive_pairs + negative_pairs
random.shuffle(training_pairs)

# Save dataset as CSV
output_file = Path(__file__).parent / "training_pairs.csv"

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow(["sentence1", "sentence2", "label"])
    writer.writerows(training_pairs)

# Display dataset statistics
print("\n========== TRAINING DATASET ==========")
print("Original FAQs:", len(faq_data))
print("Positive pairs:", len(positive_pairs))
print("Negative pairs:", len(negative_pairs))
print("Total training pairs:", len(training_pairs))
print("Saved to:", output_file)

print("\n========== SAMPLE PAIRS ==========")

for pair in training_pairs[:5]:
    label = "SIMILAR" if pair[2] == 1 else "DIFFERENT"

    print(f"\nLabel: {label}")
    print("Sentence 1:", pair[0])
    print("Sentence 2:", pair[1])
