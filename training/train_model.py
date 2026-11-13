
import csv
import random
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.sentence_transformer.losses import OnlineContrastiveLoss

# Paths
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "training_pairs.csv"
OUTPUT_DIR = BASE_DIR / "fine_tuned_model"

# Reproducibility
random.seed(42)
torch.manual_seed(42)

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

print("\n========== MODEL TRAINING ==========")
print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# Load training pairs
examples = []

with open(DATA_FILE, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        example = InputExample(
            texts=[
                row["sentence1"],
                row["sentence2"]
            ],
            label=float(row["label"])
        )

        examples.append(example)

print("Training pairs loaded:", len(examples))

# Shuffle examples
random.shuffle(examples)

# Create DataLoader
train_dataloader = DataLoader(
    examples,
    shuffle=True,
    batch_size=8,
    num_workers=0
)

# Load original MiniLM model
print("\nLoading original MiniLM model...")

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device=device
)

print("Model loaded successfully!")

# Contrastive loss
train_loss = OnlineContrastiveLoss(
    model=model,
    margin=0.5
)

print("\nStarting fine-tuning...")

# Training configuration
epochs = 2

warmup_steps = max(
    1,
    int(len(train_dataloader) * epochs * 0.1)
)

model.fit(
    train_objectives=[
        (train_dataloader, train_loss)
    ],
    epochs=epochs,
    warmup_steps=warmup_steps,
    optimizer_params={
        "lr": 2e-5
    },
    output_path=str(OUTPUT_DIR),
    show_progress_bar=True
)

print("\n========== TRAINING COMPLETE ==========")
print("Fine-tuned model saved at:")
print(OUTPUT_DIR)
