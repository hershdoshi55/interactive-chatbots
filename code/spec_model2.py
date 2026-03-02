import random
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    T5TokenizerFast,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

EXTRA_CLAUSES = [
    "with additional details for clarity.",
    "providing more context for better understanding.",
    "including a brief explanation.",
    "which may require further elaboration.",
    "focusing on key aspects of the topic.",
]

def build_output_json(completed, overprojected):
    return json.dumps({
        "completed_user_message": completed,
        "overprojected_user_message": overprojected
    }, ensure_ascii=False)

class SpeculativeDataset(Dataset):
    """
    Given full sentences/questions from WikiText & SQuAD,
    generate (prefix -> JSON completion) training pairs.
    """
    def __init__(self, tokenizer, split="train", max_samples=2000):
        wiki_raw = load_dataset("wikitext", "wikitext-2-raw-v1")[split]["text"]
        squad_split = "train" if split == "train" else "validation"
        squad_raw = load_dataset("squad")[squad_split]["question"]

        combined = []

        for line in wiki_raw:
            line = line.strip()
            if len(line.split()) >= 8:
                combined.append(line)

        for q in squad_raw:
            q = q.strip()
            if len(q.split()) >= 6:
                combined.append(q)

        random.shuffle(combined)

        self.samples = []
        for sent in combined:
            words = sent.split()
            if len(words) < 8:
                continue

            # random prefix cutoff (prefix only, not whole sentence)
            cutoff = random.randint(3, len(words) - 3)
            prefix = " ".join(words[:cutoff]).strip()

            # full completion = full original sentence
            completed = sent.strip()

            # add a simple extra clause for overprojection
            clause = random.choice(EXTRA_CLAUSES)
            overprojected = completed.rstrip(".") + ", " + clause

            target_json = build_output_json(completed, overprojected)

            self.samples.append((prefix, target_json))

            if len(self.samples) >= max_samples:
                break

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prefix, target_json = self.samples[idx]

        model_input = f"prefix: {prefix}"
        encoded = self.tokenizer(
            model_input,
            truncation=True,
            padding="max_length",
            max_length=96,
            return_tensors="pt"
        )

        labels = self.tokenizer(
            target_json,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels["input_ids"].squeeze(0),
        }
        return item


def main():
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    train_ds = SpeculativeDataset(tokenizer, split="train")
    val_ds = SpeculativeDataset(tokenizer, split="validation")

    training_args = TrainingArguments(
        output_dir="./speculative_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=4e-5,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
        no_cuda=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()
    trainer.save_model("./speculative_model")
    tokenizer.save_pretrained("./speculative_model")


if __name__ == "__main__":
    main()
