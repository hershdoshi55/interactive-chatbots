import random
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

# load the dataset to train the verification model
class PrefixQualityDataset(Dataset):
    def __init__(self, tokenizer, split="train", max_samples=2000):
        # Wikitext sentences, allows the model to recognize simple sentences 
        wiki_raw = load_dataset("wikitext", "wikitext-2-raw-v1")[split]["text"]

        # questions, allows the model to recognize questions, and can predict when to interrupt
        squad_split = "train" if split == "train" else "validation"
        squad_raw = load_dataset("squad")[squad_split]["question"]

        combined = [] # train on both sentences and questions

        # dataset filter, don't use short sentences 
        for line in wiki_raw:
            line = line.strip()
            if len(line.split()) >= 6:
                combined.append(line)

        # dataset filter, don't use short questions 
        for q in squad_raw:
            q = q.strip()
            if len(q.split()) >= 4:
                combined.append(q)

        # shuffle, allows for better data generability 
        random.shuffle(combined)

        self.samples = []
        for line in combined:
            words = line.split()
            if len(words) < 6: # skip if length < 6
                continue
            # we don't want to ever interrupt for something really small (like 1-2 words), seems unreasonable
            cutoff = random.randint(3, len(words) - 1)

            # just label as "interruptable" up to a certain random index
            interruptable = " ".join(words[:cutoff])
            self.samples.append((interruptable, 1))

            # label as "uninterruptable1" for random shuffled words, doesn't make sense to interrupt a sentence with random words 
            shuffled = words[:]
            random.shuffle(shuffled)
            uninterruptable1 = " ".join(shuffled[:cutoff])
            self.samples.append((uninterruptable1, 0))

            # label as "uninterruptable2" for random middle of sentence words 
            max_start = len(words) - cutoff - 1
            if max_start >= 1:
                start = random.randint(1, max_start)
                uninterruptable2 = " ".join(words[start:start + cutoff])
                self.samples.append((uninterruptable2, 0))

            # label as "uninterruptable3" for random short sentences (1-2 words)
            uninterruptable3 = " ".join(words[:random.randint(1, 2)])
            self.samples.append((uninterruptable3, 0))

            if len(self.samples) >= max_samples:
                break

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = torch.tensor(label, dtype=torch.long)
        return enc


def main():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )
    # use BERT and train
    # split the datasets
    train_ds = PrefixQualityDataset(tokenizer, split="train")
    val_ds = PrefixQualityDataset(tokenizer, split="validation")

    # small training so it doesn't take too long
    training_args = TrainingArguments(
        output_dir="./verification_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=4e-5,
        weight_decay=0.01,
        logging_steps=50,
        remove_unused_columns=False,
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
    trainer.save_model("./verification_model")
    tokenizer.save_pretrained("./verification_model")


if __name__ == "__main__":
    main()
