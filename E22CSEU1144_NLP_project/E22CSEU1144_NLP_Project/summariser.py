import os
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
import torch

print(torch.cuda.is_available())
# Configuration
DATASET_PATH = "./Reviews.csv"  # Update with your dataset path
MODEL_NAME = "t5-small"  # T5 model for summarization
BATCH_SIZE = 8
MAX_LEN = 200  # Maximum sequence length
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ReviewsDataset(Dataset):
    def _init_(self, texts, summaries, tokenizer, max_len):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _len_(self):
        return len(self.texts)

    def _getitem_(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]

        # Prepare input and target text
        input_text = f"summarize: {text}"
        target_text = summary

        # Tokenize input and target
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        targets = self.tokenizer(
            target_text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze(),
        }

def load_data(dataset_path):
    """Load dataset and preprocess"""
    df = pd.read_csv(dataset_path, on_bad_lines="skip")
    texts = df["Text"].astype(str).tolist()
    summaries = df["Summary"].astype(str).tolist()
    return texts, summaries

def train_model():
    """Train the T5 model"""
    print("Loading data...")
    texts, summaries = load_data(DATASET_PATH)

    print("Loading tokenizer and model...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

    print("Preparing dataset and dataloader...")
    dataset = ReviewsDataset(texts, summaries, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    print("Starting training...")
    model.train()

    num_epochs = 3  # Total number of epochs
    total_steps = len(dataloader) * num_epochs  # Total steps (batches per epoch * num_epochs)
    step = 0  # Initialize step counter

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            step += 1
            print(f"Running step: {step}/{total_steps} (Epoch {epoch + 1}/{num_epochs})", end="\r")

            optimizer.zero_grad()

            # Move batch to device
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f"\nEpoch {epoch + 1} completed. Average Loss: {total_loss / len(dataloader):.4f}")




    # Save the trained model
    print("Saving the model...")
    model.save_pretrained("t5_summarizer")
    tokenizer.save_pretrained("t5_summarizer")

def summarize(text, model, tokenizer, max_len=MAX_LEN):
    """Generate summary for a given text"""
    input_text = f"summarize: {text}"
    inputs = tokenizer(
        input_text,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(DEVICE)

    summary_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_len,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if _name_ == "_main_":
    train_model()

    # Load the trained model and test summarization
    print("Loading trained model for inference...")
    tokenizer = T5Tokenizer.from_pretrained("t5_summarizer")
    model = T5ForConditionalGeneration.from_pretrained("t5_summarizer").to(DEVICE)


    test_text = "I have bought several of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than processed meat and it smells better. My Labrador is finicky and she appreciates this product better than most."
    print("Generated Summary:")
    print(summarize(test_text, model, tokenizer))