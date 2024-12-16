import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from data_loading import load_aspect_data
from model_definition import BertForAspects
from utils import multi_label_accuracy_fn, save_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(epochs=60, lr=1e-5, batch_size=16):
    train_loader, val_loader, tokenizer = load_aspect_data(batch_size=batch_size)
    model = BertForAspects(num_labels=8).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Настраиваем шедулер для лернинг рейта
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_val_acc = 0.0
    no_improve_epochs = 0
    patience = 50  # ранняя остановка после 3 эпох без улучшений

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_acc += multi_label_accuracy_fn(logits, labels)

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)

        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                val_acc += multi_label_accuracy_fn(logits, labels)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            save_model(model, "models/best_aspect_model.pt")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopping triggered")
                break

if __name__ == '__main__':
    train_model()
