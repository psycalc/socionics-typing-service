import argparse
import torch
from transformers import BertTokenizer
from model_definition import BertForAspects

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Словарь для человекопонятных названий функций
function_names = {
    0: "Ti (Ли) – интровертная логика",
    1: "Te (Лэ) – экстравертная логика",
    2: "Fi (Эи) – интровертная этика",
    3: "Fe (Ээ) – экстравертная этика",
    4: "Ni (Ии) – интровертная интуиция",
    5: "Ne (Иэ) – экстравертная интуиция",
    6: "Si (Си) – интровертная сенсорика",
    7: "Se (Сэ) – экстравертная сенсорика",
}

def predict_aspect(text, threshold=0.5):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForAspects(num_labels=8)
    model.load_state_dict(torch.load("models/best_aspect_model.pt", map_location=DEVICE))
    model.eval()
    model.to(DEVICE)

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding='max_length',
        max_length=128
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # shape (8,)
        predicted_aspects = [i for i, p in enumerate(probs) if p > threshold]
        return predicted_aspects, probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Предложение для определения функции")
    parser.add_argument("--human-readable", action='store_true', help="Вывести результаты в человекочитаемом формате")
    parser.add_argument("--threshold", type=float, default=0.5, help="Порог вероятности для выбора аспекта (по умолчанию 0.5)")
    args = parser.parse_args()

    label, probabilities = predict_aspect(args.text, threshold=args.threshold)
    
    if args.human_readable:
        # Человекочитаемый формат
        print(f"Текст: {args.text}\n")
        print("Вероятности по аспектам:")
        for i, p in enumerate(probabilities):
            percent = p * 100
            aspect_name = function_names[i]
            mark = "✓" if i in label else " "
            print(f"[{mark}] {aspect_name}: {percent:.2f}%")
        print(f"\nВыбраны аспекты (порог > {args.threshold*100:.0f}%):")
        if label:
            for i in label:
                print(f" - {function_names[i]}")
        else:
            print("Нет аспектов, превысивших порог.")
    else:
        # Обычный формат
        print("Predicted function:", label)
        print("Probabilities:", probabilities)
