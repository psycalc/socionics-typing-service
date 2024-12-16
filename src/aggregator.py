import argparse
import torch
from transformers import BertTokenizer
from model_definition import BertForAspects

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Функции от 0 до 7:
# 0: Ti, 1: Te, 2: Fi, 3: Fe, 4: Ni, 5: Ne, 6: Si, 7: Se
function_names = {
    0: "Ti",
    1: "Te",
    2: "Fi",
    3: "Fe",
    4: "Ni",
    5: "Ne",
    6: "Si",
    7: "Se"
}

def predict_aspects_for_texts(texts):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForAspects(num_labels=8)
    model.load_state_dict(torch.load("models/best_aspect_model.pt", map_location=DEVICE))
    model.eval()
    model.to(DEVICE)

    aspect_counts = {i:0 for i in range(8)}

    for text in texts:
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
            probs = torch.sigmoid(logits)
            aspect_indices = (probs > 0.5).nonzero(as_tuple=True)[1].tolist()  # индексы аспектов > 0.5
            # обновить aspect_counts для каждого из выбранных аспектов
            for ai in aspect_indices:
                aspect_counts[ai] += 1
            pred_label = torch.argmax(probs, dim=1).item()
            aspect_counts[pred_label] += 1

    return aspect_counts

def determine_type(aspect_counts):
    # Упрощённый пример: находим 2 самые часто встречающиеся функции
    sorted_aspects = sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)
    top_two = sorted_aspects[:2]

    # Для упрощения просто выведем топ-2 функции
    f1 = function_names[top_two[0][0]]
    f2 = function_names[top_two[1][0]]

    # Здесь вы можете прописать логику определения типа на основе комбинации функций.
    # Пока просто вернём строку.
    return f"Часто встречаются {f1} и {f2}. Подумайте, какой это тип."

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", type=str, required=True, help="Путь к файлу с предложениями")
    args = parser.parse_args()

    with open(args.text_file, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    aspect_counts = predict_aspects_for_texts(lines)
    print("Aspect counts:", {function_names[k]: v for k,v in aspect_counts.items()})
    predicted_type = determine_type(aspect_counts)
    print("Predicted type hint:", predicted_type)
