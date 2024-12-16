import tkinter as tk
from tkinter import ttk
import pandas as pd

# Путь к исходному файлу
INPUT_CSV = "data/aspect_val.csv"
# Путь к файлу с обновлённой разметкой
OUTPUT_CSV = "data/aspect_val_relabelled.csv"

df = pd.read_csv(INPUT_CSV)

aspects = [
    (0, "Ti (Ли) – интровертная логика"),
    (1, "Te (Лэ) – экстравертная логика"),
    (2, "Fi (Эи) – интровертная этика"),
    (3, "Fe (Ээ) – экстравертная этика"),
    (4, "Ni (Ии) – интровертная интуиция"),
    (5, "Ne (Иэ) – экстравертная интуиция"),
    (6, "Si (Си) – интровертная сенсорика"),
    (7, "Se (Сэ) – экстравертная сенсорика"),
]

res_df = pd.DataFrame(columns=["text", "labels"])

class RelabelApp:
    def __init__(self, master, df, aspects):
        self.master = master
        self.df = df
        self.aspects = aspects
        self.index = 0
        
        self.master.title("Переразметка Аспектов")

        # Отображение прогресса
        self.progress_label = tk.Label(master, text="")
        self.progress_label.pack(pady=5)

        self.text_label = tk.Label(master, text="", wraplength=600, justify="left")
        self.text_label.pack(pady=10)
        
        self.checkframe = tk.Frame(master)
        self.checkframe.pack(pady=10)
        
        self.vars = []
        for i, (idx, desc) in enumerate(self.aspects):
            var = tk.BooleanVar()
            cb = tk.Checkbutton(self.checkframe, text=desc, variable=var, anchor="w", justify="left")
            cb.grid(row=i, column=0, sticky="w")
            self.vars.append(var)
        
        # Фрейм для кнопок управления
        self.button_frame = tk.Frame(master)
        self.button_frame.pack(pady=10)
        
        self.save_button = tk.Button(self.button_frame, text="Сохранить и далее", command=self.save_and_next)
        self.save_button.grid(row=0, column=0, padx=5)
        
        self.skip_button = tk.Button(self.button_frame, text="Пропустить (сохранить без изменений)", command=self.skip)
        self.skip_button.grid(row=0, column=1, padx=5)
        
        self.quit_button = tk.Button(self.button_frame, text="Выход", command=self.quit_app)
        self.quit_button.grid(row=0, column=2, padx=5)

        # Фрейм для перехода к нужному индексу
        self.jump_frame = tk.Frame(master)
        self.jump_frame.pack(pady=10)

        tk.Label(self.jump_frame, text="Перейти к индексу:").grid(row=0, column=0, padx=5)
        self.jump_entry = tk.Entry(self.jump_frame, width=5)
        self.jump_entry.grid(row=0, column=1, padx=5)
        self.jump_button = tk.Button(self.jump_frame, text="Перейти", command=self.jump_to_index)
        self.jump_button.grid(row=0, column=2, padx=5)
        
        self.show_entry()

    def show_entry(self):
        if self.index < len(self.df):
            text = self.df.iloc[self.index]["text"]
            current_labels = self.df.iloc[self.index]["labels"]
            if not isinstance(current_labels, str):
                current_labels = ""
                
            self.text_label.config(text=text)
            
            # Сброс чекбоксов
            for var in self.vars:
                var.set(False)
            
            # Отметить чекбоксы согласно текущей разметке
            current_labels = current_labels.strip()
            if current_labels:
                assigned = current_labels.split()
                for a in assigned:
                    a = a.strip()
                    if a.isdigit():
                        idx = int(a)
                        if 0 <= idx < len(self.vars):
                            self.vars[idx].set(True)
            
            # Обновить прогресс
            total = len(self.df)
            current_num = self.index + 1
            left = total - current_num
            self.progress_label.config(text=f"Пример {current_num} из {total} (Осталось: {left})")

        else:
            # Конец датасета
            self.text_label.config(text="Разметка завершена.")
            for cb in self.checkframe.winfo_children():
                cb.config(state="disabled")
            self.save_button.config(state="disabled")
            self.skip_button.config(state="disabled")
            self.progress_label.config(text="Готово! Все примеры размечены.")
            
    def save_and_next(self):
        if self.index < len(self.df):
            chosen_aspects = [str(i) for i, var in enumerate(self.vars) if var.get()]
            labels_str = " ".join(chosen_aspects)
            if not labels_str.strip():
                labels_str = ""
            global res_df
            # Добавляем новую строку без append (т.к. append удален)
            res_df.loc[len(res_df)] = {"text": self.df.iloc[self.index]["text"], "labels": labels_str}
            
            self.index += 1
            self.show_entry()

    def skip(self):
        # Пропускаем: сохраняем исходную разметку без изменений
        if self.index < len(self.df):
            orig_labels = self.df.iloc[self.index]["labels"]
            if not isinstance(orig_labels, str):
                orig_labels = ""
            global res_df
            res_df.loc[len(res_df)] = {"text": self.df.iloc[self.index]["text"], "labels": orig_labels}
            self.index += 1
            self.show_entry()

    def jump_to_index(self):
        # Попытка перейти к заданному индексу
        try:
            idx = int(self.jump_entry.get())
        except ValueError:
            # Если введено не число, игнорируем
            return
        # Индекс в DataFrame 0-based
        if 0 <= idx < len(self.df):
            self.index = idx
            self.show_entry()
        # Если индекс не валиден, просто ничего не делаем или можно вывести предупреждение

    def quit_app(self):
        global res_df
        res_df.to_csv(OUTPUT_CSV, index=False)
        self.master.quit()

root = tk.Tk()
app = RelabelApp(root, df, aspects)
root.mainloop()

res_df.to_csv(OUTPUT_CSV, index=False)
print("Разметка сохранена в", OUTPUT_CSV)
