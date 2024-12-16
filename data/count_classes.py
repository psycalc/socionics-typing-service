import pandas as pd

df_train = pd.read_csv("data/aspect_train.csv")
aspect_counts = {i:0 for i in range(8)}
for row in df_train.itertuples():
    aspects = str(row.labels).split()
    for a in aspects:
        aspect_counts[int(a)] += 1
print("Training aspect counts:", aspect_counts)

df_val = pd.read_csv("data/aspect_val.csv")
val_counts = {i:0 for i in range(8)}
for row in df_val.itertuples():
    aspects = str(row.labels).split()
    for a in aspects:
        val_counts[int(a)] += 1
print("Validation aspect counts:", val_counts)
