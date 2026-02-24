import os
import pandas as pd

DATASET_PATH = "data/processed/main_dataset.csv"
OUT_PATH = "data/processed/top100_main_dataset.csv"

if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH)

    top = df.head(100)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    top.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Top100 shape:", top.shape)
