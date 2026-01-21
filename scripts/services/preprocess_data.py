import pandas as pd

from scripts.services.clean_text import clean_text


# NOTE data labeled od LLM
df_raw = pd.read_parquet("data/financial_sentiment.parquet")
df = pd.DataFrame()
df["Sentence"] = df_raw["text"]
df["Sentiment"] = df_raw["label"]

df['Sentence'] = df['Sentence'].apply(clean_text)
print(df.head())
print(df[df["Sentence"] == ""])
df = df[df["Sentence"] != ""]
# NOTE PRO EFEKTIVITU TRAINING JEN NAHODNE RADKY - SNAD BY MELO DOSTATECNE REPREZENTOVAT DATASET - 15K / 100K
df = df.sample(frac=0.15, replace=False, random_state=42)
df.to_csv('data/all_data.csv', index=False)