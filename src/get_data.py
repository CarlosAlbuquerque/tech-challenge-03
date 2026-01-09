# src/get_data.py
import os
import pandas as pd
from datasets import load_dataset

if not os.path.exists("./data"):
    os.makedirs("./data")

print("⏳ Baixando dataset MedQuAD do Hugging Face (pode levar alguns segundos)...")

dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")

df = pd.DataFrame(dataset)

df = df[['Question', 'Answer']]
df.columns = ['question', 'answer']

df_amostra = df.head(4000)

caminho_csv = "./data/base_medica.csv"
df_amostra.to_csv(caminho_csv, index=False)

print(f"✅ Sucesso! Base criada em '{caminho_csv}' com {len(df_amostra)} linhas.")
print("Exemplo de dados baixados:")
print(df_amostra.head(3))