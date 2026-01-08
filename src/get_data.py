# src/get_data.py
import os
import pandas as pd
from datasets import load_dataset

# Cria a pasta data se não existir
if not os.path.exists("./data"):
    os.makedirs("./data")

print("⏳ Baixando dataset MedQuAD do Hugging Face (pode levar alguns segundos)...")

# 1. Carrega o dataset "keivalya/MedQuad-MedicalQnADataset" 
# (Esta é uma versão limpa e organizada do MedQuAD oficial citado no PDF)
dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")

# 2. Converte para Pandas DataFrame
df = pd.DataFrame(dataset)

# 3. Limpeza: Renomear colunas para o padrão que nosso código espera
# O dataset original tem colunas 'Question' e 'Answer'
df = df[['Question', 'Answer']]
df.columns = ['question', 'answer']

# 4. Filtragem (Opcional): Pegar apenas 1000 exemplos para não pesar no seu PC
# Se quiser tudo, remova o .head(1000)
df_amostra = df.head(1000)

# 5. Salvar como CSV
caminho_csv = "./data/base_medica.csv"
df_amostra.to_csv(caminho_csv, index=False)

print(f"✅ Sucesso! Base criada em '{caminho_csv}' com {len(df_amostra)} linhas.")
print("Exemplo de dados baixados:")
print(df_amostra.head(3))