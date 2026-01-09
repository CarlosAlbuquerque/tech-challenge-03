import pandas as pd
import os
from deep_translator import GoogleTranslator
from tqdm import tqdm 

ARQUIVO_ENTRADA = "./data/base_medica.csv"
ARQUIVO_SAIDA = "./data/base_medica_pt.csv"
print("🔄 Iniciando tradução da base de dados...")

if not os.path.exists(ARQUIVO_ENTRADA):
    print("❌ Arquivo de entrada não encontrado.")
    exit()

df = pd.read_csv(ARQUIVO_ENTRADA)

translator = GoogleTranslator(source='auto', target='pt')

def traduzir_texto(texto):
    try:
        if not isinstance(texto, str) or len(texto) < 2:
            return texto
        return translator.translate(texto)
    except Exception as e:
        return texto 

tqdm.pandas(desc="Traduzindo Perguntas")
df['question'] = df['question'].progress_apply(traduzir_texto)

tqdm.pandas(desc="Traduzindo Respostas")
df['answer'] = df['answer'].progress_apply(traduzir_texto)

df.to_csv(ARQUIVO_SAIDA, index=False)

print(f"✅ Sucesso! Base traduzida salva em: {ARQUIVO_SAIDA}")
print("Amostra:")
print(df.head(2))