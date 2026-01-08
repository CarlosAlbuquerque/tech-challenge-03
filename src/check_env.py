import sys
import os

print("--- DIAGNÓSTICO DE AMBIENTE ---")
print(f"1. Python Executável: {sys.executable}")
print(f"2. Versão: {sys.version}")
print(f"3. Onde estou rodando: {os.getcwd()}")
print("\nTentando importar LangChain...")

try:
    import langchain
    print(f"✅ LangChain encontrado em: {langchain.__file__}")
    
    from langchain.chains import RetrievalQA
    print("✅ Sucesso! RetrievalQA importado.")
except ImportError as e:
    print(f"❌ ERRO DE IMPORTAÇÃO: {e}")
except Exception as e:
    print(f"❌ OUTRO ERRO: {e}")