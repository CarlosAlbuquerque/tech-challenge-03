import os
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# --- 1. CONFIGURAÇÃO DE CAMINHOS ---
base_dir = os.path.dirname(os.path.abspath(__file__)) 
adapter_path = os.path.join(base_dir, "..", "models")
csv_path = os.path.join(base_dir, "..", "data", "base_medica.csv")

print("⚙️ Carregando Sistema (Modo Manual RAG)...")

# --- 2. CARREGAR MODELO ---
try:
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    # Carrega Fine-Tuning
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("✅ Modelo carregado.")
except Exception as e:
    print(f"❌ Erro no modelo: {e}")
    exit()

# --- 3. BANCO DE DADOS (MEMÓRIA OU CSV) ---
# Se o CSV existir, usa ele. Se não, usa memória.
protocolos_memoria = [
    "PROTOCOLO DOR DE CABEÇA: Dipirona 1g se leve. Sumatriptano se enxaqueca.",
    "PROTOCOLO IAM (INFARTO): Monitorização, Oxigênio, AAS 300mg e Clopidogrel 300mg.",
    "SEGURANÇA: Nunca prescrever controlados sem validação humana."
]

print("📚 Indexando dados...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Tenta carregar do CSV se possível (bônus), senão usa memória
if os.path.exists(csv_path):
    from langchain_community.document_loaders import CSVLoader
    loader = CSVLoader(csv_path, encoding='utf-8')
    docs = loader.load()
    vector_db = FAISS.from_documents(docs, embeddings)
    print(f"✅ Base CSV carregada ({len(docs)} itens).")
else:
    vector_db = FAISS.from_texts(protocolos_memoria, embeddings)
    print("✅ Base de memória carregada.")

# --- 4. A LÓGICA RAG (MANUAL) ---
# Aqui substituímos o 'RetrievalQA' por lógica pura. Funciona sempre.
def consultar_assistente(pergunta):
    
    # PASSO A: Busca (Retrieval)
    # Busca os 2 documentos mais parecidos com a pergunta
    docs_encontrados = vector_db.similarity_search(pergunta, k=2)
    
    # Junta o texto dos documentos numa string só
    contexto = "\n".join([doc.page_content for doc in docs_encontrados])
    
    # PASSO B: Construção do Prompt (Augmentation)
    prompt_final = f"""<|system|>
Você é um assistente médico. Responda à dúvida usando APENAS o contexto abaixo.
CONTEXTO:
{contexto}
</s>
<|user|>
{pergunta}
</s>
<|assistant|>
"""
    
    # PASSO C: Geração (Generation)
    inputs = tokenizer(prompt_final, return_tensors="pt")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False, # Determinístico
        repetition_penalty=1.2
    )
    
    resposta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Limpeza para pegar só a resposta
    if "<|assistant|>" in resposta_completa:
        return resposta_completa.split("<|assistant|>\n")[-1]
    return resposta_completa

# --- 5. LOOP DE INTERAÇÃO ---
if __name__ == "__main__":
    print("\n" + "="*40)
    print("🏥 CHAT MÉDICO ATIVO (CTRL+C para sair)")
    print("="*40)
    while True:
        try:
            p = input("\n👨‍⚕️ Pergunta: ")
            if p.lower() in ['sair', 'exit']: break
            
            print("🔍 Pesquisando nos protocolos...")
            res = consultar_assistente(p)
            print(f"🤖 Resposta: {res}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Erro: {e}")