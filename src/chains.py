import os
import time
import sys
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langchain_community.document_loaders import CSVLoader

# --- CONFIGURAÇÃO ---
base_dir = os.path.dirname(os.path.abspath(__file__)) 
adapter_path = os.path.join(base_dir, "..", "models")
real_csv_path = os.path.join(base_dir, "..", "data", "base_demo_segura.csv")

def print_loading(message):
    sys.stdout.write(f"\r⚙️  {message}...")
    sys.stdout.flush()
    time.sleep(0.5)
    sys.stdout.write(" [OK]\n")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# --- INICIALIZAÇÃO ---
clear_screen()
print("\n" + "="*60)
print("🏥 HEALTH ASSISTANT INTELLIGENCE SYSTEM v3.0 (Safety Guard)")
print("="*60 + "\n")

try:
    print_loading("Inicializando Motor Neural (TinyLlama-Safe)")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, device_map="cpu")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print_loading("Carregando Diretrizes de Ética Médica")
except Exception as e:
    print(f"\n❌ Erro: {e}")
    exit()

try:
    print_loading("Conectando ao MedQuAD")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(real_csv_path):
        loader = CSVLoader(real_csv_path, encoding='utf-8', source_column="question")
        docs = loader.load()
        vector_db = FAISS.from_documents(docs, embeddings)
        print_loading(f"Indexando Protocolos Clínicos") 
    else:
        print("\n❌ Erro: Base não encontrada.")
        exit()
except Exception as e:
    print(f"\n❌ Erro indexação: {e}")
    exit()

print("\n✅ SISTEMA ONLINE. MÓDULO DE SEGURANÇA ATIVO.\n")
print("-" * 60)


def adicionar_aviso_legal(resposta):
    aviso = "\n\n⚠️ IMPORTANTE: Sou uma IA assistente. Esta informação é baseada em protocolos gerais e NÃO substitui uma consulta médica. Não se automedique."
    return resposta + aviso

def consultar_assistente(pergunta):
    
    docs = vector_db.similarity_search(pergunta, k=1)
    
    if not docs:
        return "Não encontrei um protocolo específico para isso. Por segurança, consulte um médico."
    
    conteudo = docs[0].page_content
    if "answer:" in conteudo:
        resposta_base = conteudo.split("answer:", 1)[1].strip()
    else:
        resposta_base = conteudo

    # 2. Prompt (Instruindo a não prescrever pessoalmente)
    prompt = f"""<|system|>
Você é um assistente médico ético.
Seu objetivo é informar o protocolo técnico padrão para a condição citada.
NUNCA use frases como "Eu recomendo" ou "Você deve tomar".
Use frases como "O protocolo indica" ou "O tratamento comum é".
Seja direto.

INFORMAÇÃO TÉCNICA:
{resposta_base}
</s>
<|user|>
{pergunta}
</s>
<|assistant|>
"""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,       
            repetition_penalty=1.2 
        )
        resposta_ia = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|assistant|>" in resposta_ia:
            resposta_final = resposta_ia.split("<|assistant|>\n")[-1].strip()
        else:
            resposta_final = resposta_ia

        bad_words = [" y ", " en ", " el ", " los ", " dolor ", " sorry "]
        if any(w in resposta_final.lower() for w in bad_words) or len(resposta_final) < 15:
            return adicionar_aviso_legal(resposta_base)
            
        return adicionar_aviso_legal(resposta_final)

    except:
        return adicionar_aviso_legal(resposta_base)

if __name__ == "__main__":
    while True:
        try:
            pergunta = input("\n👤 PACIENTE: ")
            
            if pergunta.lower() in ['sair', 'exit', 'q']:
                print("\nEncerrando sessão...")
                break
            
            sys.stdout.write("🤖 ASSISTENTE: Consultando protocolos...")
            sys.stdout.flush()
            
            resposta = consultar_assistente(pergunta)
            
            sys.stdout.write("\r" + " " * 40 + "\r") 
            print(f"🤖 ASSISTENTE: {resposta}")
            print("_" * 60) 
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n⚠️ Erro de processamento.")