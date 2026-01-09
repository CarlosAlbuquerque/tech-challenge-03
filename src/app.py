import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

caminho_modelo_base = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
caminho_adapter = "./models" 

print("⏳ Iniciando o Assistente Médico...")

tokenizer = AutoTokenizer.from_pretrained(caminho_modelo_base)
base_model = AutoModelForCausalLM.from_pretrained(
    caminho_modelo_base,
    dtype=torch.float32, 
    device_map="cpu"
)

if os.path.exists(caminho_adapter):
    model = PeftModel.from_pretrained(base_model, caminho_adapter)
    print("✅ Modelo carregado!")
else:
    print("❌ Pasta 'models' não encontrada.")
    exit()

def consultar_medico(pergunta):
    prompt = f"<|user|>\nAtue como um médico. {pergunta}</s>\n<|assistant|>\n"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=100,
        do_sample=False,       
        repetition_penalty=1.2 
    )
    
    resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "<|assistant|>" in resposta:
        return resposta.split("<|assistant|>\n")[-1]
    return resposta

if __name__ == "__main__":
    while True:
        d = input("\n👨‍⚕️ Pergunta: ")
        if d == 'sair': break
        print(f"🤖: {consultar_medico(d)}")