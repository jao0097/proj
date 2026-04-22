import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

print("=== Teste interativo com Groq (Llama 3.3) ===")
print("Digite suas perguntas. Para sair, digite 'sair' ou apenas Enter.\n")

while True:
    pergunta = input("Você: ").strip()
    if pergunta.lower() in ("sair", ""):
        print("Encerrando teste.")
        break

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": pergunta}],
            model="llama-3.3-70b-versatile",   # ou "llama-3.1-8b-instant" para respostas mais rápidas
            temperature=0.7,
            max_tokens=200,
        )
        resposta = chat_completion.choices[0].message.content
        print(f"Groq: {resposta}\n")
    except Exception as e:
        print(f"Erro: {e}\n")