import os
from groq import Groq

MODELO_VALIDACAO = os.getenv("GROQ_MODELO_VALIDACAO", "llama-3.1-8b-instant")


def validar_chave_groq() -> None:
    chave = os.getenv("GROQ_API_KEY")
    if not chave:
        print("❌ GROQ_API_KEY não encontrada no ambiente.")
        print("Exemplo: export GROQ_API_KEY='sua_chave_aqui'")
        return

    try:
        client = Groq(api_key=chave)
        resposta = client.chat.completions.create(
            model=MODELO_VALIDACAO,
            temperature=0,
            messages=[{"role": "user", "content": "Responda apenas: OK"}],
        )
        texto = resposta.choices[0].message.content if resposta.choices else ""
        print(f"✅ Chave válida. Modelo usado: {MODELO_VALIDACAO}")
        print("Resposta da API:", texto)
    except Exception as erro:
        print("❌ Chave inválida ou erro ao acessar API.")
        print(f"Detalhes: {erro}")
        print(
            "Dica: teste com outro modelo via variável de ambiente, ex:\n"
            "export GROQ_MODELO_VALIDACAO='llama-3.3-70b-versatile'"
        )


if __name__ == "__main__":
    validar_chave_groq()
