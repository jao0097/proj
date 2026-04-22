import os
import json
import time
import difflib
import chromadb
from groq import Groq
from chromadb.utils import embedding_functions

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÕES — edite apenas aqui
# ══════════════════════════════════════════════════════════════════════════════

GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "sua_chave_groq_aqui")
PASTA_TRANSCRICOES = "/home/joao/Documentos/transcricoes"
PASTA_CHROMA       = "/home/joao/Documentos/chroma_db"
ARQUIVO_LOG_DEBUG  = os.path.join(PASTA_TRANSCRICOES, "debug-indexacao.log")

# Modelos (sobrescreva via variáveis de ambiente, se quiser)
MODELO_RAPIDO  = os.getenv("GROQ_MODELO_RAPIDO", "llama-3.1-8b-instant")
MODELO_POTENTE = os.getenv("GROQ_MODELO_POTENTE", "llama-3.3-70b-versatile")


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict):
    payload = {
        "sessionId": "31946b",
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        os.makedirs(PASTA_TRANSCRICOES, exist_ok=True)
        with open(ARQUIVO_LOG_DEBUG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except OSError:
        # Logging nunca deve interromper o pipeline principal.
        pass

# ══════════════════════════════════════════════════════════════════════════════
#  INICIALIZAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

# #region agent log
_debug_log("H1", "indexaçao.py:init", "Initialization started", {"transcricoesPath": PASTA_TRANSCRICOES, "chromaPath": PASTA_CHROMA, "apiKeyPlaceholder": GROQ_API_KEY == "sua_chave_groq_aqui"})
# #endregion
if GROQ_API_KEY == "sua_chave_groq_aqui":
    raise RuntimeError(
        "GROQ_API_KEY não configurada. Defina a variável de ambiente antes de rodar. "
        "Exemplo: export GROQ_API_KEY='sua_chave_real'"
    )

groq_client = Groq(api_key=GROQ_API_KEY)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

chroma_client = chromadb.PersistentClient(path=PASTA_CHROMA)

collection = chroma_client.get_or_create_collection(
    name="transcricoes_youtube",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITÁRIO GROQ — função base reutilizada em tudo
# ══════════════════════════════════════════════════════════════════════════════

def chamar_groq(
    prompt_sistema: str,
    prompt_usuario: str,
    modelo: str = MODELO_RAPIDO,
    json_mode: bool = False
) -> str:
    """
    Chama a API do Groq e retorna o texto da resposta.
    json_mode=True garante que a resposta será um JSON válido.
    """
    kwargs = {
        "model": modelo,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": prompt_sistema},
            {"role": "user",   "content": prompt_usuario},
        ],
    }

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    # #region agent log
    _debug_log("H2", "indexaçao.py:chamar_groq_before", "Calling Groq API", {"model": modelo, "jsonMode": json_mode})
    # #endregion
    resposta = groq_client.chat.completions.create(**kwargs)
    # #region agent log
    _debug_log("H2", "indexaçao.py:chamar_groq_after", "Groq API call succeeded", {"hasChoices": bool(getattr(resposta, "choices", None))})
    # #endregion
    return resposta.choices[0].message.content


# ══════════════════════════════════════════════════════════════════════════════
#  ETAPA 1 — LIMPEZA DA TRANSCRIÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def limpar_transcricao(texto_bruto: str) -> str:
    """
    Usa o Groq para remover ruídos típicos de transcrições automáticas:
    marcações de [música], repetições, erros de pontuação etc.
    Não altera o conteúdo, apenas formata.
    """
    print("  🧹 Limpando transcrição com Groq...")

    SISTEMA = """Você é um editor de texto especializado em transcrições de vídeos.
Sua única função é limpar e formatar o texto recebido.

Regras obrigatórias:
- Remova marcações como [música], [aplausos], [risadas], [inaudível]
- Corrija pontuação e capitalização óbvias
- Remova palavras repetidas em sequência (ex: "e e e então")
- Adicione parágrafos onde há mudança clara de assunto
- NÃO resuma, NÃO altere o conteúdo, NÃO adicione informações
- Retorne APENAS o texto limpo, sem comentários seus"""

    USUARIO = f"Limpe essa transcrição:\n\n{texto_bruto}"

    return chamar_groq(SISTEMA, USUARIO, modelo=MODELO_RAPIDO)


# ══════════════════════════════════════════════════════════════════════════════
#  ETAPA 2 — GERAÇÃO DE METADADOS COM GROQ
# ══════════════════════════════════════════════════════════════════════════════

def gerar_metadados_ia(titulo: str, transcricao_limpa: str) -> dict:
    """
    Usa o Groq para extrair metadados semânticos da transcrição:
    resumo, palavras-chave, tópicos, nível técnico etc.
    Retorna um dicionário pronto para ser mesclado no .json existente.
    """
    print("  📋 Gerando metadados com Groq...")

    SISTEMA = """Você é um assistente que analisa transcrições de vídeos do YouTube.
Retorne APENAS um JSON válido, sem texto adicional, sem markdown."""

    USUARIO = f"""Analise a transcrição abaixo e retorne um JSON com exatamente essas chaves:

{{
  "resumo": "resumo do vídeo em 3 linhas máximo",
  "palavras_chave": ["palavra1", "palavra2", "palavra3", "palavra4", "palavra5"],
  "tema_principal": "uma frase descrevendo o tema central",
  "topicos_abordados": ["tópico1", "tópico2", "tópico3"],
  "nivel_tecnico": "iniciante | intermediário | avançado",
  "linguagem": "pt | en | outro"
}}

Título do vídeo: {titulo}

Transcrição (primeiros 4000 caracteres):
{transcricao_limpa[:4000]}"""

    resposta = chamar_groq(SISTEMA, USUARIO, json_mode=True)
    return json.loads(resposta)


def atualizar_json(caminho_json: str, novos_dados: dict):
    """Mescla os metadados gerados pelo Groq no .json existente."""
    with open(caminho_json, "r", encoding="utf-8") as f:
        dados = json.load(f)

    dados.update(novos_dados)
    dados["indexado_chroma"] = False  # ainda não foi pro ChromaDB

    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  ETAPA 3 — CHUNKING E INDEXAÇÃO NO CHROMADB
# ══════════════════════════════════════════════════════════════════════════════

def fazer_chunks(texto: str, tamanho: int = 400, overlap: int = 50) -> list[str]:
    """
    Divide o texto em chunks de 'tamanho' palavras,
    com 'overlap' palavras de sobreposição entre chunks consecutivos.
    Isso evita cortar ideias no meio.
    """
    palavras = texto.split()
    chunks   = []
    passo    = tamanho - overlap

    for i in range(0, len(palavras), passo):
        chunk = " ".join(palavras[i : i + tamanho])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def indexar_video(caminho_txt: str, caminho_json: str):
    """
    Lê o par .txt + .json, faz o chunking da transcrição
    e indexa tudo no ChromaDB com os metadados do .json.
    Marca 'indexado_chroma: true' no .json ao finalizar.
    """
    with open(caminho_json, "r", encoding="utf-8") as f:
        metadados = json.load(f)

    # evita indexar duas vezes
    if metadados.get("indexado_chroma") is True:
        print(f"  ⏭️  Já indexado: {metadados['titulo']}")
        return

    with open(caminho_txt, "r", encoding="utf-8") as f:
        texto = f.read()

    chunks = fazer_chunks(texto)

    if not chunks:
        print(f"  ⚠️  Arquivo vazio: {caminho_txt}")
        return

    ids        = [f"{metadados['titulo']}_chunk_{i}" for i in range(len(chunks))]
    documentos = chunks
    metas      = [
        {
            "titulo":       metadados.get("titulo", "Desconhecido"),
            "url":          metadados.get("url", ""),
            "canal":        metadados.get("canal", ""),
            "tema":         metadados.get("tema_principal", ""),
            "nivel":        metadados.get("nivel_tecnico", ""),
            "chunk_index":  i,
            "total_chunks": len(chunks),
        }
        for i in range(len(chunks))
    ]

    collection.add(ids=ids, documents=documentos, metadatas=metas)

    # marca como indexado
    metadados["indexado_chroma"] = True
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(metadados, f, ensure_ascii=False, indent=2)

    print(f"  ✅ Indexado: {metadados['titulo']} — {len(chunks)} chunks")


# ══════════════════════════════════════════════════════════════════════════════
#  ETAPA 4 — CLASSIFICAÇÃO DA PERGUNTA
# ══════════════════════════════════════════════════════════════════════════════

def buscar_titulos_no_chroma() -> list[str]:
    """Retorna todos os títulos únicos que estão no ChromaDB."""
    resultado = collection.get(include=["metadatas"])
    titulos   = list({m["titulo"] for m in resultado["metadatas"]})
    return titulos


def classificar_pergunta(pergunta: str, titulos: list[str]) -> dict:
    """
    Usa o Groq para entender o tipo de pergunta e decidir
    quantos chunks buscar e se deve filtrar por vídeo específico.
    """
    print("  🔀 Classificando pergunta...")

    SISTEMA = """Você é um classificador de perguntas para um sistema RAG.
Retorne APENAS um JSON válido."""

    lista_titulos = "\n".join(f"- {t}" for t in titulos)

    USUARIO = f"""Classifique a pergunta abaixo com base nos vídeos disponíveis.

Vídeos disponíveis:
{lista_titulos}

Pergunta: "{pergunta}"

Retorne um JSON com exatamente essas chaves:
{{
  "tipo": "especifica | geral | comparativa | fora_de_escopo",
  "video_alvo": "título exato do vídeo se tipo for especifica, senão null",
  "n_chunks": 3,
  "raciocinio": "uma linha explicando a classificação"
}}

Regras para n_chunks:
- especifica:      3
- geral:           5
- comparativa:     6
- fora_de_escopo:  0"""

    resposta = chamar_groq(SISTEMA, USUARIO, json_mode=True)
    return json.loads(resposta)


# ══════════════════════════════════════════════════════════════════════════════
#  ETAPA 5 — BUSCA NO CHROMADB
# ══════════════════════════════════════════════════════════════════════════════

def buscar_chunks(pergunta: str, classificacao: dict) -> list[dict]:
    """
    Busca os chunks mais relevantes no ChromaDB
    usando a classificação do Groq para decidir filtros e quantidade.
    """
    n        = classificacao.get("n_chunks", 3)
    tipo     = classificacao.get("tipo")
    alvo     = classificacao.get("video_alvo")
    where    = {"titulo": alvo} if tipo == "especifica" and alvo else None

    kwargs = {
        "query_texts": [pergunta],
        "n_results":   n,
    }
    if where:
        kwargs["where"] = where

    resultado = collection.query(**kwargs)

    chunks = []
    for doc, meta in zip(resultado["documents"][0], resultado["metadatas"][0]):
        chunks.append({
            "texto":  doc,
            "titulo": meta.get("titulo", "Desconhecido"),
            "url":    meta.get("url", ""),
            "chunk":  meta.get("chunk_index", 0),
        })

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
#  ETAPA 6 — RESPOSTA FINAL COM GROQ
# ══════════════════════════════════════════════════════════════════════════════

def responder(pergunta: str, chunks: list[dict]) -> str:
    """
    Monta o contexto com os chunks encontrados e envia ao Groq
    para gerar a resposta final, citando as fontes.
    """
    print("  💬 Gerando resposta com Groq...")

    blocos = []
    for i, c in enumerate(chunks):
        bloco = f"[Fonte {i+1}]\nVídeo: {c['titulo']}\nURL: {c['url']}\nTrecho:\n{c['texto']}"
        blocos.append(bloco)

    contexto = "\n\n" + "─" * 60 + "\n\n".join(blocos) + "\n\n" + "─" * 60

    SISTEMA = """Você é um assistente especializado em responder perguntas
com base em transcrições de vídeos do YouTube.

Regras obrigatórias:
- Use APENAS as informações do contexto fornecido
- Sempre cite de qual vídeo veio a informação (ex: "No vídeo X...")
- Se a resposta não estiver no contexto, diga exatamente:
  'Não encontrei essa informação nos vídeos disponíveis.'
- Nunca invente informações
- Seja direto e objetivo
- Ao final, liste as fontes usadas com título e URL"""

    USUARIO = f"""Contexto dos vídeos:
{contexto}

Pergunta: {pergunta}"""

    return chamar_groq(SISTEMA, USUARIO, modelo=MODELO_POTENTE)


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINES PRINCIPAIS
# ══════════════════════════════════════════════════════════════════════════════

def pipeline_processar_pasta():
    """
    Varre /home/joao/Documentos/transcricoes, e para cada par .txt + .json:
    1. Limpa a transcrição com Groq
    2. Gera metadados com Groq
    3. Indexa no ChromaDB
    Pula arquivos que já foram indexados.
    """
    # #region agent log
    _debug_log("H3", "indexaçao.py:pipeline_start", "Starting folder processing", {"path": PASTA_TRANSCRICOES, "pathExists": os.path.exists(PASTA_TRANSCRICOES)})
    # #endregion
    arquivos = os.listdir(PASTA_TRANSCRICOES)
    txts     = [a for a in arquivos if a.endswith(".txt")]
    # #region agent log
    _debug_log("H3", "indexaçao.py:txt_scan", "Scanned transcript files", {"totalFiles": len(arquivos), "txtCount": len(txts)})
    # #endregion

    if not txts:
        print("Nenhum .txt encontrado na pasta.")
        return

    print(f"\n📂 {len(txts)} arquivo(s) encontrado(s)\n{'─'*50}")

    for nome_txt in txts:
        nome_base    = nome_txt.replace(".txt", "")
        caminho_txt  = os.path.join(PASTA_TRANSCRICOES, nome_txt)
        caminho_json = os.path.join(PASTA_TRANSCRICOES, f"{nome_base}.json")

        if not os.path.exists(caminho_json):
            print(f"⚠️  JSON não encontrado para: {nome_txt} — pulando")
            continue

        with open(caminho_json, "r", encoding="utf-8") as f:
            metadados = json.load(f)
        # #region agent log
        _debug_log("H4", "indexaçao.py:json_loaded", "Loaded metadata JSON", {"jsonPath": caminho_json, "hasTitulo": "titulo" in metadados, "indexado": metadados.get("indexado_chroma")})
        # #endregion

        if metadados.get("indexado_chroma") is True:
            print(f"⏭️  Já indexado: {metadados['titulo']}")
            continue

        print(f"\n🎬 Processando: {metadados['titulo']}")

        # lê o txt atual
        with open(caminho_txt, "r", encoding="utf-8") as f:
            texto_bruto = f.read()
        # #region agent log
        _debug_log("H5", "indexaçao.py:txt_loaded", "Loaded transcript text", {"txtPath": caminho_txt, "textLen": len(texto_bruto)})
        # #endregion

        # etapa 1 — limpa
        texto_limpo = limpar_transcricao(texto_bruto)

        # sobrescreve o .txt com a versão limpa
        with open(caminho_txt, "w", encoding="utf-8") as f:
            f.write(texto_limpo)

        # etapa 2 — metadados
        metadados_ia = gerar_metadados_ia(metadados["titulo"], texto_limpo)
        atualizar_json(caminho_json, metadados_ia)

        # etapa 3 — indexa
        indexar_video(caminho_txt, caminho_json)

    print(f"\n🏁 Pronto! Total de chunks no banco: {collection.count()}")


def _normalizar_texto_busca(texto: str) -> str:
    return " ".join(texto.lower().strip().split())


def _tokenizar_texto_busca(texto: str) -> set[str]:
    tokens = []
    for parte in _normalizar_texto_busca(texto).split():
        limpo = "".join(ch for ch in parte if ch.isalnum())
        if len(limpo) >= 3:
            tokens.append(limpo)
    return set(tokens)


def _pontuar_relevancia_video(consulta: str, candidato: dict) -> float:
    """
    Calcula score de relevância combinando:
    - similaridade de string (difflib)
    - sobreposição de tokens
    """
    alvo = _normalizar_texto_busca(
        f"{candidato['titulo']} {candidato['nome_txt']} {candidato.get('amostra_texto', '')}"
    )
    consulta_norm = _normalizar_texto_busca(consulta)

    if not consulta_norm or not alvo:
        return 0.0

    ratio = difflib.SequenceMatcher(None, consulta_norm, alvo).ratio()

    tokens_consulta = _tokenizar_texto_busca(consulta_norm)
    tokens_alvo = _tokenizar_texto_busca(alvo)
    inter = len(tokens_consulta & tokens_alvo)
    token_score = inter / max(1, len(tokens_consulta))

    # Peso maior para tokens em comum, depois similaridade geral.
    return (0.7 * token_score) + (0.3 * ratio)


def _carregar_candidatos_videos_novos() -> list[dict]:
    """Retorna vídeos ainda não indexados com caminhos e metadados básicos."""
    candidatos = []
    if not os.path.exists(PASTA_TRANSCRICOES):
        return candidatos

    arquivos = os.listdir(PASTA_TRANSCRICOES)
    txts = [a for a in arquivos if a.endswith(".txt")]

    for nome_txt in txts:
        nome_base = nome_txt.replace(".txt", "")
        caminho_txt = os.path.join(PASTA_TRANSCRICOES, nome_txt)
        caminho_json = os.path.join(PASTA_TRANSCRICOES, f"{nome_base}.json")

        if not os.path.exists(caminho_json):
            continue

        try:
            with open(caminho_json, "r", encoding="utf-8") as f:
                metadados = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if metadados.get("indexado_chroma") is True:
            continue

        titulo = metadados.get("titulo", nome_base)
        amostra_texto = ""
        try:
            with open(caminho_txt, "r", encoding="utf-8") as f:
                amostra_texto = f.read(2500)
        except OSError:
            amostra_texto = ""
        candidatos.append(
            {
                "titulo": titulo,
                "nome_txt": nome_txt,
                "caminho_txt": caminho_txt,
                "caminho_json": caminho_json,
                "amostra_texto": amostra_texto,
            }
        )

    return candidatos


def _selecionar_video_por_input(candidatos: list[dict], nome_digitado: str) -> dict | None:
    """Seleciona por relevância textual (título + arquivo + conteúdo)."""
    termo = _normalizar_texto_busca(nome_digitado)
    if not termo:
        return None

    ranqueados = []
    for candidato in candidatos:
        score = _pontuar_relevancia_video(termo, candidato)
        ranqueados.append((score, candidato))

    ranqueados.sort(key=lambda x: x[0], reverse=True)
    melhor_score, melhor_candidato = ranqueados[0]

    # Match forte: já retorna direto.
    if melhor_score >= 0.35:
        return melhor_candidato

    # Match fraco: mostra opções para escolha manual.
    print("\nNão tive confiança alta no match automático.")
    print("Escolha um dos vídeos mais próximos:")
    top = ranqueados[:5]
    for i, (score, c) in enumerate(top, start=1):
        print(f"  {i}. [{score:.2f}] {c['titulo']} ({c['nome_txt']})")

    escolha = input("Número correto (ou Enter para cancelar): ").strip()
    if escolha.isdigit():
        idx = int(escolha) - 1
        if 0 <= idx < len(top):
            return top[idx][1]
    return None


def processar_video_especifico(caminho_txt: str, caminho_json: str):
    with open(caminho_json, "r", encoding="utf-8") as f:
        metadados = json.load(f)

    if metadados.get("indexado_chroma") is True:
        print(f"⏭️  Já indexado: {metadados['titulo']}")
        return

    print(f"\n🎬 Processando: {metadados.get('titulo', os.path.basename(caminho_txt))}")

    with open(caminho_txt, "r", encoding="utf-8") as f:
        texto_bruto = f.read()

    texto_limpo = limpar_transcricao(texto_bruto)
    with open(caminho_txt, "w", encoding="utf-8") as f:
        f.write(texto_limpo)

    metadados_ia = gerar_metadados_ia(metadados["titulo"], texto_limpo)
    atualizar_json(caminho_json, metadados_ia)
    indexar_video(caminho_txt, caminho_json)


def pipeline_processar_video_novo_por_nome():
    """
    Pergunta o nome do vídeo novo, confirma correspondência com arquivos
    não indexados na pasta de transcrições e processa apenas esse vídeo.
    """
    candidatos = _carregar_candidatos_videos_novos()
    if not candidatos:
        print("Nenhum vídeo novo (.txt + .json não indexado) encontrado.")
        return

    nome_digitado = input(
        "Qual o nome/tema do vídeo novo? (pode descrever assunto, palavras faladas etc.): "
    ).strip()
    video = _selecionar_video_por_input(candidatos, nome_digitado)

    if not video:
        print("Não consegui identificar o vídeo novo com esse nome.")
        return

    print(f"\nVídeo encontrado: {video['titulo']} ({video['nome_txt']})")
    confirmar = input("É esse mesmo o vídeo novo? (s/n): ").strip().lower()
    if confirmar not in ("s", "sim", "y", "yes"):
        print("Processo cancelado pelo usuário.")
        return

    processar_video_especifico(video["caminho_txt"], video["caminho_json"])
    print(f"\n🏁 Pronto! Total de chunks no banco: {collection.count()}")


def pipeline_perguntar(pergunta: str) -> str:
    """
    Recebe uma pergunta do usuário e retorna a resposta com base
    nos vídeos indexados. Fluxo completo:
    1. Classifica a pergunta com Groq
    2. Busca chunks relevantes no ChromaDB
    3. Responde com Groq usando o contexto encontrado
    """
    print(f"\n🔍 Pergunta: {pergunta}\n{'─'*50}")

    titulos = buscar_titulos_no_chroma()

    if not titulos:
        return "Nenhum vídeo indexado ainda. Rode pipeline_processar_pasta() primeiro."

    # etapa 4 — classifica
    classificacao = classificar_pergunta(pergunta, titulos)
    print(f"  Tipo: {classificacao['tipo']} | Raciocínio: {classificacao['raciocinio']}")

    if classificacao["tipo"] == "fora_de_escopo":
        return "Essa pergunta está fora do escopo dos vídeos disponíveis."

    # etapa 5 — busca
    chunks = buscar_chunks(pergunta, classificacao)

    if not chunks:
        return "Não encontrei chunks relevantes para essa pergunta."

    # etapa 6 — responde
    resposta = responder(pergunta, chunks)

    print("─" * 50)
    return resposta


# ══════════════════════════════════════════════════════════════════════════════
#  EXECUÇÃO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Modo interativo: processa apenas um vídeo novo por nome ──────────────
    pipeline_processar_video_novo_por_nome()
