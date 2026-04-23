"""
transcrever_e_indexar.py
━━━━━━━━━━━━━━━━━━━━━━━━
Pipeline completo: baixa a transcrição do YouTube e já indexa no ChromaDB.

Dependências:
    pip install youtube-transcript-api chromadb groq sentence-transformers

Variáveis de ambiente obrigatórias:
    GROQ_API_KEY  — chave da API Groq

Opcionais (velocidade / custo):
    LIMPEZA_TRANSCRICAO=local|ia|nenhuma  (padrão: local — rápido, 0 tokens na limpeza)

Uso:
    python transcrever_e_indexar.py              # pede só a URL → transcreve, metadados (Groq) e indexa
    python transcrever_e_indexar.py "URL"      # mesmo fluxo sem menu
    python transcrever_e_indexar.py --menu     # menu antigo (pergunta, pasta, etc.)
    python transcrever_e_indexar.py --pergunta "texto"
    python transcrever_e_indexar.py --processar-pasta

Menos tokens (velocidade):
    export LIMPEZA_TRANSCRICAO=local          # padrão — limpeza sem Groq
    export GROQ_METADADOS_MAX_CHARS=2000    # menos texto no prompt de metadados
    export CHROMA_CHUNK_PALAVRAS=600         # menos chunks = menos embeddings locais (mais rápido)
"""

# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import json
import math
import os
import re
import sys
import time
import difflib
import unicodedata
from datetime import date
from urllib.parse import urlencode
from urllib.request import urlopen

from youtube_transcript_api import YouTubeTranscriptApi
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÕES — edite apenas aqui
# ══════════════════════════════════════════════════════════════════════════════

GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "sua_chave_groq_aqui")
PASTA_TRANSCRICOES = os.path.expanduser("~/Documentos/transcricoes")
PASTA_CHROMA       = os.path.expanduser("~/Documentos/chroma_db")
ARQUIVO_LOG_DEBUG  = os.path.join(PASTA_TRANSCRICOES, "debug-indexacao.log")

MODELO_RAPIDO  = os.getenv("GROQ_MODELO_RAPIDO",  "groq/compound-mini")
MODELO_POTENTE = os.getenv("GROQ_MODELO_POTENTE", "llama-3.3-70b-versatile")
# Limpeza em lote: não use compound — o sistema Compound usa 70B por baixo e esgota TPM (429) rápido.
MODELO_LIMPEZA = os.getenv("GROQ_MODELO_LIMPEZA", "llama-3.1-8b-instant")
# local = rápido, 0 tokens Groq | ia = limpeza com LLM (lenta) | nenhuma = texto bruto
LIMPEZA_TRANSCRICAO = os.getenv("LIMPEZA_TRANSCRICAO", "local").strip().lower()
# Só usados no modo ia: blocos maiores = menos chamadas (menos tokens de system repetido).
MAX_CHARS_LIMPEZA_GROQ = int(os.getenv("GROQ_LIMPEZA_MAX_CHARS", "7500"))
PAUSA_ENTRE_BLOCOS_LIMPEZA_S = float(os.getenv("GROQ_LIMPEZA_PAUSA_S", "1.25"))
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "10"))
# Indexação: chunks maiores = menos embeddings locais (mais rápido, mesmo custo API).
CHROMA_CHUNK_PALAVRAS = int(os.getenv("CHROMA_CHUNK_PALAVRAS", "500"))
CHROMA_CHUNK_OVERLAP = int(os.getenv("CHROMA_CHUNK_OVERLAP", "50"))
# Metadados (Groq): só um trecho da transcrição — menor = menos tokens de entrada.
GROQ_METADADOS_MAX_CHARS = int(os.getenv("GROQ_METADADOS_MAX_CHARS", "4000"))


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDAÇÃO INICIAL
# ══════════════════════════════════════════════════════════════════════════════

if GROQ_API_KEY == "sua_chave_groq_aqui":
    raise RuntimeError(
        "GROQ_API_KEY não configurada. Defina a variável de ambiente antes de rodar.\n"
        "Exemplo: export GROQ_API_KEY='sua_chave_real'"
    )

os.makedirs(PASTA_TRANSCRICOES, exist_ok=True)
os.makedirs(PASTA_CHROMA, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  INICIALIZAÇÃO DE CLIENTES
# ══════════════════════════════════════════════════════════════════════════════

groq_client = Groq(api_key=GROQ_API_KEY)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

chroma_client = chromadb.PersistentClient(path=PASTA_CHROMA)

collection = chroma_client.get_or_create_collection(
    name="transcricoes_youtube",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},
)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITÁRIOS — TRANSCRIÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def extract_video_id(video_input: str) -> str:
    value = video_input.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value
    patterns = [
        r"(?:v=|\/)([A-Za-z0-9_-]{11})(?:[?&].*)?$",
        r"youtu\.be\/([A-Za-z0-9_-]{11})(?:[?&].*)?$",
        r"youtube\.com\/shorts\/([A-Za-z0-9_-]{11})(?:[?&].*)?$",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    raise ValueError("Não foi possível extrair o ID do vídeo a partir da URL informada.")


def sanitize_filename(name: str) -> str:
    normalized    = unicodedata.normalize("NFKD", name)
    no_accents    = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    lowered       = no_accents.lower()
    underscored   = re.sub(r"\s+", "_", lowered)
    cleaned       = re.sub(r"[^a-z0-9_]", "", underscored)
    compacted     = re.sub(r"_+", "_", cleaned).strip("_")
    return compacted or "transcricao"


def get_video_metadata(video_id: str) -> dict:
    query = urlencode({"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"})
    with urlopen(f"https://www.youtube.com/oembed?{query}") as response:
        data = json.loads(response.read().decode("utf-8"))
    return {
        "titulo": data.get("title", video_id),
        "canal":  data.get("author_name", "Canal desconhecido"),
    }


def estimate_duration_seconds(transcript_items) -> int:
    if not transcript_items:
        return 0
    last = transcript_items[-1]
    if isinstance(last, dict):
        start    = float(last.get("start", 0))
        duration = float(last.get("duration", 0))
    else:
        start    = float(getattr(last, "start", 0))
        duration = float(getattr(last, "duration", 0))
    return int(math.ceil(start + duration))


def format_duration_hhmmss(total_seconds: int) -> str:
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITÁRIOS — GROQ
# ══════════════════════════════════════════════════════════════════════════════

def _pausa_retry_groq(erro: BaseException, tentativa: int) -> float:
    """Extrai 'try again in Xs' da mensagem da API ou usa backoff."""
    texto = str(erro)
    m = re.search(r"try again in ([\d.]+)\s*s", texto, re.I)
    if m:
        return float(m.group(1)) + 0.75
    resp = getattr(erro, "response", None)
    if resp is not None:
        ra = resp.headers.get("retry-after")
        if ra:
            try:
                return float(ra) + 0.75
            except ValueError:
                pass
    return min(120.0, 2.5 * (1.65 ** tentativa))


def _groq_eh_retryavel(erro: BaseException) -> bool:
    status = getattr(erro, "status_code", None)
    if status in (429, 503):
        return True
    if status == 413:
        body = str(erro).lower()
        return "rate_limit" in body or "tokens per minute" in body or "tpm" in body
    return False


def chamar_groq(
    prompt_sistema: str,
    prompt_usuario: str,
    modelo: str = MODELO_RAPIDO,
    json_mode: bool = False,
) -> str:
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

    ultimo: BaseException | None = None
    for tentativa in range(GROQ_MAX_RETRIES):
        try:
            resposta = groq_client.chat.completions.create(**kwargs)
            return resposta.choices[0].message.content
        except Exception as e:
            ultimo = e
            if _groq_eh_retryavel(e) and tentativa < GROQ_MAX_RETRIES - 1:
                espera = _pausa_retry_groq(e, tentativa)
                print(f"     ⏳ Limite da API; aguardando {espera:.0f}s e tentando de novo...")
                time.sleep(espera)
                continue
            raise
    assert ultimo is not None
    raise ultimo


# ══════════════════════════════════════════════════════════════════════════════
#  ETAPA 1 — LIMPEZA
# ══════════════════════════════════════════════════════════════════════════════

_RE_TAG_TRANSCRICAO = re.compile(r"\[[^\]]+\]")
_RE_ESPACO_HORIZONTAL = re.compile(r"[ \t]+")
_RE_LINHAS_EXTRAS = re.compile(r"\n{3,}")
_RE_PALAVRA_REPETIDA = re.compile(r"(\b\w{1,16}\b)(\s+\1\b)+", re.IGNORECASE)


def limpar_transcricao_local(texto: str) -> str:
    """Remove marcações entre colchetes, normaliza espaços e repetições óbvias — sem API."""
    if not texto:
        return ""
    t = _RE_TAG_TRANSCRICAO.sub(" ", texto)
    t = _RE_ESPACO_HORIZONTAL.sub(" ", t)
    linhas = [ln.strip() for ln in t.splitlines()]
    t = "\n".join(ln for ln in linhas if ln)
    t = _RE_LINHAS_EXTRAS.sub("\n\n", t)
    t = _RE_PALAVRA_REPETIDA.sub(r"\1", t)
    return t.strip()


def _dividir_texto_para_limpeza(texto: str, max_chars: int) -> list[str]:
    """Parte o texto em blocos que cabem numa requisição (evita 413 request too large / TPM)."""
    texto = texto or ""
    if len(texto) <= max_chars:
        return [texto]
    blocos: list[str] = []
    i = 0
    n = len(texto)
    while i < n:
        end = min(i + max_chars, n)
        if end < n:
            trecho = texto[i:end]
            nl = trecho.rfind("\n")
            if nl > max_chars // 3:
                end = i + nl + 1
        blocos.append(texto[i:end])
        i = end
    return blocos


def limpar_transcricao(texto_bruto: str) -> str:
    modo = LIMPEZA_TRANSCRICAO
    if modo in ("nenhuma", "none", "off", "0", "raw"):
        print("  ⏭️  Limpeza desativada — usando texto bruto.")
        return (texto_bruto or "").strip()
    if modo in ("local", "rapida", "fast", "regex"):
        print("  🧹 Limpeza rápida (local, sem tokens na API)...")
        return limpar_transcricao_local(texto_bruto)

    print("  🧹 Limpando transcrição com Groq (modo ia)...")
    SISTEMA = """Você é um editor de texto especializado em transcrições de vídeos.
Sua única função é limpar e formatar o texto recebido.

Regras obrigatórias:
- Remova marcações como [música], [aplausos], [risadas], [inaudível]
- Corrija pontuação e capitalização óbvias
- Remova palavras repetidas em sequência (ex: "e e e então")
- Adicione parágrafos onde há mudança clara de assunto
- NÃO resuma, NÃO altere o conteúdo, NÃO adicione informações
- Retorne APENAS o texto limpo, sem comentários seus
- Se o usuário indicar parte N de M, limpe só esse trecho; não antecipe nem resuma outras partes."""

    partes = _dividir_texto_para_limpeza(texto_bruto, MAX_CHARS_LIMPEZA_GROQ)
    total = len(partes)
    if total > 1:
        print(f"     ({total} blocos de até {MAX_CHARS_LIMPEZA_GROQ} caracteres — vídeo longo)")

    saidas: list[str] = []
    for idx, bloco in enumerate(partes, start=1):
        if total > 1:
            cabecalho = f"Trecho {idx} de {total} da mesma transcrição (ordem cronológica).\n\n"
        else:
            cabecalho = ""
        saidas.append(
            chamar_groq(
                SISTEMA,
                f"{cabecalho}Limpe essa transcrição:\n\n{bloco}",
                modelo=MODELO_LIMPEZA,
            )
        )
        if idx < total and PAUSA_ENTRE_BLOCOS_LIMPEZA_S > 0:
            time.sleep(PAUSA_ENTRE_BLOCOS_LIMPEZA_S)

    return "\n\n".join(s.strip() for s in saidas if s and s.strip())


# ══════════════════════════════════════════════════════════════════════════════
#  ETAPA 2 — METADADOS COM IA
# ══════════════════════════════════════════════════════════════════════════════

def gerar_metadados_ia(titulo: str, transcricao_limpa: str) -> dict:
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

Transcrição (primeiros {GROQ_METADADOS_MAX_CHARS} caracteres):
{transcricao_limpa[:GROQ_METADADOS_MAX_CHARS]}"""
    resposta = chamar_groq(SISTEMA, USUARIO, json_mode=True)
    return json.loads(resposta)


def atualizar_json(caminho_json: str, novos_dados: dict):
    with open(caminho_json, "r", encoding="utf-8") as f:
        dados = json.load(f)
    dados.update(novos_dados)
    dados["indexado_chroma"] = False
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  ETAPA 3 — CHUNKING E INDEXAÇÃO NO CHROMADB
# ══════════════════════════════════════════════════════════════════════════════

def fazer_chunks(
    texto: str,
    tamanho: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    t = CHROMA_CHUNK_PALAVRAS if tamanho is None else tamanho
    o = CHROMA_CHUNK_OVERLAP if overlap is None else overlap
    palavras = texto.split()
    chunks   = []
    passo    = max(1, t - o)
    for i in range(0, len(palavras), passo):
        chunk = " ".join(palavras[i : i + t])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def indexar_video(caminho_txt: str, caminho_json: str):
    with open(caminho_json, "r", encoding="utf-8") as f:
        metadados = json.load(f)

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

    metadados["indexado_chroma"] = True
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(metadados, f, ensure_ascii=False, indent=2)

    print(f"  ✅ Indexado: {metadados['titulo']} — {len(chunks)} chunks")


# ══════════════════════════════════════════════════════════════════════════════
#  ETAPA 4 — CLASSIFICAÇÃO DA PERGUNTA
# ══════════════════════════════════════════════════════════════════════════════

def buscar_titulos_no_chroma() -> list[str]:
    resultado = collection.get(include=["metadatas"])
    return list({m["titulo"] for m in resultado["metadatas"]})


def classificar_pergunta(pergunta: str, titulos: list[str]) -> dict:
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
    n     = classificacao.get("n_chunks", 3)
    tipo  = classificacao.get("tipo")
    alvo  = classificacao.get("video_alvo")
    where = {"titulo": alvo} if tipo == "especifica" and alvo else None

    kwargs = {"query_texts": [pergunta], "n_results": n}
    if where:
        kwargs["where"] = where

    resultado = collection.query(**kwargs)
    return [
        {
            "texto":  doc,
            "titulo": meta.get("titulo", "Desconhecido"),
            "url":    meta.get("url", ""),
            "chunk":  meta.get("chunk_index", 0),
        }
        for doc, meta in zip(resultado["documents"][0], resultado["metadatas"][0])
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  ETAPA 6 — RESPOSTA FINAL COM GROQ
# ══════════════════════════════════════════════════════════════════════════════

def responder(pergunta: str, chunks: list[dict]) -> str:
    print("  💬 Gerando resposta com Groq...")
    blocos = [
        f"[Fonte {i+1}]\nVídeo: {c['titulo']}\nURL: {c['url']}\nTrecho:\n{c['texto']}"
        for i, c in enumerate(chunks)
    ]
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
    USUARIO = f"Contexto dos vídeos:\n{contexto}\n\nPergunta: {pergunta}"
    return chamar_groq(SISTEMA, USUARIO, modelo=MODELO_POTENTE)


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINES
# ══════════════════════════════════════════════════════════════════════════════

def _parece_url_ou_id_youtube(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return True
    low = s.lower()
    return "youtube.com" in low or "youtu.be" in low


def pipeline_transcrever_e_indexar(video_url_input: str | None = None):
    """
    Pipeline completo:
      1. Pede a URL do YouTube
      2. Baixa a transcrição
      3. Salva .txt e .json
      4. Limpa a transcrição com Groq
      5. Gera metadados com Groq
      6. Indexa no ChromaDB
    """
    # ── 1. URL ────────────────────────────────────────────────────────────────
    if video_url_input is None:
        video_url_input = input("Cole a URL (ou ID) do vídeo do YouTube: ").strip()
    video_id        = extract_video_id(video_url_input)
    video_url       = f"https://youtube.com/watch?v={video_id}"
    data_coleta     = date.today().isoformat()

    print(f"\n📥 Baixando transcrição: {video_url}")

    # ── 2. Transcrição ────────────────────────────────────────────────────────
    api        = YouTubeTranscriptApi()
    transcript = api.fetch(video_id, languages=["pt", "pt-BR"])

    texto_bruto       = " ".join(
        (t["text"] if isinstance(t, dict) else t.text) for t in transcript
    )
    duracao_segundos  = estimate_duration_seconds(transcript)
    duracao_formatada = format_duration_hhmmss(duracao_segundos)
    metadata          = get_video_metadata(video_id)
    titulo_video      = metadata["titulo"]
    canal_video       = metadata["canal"]

    print(f"   Título : {titulo_video}")
    print(f"   Canal  : {canal_video}")
    print(f"   Duração: {duracao_formatada}")

    # ── 3. Salvar .txt e .json ────────────────────────────────────────────────
    base_name    = sanitize_filename(titulo_video)
    txt_destino  = os.path.join(PASTA_TRANSCRICOES, f"{base_name}.txt")
    json_destino = os.path.join(PASTA_TRANSCRICOES, f"{base_name}.json")

    txt_conteudo = (
        f"TÍTULO: {titulo_video}\n"
        f"URL: {video_url}\n"
        f"DATA DE INDEXAÇÃO: {data_coleta}\n"
        f"DURAÇÃO ESTIMADA: {duracao_formatada}\n"
        f"[TRANSCRIÇÃO]\n"
        f"{texto_bruto}"
    )
    with open(txt_destino, "w", encoding="utf-8") as f:
        f.write(txt_conteudo)

    json_conteudo = {
        "titulo":          titulo_video,
        "url":             video_url,
        "canal":           canal_video,
        "data_coleta":     data_coleta,
        "duracao_segundos": duracao_segundos,
        "idioma":          "pt",
        "indexado_chroma": False,
    }
    with open(json_destino, "w", encoding="utf-8") as f:
        json.dump(json_conteudo, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Transcrição bruta salva em: {txt_destino}")

    # ── 4. Limpeza ────────────────────────────────────────────────────────────
    print(f"\n🎬 Processando: {titulo_video}")
    texto_limpo = limpar_transcricao(texto_bruto)
    with open(txt_destino, "w", encoding="utf-8") as f:
        f.write(texto_limpo)

    # ── 5. Metadados ──────────────────────────────────────────────────────────
    metadados_ia = gerar_metadados_ia(titulo_video, texto_limpo)
    atualizar_json(json_destino, metadados_ia)

    # ── 6. Indexação ──────────────────────────────────────────────────────────
    indexar_video(txt_destino, json_destino)

    print(f"\n🏁 Pronto! Total de chunks no banco: {collection.count()}")


def pipeline_perguntar(pergunta: str) -> str:
    """Busca nos vídeos indexados e responde usando Groq."""
    print(f"\n🔍 Pergunta: {pergunta}\n{'─'*50}")
    titulos = buscar_titulos_no_chroma()
    if not titulos:
        return "Nenhum vídeo indexado ainda. Transcreva um vídeo primeiro."

    classificacao = classificar_pergunta(pergunta, titulos)
    print(f"  Tipo: {classificacao['tipo']} | Raciocínio: {classificacao['raciocinio']}")

    if classificacao["tipo"] == "fora_de_escopo":
        return "Essa pergunta está fora do escopo dos vídeos disponíveis."

    chunks = buscar_chunks(pergunta, classificacao)
    if not chunks:
        return "Não encontrei chunks relevantes para essa pergunta."

    resposta = responder(pergunta, chunks)
    print("─" * 50)
    return resposta


def pipeline_processar_pasta():
    """
    Processa todos os pares .txt + .json ainda não indexados
    que já existam na pasta de transcrições (compatibilidade com o fluxo antigo).
    """
    arquivos = os.listdir(PASTA_TRANSCRICOES)
    txts     = [a for a in arquivos if a.endswith(".txt")]

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

        if metadados.get("indexado_chroma") is True:
            print(f"⏭️  Já indexado: {metadados['titulo']}")
            continue

        print(f"\n🎬 Processando: {metadados['titulo']}")

        with open(caminho_txt, "r", encoding="utf-8") as f:
            texto_bruto = f.read()

        texto_limpo = limpar_transcricao(texto_bruto)
        with open(caminho_txt, "w", encoding="utf-8") as f:
            f.write(texto_limpo)

        metadados_ia = gerar_metadados_ia(metadados["titulo"], texto_limpo)
        atualizar_json(caminho_json, metadados_ia)
        indexar_video(caminho_txt, caminho_json)

    print(f"\n🏁 Pronto! Total de chunks no banco: {collection.count()}")


# ══════════════════════════════════════════════════════════════════════════════
#  MENU INTERATIVO
# ══════════════════════════════════════════════════════════════════════════════

def menu():
    opcoes = {
        "1": ("Transcrever e indexar novo vídeo (URL)",  pipeline_transcrever_e_indexar),
        "2": ("Processar .txt/.json já salvos na pasta", pipeline_processar_pasta),
        "3": ("Fazer uma pergunta sobre os vídeos",      None),
        "0": ("Sair",                                    None),
    }

    while True:
        print("\n" + "═" * 50)
        print("  📼  Transcrever & Indexar YouTube")
        print("═" * 50)
        for k, (desc, _) in opcoes.items():
            print(f"  [{k}] {desc}")
        print("═" * 50)

        escolha = input("Opção: ").strip()

        if escolha == "0":
            print("Até logo!")
            break
        elif escolha == "1":
            pipeline_transcrever_e_indexar()
        elif _parece_url_ou_id_youtube(escolha):
            print("(URL/ID detectado na opção — iniciando transcrição e indexação.)")
            try:
                pipeline_transcrever_e_indexar(video_url_input=escolha)
            except ValueError as exc:
                print(f"Não deu para usar essa entrada como vídeo: {exc}")
        elif escolha == "2":
            pipeline_processar_pasta()
        elif escolha == "3":
            pergunta = input("Sua pergunta: ").strip()
            if pergunta:
                print("\n" + pipeline_perguntar(pergunta))
        else:
            print("Opção inválida.")


def _main_cli(argv: list[str]) -> None:
    """Fluxo padrão: só URL → pipeline completo. Use --menu para o menu interativo."""
    args = argv[1:]
    if not args:
        print("Cole a URL (ou ID) do YouTube — transcrição, metadados e indexação serão feitos em sequência.\n")
        pipeline_transcrever_e_indexar()
        return
    if args[0] in ("--menu", "-m", "menu"):
        menu()
        return
    if args[0] == "--pergunta" and len(args) >= 2:
        pergunta = " ".join(args[1:]).strip()
        if not pergunta:
            print("Use: --pergunta \"sua pergunta\"")
            return
        print("\n" + pipeline_perguntar(pergunta))
        return
    if args[0] in ("--processar-pasta", "-p"):
        pipeline_processar_pasta()
        return
    if args[0].startswith("-"):
        print("Opções: URL | --menu | --pergunta \"...\" | --processar-pasta")
        return
    url = " ".join(args).strip()
    pipeline_transcrever_e_indexar(video_url_input=url)


if __name__ == "__main__":
    if os.getenv("PIPELINE_MENU", "").strip() in ("1", "true", "yes", "sim"):
        menu()
    else:
        _main_cli(sys.argv)


"""
chat_ia.py
━━━━━━━━━━
Módulo de chat conversacional para o sistema RAG de transcrições YouTube.
Integra ao transcrever_e_indexar.py sem modificar o código existente.

Funcionalidades novas:
  ✅ Histórico multi-turno (memória dentro da sessão)
  ✅ Streaming de tokens em tempo real
  ✅ Reformulação de query com base no contexto anterior
  ✅ Sessões salvas em JSON (persistência entre execuções)
  ✅ Comando /novo, /historico, /salvar, /carregar, /sair

Uso standalone:
    python chat_ia.py                      # nova sessão
    python chat_ia.py --sessao minha.json  # continua sessão salva

Uso como módulo:
    from chat_ia import ChatRAG
    chat = ChatRAG(collection, groq_client, modelo=MODELO_POTENTE)
    chat.iniciar()
"""

import json
import os
import sys
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator

# ── Importa o contexto do script principal (quando usado como módulo) ──────────
# Quando rodado standalone, importa as dependências diretamente.
try:
    from transcrever_e_indexar import (
        collection,
        groq_client,
        MODELO_POTENTE,
        MODELO_RAPIDO,
        PASTA_TRANSCRICOES,
        buscar_titulos_no_chroma,
        classificar_pergunta,
        buscar_chunks,
        _groq_eh_retryavel,
        _pausa_retry_groq,
        GROQ_MAX_RETRIES,
    )
    _MODO_MODULO = True
except ImportError:
    _MODO_MODULO = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÕES DO CHAT
# ══════════════════════════════════════════════════════════════════════════════

CHAT_MAX_TURNOS_HISTORICO = int(os.getenv("CHAT_MAX_TURNOS", "10"))
# Quantos turnos passados são enviados ao modelo (evita explodir o contexto)

CHAT_PASTA_SESSOES = os.path.expanduser("~/Documentos/chat_sessoes")
os.makedirs(CHAT_PASTA_SESSOES, exist_ok=True)

SYSTEM_CHAT = """Você é um assistente especializado em responder perguntas sobre \
transcrições de vídeos do YouTube indexadas em uma base de conhecimento local.

Regras obrigatórias:
- Baseie suas respostas EXCLUSIVAMENTE no contexto fornecido (trechos dos vídeos).
- Sempre cite de qual vídeo veio a informação: "No vídeo X..." ou "Segundo o canal Y...".
- Se a resposta não estiver no contexto, diga exatamente:
  'Não encontrei essa informação nos vídeos disponíveis.'
- Nunca invente informações ou complete com conhecimento externo.
- Você tem acesso ao histórico da conversa — use-o para responder perguntas de \
  acompanhamento como "e sobre isso?", "pode detalhar?", "qual era mesmo o nome?".
- Seja direto, claro e objetivo.
- Ao final de cada resposta, liste as fontes usadas (título + URL).
- Se o usuário perguntar algo fora do escopo dos vídeos, oriente-o a transcrever \
  novos vídeos relacionados."""

SYSTEM_REFORMULAR = """Você reformula perguntas de acompanhamento em perguntas \
independentes, completas e ricas em contexto, para melhorar a busca semântica \
em uma base vetorial.

Regras:
- Se a pergunta já for completa e independente, retorne-a SEM modificações.
- Se depender do histórico (ex: "e o outro?", "pode detalhar?", "por que isso?"), \
  reescreva incorporando o contexto necessário do histórico.
- Retorne APENAS a pergunta reformulada, sem explicações, sem prefixos, \
  sem aspas, sem ponto final extra.
- Máximo de 2 frases."""


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class ChatRAG:
    """
    Chat conversacional com RAG, streaming e persistência de sessão.

    Parâmetros:
        collection   — instância chromadb.Collection
        groq_client  — instância groq.Groq
        modelo       — modelo Groq para respostas (padrão: llama-3.3-70b-versatile)
        modelo_rapido— modelo Groq para reformulação/classificação
        max_turnos   — quantos turnos do histórico enviar ao modelo
    """

    def __init__(
        self,
        collection,
        groq_client,
        modelo: str = "llama-3.3-70b-versatile",
        modelo_rapido: str = "llama-3.1-8b-instant",
        max_turnos: int = CHAT_MAX_TURNOS_HISTORICO,
    ):
        self.collection    = collection
        self.groq          = groq_client
        self.modelo        = modelo
        self.modelo_rapido = modelo_rapido
        self.max_turnos    = max_turnos
        self.historico: list[dict] = []  # {"role": "user"|"assistant", "content": "..."}
        self.sessao_path: str | None = None

    # ── Histórico ──────────────────────────────────────────────────────────────

    def _historico_recente(self) -> list[dict]:
        """Retorna apenas os últimos N turnos (user+assistant = 1 turno = 2 msgs)."""
        return self.historico[-(self.max_turnos * 2):]

    def _adicionar(self, role: str, content: str):
        self.historico.append({"role": role, "content": content})

    # ── Streaming ──────────────────────────────────────────────────────────────

    def _stream_groq(
        self,
        mensagens: list[dict],
        modelo: str | None = None,
    ) -> str:
        """
        Faz chamada streaming ao Groq, imprime tokens em tempo real
        e retorna o texto completo acumulado.
        """
        modelo = modelo or self.modelo
        ultimo: BaseException | None = None

        for tentativa in range(GROQ_MAX_RETRIES if _MODO_MODULO else 5):
            try:
                stream = self.groq.chat.completions.create(
                    model=modelo,
                    temperature=0.25,
                    messages=mensagens,
                    stream=True,
                )
                texto_completo = ""
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    print(delta, end="", flush=True)
                    texto_completo += delta
                print()  # quebra de linha após streaming
                return texto_completo

            except Exception as e:
                ultimo = e
                retryavel = (
                    _groq_eh_retryavel(e)
                    if _MODO_MODULO
                    else getattr(e, "status_code", None) in (429, 503)
                )
                if retryavel and tentativa < (GROQ_MAX_RETRIES if _MODO_MODULO else 4):
                    espera = (
                        _pausa_retry_groq(e, tentativa)
                        if _MODO_MODULO
                        else min(60.0, 2.5 * (1.65 ** tentativa))
                    )
                    print(f"\n  ⏳ Limite da API; aguardando {espera:.0f}s...")
                    time.sleep(espera)
                    continue
                raise

        assert ultimo is not None
        raise ultimo

    def _chamar_rapido(self, system: str, user: str) -> str:
        """Chamada não-streaming para tarefas rápidas (reformulação, classificação)."""
        resposta = self.groq.chat.completions.create(
            model=self.modelo_rapido,
            temperature=0.1,
            messages=[
                {"role": "system",  "content": system},
                {"role": "user",    "content": user},
            ],
        )
        return resposta.choices[0].message.content.strip()

    # ── Reformulação de query ──────────────────────────────────────────────────

    def reformular_query(self, pergunta: str) -> str:
        """
        Se houver histórico, reformula a pergunta para incorporar contexto anterior.
        Perguntas independentes são retornadas sem alteração.
        """
        if len(self.historico) < 2:
            return pergunta  # primeiro turno: não há contexto

        resumo_historico = "\n".join(
            f"{m['role'].upper()}: {m['content'][:300]}"
            for m in self._historico_recente()[-6:]  # últimos 3 turnos
        )
        prompt = (
            f"Histórico recente:\n{resumo_historico}\n\n"
            f"Nova mensagem do usuário: {pergunta}\n\n"
            "Reformule a mensagem acima em uma pergunta independente e completa:"
        )
        try:
            reformulada = self._chamar_rapido(SYSTEM_REFORMULAR, prompt)
            if reformulada and reformulada != pergunta:
                print(f"  🔄 Query reformulada: {reformulada[:120]}{'…' if len(reformulada)>120 else ''}")
            return reformulada or pergunta
        except Exception:
            return pergunta  # fallback seguro

    # ── Pipeline de resposta ───────────────────────────────────────────────────

    def _buscar_titulos(self) -> list[str]:
        if _MODO_MODULO:
            return buscar_titulos_no_chroma()
        resultado = self.collection.get(include=["metadatas"])
        return list({m["titulo"] for m in resultado["metadatas"]})

    def _classificar(self, query: str, titulos: list[str]) -> dict:
        if _MODO_MODULO:
            return classificar_pergunta(query, titulos)
        lista = "\n".join(f"- {t}" for t in titulos)
        prompt = (
            f"Vídeos:\n{lista}\n\nPergunta: \"{query}\"\n\n"
            "JSON com: tipo (especifica|geral|comparativa|fora_de_escopo), "
            "video_alvo (str|null), n_chunks (int), raciocinio (str)"
        )
        raw = self._chamar_rapido("Classificador RAG. Retorne APENAS JSON válido.", prompt)
        return json.loads(raw)

    def _buscar(self, query: str, classificacao: dict) -> list[dict]:
        if _MODO_MODULO:
            return buscar_chunks(query, classificacao)
        n     = classificacao.get("n_chunks", 4)
        tipo  = classificacao.get("tipo")
        alvo  = classificacao.get("video_alvo")
        where = {"titulo": alvo} if tipo == "especifica" and alvo else None
        kwargs: dict = {"query_texts": [query], "n_results": n}
        if where:
            kwargs["where"] = where
        resultado = self.collection.query(**kwargs)
        return [
            {
                "texto":  doc,
                "titulo": meta.get("titulo", "?"),
                "url":    meta.get("url", ""),
                "chunk":  meta.get("chunk_index", 0),
            }
            for doc, meta in zip(
                resultado["documents"][0], resultado["metadatas"][0]
            )
        ]

    def responder(self, pergunta_original: str) -> str:
        """
        Pipeline completo de resposta:
        1. Reformula query com base no histórico
        2. Classifica e busca chunks relevantes
        3. Gera resposta com streaming, passando o histórico ao modelo
        """
        titulos = self._buscar_titulos()
        if not titulos:
            return "⚠️  Nenhum vídeo indexado ainda. Use a opção de transcrever primeiro."

        # 1. Reformulação
        query_busca = self.reformular_query(pergunta_original)

        # 2. Classificação + busca
        print("  🔍 Buscando no banco de conhecimento...")
        classificacao = self._classificar(query_busca, titulos)
        tipo = classificacao.get("tipo", "geral")
        print(f"  📂 Tipo: {tipo} | {classificacao.get('raciocinio','')[:80]}")

        if tipo == "fora_de_escopo":
            return (
                "Essa pergunta está fora do escopo dos vídeos disponíveis. "
                "Tente transcrever um vídeo sobre esse tema."
            )

        chunks = self._buscar(query_busca, classificacao)
        if not chunks:
            return "Não encontrei trechos relevantes para essa pergunta."

        # 3. Montar contexto de fontes
        blocos_contexto = "\n\n".join(
            f"[Fonte {i+1}] Vídeo: {c['titulo']} | URL: {c['url']}\n{c['texto']}"
            for i, c in enumerate(chunks)
        )

        # 4. Montar mensagens (system + histórico + pergunta atual com contexto)
        mensagem_usuario_completa = (
            f"Contexto dos vídeos (use APENAS estas informações):\n"
            f"{'─'*60}\n{blocos_contexto}\n{'─'*60}\n\n"
            f"Pergunta: {pergunta_original}"
        )

        mensagens = (
            [{"role": "system", "content": SYSTEM_CHAT}]
            + self._historico_recente()
            + [{"role": "user", "content": mensagem_usuario_completa}]
        )

        # 5. Streaming
        print("\n  💬 Resposta:\n" + "─" * 60)
        resposta = self._stream_groq(mensagens)
        print("─" * 60)

        # 6. Atualizar histórico
        # Armazena a pergunta original (sem o contexto) para o histórico
        self._adicionar("user", pergunta_original)
        self._adicionar("assistant", resposta)

        return resposta

    # ── Sessão ─────────────────────────────────────────────────────────────────

    def salvar_sessao(self, caminho: str | None = None) -> str:
        if caminho is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            caminho = os.path.join(CHAT_PASTA_SESSOES, f"sessao_{ts}.json")
        dados = {
            "criado_em":    datetime.now().isoformat(),
            "total_turnos": len(self.historico) // 2,
            "historico":    self.historico,
        }
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
        self.sessao_path = caminho
        return caminho

    def carregar_sessao(self, caminho: str):
        with open(caminho, "r", encoding="utf-8") as f:
            dados = json.load(f)
        self.historico   = dados.get("historico", [])
        self.sessao_path = caminho
        print(
            f"  📂 Sessão carregada: {len(self.historico)//2} turno(s) anteriores "
            f"de {dados.get('criado_em','?')[:16]}"
        )

    def mostrar_historico(self):
        if not self.historico:
            print("  (histórico vazio)")
            return
        for i, msg in enumerate(self.historico):
            role  = "🧑 Você" if msg["role"] == "user" else "🤖 IA"
            texto = msg["content"][:200]
            reticencias = "…" if len(msg["content"]) > 200 else ""
            print(f"\n{role}: {texto}{reticencias}")

    # ── Loop principal ─────────────────────────────────────────────────────────

    def iniciar(self, sessao_inicial: str | None = None):
        """Inicia o loop interativo de chat."""
        if sessao_inicial and os.path.exists(sessao_inicial):
            self.carregar_sessao(sessao_inicial)

        titulos = self._buscar_titulos()
        n_videos = len(titulos)

        print("\n" + "═" * 60)
        print("  🤖  Chat IA — Base de Conhecimento YouTube")
        print("═" * 60)
        print(f"  {n_videos} vídeo(s) indexado(s)" if n_videos else "  ⚠️  Nenhum vídeo indexado")
        if titulos:
            for t in titulos[:5]:
                print(f"    • {t[:65]}")
            if n_videos > 5:
                print(f"    … e mais {n_videos-5}")
        print("  Para sair: /sair")
        print("═" * 60 + "\n")

        while True:
            try:
                entrada = input("Você: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nEncerrando o chat.")
                break

            if not entrada:
                continue

            # ── Comandos especiais ────────────────────────────────────────────
            if entrada.lower() in ("/sair", "/exit", "/quit", "sair"):
                if self.historico:
                    salvo = self.salvar_sessao()
                    print(f"  💾 Sessão salva em: {salvo}")
                print("Até logo!")
                break

            elif entrada.lower() in ("/novo", "/reset", "/limpar"):
                self.historico = []
                print("  🆕 Histórico limpo. Nova conversa iniciada.\n")
                continue

            elif entrada.lower() in ("/historico", "/hist"):
                self.mostrar_historico()
                print()
                continue

            elif entrada.lower() == "/salvar":
                salvo = self.salvar_sessao()
                print(f"  💾 Sessão salva em: {salvo}\n")
                continue

            elif entrada.lower().startswith("/carregar "):
                arq = entrada[10:].strip()
                if os.path.exists(arq):
                    self.carregar_sessao(arq)
                else:
                    print(f"  ❌ Arquivo não encontrado: {arq}")
                print()
                continue

            # ── Resposta ─────────────────────────────────────────────────────
            print()
            try:
                self.responder(entrada)
            except Exception as e:
                print(f"\n  ❌ Erro: {e}")
            print()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT STANDALONE
# ══════════════════════════════════════════════════════════════════════════════

def _standalone_main():
    """
    Modo standalone: importa tudo do transcrever_e_indexar.py.
    Precisa estar na mesma pasta que o script principal.
    """
    if not _MODO_MODULO:
        print("❌ Erro: transcrever_e_indexar.py não encontrado no diretório atual.")
        sys.exit(1)

    sessao = None
    args = sys.argv[1:]
    if args and args[0] == "--sessao" and len(args) >= 2:
        sessao = args[1]

    chat = ChatRAG(
        collection=collection,
        groq_client=groq_client,
        modelo=MODELO_POTENTE,
        modelo_rapido=MODELO_RAPIDO,
    )
    chat.iniciar(sessao_inicial=sessao)


if __name__ == "__main__":
    _standalone_main()