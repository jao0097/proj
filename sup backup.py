"""
sup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sistema RAG: coleta, indexação e Q&A com classificação inteligente de perguntas.

Fontes suportadas:
  • Sites / blogs   → crawl automático de listagem, raspa artigos
  • YouTube         → transcrição automática de vídeos (1 ou vários)
  • Arquivos locais → processa pares .txt + .json já salvos em disco

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTALAÇÃO
  pip install requests beautifulsoup4 chromadb groq sentence-transformers youtube-transcript-api
  export GROQ_API_KEY='sua_chave'

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USO — INDEXAR

  Crawl de site (descobre e indexa todos os artigos):
    sup --crawl "https://site.com/blog"
    sup --crawl "https://site.com/blog" --filtro "/posts/"
    sup --crawl "https://site.com/blog" --sem-paginacao

  Artigo avulso:
    sup --artigo "https://site.com/post"

  Vídeo(s) do YouTube — aceita 1 ou vários (URL, youtu.be ou ID de 11 chars):
    sup --video "https://youtube.com/watch?v=ID"
    sup --video "url1" "url2" "url3"
    sup --video lista.txt          # arquivo com uma URL por linha

  Arquivos locais (.txt + .json gerados pelo sistema):
    sup --local

  Localizar arquivo local por nome / tema:
    sup --buscar-local

  Forçar reindexação de uma URL já processada:
    sup --reindexar "https://..."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USO — CONSULTAR

  Fazer uma pergunta à base:
    sup --pergunta "sua dúvida aqui"

  Ver status e estatísticas:
    sup --status

  Menu interativo:
    sup --menu

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VARIÁVEIS DE AMBIENTE (opcionais)
  RAG_PASTA_SAIDA          pasta de saída dos arquivos
  RAG_PASTA_CHROMA         pasta do banco vetorial
  GROQ_MODELO_RAPIDO       modelo para metadados/classificação  (padrão: llama-3.1-8b-instant)
  GROQ_MODELO_POTENTE      modelo para respostas finais         (padrão: llama-3.3-70b-versatile)
  GROQ_MODELO_LIMPEZA      modelo para limpeza de transcrição   (padrão: llama-3.1-8b-instant)
  CHROMA_CHUNK_PALAVRAS    tamanho do chunk em palavras         (padrão: 500)
  CHROMA_CHUNK_OVERLAP     sobreposição entre chunks            (padrão: 50)
  LIMPEZA_TRANSCRICAO      local | ia | nenhuma                 (padrão: local)
  GROQ_LIMPEZA_MAX_CHARS   máx. chars por bloco de limpeza IA  (padrão: 7500)
  GROQ_LIMPEZA_PAUSA_S     pausa entre blocos de limpeza        (padrão: 1.25)
  GROQ_METADADOS_MAX_CHARS máx. chars enviados para metadados   (padrão: 4000)
  TIMEOUT_REQUISICAO       timeout HTTP em segundos             (padrão: 15)
  PAUSA_ENTRE_PAGINAS      pausa entre requisições web          (padrão: 1.0)
  GROQ_MAX_RETRIES         tentativas em caso de rate-limit     (padrão: 10)
"""

# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import difflib
import json
import math
import os
import re
import sys
import time
import unicodedata
from datetime import date
from typing import List, Optional, Set
from urllib.parse import urljoin, urlparse, urlencode
from urllib.request import urlopen

import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÕES  ← edite aqui ou via variáveis de ambiente
# ══════════════════════════════════════════════════════════════════════════════

GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "sua_chave_groq_aqui")

_PASTA_PADRAO       = os.path.expanduser("~/Documentos/rag_base")
PASTA_SAIDA         = os.getenv("RAG_PASTA_SAIDA",  _PASTA_PADRAO)
PASTA_CHROMA        = os.getenv("RAG_PASTA_CHROMA", os.path.join(PASTA_SAIDA, "chroma_db"))
ARQUIVO_PROCESSADAS = os.path.join(PASTA_SAIDA, ".urls_processadas.json")

# Modelos Groq — 3 papéis distintos para controle de custo/velocidade
MODELO_RAPIDO  = os.getenv("GROQ_MODELO_RAPIDO",  "llama-3.1-8b-instant")   # metadados, classificação
MODELO_POTENTE = os.getenv("GROQ_MODELO_POTENTE", "llama-3.3-70b-versatile") # resposta final
MODELO_LIMPEZA = os.getenv("GROQ_MODELO_LIMPEZA", "llama-3.1-8b-instant")   # limpeza em lotes

# Chunking
CHUNK_PALAVRAS      = int(os.getenv("CHROMA_CHUNK_PALAVRAS", "500"))
CHUNK_OVERLAP       = int(os.getenv("CHROMA_CHUNK_OVERLAP",  "50"))

# Limpeza de transcrição: "local" (0 tokens) | "ia" (Groq) | "nenhuma" (bruto)
LIMPEZA_TRANSCRICAO = os.getenv("LIMPEZA_TRANSCRICAO", "local").strip().lower()
LIMPEZA_MAX_CHARS   = int(os.getenv("GROQ_LIMPEZA_MAX_CHARS", "7500"))   # max chars por bloco IA
LIMPEZA_PAUSA_S     = float(os.getenv("GROQ_LIMPEZA_PAUSA_S", "1.25"))   # pausa entre blocos

# Metadados
METADADOS_MAX_CHARS = int(os.getenv("GROQ_METADADOS_MAX_CHARS", "4000"))

# Requisições web
TIMEOUT           = int(os.getenv("TIMEOUT_REQUISICAO", "15"))
PAUSA_ENTRE_REQS  = float(os.getenv("PAUSA_ENTRE_PAGINAS", "1.0"))
GROQ_MAX_RETRIES  = int(os.getenv("GROQ_MAX_RETRIES", "10"))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
}

# Seletores de conteúdo principal (ordem de prioridade)
SELETORES_CONTEUDO = [
    "article", "main",
    "[class*='post-content']", "[class*='article-content']",
    "[class*='entry-content']", "[class*='article-body']",
    "[class*='post-body']",    "[class*='content']",
    "div#content", "div#main",
]

# Elementos removidos antes de extrair o texto
SELETORES_LIXO = [
    "nav", "header", "footer", "aside", "form",
    "script", "style", "noscript", "iframe",
    "[class*='sidebar']",    "[class*='menu']",       "[class*='social']",
    "[class*='share']",      "[class*='comment']",    "[class*='related']",
    "[class*='newsletter']", "[class*='popup']",      "[class*='cookie']",
    "[class*='banner']",     "[class*='ad-']",        "[class*='ads']",
]


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDAÇÃO E INICIALIZAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

if GROQ_API_KEY == "sua_chave_groq_aqui":
    sys.exit("❌ GROQ_API_KEY não configurada.\n   export GROQ_API_KEY='sua_chave_real'")

os.makedirs(PASTA_SAIDA,  exist_ok=True)
os.makedirs(PASTA_CHROMA, exist_ok=True)

# Session HTTP reutilizável (mais eficiente que requests.get() isolado)
http = requests.Session()
http.headers.update(HEADERS)

groq_client  = Groq(api_key=GROQ_API_KEY)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

chroma  = chromadb.PersistentClient(path=PASTA_CHROMA)
colecao = chroma.get_or_create_collection(
    name="base_conhecimento",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},
)


# ══════════════════════════════════════════════════════════════════════════════
#  DEDUPLICAÇÃO DE URLs
# ══════════════════════════════════════════════════════════════════════════════

def carregar_processadas() -> Set[str]:
    if os.path.exists(ARQUIVO_PROCESSADAS):
        with open(ARQUIVO_PROCESSADAS, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def marcar_processada(url: str, processadas: Set[str]) -> None:
    processadas.add(url)
    with open(ARQUIVO_PROCESSADAS, "w", encoding="utf-8") as f:
        json.dump(sorted(processadas), f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  GROQ — chamada com retry automático e backoff exponencial
# ══════════════════════════════════════════════════════════════════════════════

def _retry_wait(erro: BaseException, tentativa: int) -> float:
    """Extrai o tempo de espera da mensagem da API ou usa backoff exponencial."""
    m = re.search(r"try again in ([\d.]+)\s*s", str(erro), re.I)
    if m:
        return float(m.group(1)) + 0.75
    ra = getattr(getattr(erro, "response", None), "headers", {}).get("retry-after")
    try:
        return float(ra) + 0.75
    except (TypeError, ValueError):
        pass
    return min(120.0, 2.5 * (1.65 ** tentativa))


def _groq_retryavel(erro: BaseException) -> bool:
    status = getattr(erro, "status_code", None)
    if status in (429, 503):
        return True
    if status == 413:
        body = str(erro).lower()
        return "rate_limit" in body or "tokens per minute" in body or "tpm" in body
    return False


def chamar_groq(
    sistema: str,
    usuario: str,
    modelo: str = MODELO_RAPIDO,
    json_mode: bool = False,
) -> str:
    kwargs: dict = {
        "model": modelo,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": sistema},
            {"role": "user",   "content": usuario},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    ultimo: Optional[BaseException] = None
    for tentativa in range(GROQ_MAX_RETRIES):
        try:
            return groq_client.chat.completions.create(**kwargs).choices[0].message.content
        except Exception as e:
            ultimo = e
            if _groq_retryavel(e) and tentativa < GROQ_MAX_RETRIES - 1:
                w = _retry_wait(e, tentativa)
                print(f"   ⏸️  Groq rate-limit — aguardando {w:.1f}s "
                      f"(tentativa {tentativa + 1}/{GROQ_MAX_RETRIES})")
                time.sleep(w)
                continue
            raise
    assert ultimo is not None
    raise ultimo


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITÁRIOS GERAIS
# ══════════════════════════════════════════════════════════════════════════════

def slugify(texto: str, limite: int = 60) -> str:
    s = unicodedata.normalize("NFKD", texto)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9_]", "", re.sub(r"\s+", "_", s.lower()))
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:limite] or "item"


def chunkar(texto: str) -> List[str]:
    """Divide em chunks de palavras com sobreposição configurável."""
    palavras = texto.split()
    step = max(1, CHUNK_PALAVRAS - CHUNK_OVERLAP)
    return [
        " ".join(palavras[i : i + CHUNK_PALAVRAS])
        for i in range(0, len(palavras), step)
        if palavras[i : i + CHUNK_PALAVRAS]
    ]


def gerar_metadados(titulo: str, conteudo: str, tipo: str) -> dict:
    """
    Gera metadados semânticos enriquecidos via Groq.
    Funciona para artigos web e transcrições de vídeo.
    """
    raw = chamar_groq(
        sistema=(
            "Você é um assistente que analisa conteúdo textual. "
            "Retorne APENAS um JSON válido, sem texto adicional, sem markdown."
        ),
        usuario=(
            f"Analise o conteúdo abaixo e retorne JSON com exatamente essas chaves:\n\n"
            f'{{"resumo": "até 3 frases", '
            f'"palavras_chave": ["p1","p2","p3","p4","p5"], '
            f'"tema_principal": "uma frase", '
            f'"topicos_abordados": ["t1","t2","t3"], '
            f'"nivel_tecnico": "iniciante|intermediário|avançado", '
            f'"linguagem": "pt|en|outro"}}\n\n'
            f"Tipo: {tipo}\nTítulo: {titulo}\n\n"
            f"{conteudo[:METADADOS_MAX_CHARS]}"
        ),
        json_mode=True,
    )
    fallback = {
        "resumo": conteudo[:200],
        "palavras_chave": [],
        "tema_principal": "",
        "topicos_abordados": [],
        "nivel_tecnico": "intermediário",
        "linguagem": "pt",
    }
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return fallback

    if not isinstance(parsed, dict):
        return fallback

    return parsed


def indexar_conteudo(
    conteudo: str,
    titulo: str,
    url: str,
    tipo: str,
    meta_extra: dict,
    meta_ia: Optional[dict] = None,
) -> int:
    """
    Divide em chunks e faz upsert no ChromaDB.
    Armazena chunk_index, total_chunks e todos os campos de meta_ia.
    Retorna o número de chunks indexados.
    """
    chunks = chunkar(conteudo)
    if not chunks:
        return 0

    slug  = slugify(url)[:80]
    total = len(chunks)
    ids   = [f"{slug}_c{i}" for i in range(total)]

    # Campos IA achatados para o ChromaDB (apenas strings)
    ia_fields: dict = {}
    if meta_ia:
        ia_fields["resumo"]            = str(meta_ia.get("resumo", ""))[:500]
        ia_fields["palavras_chave"]    = ", ".join(meta_ia.get("palavras_chave", []))
        ia_fields["tema_principal"]    = str(meta_ia.get("tema_principal", ""))
        ia_fields["topicos_abordados"] = ", ".join(meta_ia.get("topicos_abordados", []))
        ia_fields["nivel_tecnico"]     = str(meta_ia.get("nivel_tecnico", ""))
        ia_fields["linguagem"]         = str(meta_ia.get("linguagem", "pt"))

    metas = [
        {
            "titulo":       titulo,
            "url":          url,
            "tipo":         tipo,
            "data":         date.today().isoformat(),
            "chunk_index":  i,
            "total_chunks": total,
            **{k: str(v) for k, v in meta_extra.items()},
            **ia_fields,
        }
        for i in range(total)
    ]

    colecao.upsert(ids=ids, documents=chunks, metadatas=metas)
    return total


def salvar_arquivos(
    titulo: str,
    url: str,
    conteudo: str,
    meta_ia: dict,
    tipo: str,
    extra: dict,
) -> None:
    """Salva .txt e .json em PASTA_SAIDA. Evita colisão de slug entre URLs diferentes."""
    slug = slugify(titulo)
    base = os.path.join(PASTA_SAIDA, slug)

    sufixo = 1
    while os.path.exists(f"{base}.json"):
        if _ler_url_json(f"{base}.json") == url:
            break  # mesma URL, sobrescreve normalmente
        base = os.path.join(PASTA_SAIDA, f"{slug}_{sufixo}")
        sufixo += 1

    with open(f"{base}.txt", "w", encoding="utf-8") as f:
        f.write(
            f"TIPO: {tipo}\n"
            f"TÍTULO: {titulo}\n"
            f"URL: {url}\n"
            f"DATA: {date.today().isoformat()}\n"
            f"RESUMO: {meta_ia.get('resumo', '')}\n"
            f"[CONTEÚDO]\n"
            f"{conteudo}"
        )

    with open(f"{base}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "tipo":             tipo,
                "titulo":           titulo,
                "url":              url,
                "data_coleta":      date.today().isoformat(),
                "resumo":           meta_ia.get("resumo", ""),
                "palavras_chave":   meta_ia.get("palavras_chave", []),
                "tema_principal":   meta_ia.get("tema_principal", ""),
                "topicos_abordados":meta_ia.get("topicos_abordados", []),
                "nivel_tecnico":    meta_ia.get("nivel_tecnico", ""),
                "linguagem":        meta_ia.get("linguagem", "pt"),
                "tamanho_chars":    len(conteudo),
                "indexado_chroma":  True,
                **extra,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def _ler_url_json(caminho: str) -> str:
    try:
        with open(caminho, encoding="utf-8") as f:
            return json.load(f).get("url", "")
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
#  CATÁLOGO DA PASTA CENTRAL
# ══════════════════════════════════════════════════════════════════════════════

def salvar_catalogo() -> None:
    """
    Gera (ou atualiza) o arquivo _CATALOGO.md na PASTA_SAIDA com um resumo
    legível de todo o conteúdo indexado: artigos por domínio, vídeos por canal,
    estatísticas gerais e data da última atualização.
    Chamado automaticamente após cada indexação e com --status.
    """
    if not os.path.exists(PASTA_SAIDA):
        return

    arquivos_json = [
        f for f in os.listdir(PASTA_SAIDA)
        if f.endswith(".json") and not f.startswith("_")
    ]

    artigos: list = []
    videos:  list = []

    for nome in sorted(arquivos_json):
        caminho = os.path.join(PASTA_SAIDA, nome)
        try:
            with open(caminho, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        if not isinstance(meta, dict):
            continue

        tipo   = meta.get("tipo", "")
        titulo = meta.get("titulo", nome[:-5])
        url    = meta.get("url", "")
        resumo = meta.get("resumo", "")
        data   = meta.get("data_coleta", "")
        chars  = meta.get("tamanho_chars", 0)
        tags   = ", ".join(meta.get("palavras_chave", [])) or "—"
        nivel  = meta.get("nivel_tecnico", "")

        entrada = {
            "titulo": titulo,
            "url":    url,
            "resumo": resumo,
            "data":   data,
            "chars":  chars,
            "tags":   tags,
            "nivel":  nivel,
        }

        if tipo == "video_youtube":
            entrada["canal"]   = meta.get("canal", "")
            entrada["duracao"] = meta.get("duracao", "")
            videos.append(entrada)
        else:
            entrada["dominio"] = meta.get("dominio", urlparse(url).netloc if url else "")
            artigos.append(entrada)

    # ── Agrupa artigos por domínio ────────────────────────────────────────
    dominios: dict = {}
    for a in artigos:
        d = a["dominio"] or "local"
        dominios.setdefault(d, []).append(a)

    # ── Agrupa vídeos por canal ───────────────────────────────────────────
    canais: dict = {}
    for v in videos:
        c = v["canal"] or "Sem canal"
        canais.setdefault(c, []).append(v)

    # ── Escreve o Markdown ────────────────────────────────────────────────
    linhas: list = []
    def w(s=""): linhas.append(s)

    w("# 📚 Catálogo da Base de Conhecimento")
    w()
    w(f"> Gerado automaticamente por **sup** em {date.today().isoformat()}")
    w(f"> Pasta: `{PASTA_SAIDA}`")
    w()
    w("---")
    w()
    w("## 📊 Resumo")
    w()
    w(f"| Item | Quantidade |")
    w(f"|------|-----------|")
    w(f"| Artigos indexados | {len(artigos)} |")
    w(f"| Vídeos indexados  | {len(videos)} |")
    w(f"| Domínios de artigos | {len(dominios)} |")
    w(f"| Canais de vídeo   | {len(canais)} |")
    w(f"| Chunks no banco   | {colecao.count()} |")
    w()

    if dominios:
        w("---")
        w()
        w("## 📄 Artigos")
        w()
        for dominio, itens in sorted(dominios.items()):
            w(f"### 🌐 {dominio} ({len(itens)} artigo(s))")
            w()
            for a in sorted(itens, key=lambda x: x["titulo"].lower()):
                link = f"[{a['titulo']}]({a['url']})" if a["url"] else a["titulo"]
                w(f"#### {link}")
                if a["resumo"]:
                    w(f"> {a['resumo']}")
                meta_line = []
                if a["data"]:   meta_line.append(f"📅 {a['data']}")
                if a["chars"]:  meta_line.append(f"📝 {a['chars']:,} chars")
                if a["nivel"]:  meta_line.append(f"🎓 {a['nivel']}")
                if a["tags"] != "—": meta_line.append(f"🏷️ {a['tags']}")
                if meta_line:
                    w("  " + "  ·  ".join(meta_line))
                w()

    if canais:
        w("---")
        w()
        w("## 🎬 Vídeos YouTube")
        w()
        for canal, itens in sorted(canais.items()):
            w(f"### 📺 {canal} ({len(itens)} vídeo(s))")
            w()
            for v in sorted(itens, key=lambda x: x["titulo"].lower()):
                link = f"[{v['titulo']}]({v['url']})" if v["url"] else v["titulo"]
                w(f"#### {link}")
                if v["resumo"]:
                    w(f"> {v['resumo']}")
                meta_line = []
                if v["data"]:    meta_line.append(f"📅 {v['data']}")
                if v["duracao"]: meta_line.append(f"⏱️ {v['duracao']}")
                if v["chars"]:   meta_line.append(f"📝 {v['chars']:,} chars")
                if v["nivel"]:   meta_line.append(f"🎓 {v['nivel']}")
                if v["tags"] != "—": meta_line.append(f"🏷️ {v['tags']}")
                if meta_line:
                    w("  " + "  ·  ".join(meta_line))
                w()

    caminho_catalogo = os.path.join(PASTA_SAIDA, "_CATALOGO.md")
    with open(caminho_catalogo, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))

    print(f"  📋 Catálogo atualizado: {caminho_catalogo}")

_RE_TAG     = re.compile(r"\[[^\]]+\]")
_RE_ESPACO  = re.compile(r"[ \t]+")
_RE_NEWLINE = re.compile(r"\n{3,}")
_RE_REPETE  = re.compile(r"(\b\w{1,16}\b)(\s+\1\b)+", re.IGNORECASE)


def _limpar_local(texto: str) -> str:
    """Limpeza por regex — sem tokens Groq, muito rápida."""
    if not texto:
        return ""
    t = _RE_TAG.sub(" ", texto)
    t = _RE_ESPACO.sub(" ", t)
    linhas = [ln.strip() for ln in t.splitlines()]
    t = "\n".join(ln for ln in linhas if ln)
    t = _RE_NEWLINE.sub("\n\n", t)
    t = _RE_REPETE.sub(r"\1", t)
    return t.strip()


def _dividir_para_limpeza(texto: str) -> List[str]:
    """
    Parte em blocos de até LIMPEZA_MAX_CHARS chars, quebrando em newlines
    para preservar contexto. Evita 413 / esgotamento de TPM na API Groq.
    """
    if len(texto) <= LIMPEZA_MAX_CHARS:
        return [texto]
    blocos: List[str] = []
    i, n = 0, len(texto)
    while i < n:
        end = min(i + LIMPEZA_MAX_CHARS, n)
        if end < n:
            trecho = texto[i:end]
            nl = trecho.rfind("\n")
            if nl > LIMPEZA_MAX_CHARS // 3:
                end = i + nl + 1
        blocos.append(texto[i:end])
        i = end
    return blocos


_SISTEMA_LIMPEZA = (
    "Você é um editor de texto especializado em transcrições de vídeos. "
    "Sua única função é limpar e formatar o texto recebido.\n\n"
    "Regras:\n"
    "- Remova marcações como [música], [aplausos], [risadas], [inaudível]\n"
    "- Corrija pontuação e capitalização óbvias\n"
    "- Remova palavras repetidas em sequência (ex: 'e e e então')\n"
    "- Adicione parágrafos onde há mudança clara de assunto\n"
    "- NÃO resuma, NÃO altere o conteúdo, NÃO adicione informações\n"
    "- Retorne APENAS o texto limpo, sem comentários\n"
    "- Se indicado 'Trecho N de M', limpe só esse trecho; não antecipe outros"
)


def limpar_transcricao(texto_bruto: str) -> str:
    """
    Limpa a transcrição de vídeo conforme LIMPEZA_TRANSCRICAO:
      local  (padrão) → regex, 0 tokens, muito rápido
      ia              → Groq em blocos, melhor qualidade
      nenhuma         → texto bruto
    """
    modo = LIMPEZA_TRANSCRICAO
    if modo in ("nenhuma", "none", "off", "raw"):
        print("  ⏭️  Limpeza desativada — texto bruto.")
        return (texto_bruto or "").strip()
    if modo in ("local", "rapida", "fast", "regex"):
        print("  🧹 Limpeza local (regex, 0 tokens)...")
        return _limpar_local(texto_bruto)

    # modo ia
    print("  🧹 Limpando com Groq (modo ia)...")
    partes = _dividir_para_limpeza(texto_bruto)
    total  = len(partes)
    if total > 1:
        print(f"     ({total} blocos de até {LIMPEZA_MAX_CHARS} chars)")
    saidas: List[str] = []
    for idx, bloco in enumerate(partes, start=1):
        cab = f"Trecho {idx} de {total}.\n\n" if total > 1 else ""
        saidas.append(chamar_groq(
            _SISTEMA_LIMPEZA,
            f"{cab}Limpe:\n\n{bloco}",
            modelo=MODELO_LIMPEZA,
        ))
        if idx < total:
            time.sleep(LIMPEZA_PAUSA_S)
    return "\n\n".join(s.strip() for s in saidas if s.strip())


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO: ARTIGOS WEB
# ══════════════════════════════════════════════════════════════════════════════

def _get_soup(url: str) -> Optional[BeautifulSoup]:
    try:
        r = http.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return BeautifulSoup(r.text, "html.parser")
    except requests.RequestException as e:
        print(f"   ⚠️  GET falhou ({url}): {e}")
        return None


def raspar_artigo(url: str) -> Optional[dict]:
    """Raspa artigo web com seletores inteligentes e remoção de lixo."""
    soup = _get_soup(url)
    if soup is None:
        return None

    for sel in SELETORES_LIXO:
        for el in soup.select(sel):
            el.decompose()

    h1    = soup.find("h1")
    title = soup.find("title")
    titulo = (
        h1.get_text(strip=True)    if h1    else
        title.get_text(strip=True) if title else
        urlparse(url).path.split("/")[-1] or "Sem título"
    )

    el = next(
        (soup.select_one(s) for s in SELETORES_CONTEUDO if soup.select_one(s)),
        soup.find("body") or soup,
    )
    conteudo = re.sub(r"\s{2,}", " ", el.get_text(separator=" ", strip=True))

    if len(conteudo) < 150:
        print(f"   ⚠️  Conteúdo muito curto ({len(conteudo)} chars) — pulando")
        return None

    return {"titulo": titulo, "conteudo": conteudo, "url": url}


def processar_artigo(url: str, processadas: Set[str]) -> int:
    """
    Raspa, gera metadados, indexa e salva um artigo.
    Retorna: chunks indexados (≥1=ok) | 0=já processado | -1=erro.
    """
    if url in processadas:
        return 0

    artigo = raspar_artigo(url)
    if not artigo:
        return -1

    print(f"   ✓ \"{artigo['titulo'][:65]}\" — {len(artigo['conteudo'])} chars")

    meta_ia = gerar_metadados(artigo["titulo"], artigo["conteudo"], "artigo")
    dominio = urlparse(url).netloc

    n = indexar_conteudo(
        conteudo=artigo["conteudo"],
        titulo=artigo["titulo"],
        url=url,
        tipo="artigo",
        meta_extra={"dominio": dominio},
        meta_ia=meta_ia,
    )
    salvar_arquivos(
        titulo=artigo["titulo"],
        url=url,
        conteudo=artigo["conteudo"],
        meta_ia=meta_ia,
        tipo="artigo",
        extra={"dominio": dominio},
    )
    marcar_processada(url, processadas)
    print(f"   ✓ {n} chunks indexados")
    salvar_catalogo()
    return n


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO: VÍDEOS YOUTUBE
# ══════════════════════════════════════════════════════════════════════════════

_RE_YT_ID = re.compile(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})")


def extrair_video_id(url: str) -> Optional[str]:
    """Aceita URL completa, youtu.be, shorts ou ID puro de 11 chars."""
    s = url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    m = _RE_YT_ID.search(s)
    return m.group(1) if m else None


def _metadados_yt(video_id: str) -> dict:
    """Obtém título e canal via oEmbed — sem precisar de API key."""
    query = urlencode({
        "url":    f"https://www.youtube.com/watch?v={video_id}",
        "format": "json",
    })
    try:
        with urlopen(f"https://www.youtube.com/oembed?{query}", timeout=10) as r:
            data = json.loads(r.read().decode())
        return {
            "titulo": data.get("title",       video_id),
            "canal":  data.get("author_name", ""),
        }
    except Exception:
        return {"titulo": video_id, "canal": ""}


def processar_video(url: str, processadas: Set[str]) -> int:
    """
    Transcreve, limpa, gera metadados, indexa e salva um vídeo do YouTube.
    Retorna: chunks indexados (≥1=ok) | 0=já processado | -1=erro.
    """
    if url in processadas:
        return 0

    video_id = extrair_video_id(url)
    if not video_id:
        print(f"   ⚠️  Não foi possível extrair ID do vídeo: {url}")
        return -1

    print(f"   📹 Baixando transcrição: {url}")
    try:
        api         = YouTubeTranscriptApi()
        transcript  = api.fetch(video_id, languages=["pt", "pt-BR", "en"])
        texto_bruto = " ".join(
            t["text"] if isinstance(t, dict) else t.text for t in transcript
        )
    except Exception as e:
        print(f"   ⚠️  Transcrição indisponível: {e}")
        return -1

    # Duração estimada a partir do último item da transcrição
    ultimo  = transcript[-1]
    start   = float(ultimo["start"]    if isinstance(ultimo, dict) else getattr(ultimo, "start",    0))
    dur     = float(ultimo["duration"] if isinstance(ultimo, dict) else getattr(ultimo, "duration", 0))
    dur_s   = int(math.ceil(start + dur))
    duracao = f"{dur_s // 3600:02d}:{(dur_s % 3600) // 60:02d}:{dur_s % 60:02d}"

    meta_yt = _metadados_yt(video_id)
    titulo  = meta_yt["titulo"]
    canal   = meta_yt["canal"]
    print(f"   ✓ \"{titulo[:65]}\" — {duracao}")

    conteudo = limpar_transcricao(texto_bruto)
    meta_ia  = gerar_metadados(titulo, conteudo, "video_youtube")

    n = indexar_conteudo(
        conteudo=conteudo,
        titulo=titulo,
        url=url,
        tipo="video_youtube",
        meta_extra={"canal": canal, "duracao": duracao, "video_id": video_id},
        meta_ia=meta_ia,
    )
    salvar_arquivos(
        titulo=titulo,
        url=url,
        conteudo=conteudo,
        meta_ia=meta_ia,
        tipo="video_youtube",
        extra={"canal": canal, "duracao": duracao, "video_id": video_id},
    )
    marcar_processada(url, processadas)
    print(f"   ✓ {n} chunks indexados")
    salvar_catalogo()
    return n


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO: PROCESSAMENTO DE ARQUIVOS LOCAIS (.txt + .json)
# ══════════════════════════════════════════════════════════════════════════════

def pipeline_processar_pasta() -> None:
    """
    Varre PASTA_SAIDA e processa pares .txt + .json ainda não indexados.
    Compatível com arquivos gerados por versões anteriores do sistema.
    """
    if not os.path.exists(PASTA_SAIDA):
        print(f"Pasta não encontrada: {PASTA_SAIDA}")
        return

    txts = [a for a in os.listdir(PASTA_SAIDA) if a.endswith(".txt")]
    if not txts:
        print("Nenhum .txt encontrado na pasta.")
        return

    print(f"\n📂 {len(txts)} arquivo(s) encontrado(s)\n{'─'*50}")

    for nome_txt in txts:
        nome_base    = nome_txt[:-4]
        caminho_txt  = os.path.join(PASTA_SAIDA, nome_txt)
        caminho_json = os.path.join(PASTA_SAIDA, f"{nome_base}.json")

        if not os.path.exists(caminho_json):
            print(f"⚠️  JSON não encontrado para: {nome_txt} — pulando")
            continue

        try:
            with open(caminho_json, encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️  JSON inválido ({caminho_json}): {e} — pulando")
            continue

        if meta.get("indexado_chroma") is True:
            print(f"⏭️  Já indexado: {meta.get('titulo', nome_base)}")
            continue

        titulo = meta.get("titulo", nome_base)
        tipo   = meta.get("tipo", "video_youtube")
        print(f"\n🎬 Processando: {titulo}")

        with open(caminho_txt, encoding="utf-8") as f:
            texto_bruto = f.read()

        texto_limpo = limpar_transcricao(texto_bruto)
        with open(caminho_txt, "w", encoding="utf-8") as f:
            f.write(texto_limpo)

        meta_ia = gerar_metadados(titulo, texto_limpo, tipo)
        extra   = {k: v for k, v in meta.items()
                   if k in ("canal", "dominio", "duracao", "video_id")}

        n = indexar_conteudo(
            conteudo=texto_limpo,
            titulo=titulo,
            url=meta.get("url", ""),
            tipo=tipo,
            meta_extra=extra,
            meta_ia=meta_ia,
        )

        meta.update(meta_ia)
        meta["indexado_chroma"] = True
        with open(caminho_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # Registra no histórico de URLs processadas (se tiver URL válida)
        url_meta = meta.get("url", "")
        if url_meta:
            processadas_set = carregar_processadas()
            marcar_processada(url_meta, processadas_set)

        print(f"  ✅ {titulo} — {n} chunks indexados")

    print(f"\n🏁 Concluído! Total de chunks no banco: {colecao.count()}")
    salvar_catalogo()


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO: LOTE DE VÍDEOS YOUTUBE
# ══════════════════════════════════════════════════════════════════════════════

def _normalizar_url_yt(url: str) -> str:
    """Normaliza ID puro (11 chars) para URL completa."""
    url = url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return f"https://www.youtube.com/watch?v={url}"
    return url


def _ler_urls_arquivo(caminho: str) -> List[str]:
    """
    Lê URLs de um arquivo de texto — uma por linha.
    Ignora linhas em branco e comentários iniciados com '#'.
    """
    if not os.path.exists(caminho):
        print(f"❌ Arquivo não encontrado: {caminho}")
        return []
    with open(caminho, encoding="utf-8") as f:
        linhas = f.readlines()
    urls = []
    for linha in linhas:
        linha = linha.strip()
        if not linha or linha.startswith("#"):
            continue
        urls.append(_normalizar_url_yt(linha))
    return urls


def pipeline_artigos_em_lote(urls: List[str]) -> None:
    """
    Processa uma lista de URLs de artigos em sequência.
    Pula URLs inválidas e já indexadas, com resumo ao final.
    """
    urls_validas: List[str] = []
    for url in urls:
        u = url.strip()
        if not _eh_url(u):
            print(f"⚠️  Ignorando (URL inválida): {u}")
            continue
        if _eh_url_youtube(u):
            print(f"⚠️  Ignorando (é YouTube, use opção de vídeos): {u}")
            continue
        urls_validas.append(u.rstrip("/"))

    if not urls_validas:
        print("Nenhuma URL de artigo válida para processar.")
        return

    processadas  = carregar_processadas()
    novas        = [u for u in urls_validas if u not in processadas]
    ja_indexadas = len(urls_validas) - len(novas)
    total        = len(novas)

    print(f"\n{'═'*55}")
    print(f"  📄 LOTE DE ARTIGOS — {total} novo(s) | {ja_indexadas} já indexado(s)")
    print(f"{'═'*55}")

    if not novas:
        print("✅ Todos os artigos já estavam indexados. Nada a fazer.")
        return

    total_chunks = 0
    erros = 0

    for i, url in enumerate(novas, start=1):
        print(f"\n[{i}/{total}] 📄 {url}")
        n = processar_artigo(url, processadas)
        if n < 0:
            erros += 1
        else:
            total_chunks += n
        if i < total:
            time.sleep(PAUSA_ENTRE_REQS)

    print(f"\n{'═'*55}")
    print(f"  ✅ {total - erros} indexado(s) | {erros} erro(s)")
    print(f"  📦 Chunks nesta execução    : {total_chunks}")
    print(f"  📦 Total de chunks no banco : {colecao.count()}")
    print(f"{'═'*55}")
    salvar_catalogo()


def pipeline_videos_em_lote(urls: List[str]) -> None:
    """
    Processa uma lista de URLs/IDs de vídeos do YouTube em sequência.
    Pula URLs já indexadas, exibe progresso e resumo ao final.
    """
    # Normaliza todas as URLs e filtra inválidas
    urls_normalizadas = []
    for url in urls:
        url = _normalizar_url_yt(url)
        if not _eh_url_youtube(url):
            print(f"⚠️  Ignorando (não é YouTube): {url}")
            continue
        urls_normalizadas.append(url)

    if not urls_normalizadas:
        print("Nenhuma URL válida para processar.")
        return

    processadas   = carregar_processadas()
    novas         = [u for u in urls_normalizadas if u not in processadas]
    ja_indexadas  = len(urls_normalizadas) - len(novas)
    total         = len(novas)

    print(f"\n{'═'*55}")
    print(f"  🎬 LOTE DE VÍDEOS — {total} novo(s) | {ja_indexadas} já indexado(s)")
    print(f"{'═'*55}")

    if not novas:
        print("✅ Todos os vídeos já estavam indexados. Nada a fazer.")
        return

    total_chunks = 0
    erros        = 0

    for i, url in enumerate(novas, start=1):
        print(f"\n[{i}/{total}] 🎬 {url}")
        n = processar_video(url, processadas)
        if n < 0:
            erros += 1
        else:
            total_chunks += n
        # Pausa entre requisições para não sobrecarregar a API
        if i < total:
            time.sleep(PAUSA_ENTRE_REQS)

    print(f"\n{'═'*55}")
    print(f"  ✅ {total - erros} indexado(s) | {erros} erro(s)")
    print(f"  📦 Chunks nesta execução    : {total_chunks}")
    print(f"  📦 Total de chunks no banco : {colecao.count()}")
    print(f"{'═'*55}")


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO: BUSCA FUZZY DE ARQUIVO LOCAL (indexação por nome/tema)
# ══════════════════════════════════════════════════════════════════════════════

def _normalizar(texto: str) -> str:
    return " ".join(texto.lower().strip().split())


def _tokenizar(texto: str) -> Set[str]:
    tokens: Set[str] = set()
    for parte in _normalizar(texto).split():
        limpo = "".join(ch for ch in parte if ch.isalnum())
        if len(limpo) >= 3:
            tokens.add(limpo)
    return tokens


def _score_relevancia(consulta: str, candidato: dict) -> float:
    """
    Score combinado:
      70% sobreposição de tokens (palavras com ≥3 chars em comum)
      30% similaridade de string via difflib
    """
    alvo = _normalizar(
        f"{candidato['titulo']} {candidato['nome_txt']} {candidato.get('amostra', '')}"
    )
    cons = _normalizar(consulta)
    if not cons or not alvo:
        return 0.0
    ratio       = difflib.SequenceMatcher(None, cons, alvo).ratio()
    t_cons      = _tokenizar(cons)
    t_alvo      = _tokenizar(alvo)
    token_score = len(t_cons & t_alvo) / max(1, len(t_cons))
    return 0.7 * token_score + 0.3 * ratio


def _candidatos_locais() -> List[dict]:
    """Retorna arquivos locais (.txt + .json) ainda não indexados."""
    candidatos = []
    if not os.path.exists(PASTA_SAIDA):
        return candidatos

    for nome_txt in os.listdir(PASTA_SAIDA):
        if not nome_txt.endswith(".txt"):
            continue
        nome_base    = nome_txt[:-4]
        caminho_txt  = os.path.join(PASTA_SAIDA, nome_txt)
        caminho_json = os.path.join(PASTA_SAIDA, f"{nome_base}.json")

        if not os.path.exists(caminho_json):
            continue
        try:
            with open(caminho_json, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        if meta.get("indexado_chroma") is True:
            continue

        try:
            with open(caminho_txt, encoding="utf-8") as f:
                amostra = f.read(2500)
        except OSError:
            amostra = ""

        candidatos.append({
            "titulo":       meta.get("titulo", nome_base),
            "nome_txt":     nome_txt,
            "caminho_txt":  caminho_txt,
            "caminho_json": caminho_json,
            "amostra":      amostra,
        })
    return candidatos


def _selecionar_candidato(candidatos: List[dict], consulta: str) -> Optional[dict]:
    """
    Seleciona o candidato mais relevante por score combinado.
    Abaixo do limiar de confiança (0.35), apresenta as opções ao usuário.
    """
    if not candidatos or not consulta.strip():
        return None

    ranqueados = sorted(
        ((_score_relevancia(consulta, c), c) for c in candidatos),
        reverse=True,
    )
    melhor_score, melhor = ranqueados[0]

    if melhor_score >= 0.35:
        return melhor

    print("\nMatch automático incerto. Candidatos mais próximos:")
    top = ranqueados[:5]
    for i, (score, c) in enumerate(top, start=1):
        print(f"  {i}. [{score:.2f}] {c['titulo']} ({c['nome_txt']})")

    escolha = input("Número correto (ou Enter para cancelar): ").strip()
    if escolha.isdigit():
        idx = int(escolha) - 1
        if 0 <= idx < len(top):
            return top[idx][1]
    return None


def pipeline_indexar_video_por_nome() -> None:
    """
    Pede uma descrição do arquivo local, encontra via busca fuzzy
    e indexa interativamente após confirmação do usuário.
    """
    candidatos = _candidatos_locais()
    if not candidatos:
        print(f"Nenhum arquivo não indexado encontrado em: {PASTA_SAIDA}")
        return

    consulta  = input("Descreva o vídeo/artigo (nome, tema, palavras do conteúdo): ").strip()
    candidato = _selecionar_candidato(candidatos, consulta)

    if not candidato:
        print("Não foi possível identificar o arquivo.")
        return

    print(f"\nEncontrado: {candidato['titulo']} ({candidato['nome_txt']})")
    confirma = input("É esse? (s/n): ").strip().lower()
    if confirma not in ("s", "sim", "y", "yes"):
        print("Cancelado.")
        return

    try:
        with open(candidato["caminho_json"], encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        print(f"Erro ao ler JSON: {e}")
        return

    titulo = meta.get("titulo", candidato["nome_txt"])
    tipo   = meta.get("tipo", "video_youtube")
    print(f"\n🎬 Processando: {titulo}")

    with open(candidato["caminho_txt"], encoding="utf-8") as f:
        texto_bruto = f.read()

    texto_limpo = limpar_transcricao(texto_bruto)
    with open(candidato["caminho_txt"], "w", encoding="utf-8") as f:
        f.write(texto_limpo)

    meta_ia = gerar_metadados(titulo, texto_limpo, tipo)
    extra   = {k: v for k, v in meta.items()
               if k in ("canal", "dominio", "duracao", "video_id")}

    n = indexar_conteudo(
        conteudo=texto_limpo,
        titulo=titulo,
        url=meta.get("url", ""),
        tipo=tipo,
        meta_extra=extra,
        meta_ia=meta_ia,
    )
    meta.update(meta_ia)
    meta["indexado_chroma"] = True
    with open(candidato["caminho_json"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Registra no histórico de URLs processadas (se tiver URL válida)
    url_meta = meta.get("url", "")
    if url_meta:
        processadas_set = carregar_processadas()
        marcar_processada(url_meta, processadas_set)

    print(f"\n🏁 Concluído! {n} chunks indexados. Total no banco: {colecao.count()}")
    salvar_catalogo()


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO: DESCOBERTA DE LINKS (CRAWLER WEB)
# ══════════════════════════════════════════════════════════════════════════════

def _eh_url_youtube(url: str) -> bool:
    return (
        "youtube.com/watch" in url or
        "youtu.be/" in url or
        "youtube.com/shorts/" in url
    )


def _eh_paginacao(tag_a, href: str, url_base: str) -> bool:
    """Detecta links de paginação por texto, classe CSS ou padrão de URL."""
    texto   = tag_a.get_text(strip=True).lower()
    classes = " ".join(tag_a.get("class", [])).lower()
    gatilhos_texto = {"próximo", "proximo", "next", "seguinte", "›", "»", ">", "mais"}
    if any(p in texto for p in gatilhos_texto):
        return True
    if any(p in classes for p in ("next", "pagination", "pager", "proximo")):
        return True
    if re.search(r"(/page/\d+|[?&]p(?:age)?=\d+)$", href):
        return True
    base_path = urlparse(url_base).path.rstrip("/")
    if re.match(rf"^{re.escape(base_path)}/\d+$", urlparse(href).path.rstrip("/")):
        return True
    return False


def descobrir_links(
    url_base: str,
    filtro_path: Optional[str] = None,
    seguir_paginacao: bool = True,
) -> dict:
    """
    Varre a página de listagem (com suporte a paginação) e separa links em:
      - artigos : páginas do mesmo domínio que passam no filtro de path
      - videos  : links do YouTube
    Retorna {"artigos": [...], "videos": [...]}
    """
    dominio   = urlparse(url_base).netloc
    visitadas: Set[str] = set()
    artigos:   Set[str] = set()
    videos:    Set[str] = set()
    fila      = [url_base]

    print(f"\n🔍 Descobrindo links em: {url_base}")
    if filtro_path:
        print(f"   Filtro de path: '{filtro_path}'")

    while fila:
        url_atual = fila.pop(0)
        if url_atual in visitadas:
            continue
        visitadas.add(url_atual)

        soup = _get_soup(url_atual)
        if soup is None:
            continue

        for tag_a in soup.find_all("a", href=True):
            href = urljoin(url_atual, tag_a["href"]).split("#")[0].rstrip("/")
            if not href:
                continue

            if _eh_url_youtube(href):
                videos.add(href)
                continue

            parsed = urlparse(href)
            if parsed.netloc != dominio:
                continue
            if href == url_atual or href in visitadas:
                continue

            if filtro_path and filtro_path not in href:
                if seguir_paginacao and _eh_paginacao(tag_a, href, url_base):
                    fila.append(href)
                continue

            # Página de paginação: enfileira para crawl mas NÃO indexa como artigo
            if seguir_paginacao and _eh_paginacao(tag_a, href, url_base):
                fila.append(href)
                continue

            artigos.add(href)

        time.sleep(PAUSA_ENTRE_REQS)

    print(f"   ✓ {len(artigos)} artigo(s) | {len(videos)} vídeo(s) YouTube")
    return {"artigos": sorted(artigos), "videos": sorted(videos)}


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO: Q&A COM CLASSIFICAÇÃO INTELIGENTE DE PERGUNTAS
# ══════════════════════════════════════════════════════════════════════════════

def _buscar_titulos() -> List[str]:
    """Retorna todos os títulos únicos presentes no ChromaDB."""
    resultado = colecao.get(include=["metadatas"])
    if not resultado or not resultado.get("metadatas"):
        return []
    return list({m.get("titulo", "") for m in resultado["metadatas"] if m.get("titulo")})


def _classificar_pergunta(pergunta: str, titulos: List[str]) -> dict:
    """
    Usa Groq para classificar a pergunta em:
      especifica  → filtra por fonte_alvo, busca 3 chunks
      geral       → busca ampla, 5 chunks
      comparativa → busca ampla, 6 chunks
      fora_de_escopo → não busca, resposta direta
    """
    print("  🔀 Classificando pergunta...")
    lista = "\n".join(f"- {t}" for t in titulos)
    raw = chamar_groq(
        sistema="Você é um classificador de perguntas para um sistema RAG. Retorne APENAS um JSON válido.",
        usuario=(
            f"Fontes disponíveis (artigos e vídeos indexados):\n{lista}\n\n"
            f"Pergunta: \"{pergunta}\"\n\n"
            f"Retorne JSON com exatamente essas chaves:\n"
            f'{{"tipo": "especifica|geral|comparativa|fora_de_escopo", '
            f'"fonte_alvo": "título exato se especifica, senão null", '
            f'"n_chunks": 5, '
            f'"raciocinio": "uma linha"}}\n\n'
            f"Regras para n_chunks: especifica=3, geral=5, comparativa=6, fora_de_escopo=0"
        ),
        json_mode=True,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"tipo": "geral", "fonte_alvo": None, "n_chunks": 5, "raciocinio": "fallback"}


def _buscar_chunks(pergunta: str, classificacao: dict) -> List[dict]:
    n    = max(1, classificacao.get("n_chunks", 5))
    tipo = classificacao.get("tipo", "geral")
    alvo = classificacao.get("fonte_alvo")

    # ChromaDB lança exceção se n_results > total de documentos na coleção
    total_docs = colecao.count()
    if total_docs == 0:
        return []
    n = min(n, total_docs)

    kwargs: dict = {"query_texts": [pergunta], "n_results": n}
    if tipo == "especifica" and alvo:
        kwargs["where"] = {"titulo": alvo}

    try:
        resultado = colecao.query(**kwargs)
    except Exception:
        # Se o filtro falhar (ex.: fonte_alvo não encontrada), tenta sem filtro
        resultado = colecao.query(query_texts=[pergunta], n_results=n)

    docs  = resultado.get("documents", [[]])[0]
    metas = resultado.get("metadatas", [[]])[0]
    return [
        {
            "texto":  doc,
            "titulo": meta.get("titulo", ""),
            "url":    meta.get("url", ""),
            "tipo":   meta.get("tipo", ""),
            "chunk":  meta.get("chunk_index", 0),
        }
        for doc, meta in zip(docs, metas)
    ]


def pipeline_perguntar(pergunta: str) -> str:
    """
    Q&A com classificação inteligente:
    1. Classifica a pergunta (específica / geral / comparativa / fora de escopo)
    2. Busca chunks no ChromaDB (com ou sem filtro de fonte)
    3. Responde usando Groq com o contexto encontrado, citando fontes
    """
    titulos = _buscar_titulos()
    if not titulos:
        return "Base de conhecimento vazia. Indexe algum conteúdo primeiro."

    classificacao = _classificar_pergunta(pergunta, titulos)
    print(f"  Tipo: {classificacao['tipo']} | {classificacao.get('raciocinio', '')}")

    if classificacao["tipo"] == "fora_de_escopo":
        return "Essa pergunta está fora do escopo do conteúdo disponível."

    chunks = _buscar_chunks(pergunta, classificacao)
    if not chunks:
        return "Não encontrei conteúdo relevante para essa pergunta."

    blocos = []
    for i, c in enumerate(chunks):
        tipo_label = "🎬 Vídeo" if c["tipo"] == "video_youtube" else "📄 Artigo"
        blocos.append(
            f"[Fonte {i+1} — {tipo_label}]\n"
            f"Título: {c['titulo']}\n"
            f"URL: {c['url']}\n\n"
            f"{c['texto']}"
        )
    contexto = ("\n\n" + "─" * 60 + "\n").join(blocos)

    print("  💬 Gerando resposta com Groq...")
    return chamar_groq(
        sistema=(
            "Você responde perguntas com base em artigos e transcrições de vídeos. "
            "Use APENAS o contexto fornecido. Cite as fontes pelo título e URL. "
            "Ao final, liste todas as fontes usadas. "
            "Se a informação não estiver no contexto, diga exatamente: "
            "'Não encontrei essa informação na base de conhecimento.'"
        ),
        usuario=f"Pergunta: {pergunta}\n\nContexto:\n{contexto}",
        modelo=MODELO_POTENTE,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL: CRAWLER
# ══════════════════════════════════════════════════════════════════════════════

def pipeline_crawler(
    url_listagem: str,
    filtro_path: Optional[str] = None,
    sem_paginacao: bool = False,
) -> None:
    """Descobre todos os links da listagem e processa artigos + vídeos."""
    processadas = carregar_processadas()

    encontrados = descobrir_links(
        url_listagem,
        filtro_path=filtro_path,
        seguir_paginacao=not sem_paginacao,
    )
    artigos = [u for u in encontrados["artigos"] if u not in processadas]
    videos  = [u for u in encontrados["videos"]  if u not in processadas]
    pulados = (
        (len(encontrados["artigos"]) - len(artigos)) +
        (len(encontrados["videos"])  - len(videos))
    )

    print(f"\n📋 Novos: {len(artigos)} artigo(s) + {len(videos)} vídeo(s)"
          f"  |  {pulados} já processado(s)")

    if not artigos and not videos:
        print("✅ Tudo já indexado. Nada a fazer.")
        return

    total_chunks = 0
    erros        = 0
    total        = len(artigos) + len(videos)
    i            = 0

    for url in artigos:
        i += 1
        print(f"\n[{i}/{total}] 📄 {url}")
        n = processar_artigo(url, processadas)
        if n < 0:
            erros += 1
        else:
            total_chunks += n
        if i < total:
            time.sleep(PAUSA_ENTRE_REQS)

    for url in videos:
        i += 1
        print(f"\n[{i}/{total}] 🎬 {url}")
        n = processar_video(url, processadas)
        if n < 0:
            erros += 1
        else:
            total_chunks += n
        if i < total:
            time.sleep(PAUSA_ENTRE_REQS)

    print(f"\n{'═'*55}")
    print(f"  ✅ {total - erros} indexado(s) | {erros} erro(s)")
    print(f"  📦 Chunks nesta execução    : {total_chunks}")
    print(f"  📦 Total de chunks no banco : {colecao.count()}")
    print(f"  💾 Arquivos em: {PASTA_SAIDA}")
    print(f"{'═'*55}")
    salvar_catalogo()


# ══════════════════════════════════════════════════════════════════════════════
#  STATUS DA BASE
# ══════════════════════════════════════════════════════════════════════════════

def status_indice() -> None:
    processadas   = carregar_processadas()
    arquivos_json = (
        [f for f in os.listdir(PASTA_SAIDA) if f.endswith(".json")]
        if os.path.exists(PASTA_SAIDA) else []
    )
    artigos_count = sum(1 for u in processadas if "youtube" not in u)
    videos_count  = sum(1 for u in processadas if "youtube" in u)
    dominios: dict = {}
    for u in processadas:
        if "youtube" not in u:
            d = urlparse(u).netloc
            dominios[d] = dominios.get(d, 0) + 1

    print(f"\n{'═'*55}")
    print("  📊 STATUS DA BASE DE CONHECIMENTO")
    print(f"{'═'*55}")
    print(f"  Pasta            : {PASTA_SAIDA}")
    print(f"  Artigos indexados: {artigos_count}")
    print(f"  Vídeos YT        : {videos_count}")
    print(f"  Arquivos .json   : {len(arquivos_json)}")
    print(f"  Chunks no banco  : {colecao.count()}")
    if dominios:
        print("\n  Domínios de artigos:")
        for d, n in sorted(dominios.items(), key=lambda x: -x[1]):
            print(f"    {d:<42} {n:>3} artigo(s)")
    print(f"{'═'*55}")
    salvar_catalogo()


# ══════════════════════════════════════════════════════════════════════════════
#  MENU INTERATIVO
# ══════════════════════════════════════════════════════════════════════════════

def _parece_youtube(s: str) -> bool:
    s = (s or "").strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return True
    low = s.lower()
    return "youtube.com" in low or "youtu.be" in low


def menu() -> None:
    L = 52  # largura do menu
    def _linha(c="═"): print(f"  {c * L}")
    def _titulo(t): print(f"  {t}")
    def _item(n, icone, label, detalhe=""):
        detalhe_fmt = f"  {detalhe}" if detalhe else ""
        print(f"    [{n}] {icone}  {label}{detalhe_fmt}")
    def _sep(): print(f"  {'─' * L}")

    while True:
        print()
        _linha()
        _titulo("🧠  RAG SYSTEM")
        _linha()
        print()
        _titulo("  INDEXAR")
        _sep()
        _item("1", "📄", "Artigos por URL",        "uma ou várias URLs em sequência")
        _item("3", "🎬", "Vídeo(s) do YouTube",   "um ou vários (URL, ID ou arquivo .txt)")
        _item("4", "📂", "Arquivos locais",        "processa .txt + .json da pasta de saída")
        _item("5", "🔍", "Buscar arquivo local",   "localiza por nome ou tema")
        _item("6", "🔄", "Forçar reindexação",     "reprocessa URL já indexada")
        print()
        _titulo("  CONSULTAR")
        _sep()
        _item("7", "💬", "Fazer uma pergunta")
        _item("8", "📊", "Status da base")
        print()
        _linha()
        _item("0", "🚪", "Sair")
        _linha()
        print()
        opcao = input("  Opção: ").strip()
        print()

        if opcao == "0":
            print("  Até logo!")
            break

        # ── 1/2. Artigos por URL (lote) ───────────────────────────────────
        elif opcao in ("1", "2"):
            print("  Cole as URLs dos artigos, uma por linha.")
            print("  Ou informe o caminho de um arquivo .txt com a lista.")
            print("  ── linha vazia para confirmar ──")
            entradas: List[str] = []
            while True:
                linha = input("  > ").strip()
                if not linha:
                    break
                entradas.append(linha)

            if not entradas:
                print("  Nenhuma entrada informada.")
                continue

            if len(entradas) == 1 and os.path.exists(entradas[0]) and not _eh_url(entradas[0]):
                urls_lote = _ler_urls_arquivo(entradas[0])
            else:
                urls_lote = entradas

            try:
                pipeline_artigos_em_lote(urls_lote)
            except Exception as e:
                print(f"  ❌ Erro ao processar lote de artigos: {e}")

        # ── 3. Vídeo(s) do YouTube ────────────────────────────────────────
        elif opcao == "3" or _parece_youtube(opcao):
            # Permite colar a URL/ID direto como opção do menu
            if _parece_youtube(opcao):
                entradas = [opcao]
            else:
                print("  Cole as URLs ou IDs dos vídeos, uma por linha.")
                print("  Ou informe o caminho de um arquivo .txt com a lista.")
                print("  ── linha vazia para confirmar ──")
                entradas = []
                while True:
                    linha = input("  > ").strip()
                    if not linha:
                        break
                    entradas.append(linha)

            if not entradas:
                print("  Nenhuma entrada informada.")
                continue

            # Se for um único caminho de arquivo, lê o arquivo
            if len(entradas) == 1 and os.path.exists(entradas[0]) and not _parece_youtube(entradas[0]):
                urls_lote = _ler_urls_arquivo(entradas[0])
            else:
                urls_lote = entradas

            if len(urls_lote) == 1:
                url = _normalizar_url_yt(urls_lote[0])
                processadas = carregar_processadas()
                n = processar_video(url, processadas)
                _exibir_resultado_avulso(n)
                print(f"  📦 Total de chunks no banco: {colecao.count()}")
            else:
                pipeline_videos_em_lote(urls_lote)

        # ── 4. Arquivos locais ────────────────────────────────────────────
        elif opcao == "4":
            pipeline_processar_pasta()

        # ── 5. Buscar arquivo local ───────────────────────────────────────
        elif opcao == "5":
            pipeline_indexar_video_por_nome()

        # ── 6. Forçar reindexação ─────────────────────────────────────────
        elif opcao == "6":
            url = input("  URL para reindexar: ").strip()
            processadas = carregar_processadas()
            processadas.discard(url)
            with open(ARQUIVO_PROCESSADAS, "w", encoding="utf-8") as f:
                json.dump(sorted(processadas), f, ensure_ascii=False, indent=2)
            url = _normalizar_url_yt(url)
            if _eh_url_youtube(url):
                n = processar_video(url, processadas)
            elif _eh_url(url):
                n = processar_artigo(url, processadas)
            else:
                print("  ❌ URL inválida.")
                continue
            _exibir_resultado_avulso(n)
            print(f"  📦 Total de chunks no banco: {colecao.count()}")

        # ── 7. Fazer uma pergunta ─────────────────────────────────────────
        elif opcao == "7":
            pergunta = input("  Sua pergunta: ").strip()
            if not pergunta:
                continue
            print(f"\n  🔍 {pergunta}")
            print(f"  {'─' * L}")
            print(pipeline_perguntar(pergunta))

        # ── 8. Status da base ─────────────────────────────────────────────
        elif opcao == "8":
            status_indice()

        else:
            print("  ❌ Opção inválida. Tente novamente.")


def _exibir_resultado_avulso(n: int) -> None:
    if n < 0:
        print("   ❌ Falha ao processar.")
    elif n == 0:
        print("   ℹ️  Já estava indexado.")
    else:
        print(f"   ✅ {n} chunks indexados.")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _eh_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def _main(argv: list) -> None:
    args = argv[1:]

    # ── Sem argumentos: abre menu persistente ─────────────────────────────
    if not args:
        menu()
        return

    # ── --ajuda / --help ────────────────────────────────────────────────
    if args[0] in ("-h", "--help", "--ajuda"):
        print(__doc__)
        return

    # ── --menu ───────────────────────────────────────────────────────────
    if args[0] in ("--menu", "-m"):
        menu()
        return

    # ── --status ─────────────────────────────────────────────────────────
    if args[0] == "--status":
        status_indice()
        return

    # ── --pergunta ───────────────────────────────────────────────────────
    if args[0] == "--pergunta":
        pergunta = " ".join(args[1:]).strip()
        if not pergunta:
            sys.exit('  Use: --pergunta "sua pergunta aqui"')
        print(f"\n  🔍 {pergunta}\n  {'─'*55}")
        print(pipeline_perguntar(pergunta))
        return

    # ── --artigo / --artigos / URL direta (somente links informados) ─────
    if args[0] in ("--artigo", "--artigos", "--crawl", "--site") or (args and _eh_url(args[0]) and not _eh_url_youtube(args[0])):
        entradas = args[1:] if args[0] in ("--artigo", "--artigos", "--crawl", "--site") else args
        if not entradas:
            sys.exit(
                "  Use:\n"
                '    --artigos "https://site.com/post-1" "https://site.com/post-2"\n'
                "    --artigos lista.txt"
            )

        if len(entradas) == 1 and os.path.exists(entradas[0]) and not _eh_url(entradas[0]):
            urls = _ler_urls_arquivo(entradas[0])
            if not urls:
                sys.exit("  Nenhuma URL válida encontrada no arquivo.")
        else:
            urls = entradas

        pipeline_artigos_em_lote(urls)
        return

    # ── --video  (único, vários ou arquivo .txt) ──────────────────────────
    if args[0] in ("--video", "--videos"):
        entradas = args[1:]
        if not entradas:
            sys.exit(
                "  Use:\n"
                '    --video "https://youtube.com/watch?v=ID"\n'
                '    --video "url1" "url2" "url3"\n'
                "    --video lista.txt"
            )
        # Se for um único argumento que é um arquivo existente → lê o arquivo
        if len(entradas) == 1 and os.path.exists(entradas[0]) and not _parece_youtube(entradas[0]):
            urls = _ler_urls_arquivo(entradas[0])
            if not urls:
                sys.exit("  Nenhuma URL válida encontrada no arquivo.")
        else:
            urls = entradas

        if len(urls) == 1:
            url = _normalizar_url_yt(urls[0])
            processadas = carregar_processadas()
            print(f"\n  🎬 Indexando vídeo: {url}")
            n = processar_video(url, processadas)
            _exibir_resultado_avulso(n)
            print(f"  📦 Total de chunks no banco: {colecao.count()}")
        else:
            pipeline_videos_em_lote(urls)
        return

    # ── --local ───────────────────────────────────────────────────────────
    if args[0] in ("--local", "--pasta", "-p"):
        pipeline_processar_pasta()
        return

    # ── --buscar-local ────────────────────────────────────────────────────
    if args[0] in ("--buscar-local", "--indexar-video"):
        pipeline_indexar_video_por_nome()
        return

    # ── --reindexar ───────────────────────────────────────────────────────
    if args[0] == "--reindexar":
        if len(args) < 2:
            sys.exit('  Use: --reindexar "https://..."')
        url = _normalizar_url_yt(args[1])
        processadas = carregar_processadas()
        processadas.discard(url)
        with open(ARQUIVO_PROCESSADAS, "w", encoding="utf-8") as f:
            json.dump(sorted(processadas), f, ensure_ascii=False, indent=2)
        print(f"\n  🔄 Reindexando: {url}")
        if _eh_url_youtube(url):
            n = processar_video(url, processadas)
        elif _eh_url(url):
            n = processar_artigo(url, processadas)
        else:
            sys.exit(f"  ❌ URL inválida: {url}")
        _exibir_resultado_avulso(n)
        print(f"  📦 Total de chunks no banco: {colecao.count()}")
        return

    # ── Argumento não reconhecido ─────────────────────────────────────────
    sys.exit(
        f"  ❌ Argumento não reconhecido: '{args[0]}'\n"
        f"  Use --ajuda para ver todas as opções."
    )


if __name__ == "__main__":
    _main(sys.argv)
