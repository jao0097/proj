import json
import math
import os
import re
import unicodedata
from datetime import date
from urllib.parse import urlencode
from urllib.request import urlopen

from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(video_input):
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
    raise ValueError("Nao foi possivel extrair o ID do video a partir da URL informada.")


def sanitize_filename(name):
    normalized = unicodedata.normalize("NFKD", name)
    no_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    lowered = no_accents.lower()
    with_underscores = re.sub(r"\s+", "_", lowered)
    cleaned = re.sub(r"[^a-z0-9_]", "", with_underscores)
    compacted = re.sub(r"_+", "_", cleaned).strip("_")
    return compacted or "transcricao"


def get_video_metadata(video_id):
    query = urlencode(
        {
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "format": "json",
        }
    )
    with urlopen(f"https://www.youtube.com/oembed?{query}") as response:
        data = json.loads(response.read().decode("utf-8"))
    return {
        "titulo": data.get("title", video_id),
        "canal": data.get("author_name", "Canal desconhecido"),
    }


def estimate_duration_seconds(transcript_items):
    if not transcript_items:
        return 0
    last = transcript_items[-1]
    if isinstance(last, dict):
        start = float(last.get("start", 0))
        duration = float(last.get("duration", 0))
    else:
        start = float(getattr(last, "start", 0))
        duration = float(getattr(last, "duration", 0))
    return int(math.ceil(start + duration))


def format_duration_hhmmss(total_seconds):
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

video_url_input = input("Cole a URL (ou ID) do video do YouTube: ").strip()
video_id = extract_video_id(video_url_input)
video_url = f"https://youtube.com/watch?v={video_id}"
data_coleta = date.today().isoformat()
idioma = "pt"

api = YouTubeTranscriptApi()
transcript = api.fetch(video_id, languages=["pt", "pt-BR"])

# Juntando tudo num texto limpo
texto_completo = " ".join([(trecho["text"] if isinstance(trecho, dict) else trecho.text) for trecho in transcript])
duracao_segundos = estimate_duration_seconds(transcript)
duracao_formatada = format_duration_hhmmss(duracao_segundos)
metadata = get_video_metadata(video_id)
titulo_video = metadata["titulo"]
canal_video = metadata["canal"]

# Salvando arquivos .txt e .json
documents_dir = os.path.expanduser("~/Documentos/transcricoes")
os.makedirs(documents_dir, exist_ok=True)

base_name = sanitize_filename(titulo_video)
txt_destino = os.path.join(documents_dir, f"{base_name}.txt")
json_destino = os.path.join(documents_dir, f"{base_name}.json")

txt_conteudo = (
    f"TÍTULO: {titulo_video}\n"
    f"URL: {video_url}\n"
    f"DATA DE INDEXAÇÃO: {data_coleta}\n"
    f"DURAÇÃO ESTIMADA: {duracao_formatada}\n"
    f"[TRANSCRIÇÃO]\n"
    f"{texto_completo}"
)

with open(txt_destino, "w", encoding="utf-8") as txt_file:
    txt_file.write(txt_conteudo)

json_conteudo = {
    "titulo": titulo_video,
    "url": video_url,
    "canal": canal_video,
    "data_coleta": data_coleta,
    "duracao_segundos": duracao_segundos,
    "idioma": idioma,
    "indexado_chroma": False,
}
with open(json_destino, "w", encoding="utf-8") as json_file:
    json.dump(json_conteudo, json_file, ensure_ascii=False, indent=2)

print(f"Transcrição salva em: {txt_destino}")
print(f"Metadados salvos em: {json_destino}")