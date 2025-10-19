# gemini_integracao_api_npa.py - Versão 3.6.0 com Extração e Uso de Contexto Adicional para Personalização
# -*- coding: utf-8 -*-

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import asyncio
import unicodedata

try:
    import google.generativeai as genai
except Exception:
    genai = None
    logging.warning("Biblioteca 'google.generativeai' não disponível; funcionalidades de IA podem ficar limitadas.")
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from chromadb.utils import embedding_functions
    from sentence_transformers import SentenceTransformer
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False
    logging.info("Chroma or sentence-transformers not available; using lightweight RAG fallback.")
import time
import json as _json
import re
from pathlib import Path
from pydantic_settings import BaseSettings

# Importa o módulo de análise existente
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.analise_clima import analisar_dados_climaticos
except (ImportError, ModuleNotFoundError):
    logging.exception("Falha ao importar 'analise_clima'. Usando função mock.")
    def analisar_dados_climaticos(payload: Dict, dia_alvo: str) -> Dict:
        return {"mock_data": "análise não pôde ser realizada", "dia_alvo": dia_alvo}

# ===== CONFIGURAÇÃO DE LOGGING =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# ===== GESTÃO DE CONFIGURAÇÕES =====
class Settings(BaseSettings):
    GEMINI_API_KEY: str
    NASA_POWER_BASE_URL: str = "https://power.larc.nasa.gov/api/temporal/daily/point"
    GEOCODING_API_URL: str = "https://nominatim.openstreetmap.org/search"
    DEFAULT_ANALYSIS_START_YEAR: int = 1981
    DEFAULT_ANALYSIS_END_YEAR: int = 2025
    
    class Config:
        env_file = ".env"
        extra = "ignore"

try:
    settings = Settings()
    if genai:
        genai.configure(api_key=settings.GEMINI_API_KEY)
except ValueError as e:
    logging.error(f"Erro ao carregar configurações: {e}")
    raise

# ===== CONFIGURAÇÃO DO MODELO GEMINI =====
if genai:
    try:
        gemini_model = genai.GenerativeModel('gemma-3-27b-it')
    except Exception as e:
        gemini_model = None
        logging.warning(f"Não foi possível inicializar o modelo Gemini: {e}")
else:
    gemini_model = None

# ===== PROMPTS PARA A IA (MELHORADOS COM CONTEXTO) =====
EXTRACTION_PROMPT_TEMPLATE = """
Você é um assistente especialista em extrair e contextualizar nomes de locais, datas, tipos de eventos e contextos adicionais de textos para geocodificação e personalização de recomendações.
Analise a consulta do usuário e retorne APENAS um objeto JSON com a seguinte estrutura.
O mais importante é extrair um nome de local o mais específico possível para evitar ambiguidade.
Para locais globais ou famosos (ex: Polo Norte, Monte Everest, Deserto do Saara), adicione um contexto geográfico.
Extraia também o contexto adicional do evento e categorize a atividade principal (ex: 'caminhada longa', 'festa social', 'evento desportivo').

Consulta: "{query}"

Estrutura JSON de saída:
{{
    "location_name": "<o nome do local o mais completo possível, ex: 'Polo Norte Geográfico, Ártico', 'Benguela, Angola', 'Torre Eiffel, Paris, França'>",
    "event_date": "<data no formato YYYY-MM-DD ou null>",
    "event_type": "<ex: 'Festa de Aniversário', 'Casamento', 'Corrida de Montanha'>",
    "event_context": "<descrição detalhada do contexto adicional extraído, ex: 'sem camisola e com os pés descalços, com olhos vendados, uns 45km de caminhada, pulando alto'>",
    "activity_category": "<'Endurance', 'Social', 'Lazer', 'Construção', 'Outro'>"
}}
"""

# <<< MELHORIA: Adicionado {event_context} para personalizar as recomendações com base no contexto extraído.
RECOMMENDATION_PROMPT_TEMPLATE = """
Você é "AURA", uma assistente especialista em meteorologia e planeamento de eventos da Cygnus-X1. A sua missão é traduzir dados climáticos complexos em conselhos práticos, claros e acionáveis para qualquer pessoa. Seja empática e direta.

NÃO use jargão técnico. NÃO inclua a frase "Com base nos dados fornecidos". Vá direto ao ponto.

Utilize os dados climáticos fornecidos para um evento do tipo '{event_type}' e categoria '{activity_category}', previsto para a data em '{event_date}', considerando o contexto adicional detalhado em '{event_context}'. Gere recomendações detalhadas e personalizadas fáceis de entender para que qualquer tipo de pessoa possa ser bem informada e saber exatamente o que fazer com base nessas informações.
Adapte as sugestões de acordo com o contexto disponibilizado, considerando fatores como:
- riscos específicos para atividades ao ar livre;
- potenciais problemas com vestimenta inadequada;
- condições físicas extremas.
Sempre substitua os marcadores '{event_type}', '{event_date}' e '{event_context}' pelos valores reais ao realizar sua análise.
Se algum dado climático estiver ausente, informe explicitamente quais informações são desconhecidas e explique como essa ausência pode impactar as recomendações.
Ao finalizar o processamento, valide em 1-2 linhas se todas as informações essenciais foram utilizadas e se as limitações estão claramente comunicadas. Se identificar alguma inconsistência, corrija antes de retornar o resultado.
Retorne exclusivamente um objeto JSON estruturado conforme o modelo abaixo, sem explicações adicionais nem markdown.

Dados de Análise:
{analysis_data}

Estrutura JSON de saída:
{{
    "resumo_executivo": "<parágrafo conciso sobre as condições gerais para a data correta do evento, considerando o contexto>",
    "nivel_risco": "<BAIXO, MODERADO, ALTO ou MUITO_ALTO>",
    "recomendacao_principal": "<a recomendação mais crítica, personalizada ao contexto>",
    "preparacoes_essenciais": ["<lista de 3-5 preparações críticas, adaptadas ao contexto>"],
    "alternativas_sugeridas": {{
        "datas_alternativas": ["<lista de até 3 sugestões de datas se aplicável>"],
        "horarios_alternativos": "<sugestão de horário (manhã, tarde, noite)>",
        "tipo_local": "<recomendação: 'ambiente fechado', 'ambiente aberto com cobertura' ou 'ambiente aberto'>"
    }},
    "itens_necessarios": ["<lista de 5-8 itens essenciais para os convidados/evento, personalizados ao contexto>"],
    "avisos_especiais": ["<lista de 2-3 avisos importantes (ex: risco de ventos fortes), considerando o contexto>"],
    "dica_especial": "<uma dica única e criativa baseada nos dados e no contexto>"
}}
"""

DASHBOARD_PROMPT_TEMPLATE = """
Com base nos dados de probabilidade de condições climáticas, gere insights para um dashboard.
Retorne APENAS um objeto JSON com a estrutura especificada, sem explicações ou markdown.

Dados de Probabilidade:
{probability_data}

Estrutura JSON de saída:
{{
    "headline": "<título impactante de 5-10 palavras sobre o clima>",
    "score_geral": 75,
    "emoji_clima": "<único emoji que representa o clima geral>",
    "cor_tema": "<código de cor hexadecimal (ex: #4A90E2)>",
    "tags": ["<lista de 3-5 tags relevantes (ex: 'Sol', 'Risco de Chuva')>"],
    "grafico_recomendado": "bar",
    "metricas_chave": [
        {{"nome": "Prob. Chuva", "valor": "15%", "tendencia": "stable"}}
    ]
}}
"""

# ===== CONFIGURAÇÃO DA API FASTAPI =====
app = FastAPI(
    title="Cygnus-X1 AI-Powered Weather Analysis",
    description="API com geocodificação robusta para análise climática em qualquer local do mundo, agora com extração de contexto para personalização.",
    version="3.6.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ... (O resto dos modelos Pydantic não muda, mas adicionamos event_context ao ExtractedInfo) ...
class SearchQueryRequest(BaseModel):
    query: str = Field(..., description="Consulta em linguagem natural do usuário.", min_length=10)
class ExtractedInfo(BaseModel):
    latitude: float; longitude: float; location_name: str; event_date: Optional[str] = None; event_type: str = "evento"; event_context: str = ""; activity_category: str = "Outro"; analysis_start_date: str; analysis_end_date: str
class QuickSummary(BaseModel):
    risk_level: str; rain_probability: float; heat_risk: float; main_risks: List[str]


class RecommendationModel(BaseModel):
    resumo_executivo: str
    nivel_risco: str
    recomendacao_principal: str
    preparacoes_essenciais: List[str]
    alternativas_sugeridas: Dict[str, Any]
    itens_necessarios: List[str]
    avisos_especiais: List[str]
    dica_especial: str


class DashboardModel(BaseModel):
    headline: str
    score_geral: int
    emoji_clima: str
    cor_tema: str
    tags: List[str]
    grafico_recomendado: str
    metricas_chave: List[Dict[str, Any]]


class ApiResponse(BaseModel):
    query_info: Dict[str, Any]
    climate_analysis: Dict[str, Any]
    ai_recommendations: RecommendationModel
    dashboard_insights: DashboardModel
    quick_summary: QuickSummary

# ===== FUNÇÕES AUXILIARES =====

def clean_and_parse_json(raw_text: str) -> Dict[str, Any]:
    json_str = raw_text.strip()
    if json_str.startswith("```") and json_str.endswith("```"):
        json_str = json_str.split('\n', 1)[1]
        json_str = json_str.rsplit('\n', 1)[0]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logging.error(f"Falha ao decodificar JSON após limpeza. Texto: '{json_str}'. Erro: {e}")
        raise ValueError("A resposta da IA não continha um JSON válido.")

async def get_coords_from_location_name(location_name: str) -> Tuple[float, float]:
    if not location_name:
        raise ValueError("O nome do local não pode ser vazio.")

    # Preprocess location name
    location_name = location_name.strip().lower()

    # Normalize accents and special characters
    location_name = ''.join(
        c for c in unicodedata.normalize('NFD', location_name) if unicodedata.category(c) != 'Mn'
    )

    # Remove common stop words
    stop_words = ["o", "a", "de", "do", "da", "dos", "das", "e"]
    location_name = ' '.join(word for word in location_name.split() if word not in stop_words)

    # Replace common problematic terms with normalized equivalents
    replacements = {
        "deserto": "desert",
        "floresta": "forest",
        "montanha": "mountain",
        "rio": "river",
        "praia": "beach",
        "cidade": "city",
        "vila": "village",
        "país": "country"
    }

    for term, replacement in replacements.items():
        if term in location_name:
            location_name = location_name.replace(term, replacement)

    params = {'q': location_name, 'format': 'json', 'limit': 1}
    headers = {'User-Agent': 'CygnusX1-WeatherApp/1.0', 'Accept-Language': 'en,pt;q=0.9'}

    async with httpx.AsyncClient() as client:
        try:
            logging.info(f"Buscando coordenadas para: '{location_name}'")
            response = await client.get(settings.GEOCODING_API_URL, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            if data and isinstance(data, list):
                location = data[0]
                lat, lon = float(location['lat']), float(location['lon'])
                found_display_name = location.get('display_name', 'N/A')
                logging.info(f"Coordenadas encontradas para '{location_name}': lat={lat}, lon={lon} (Local: {found_display_name})")
                return lat, lon

            # Fallback mechanism: Try splitting the location name
            logging.warning(f"Nenhuma coordenada encontrada para '{location_name}'. Tentando fallback com partes do nome.")
            parts = location_name.split(',')
            for part in parts:
                fallback_params = {'q': part.strip(), 'format': 'json', 'limit': 1}
                fallback_response = await client.get(settings.GEOCODING_API_URL, params=fallback_params, headers=headers)
                fallback_response.raise_for_status()
                fallback_data = fallback_response.json()

                if fallback_data and isinstance(fallback_data, list):
                    fallback_location = fallback_data[0]
                    lat, lon = float(fallback_location['lat']), float(fallback_location['lon'])
                    found_display_name = fallback_location.get('display_name', 'N/A')
                    logging.info(f"Fallback coordenadas encontradas: lat={lat}, lon={lon} (Local: {found_display_name})")
                    return lat, lon

            raise ValueError(f"Nenhuma coordenada encontrada para '{location_name}' mesmo após fallback.")

        except (httpx.RequestError, IndexError, KeyError, TypeError) as e:
            logging.error(f"Erro na API de geocodificação para '{location_name}': {e}")
            raise ValueError(f"Não foi possível obter coordenadas para '{location_name}'. Tente ser mais específico.")


# ===== RAG leve (recuperar contexto relevante a partir de ficheiros do repositório) =====
def retrieve_contexts(query: str, top_n: int = 3) -> str:
    """Recupera pequenos excertos do README e do código com base em sobreposição de tokens.

    Implementação simples: lê README.md e `src/analise_clima.py`, divide em parágrafos e pontua por tokens em comum.
    """
    try:
        repo_root = Path(project_root)
        readme_path = repo_root / 'README.md'
        analise_path = repo_root / 'src' / 'analise_clima.py'
        docs = []

        # If chroma available, use vector retrieval
        if CHROMA_AVAILABLE:
            try:
                # initialize client (in-memory)
                client = chromadb.Client(ChromaSettings())
                embedder = SentenceTransformer('all-MiniLM-L6-v2')
                ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')
                collection = None
                # create a transient collection name based on project path
                coll_name = 'nisac_local_docs'
                try:
                    collection = client.get_collection(coll_name)
                except Exception:
                    collection = client.create_collection(coll_name, embedding_function=ef)

                # index README and analise file if not already present
                to_index = []
                for p in (readme_path, analise_path):
                    if p.exists():
                        txt = p.read_text(encoding='utf-8')
                        parts = [para.strip() for para in re.split(r"\n\s*\n", txt) if para.strip()][:50]
                        for i, part in enumerate(parts):
                            doc_id = f"{p.name}_{i}"
                            try:
                                collection.get(ids=[doc_id])
                            except Exception:
                                collection.add(ids=[doc_id], documents=[part], metadatas=[{"source": str(p)}])

                query_embed = embedder.encode(query)
                results = collection.query(query_embeddings=[query_embed], n_results=top_n)
                for docs_list in results['documents']:
                    for d in docs_list:
                        docs.append(d)
                if docs:
                    return "\n\n--- Contexto relevante (via Chroma) ---\n\n" + "\n\n".join(docs[:top_n])
            except Exception as e:
                logging.warning(f"Chroma RAG falhou: {e}; fallback para método leve.")

        # Fallback leve (token overlap)
        candidates = []
        for p in (readme_path, analise_path):
            if p.exists():
                text = p.read_text(encoding='utf-8')
                parts = [para.strip() for para in re.split(r"\n\s*\n", text) if para.strip()]
                for part in parts:
                    candidates.append(part)

        q_tokens = set(re.findall(r"\w+", query.lower()))
        scored = []
        for c in candidates:
            c_tokens = set(re.findall(r"\w+", c.lower()))
            score = len(q_tokens & c_tokens)
            if score > 0:
                scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [c for _, c in scored[:top_n]]
        if selected:
            return "\n\n--- Contexto relevante (extraído dos ficheiros do projeto) ---\n\n" + "\n\n".join(selected)
    except Exception as e:
        logging.warning(f"Falha ao recuperar contexto local: {e}")
    return ""


# ===== Logging de prompts/respostas (append JSON lines) =====
def log_prompt_entry(entry: Dict[str, Any]):
    try:
        repo = Path(project_root)
        log_path = repo / (os.environ.get('AI_PROMPT_LOG_NAME', 'ai_prompts.log'))
        ts = time.strftime('%Y-%m-%dT%H:%M:%S')
        entry_record = {'timestamp': ts, **entry}
        with open(log_path, 'a', encoding='utf-8') as fh:
            fh.write(_json.dumps(entry_record, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.debug(f"Não foi possível gravar o log do prompt: {e}")


async def call_gemini_and_validate(prompt: str, model_schema: BaseModel.__class__, max_retries: int = 2) -> Dict[str, Any]:
    """Chama o modelo Gemini, tenta parsear JSON e valida com Pydantic.

    Se a resposta não for válida, reenvia ao modelo uma instrução curta pedindo correção.
    Retorna o dicionário validado.
    """
    if not gemini_model:
        raise HTTPException(status_code=503, detail="Modelo de IA indisponível.")

    last_error = None
    for attempt in range(1, max_retries + 2):
        try:
            logging.info(f"Chamando Gemini (attempt {attempt})")

            # Build system message and few-shot examples
            system_msg = (
                "Você é um assistente especialista em meteorologia e planejamento de eventos. "
                "Responda estritamente em JSON válido conforme o schema solicitado. Não adicione explicações. "
                "Se algum campo estiver ausente, retorne null nesse campo. "
            )
            few_shot_example = (
                "Exemplo de saída válida (apenas para formato):\n"
                "{\"resumo_executivo\": \"Condições majoritariamente secas...\", \"nivel_risco\": \"MODERADO\", ...}"
            )

            full_prompt = f"[SYSTEM]\n{system_msg}\n\n[EXAMPLE]\n{few_shot_example}\n\n[USER]\n{prompt}"

            # Use full_prompt for the model call
            model_call_prompt = full_prompt

            # log prompt (truncated)
            try:
                log_prompt_entry({'stage': 'request', 'attempt': attempt, 'prompt_preview': prompt[:400]})
            except Exception:
                pass

            # allow passing model params via env vars
            model_params = {}
            temp = os.environ.get('GEMINI_TEMPERATURE')
            if temp:
                try:
                    model_params['temperature'] = float(temp)
                except Exception:
                    pass

            # invoke model
            try:
                if hasattr(gemini_model, 'generate_content_async'):
                    # if the SDK supports params pass them (fallback if it doesn't)
                    resp = await gemini_model.generate_content_async(model_call_prompt, **model_params) if model_params else await gemini_model.generate_content_async(model_call_prompt)
                else:
                    resp = await gemini_model.generate_content_async(model_call_prompt)
            except TypeError:
                # some SDKs may not accept kwargs; fallback
                resp = await gemini_model.generate_content_async(model_call_prompt)
            response_text = getattr(resp, 'text', str(resp))
            # log response (truncated)
            try:
                log_prompt_entry({'stage': 'response', 'attempt': attempt, 'response_preview': response_text[:800]})
            except Exception:
                pass

            parsed = clean_and_parse_json(response_text)

            # Validar com Pydantic
            try:
                validated = model_schema.model_validate(parsed) if hasattr(model_schema, 'model_validate') else model_schema(**parsed)
                result = validated.model_dump() if hasattr(validated, 'model_dump') else dict(validated)
                # log validated
                try:
                    log_prompt_entry({'stage': 'validated', 'attempt': attempt, 'validated_preview': str(result)[:800]})
                except Exception:
                    pass
                return result
            except ValidationError as ve:
                last_error = ve
                logging.warning(f"Validação falhou: {ve}. Tentando corrigir com o modelo.")
                if attempt <= max_retries:
                    correction_prompt = (
                        "A resposta anterior não respeitou o esquema JSON exigido. "
                        "Por favor, corrija apenas o JSON e assegure que todos os campos estão presentes e válidos. "
                        "Retorne exclusivamente o JSON corrigido.")
                    prompt = prompt + "\n\n" + correction_prompt
                    continue
                else:
                    raise HTTPException(status_code=500, detail=f"Resposta da IA inválida: {ve}")

        except ValueError as e:
            last_error = e
            logging.warning(f"Falha ao parsear JSON da resposta do modelo: {e}")
            if attempt <= max_retries:
                prompt = prompt + "\n\nA resposta anterior não foi JSON válido. Corrija apenas o JSON e retorne-o." 
                continue
            raise HTTPException(status_code=500, detail="A IA não retornou JSON válido.")
        except Exception as e:
            logging.error(f"Erro inesperado ao chamar Gemini: {e}")
            last_error = e
            break

    logging.error(f"Todas as tentativas falharam: {last_error}")
    raise HTTPException(status_code=500, detail="Falha ao gerar/validar a saída da IA.")

async def extract_info_with_gemini(query: str) -> ExtractedInfo:
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(query=query)
    try:
        response = await gemini_model.generate_content_async(prompt)
        data = clean_and_parse_json(response.text)
        location_name = data.get("location_name")
        if not location_name: raise ValueError("A IA não conseguiu identificar um nome de local na sua consulta.")
    except Exception as e:
        logging.error(f"Gemini (extract name): Erro inesperado. {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao extrair o nome do local: {e}")
    try:
        lat, lon = await get_coords_from_location_name(location_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ExtractedInfo(latitude=lat, longitude=lon, location_name=location_name, event_date=data.get("event_date"), event_type=data.get("event_type") or "evento", event_context=data.get("event_context") or "", activity_category=data.get("activity_category") or "Outro", analysis_start_date=f"{settings.DEFAULT_ANALYSIS_START_YEAR}0101", analysis_end_date=f"{settings.DEFAULT_ANALYSIS_END_YEAR}1231")

async def fetch_nasa_data(params: Dict[str, Any]) -> Dict[str, Any]:
    """Busca dados da NASA POWER com retries e cache em disco opcional."""
    cache_dir = os.environ.get("NASA_CACHE_DIR")
    cache_path = None
    if cache_dir:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            key = f"nasa_{params.get('latitude')}_{params.get('longitude')}_{params.get('start')}_{params.get('end')}".replace('.', '_')
            cache_path = os.path.join(cache_dir, f"{key}.json")
            if os.path.exists(cache_path):
                logging.info(f"Carregando resposta da NASA a partir do cache: {cache_path}")
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logging.warning(f"Falha ao usar cache em disco: {e}")

    max_attempts = 3
    backoff_base = 1.0
    last_exc = None

    async with httpx.AsyncClient(timeout=45.0) as client:
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await client.get(settings.NASA_POWER_BASE_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
                if cache_path:
                    try:
                        with open(cache_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False)
                    except Exception as e:
                        logging.warning(f"Falha ao gravar cache em disco: {e}")
                return data
            except httpx.HTTPStatusError as e:
                logging.error(f"Erro na API da NASA: {e.response.status_code} - {e.response.text}")
                last_exc = HTTPException(status_code=502, detail="Erro ao comunicar com a API da NASA.")
                break
            except httpx.RequestError as e:
                logging.warning(f"Erro de rede ao contactar a NASA (tentativa {attempt}): {e}")
                last_exc = HTTPException(status_code=504, detail="Falha de conexão com o servidor da NASA.")
                if attempt < max_attempts:
                    sleep_for = backoff_base * (2 ** (attempt - 1))
                    logging.info(f"A aguardar {sleep_for}s antes da próxima tentativa...")
                    await asyncio.sleep(sleep_for)
                    continue
                break

    if last_exc:
        raise last_exc
    raise HTTPException(status_code=500, detail="Falha desconhecida ao obter dados da NASA.")

# <<< MELHORIA: A função agora aceita `event_context` além de `event_date` para dar contexto à IA.
async def generate_ai_outputs(analysis_data: Dict[str, Any], event_type: str, event_date: Optional[str], event_context: str, activity_category: str) -> Tuple[Dict, Dict]:
    # Cria uma string de data mais legível para a IA, ou um fallback genérico
    if event_date:
        try:
            date_obj = datetime.strptime(event_date, "%Y-%m-%d")
            meses = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]
            date_str_formatada = f"{date_obj.day} de {meses[date_obj.month - 1]} de {date_obj.year}"
        except (ValueError, TypeError):
            date_str_formatada = event_date
    else:
        date_str_formatada = "a data do evento"

    try:
        prompt_recommendations = RECOMMENDATION_PROMPT_TEMPLATE.format(
            event_type=event_type,
            event_date=date_str_formatada,
            event_context=event_context,
            activity_category=activity_category,
            analysis_data=json.dumps(analysis_data, indent=2, ensure_ascii=False)
        )
        prompt_dashboard = DASHBOARD_PROMPT_TEMPLATE.format(
            probability_data=json.dumps(analysis_data.get('weather_condition_probabilities', {}), ensure_ascii=False)
        )
        # Use wrapper para chamadas ao Gemini com validação das saídas
        rec = await call_gemini_and_validate(prompt_recommendations, RecommendationModel)
        dash = await call_gemini_and_validate(prompt_dashboard, DashboardModel)
        return rec, dash
    except Exception as e:
        logging.error(f"Gemini (generate): Erro inesperado ao gerar conteúdo. {e}")
        return {"error": f"Falha ao gerar recomendações: {e}"}, {"error": f"Falha ao gerar insights: {e}"}

# ===== ROTA PRINCIPAL DA API =====
@app.post("/api/analyze", response_model=ApiResponse, tags=["AI Analysis"], summary="Análise climática completa via linguagem natural")
async def analyze_with_ai(request: SearchQueryRequest):
    if not gemini_model:
        raise HTTPException(status_code=503, detail="O modelo de IA não está disponível.")

    logging.info(f"Nova consulta recebida: '{request.query}'")
    
    extracted_info = await extract_info_with_gemini(request.query)
    logging.info(f"Informações extraídas e geocodificadas: {extracted_info.model_dump()}")
    
    try:
        target_day = datetime.strptime(extracted_info.event_date, "%Y-%m-%d").strftime("%m-%d") if extracted_info.event_date else datetime.now().strftime("%m-%d")
    except (ValueError, TypeError):
        logging.warning(f"Formato de data inválido ou ausente: {extracted_info.event_date}. Usando data atual.")
        target_day = datetime.now().strftime("%m-%d")

    nasa_params = {
        "parameters": "PRECTOTCORR,T2M_MAX,T2M_MIN,RH2M,WS2M_MAX",
        "community": "RE",
        "longitude": extracted_info.longitude,
        "latitude": extracted_info.latitude,
        "start": extracted_info.analysis_start_date,
        "end": extracted_info.analysis_end_date,
        "format": "JSON",
    }
    nasa_data = await fetch_nasa_data(nasa_params)
    logging.info(f"Dados da NASA recebidos para '{extracted_info.location_name}' (lat={extracted_info.latitude}, lon={extracted_info.longitude})")
    
    try:
        climate_analysis = analisar_dados_climaticos(nasa_data, target_day)
        logging.info("Análise estatística concluída com sucesso.")
    except Exception as e:
        logging.error(f"Erro na função 'analisar_dados_climaticos': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erro interno ao analisar os dados climáticos.")
    
    # <<< MELHORIA: Passando o contexto extraído para a função de geração de IA.
    ai_recommendations, dashboard_insights = await generate_ai_outputs(
        climate_analysis,
        extracted_info.event_type,
        extracted_info.event_date,
        extracted_info.event_context,
        extracted_info.activity_category
    )
    logging.info("Recomendações e insights da IA gerados.")
    
    quick_summary_data = {
        "risk_level": ai_recommendations.get("nivel_risco", "INDETERMINADO"),
        "rain_probability": climate_analysis.get("precipitation_analysis", {}).get("any_rain", {}).get("probability_percent", 0),
        "heat_risk": climate_analysis.get("weather_condition_probabilities", {}).get("VERY_HOT", {}).get("probability_percent", 0),
        "main_risks": climate_analysis.get("event_planning_recommendations", {}).get("key_risks", [])
    }
    
    response_data = {
        "query_info": {
            "original_query": request.query,
            "extracted_location": {
                "name": extracted_info.location_name,
                "latitude": extracted_info.latitude,
                "longitude": extracted_info.longitude
            },
            "extracted_date": extracted_info.event_date,
            "event_type": extracted_info.event_type,
            "event_context": extracted_info.event_context,
            "activity_category": extracted_info.activity_category,
            "analysis_day": target_day,
            "timestamp": datetime.now().isoformat()
        },
        "climate_analysis": climate_analysis,
        "ai_recommendations": ai_recommendations,
        "dashboard_insights": dashboard_insights,
        "quick_summary": quick_summary_data
    }
    
    return JSONResponse(content=response_data, media_type="application/json; charset=utf-8")

# ... (O resto das rotas não muda) ...
@app.get("/", tags=["Status"], summary="Informações da API")
def root(): return {"api_name": app.title, "version": app.version, "documentation": "/docs"}
@app.get("/health", tags=["Status"], summary="Verificação de saúde")
def health_check(): return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# ===== EXECUÇÃO DA APLICAÇÃO =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)