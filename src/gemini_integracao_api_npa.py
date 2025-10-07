# gemini_integracao_api_npa.py - Versão 3.6.0 com Extração e Uso de Contexto Adicional para Personalização
# -*- coding: utf-8 -*-

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import asyncio

try:
    import google.generativeai as genai
except Exception:
    genai = None
    logging.warning("Biblioteca 'google.generativeai' não disponível; funcionalidades de IA podem ficar limitadas.")
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
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
        gemini_model = genai.GenerativeModel('gemma-3-4b-it')
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
Extraia também o contexto adicional do evento, como condições específicas (ex: sem roupa adequada, descalço, vendado, distância de caminhada, atividades como pular).

Consulta: "{query}"

Estrutura JSON de saída:
{{
    "location_name": "<o nome do local o mais completo possível, ex: 'Polo Norte Geográfico, Ártico', 'Benguela, Angola', 'Torre Eiffel, Paris, França'>",
    "event_date": "<data no formato YYYY-MM-DD ou null>",
    "event_type": "<tipo de evento ou null>",
    "event_context": "<descrição detalhada do contexto adicional extraído, ex: 'sem camisola e com os pés descalços, com olhos vendados, uns 45km de caminhada, pulando alto'>"
}}
"""

# <<< MELHORIA: Adicionado {event_context} para personalizar as recomendações com base no contexto extraído.
RECOMMENDATION_PROMPT_TEMPLATE = """
Você é um especialista em meteorologia e planejamento de eventos, com atuação global. Com base nos dados climáticos informados para um evento do tipo '{event_type}', programado para a data '{event_date}' e considerando o contexto adicional '{event_context}', gere recomendações detalhadas e personalizadas. As sugestões devem levar em conta riscos climáticos específicos para atividades ao ar livre, possíveis inadequações de vestimenta ou condições físicas extremas, adaptando-se ao contexto apresentado.

Antes de responder, inicie com uma checklist conceitual de etapas para análise dos dados e elaboração das recomendações (3-7 itens, nível conceitual, não detalhado). Em seguida:

Retorne SOMENTE um objeto JSON conforme a estrutura abaixo, sem explicações adicionais ou formatação markdown.

## Output Format
Retorne o JSON com a seguinte estrutura:
{
  "recomendacoes": [
    {
      "titulo": "string (título da recomendação)",
      "descricao": "string (detalhamento da recomendação)"
    },
    ...
  ],
  "avisos_climaticos": [
    "string (avisos relevantes, explicando riscos ou cuidados com o clima)"
  ],
  "ajustes_sugeridos": [
    "string (alterações ou adaptações sugeridas ao evento em função do clima)"
  ],
  "dados_utilizados": {
    "event_type": "string ou null (tipo do evento ou null se ausente)",
    "event_date": "string ou null (data do evento ou null se ausente)",
    "event_context": "string ou null (contexto adicional ou null se ausente)"
  },
  "campos_ausentes": [
    "event_type", "event_date", "event_context" (inclua no array os nomes dos campos não fornecidos, inválidos ou em branco)]
}

Se qualquer um dos campos de entrada ({event_type}, {event_date}, {event_context}) estiver ausente, inválido ou em branco, atribua null ao campo correspondente em "dados_utilizados" e adicione o nome do campo em "campos_ausentes". Antes de gerar o JSON final, valide se todos os dados necessários foram considerados e adapte as recomendações conforme necessário.
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
    latitude: float; longitude: float; location_name: str; event_date: Optional[str] = None; event_type: str = "evento"; event_context: str = ""; analysis_start_date: str; analysis_end_date: str
class QuickSummary(BaseModel):
    risk_level: str; rain_probability: float; heat_risk: float; main_risks: List[str]
class ApiResponse(BaseModel):
    query_info: Dict[str, Any]; climate_analysis: Dict[str, Any]; ai_recommendations: Dict[str, Any]; dashboard_insights: Dict[str, Any]; quick_summary: QuickSummary

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
    if not location_name: raise ValueError("O nome do local não pode ser vazio.")
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
            else: raise ValueError(f"Nenhuma coordenada encontrada para '{location_name}'.")
        except (httpx.RequestError, IndexError, KeyError, TypeError) as e:
            logging.error(f"Erro na API de geocodificação para '{location_name}': {e}")
            raise ValueError(f"Não foi possível obter coordenadas para '{location_name}'. Tente ser mais específico.")

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
    return ExtractedInfo(latitude=lat, longitude=lon, location_name=location_name, event_date=data.get("event_date"), event_type=data.get("event_type") or "evento", event_context=data.get("event_context") or "", analysis_start_date=f"{settings.DEFAULT_ANALYSIS_START_YEAR}0101", analysis_end_date=f"{settings.DEFAULT_ANALYSIS_END_YEAR}1231")

async def fetch_nasa_data(params: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=45.0) as client:
        try:
            resp = await client.get(settings.NASA_POWER_BASE_URL, params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logging.error(f"Erro na API da NASA: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=502, detail="Erro ao comunicar com a API da NASA.")
        except httpx.RequestError as e:
            logging.error(f"Erro de conexão com a API da NASA: {e}")
            raise HTTPException(status_code=504, detail="Falha de conexão com o servidor da NASA.")

# <<< MELHORIA: A função agora aceita `event_context` além de `event_date` para dar contexto à IA.
async def generate_ai_outputs(analysis_data: Dict[str, Any], event_type: str, event_date: Optional[str], event_context: str) -> Tuple[Dict, Dict]:
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
            analysis_data=json.dumps(analysis_data, indent=2, ensure_ascii=False)
        )
        prompt_dashboard = DASHBOARD_PROMPT_TEMPLATE.format(
            probability_data=json.dumps(analysis_data.get('weather_condition_probabilities', {}), ensure_ascii=False)
        )
        
        task_recommendations = gemini_model.generate_content_async(prompt_recommendations)
        task_dashboard = gemini_model.generate_content_async(prompt_dashboard)
        
        responses = await asyncio.gather(task_recommendations, task_dashboard)
        
        recommendations = clean_and_parse_json(responses[0].text)
        dashboard_insights = clean_and_parse_json(responses[1].text)
        
        return recommendations, dashboard_insights
        
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
        extracted_info.event_context
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