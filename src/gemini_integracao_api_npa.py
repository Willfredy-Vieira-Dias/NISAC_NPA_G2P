# api_integrada.py - Versão com Geocodificação Automática
# -*- coding: utf-8 -*-

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import asyncio

import google.generativeai as genai
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Importa o módulo de análise existente (assumimos que este arquivo existe e funciona)
try:
    from analise_clima import analisar_dados_climaticos
except ImportError:
    # Função mock caso o arquivo não exista, para permitir que a API inicie
    def analisar_dados_climaticos(payload: Dict, dia_alvo: str) -> Dict:
        logging.warning("Módulo 'analise_clima' não encontrado. Usando dados mock.")
        return {"mock_data": "análise não pôde ser realizada", "dia_alvo": dia_alvo}

# ===== CONFIGURAÇÃO DE LOGGING =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# ===== GESTÃO DE CONFIGURAÇÕES (BOAS PRÁTICAS) =====
class Settings(BaseSettings):
    """Carrega configurações a partir de variáveis de ambiente."""
    GEMINI_API_KEY: str
    NASA_POWER_BASE_URL: str = "https://power.larc.nasa.gov/api/temporal/daily/point"
    # <<< NOVO: URL da API de Geocodificação Nominatim (OpenStreetMap)
    GEOCODING_API_URL: str = "https://nominatim.openstreetmap.org/search"
    DEFAULT_ANALYSIS_START_YEAR: int = 1981
    DEFAULT_ANALYSIS_END_YEAR: int = 2025
    
    class Config:
        env_file = ".env"
        extra = "ignore"

try:
    settings = Settings()
    genai.configure(api_key=settings.GEMINI_API_KEY)
except ValueError as e:
    logging.error(f"Erro ao carregar configurações: {e}")
    raise

# ===== CONFIGURAÇÃO DO MODELO GEMINI =====
gemini_model = genai.GenerativeModel('models/gemma-3-4b-it')


# ===== PROMPTS PARA A IA (ATUALIZADO) =====
EXTRACTION_PROMPT_TEMPLATE = """
Você é um assistente especialista em extrair o nome de locais de textos.
Analise a consulta do usuário e retorne APENAS um objeto JSON com a seguinte estrutura. O mais importante é extrair o nome do local.

Consulta: "{query}"

Estrutura JSON de saída:
{{
    "location_name": "<o nome do local, cidade, região ou país mencionado, ex: 'Benguela, Angola' ou 'Torre Eiffel, Paris'>",
    "event_date": "<data no formato YYYY-MM-DD ou null>",
    "event_type": "<tipo de evento ou null>"
}}
"""

RECOMMENDATION_PROMPT_TEMPLATE = """
Você é um especialista em meteorologia e planeamento de eventos para qualquer parte do mundo.
Baseado nos seguintes dados climáticos para um evento do tipo '{event_type}', gere recomendações detalhadas.
Retorne APENAS um objeto JSON com a estrutura especificada abaixo, sem explicações ou markdown.

Dados de Análise:
{analysis_data}

Estrutura JSON de saída:
{{
    "resumo_executivo": "<parágrafo conciso sobre as condições gerais>",
    "nivel_risco": "<BAIXO, MODERADO, ALTO ou MUITO_ALTO>",
    "recomendacao_principal": "<a recomendação mais crítica>",
    "preparacoes_essenciais": ["<lista de 3-5 preparações críticas>"],
    "alternativas_sugeridas": {{
        "datas_alternativas": ["<lista de até 3 sugestões de datas se aplicável>"],
        "horarios_alternativos": "<sugestão de horário (manhã, tarde, noite)>",
        "tipo_local": "<recomendação: 'ambiente fechado', 'ambiente aberto com cobertura' ou 'ambiente aberto'>"
    }},
    "itens_necessarios": ["<lista de 5-8 itens essenciais para os convidados/evento>"],
    "avisos_especiais": ["<lista de 2-3 avisos importantes (ex: risco de ventos fortes)>"],
    "dica_especial": "<uma dica única e criativa baseada nos dados>"
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
    description="API com geocodificação automática para análise climática em qualquer local do mundo.",
    version="3.3.0" # Versão com Geocodificação
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== MODELOS DE DADOS (PYDANTIC) =====
class SearchQueryRequest(BaseModel):
    query: str = Field(..., description="Consulta em linguagem natural do usuário.", min_length=10)

class ExtractedInfo(BaseModel):
    latitude: float
    longitude: float
    location_name: str # Adicionado para melhor feedback
    event_date: Optional[str] = None
    event_type: str = "evento"
    analysis_start_date: str
    analysis_end_date: str

class QuickSummary(BaseModel):
    risk_level: str
    rain_probability: float
    heat_risk: float
    main_risks: List[str]

class ApiResponse(BaseModel):
    query_info: Dict[str, Any]
    climate_analysis: Dict[str, Any]
    ai_recommendations: Dict[str, Any]
    dashboard_insights: Dict[str, Any]
    quick_summary: QuickSummary

# ===== FUNÇÕES AUXILIARES =====

def clean_and_parse_json(raw_text: str) -> Dict[str, Any]:
    """
    Limpa a resposta de texto do modelo para extrair um objeto JSON.
    Remove markdown (```json ... ```) e tenta fazer o parse.
    """
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
    """
    Usa a API Nominatim (OpenStreetMap) para obter lat/lon de um nome de local.
    """
    if not location_name:
        raise ValueError("O nome do local não pode ser vazio.")

    params = {'q': location_name, 'format': 'json', 'limit': 1}
    # É boa prática da API Nominatim enviar um User-Agent descritivo.
    headers = {'User-Agent': 'CygnusX1-WeatherApp/1.0'} 
    
    async with httpx.AsyncClient() as client:
        try:
            logging.info(f"Buscando coordenadas para: '{location_name}'")
            response = await client.get(settings.GEOCODING_API_URL, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if data and isinstance(data, list):
                location = data[0]
                lat = float(location['lat'])
                lon = float(location['lon'])
                logging.info(f"Coordenadas encontradas: lat={lat}, lon={lon}")
                return lat, lon
            else:
                raise ValueError(f"Nenhuma coordenada encontrada para '{location_name}'.")

        except (httpx.RequestError, IndexError, KeyError, TypeError) as e:
            logging.error(f"Erro na API de geocodificação para '{location_name}': {e}")
            raise ValueError(f"Não foi possível obter coordenadas para '{location_name}'. Tente ser mais específico.")

async def extract_info_with_gemini(query: str) -> ExtractedInfo:
    """
    Etapa 1: Usa Gemini para extrair o NOME do local.
    Etapa 2: Usa a API de geocodificação para obter as coordenadas.
    """
    # --- Etapa 1: Extrair o nome do local com Gemini ---
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(query=query)
    try:
        response = await gemini_model.generate_content_async(prompt)
        data = clean_and_parse_json(response.text)
        location_name = data.get("location_name")
        
        if not location_name:
            raise ValueError("A IA não conseguiu identificar um nome de local na sua consulta.")

    except Exception as e:
        logging.error(f"Gemini (extract name): Erro inesperado. {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao extrair o nome do local: {e}")

    # --- Etapa 2: Obter coordenadas com a API de Geocodificação ---
    try:
        lat, lon = await get_coords_from_location_name(location_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # --- Montar o resultado final ---
    start_year = settings.DEFAULT_ANALYSIS_START_YEAR
    end_year = settings.DEFAULT_ANALYSIS_END_YEAR
    
    return ExtractedInfo(
        latitude=lat,
        longitude=lon,
        location_name=location_name,
        event_date=data.get("event_date"),
        event_type=data.get("event_type") or "evento",
        analysis_start_date=f"{start_year}0101",
        analysis_end_date=f"{end_year}1231"
    )

async def fetch_nasa_data(params: Dict[str, Any]) -> Dict[str, Any]:
    """Busca dados da API NASA POWER de forma assíncrona."""
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

async def generate_ai_outputs(analysis_data: Dict[str, Any], event_type: str) -> Tuple[Dict, Dict]:
    """Gera recomendações e insights de dashboard em paralelo com a IA."""
    try:
        prompt_recommendations = RECOMMENDATION_PROMPT_TEMPLATE.format(
            event_type=event_type, 
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
@app.post("/api/analyze",
          response_model=ApiResponse,
          tags=["AI Analysis"],
          summary="Análise climática completa via linguagem natural")
async def analyze_with_ai(request: SearchQueryRequest):
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
    
    ai_recommendations, dashboard_insights = await generate_ai_outputs(climate_analysis, extracted_info.event_type)
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
            "analysis_day": target_day,
            "timestamp": datetime.now().isoformat()
        },
        "climate_analysis": climate_analysis,
        "ai_recommendations": ai_recommendations,
        "dashboard_insights": dashboard_insights,
        "quick_summary": quick_summary_data
    }
    
    return JSONResponse(content=response_data, media_type="application/json; charset=utf-8")

# ===== ROTAS ADICIONAIS =====
@app.get("/", tags=["Status"], summary="Informações da API")
def root():
    return {
        "api_name": app.title,
        "version": app.version,
        "documentation": "/docs"
    }

@app.get("/health", tags=["Status"], summary="Verificação de saúde")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# ===== EXECUÇÃO DA APLICAÇÃO =====
if __name__ == "__main__":
    import uvicorn
    
    print("="*60)
    print(f"🚀 CYGNUS-X1 ANÁLISE CLIMÁTICA COM IA (v{app.version})")
    print("="*60)
    print(f"🤖 Modelo Gemini: {gemini_model.model_name}")
    print("🌍 Geocodificação: Ativa via OpenStreetMap Nominatim")
    print("🛰️ Fonte de Dados: NASA POWER API")
    print("📊 Módulo de Análise: Carregado")
    print("="*60)
    print("📚 Documentação Interativa: http://127.0.0.1:8000/docs")
    print("✅ Endpoint Principal: [POST] http://127.0.0.1:8000/api/analyze")
    print("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)