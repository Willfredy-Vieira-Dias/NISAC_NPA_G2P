# gemini_integracao_api_npa.py - Versão 3.6.0 com Import Direto e Contexto de Usuário
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
    logging.warning("Biblioteca 'google.generativeai' não disponível.")
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# <<< CORREÇÃO: Lógica de importação simplificada para maior robustez.
try:
    # Assume que 'analise_clima.py' está no mesmo diretório.
    from src.analise_clima import analisar_dados_climaticos
except (ImportError, ModuleNotFoundError):
    logging.exception("FALHA CRÍTICA AO IMPORTAR 'analise_clima'. Verifique se o ficheiro 'analise_clima.py' existe na mesma pasta. Usando função mock.")
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
        env_file = ".env"; extra = "ignore"
try:
    settings = Settings()
    if genai: genai.configure(api_key=settings.GEMINI_API_KEY)
except ValueError as e:
    logging.error(f"Erro ao carregar configurações: {e}"); raise

# ===== CONFIGURAÇÃO DO MODELO GEMINI =====
if genai:
    try:
        # Usando um modelo mais recente e válido
        gemini_model = genai.GenerativeModel('gemma-3-4b-it')
    except Exception as e:
        gemini_model = None; logging.warning(f"Não foi possível inicializar o modelo Gemini: {e}")
else:
    gemini_model = None

# ===== PROMPTS PARA A IA (MELHORADOS) =====

# <<< MELHORIA: Adicionado "event_context" para capturar detalhes da consulta.
EXTRACTION_PROMPT_TEMPLATE = """
Você é um assistente especialista em extrair informações de uma consulta.
Analise a consulta e retorne APENAS um objeto JSON com a estrutura abaixo.
Extraia o nome do local da forma mais completa possível e detalhes importantes sobre o evento.

Consulta: "{query}"

Estrutura JSON de saída:
{{
    "location_name": "<o nome do local, ex: 'Dubai, Emirados Árabes Unidos'>",
    "event_date": "<data no formato YYYY-MM-DD ou null>",
    "event_type": "<tipo de evento, ex: 'natação'>",
    "event_context": "<detalhes importantes do evento, ex: 'distância de 45KM, com olhos vendados'>"
}}
"""

# <<< MELHORIA: Adicionado "event_context" para gerar recomendações personalizadas.
RECOMMENDATION_PROMPT_TEMPLATE = """
Você é um especialista em meteorologia e planeamento de eventos.
Baseado nos dados climáticos para um evento do tipo '{event_type}', agendado para '{event_date}', gere recomendações.
Leve em consideração o seguinte contexto adicional do evento: '{event_context}'.
As recomendações devem ser altamente personalizadas para este contexto. Por exemplo, para uma natação longa, fale sobre temperatura da água e hipotermia. Para um evento com olhos vendados, fale sobre riscos de vento ou chuva.

Dados de Análise:
{analysis_data}

Retorne APENAS um objeto JSON com a estrutura especificada:
{{
    "resumo_executivo": "<parágrafo conciso sobre as condições, considerando o contexto do evento>",
    "nivel_risco": "<BAIXO, MODERADO, ALTO ou MUITO_ALTO>",
    "recomendacao_principal": "<a recomendação mais crítica e personalizada>",
    "preparacoes_essenciais": ["<lista de 3-5 preparações críticas e personalizadas>"],
    "alternativas_sugeridas": {{ "datas_alternativas": [], "horarios_alternativos": "", "tipo_local": "" }},
    "itens_necessarios": ["<lista de 5-8 itens essenciais, personalizados para o evento>"],
    "avisos_especiais": ["<lista de 2-3 avisos importantes, personalizados para o contexto>"],
    "dica_especial": "<uma dica única e criativa baseada em todos os dados e contexto>"
}}
"""

DASHBOARD_PROMPT_TEMPLATE = """
...
""" # (Sem alterações no prompt do dashboard)

# ===== CONFIGURAÇÃO DA API FASTAPI =====
app = FastAPI(title="Cygnus-X1 AI-Powered Weather Analysis", version="3.6.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ===== MODELOS DE DADOS (PYDANTIC) =====
class SearchQueryRequest(BaseModel):
    query: str = Field(..., min_length=10)

# <<< MELHORIA: Adicionado 'event_context' ao modelo de dados.
class ExtractedInfo(BaseModel):
    latitude: float; longitude: float; location_name: str
    event_date: Optional[str] = None
    event_type: str = "evento"
    event_context: Optional[str] = None
    analysis_start_date: str; analysis_end_date: str

class QuickSummary(BaseModel):
    risk_level: str; rain_probability: float; heat_risk: float; main_risks: List[str]

class ApiResponse(BaseModel):
    query_info: Dict[str, Any]; climate_analysis: Dict[str, Any]; ai_recommendations: Dict[str, Any]; dashboard_insights: Dict[str, Any]; quick_summary: QuickSummary

# ===== FUNÇÕES AUXILIARES =====
# ... (clean_and_parse_json e get_coords_from_location_name sem alterações) ...
def clean_and_parse_json(raw_text: str) -> Dict[str, Any]:
    json_str = raw_text.strip().removeprefix("```json").removesuffix("```").strip()
    try: return json.loads(json_str)
    except json.JSONDecodeError as e: logging.error(f"Falha ao decodificar JSON: '{json_str}'. Erro: {e}"); raise ValueError("A resposta da IA não era um JSON válido.")

async def get_coords_from_location_name(location_name: str) -> Tuple[float, float]:
    if not location_name: raise ValueError("O nome do local não pode ser vazio.")
    params = {'q': location_name, 'format': 'json', 'limit': 1}
    headers = {'User-Agent': 'CygnusX1-WeatherApp/1.0', 'Accept-Language': 'en,pt;q=0.9'}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(settings.GEOCODING_API_URL, params=params, headers=headers)
            response.raise_for_status(); data = response.json()
            if data:
                loc = data[0]; lat, lon = float(loc['lat']), float(loc['lon'])
                logging.info(f"Coordenadas para '{location_name}': lat={lat}, lon={lon} ({loc.get('display_name', 'N/A')})")
                return lat, lon
            raise ValueError(f"Nenhuma coordenada encontrada para '{location_name}'.")
        except Exception as e:
            logging.error(f"Erro na geocodificação para '{location_name}': {e}")
            raise ValueError(f"Não foi possível obter coordenadas para '{location_name}'. Tente ser mais específico.")

async def extract_info_with_gemini(query: str) -> ExtractedInfo:
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(query=query)
    try:
        response = await gemini_model.generate_content_async(prompt)
        data = clean_and_parse_json(response.text)
        location_name = data.get("location_name")
        if not location_name: raise ValueError("A IA não identificou um local na consulta.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao extrair informações: {e}")
    try:
        lat, lon = await get_coords_from_location_name(location_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    # <<< MELHORIA: Retornando o novo campo 'event_context'.
    return ExtractedInfo(
        latitude=lat, longitude=lon,
        location_name=location_name,
        event_date=data.get("event_date"),
        event_type=data.get("event_type") or "evento",
        event_context=data.get("event_context") or "Nenhum contexto adicional fornecido.",
        analysis_start_date=f"{settings.DEFAULT_ANALYSIS_START_YEAR}0101",
        analysis_end_date=f"{settings.DEFAULT_ANALYSIS_END_YEAR}1231"
    )

async def fetch_nasa_data(params: Dict[str, Any]) -> Dict[str, Any]:
    # ... (Sem alterações) ...
    async with httpx.AsyncClient(timeout=45.0) as client:
        try:
            resp = await client.get(settings.NASA_POWER_BASE_URL, params=params); resp.raise_for_status(); return resp.json()
        except httpx.HTTPStatusError as e: raise HTTPException(status_code=502, detail=f"Erro na API da NASA: {e.response.text}")
        except httpx.RequestError as e: raise HTTPException(status_code=504, detail=f"Falha de conexão com a NASA: {e}")

# <<< MELHORIA: A função agora aceita `event_context`.
async def generate_ai_outputs(analysis_data: Dict[str, Any], event_type: str, event_date: Optional[str], event_context: Optional[str]) -> Tuple[Dict, Dict]:
    if event_date:
        try:
            date_obj = datetime.strptime(event_date, "%Y-%m-%d")
            meses = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]
            date_str_formatada = f"{date_obj.day} de {meses[date_obj.month - 1]} de {date_obj.year}"
        except: date_str_formatada = event_date
    else: date_str_formatada = "a data do evento"
    
    try:
        prompt_recommendations = RECOMMENDATION_PROMPT_TEMPLATE.format(
            event_type=event_type,
            event_date=date_str_formatada,
            event_context=event_context, # <<< Passando o novo contexto para o prompt
            analysis_data=json.dumps(analysis_data, indent=2, ensure_ascii=False)
        )
        prompt_dashboard = DASHBOARD_PROMPT_TEMPLATE.format(probability_data=json.dumps(analysis_data.get('weather_condition_probabilities', {})))
        
        task_recommendations, task_dashboard = await asyncio.gather(
            gemini_model.generate_content_async(prompt_recommendations),
            gemini_model.generate_content_async(prompt_dashboard)
        )
        
        return clean_and_parse_json(task_recommendations.text), clean_and_parse_json(task_dashboard.text)
    except Exception as e:
        logging.error(f"Gemini (generate): Erro inesperado. {e}")
        return {"error": f"Falha ao gerar recomendações: {e}"}, {"error": f"Falha ao gerar insights: {e}"}

# ===== ROTA PRINCIPAL DA API =====
@app.post("/api/analyze", response_model=ApiResponse, tags=["AI Analysis"])
async def analyze_with_ai(request: SearchQueryRequest):
    if not gemini_model: raise HTTPException(status_code=503, detail="O modelo de IA não está disponível.")
    
    extracted_info = await extract_info_with_gemini(request.query)
    
    try:
        target_day = datetime.strptime(extracted_info.event_date, "%Y-%m-%d").strftime("%m-%d") if extracted_info.event_date else datetime.now().strftime("%m-%d")
    except:
        target_day = datetime.now().strftime("%m-%d")

    nasa_params = {"parameters": "PRECTOTCORR,T2M_MAX,T2M_MIN,RH2M,WS2M_MAX", "community": "RE", "longitude": extracted_info.longitude, "latitude": extracted_info.latitude, "start": extracted_info.analysis_start_date, "end": extracted_info.analysis_end_date, "format": "JSON"}
    nasa_data = await fetch_nasa_data(nasa_params)
    
    # Este 'try' é crucial, pois a função real de análise é chamada aqui.
    try:
        climate_analysis = analisar_dados_climaticos(nasa_data, target_day)
        if "mock_data" in climate_analysis:
            logging.warning("A análise climática retornou dados MOCK. Verifique a lógica de importação ou o ficheiro 'analise_clima.py'.")
    except Exception as e:
        logging.error(f"Erro na função 'analisar_dados_climaticos': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erro interno ao analisar os dados climáticos.")
    
    # <<< MELHORIA: Passando o contexto para a função de geração de IA.
    ai_recommendations, dashboard_insights = await generate_ai_outputs(
        climate_analysis,
        extracted_info.event_type,
        extracted_info.event_date,
        extracted_info.event_context
    )
    
    quick_summary_data = {
        "risk_level": ai_recommendations.get("nivel_risco", "INDETERMINADO"),
        "rain_probability": climate_analysis.get("precipitation_analysis", {}).get("any_rain", {}).get("probability_percent", 0),
        "heat_risk": climate_analysis.get("weather_condition_probabilities", {}).get("VERY_HOT", {}).get("probability_percent", 0),
        "main_risks": ai_recommendations.get("avisos_especiais", [])
    }
    
    response_data = {
        "query_info": {
            "original_query": request.query,
            "extracted_location": {"name": extracted_info.location_name, "latitude": extracted_info.latitude, "longitude": extracted_info.longitude},
            "extracted_date": extracted_info.event_date,
            "event_type": extracted_info.event_type,
            # <<< MELHORIA: Adicionando o contexto extraído à resposta para depuração.
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

# ... (Rotas de status e execução sem alterações) ...
@app.get("/", tags=["Status"])
def root(): return {"api_name": app.title, "version": app.version, "documentation": "/docs"}
@app.get("/health", tags=["Status"])
def health_check(): return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)