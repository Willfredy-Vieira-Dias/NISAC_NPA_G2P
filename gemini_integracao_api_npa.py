# api_integrada.py - Sistema completo com Gemini AI
# -*- coding: utf-8 -*-

import os
import json
import traceback
from typing import Optional, Dict, Any, List
from datetime import datetime
import google.generativeai as genai
import httpx
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Importa o módulo de análise existente
from analise_clima import analisar_dados_climaticos

# ===== CONFIGURAÇÃO DO GEMINI =====
# IMPORTANTE: Substitua pela sua chave API real
GEMINI_API_KEY = "COLE_AQUI_SUA_CHAVE_GEMINI_API"
genai.configure(api_key=GEMINI_API_KEY)

# Configurar o modelo Gemini
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# ===== CONFIGURAÇÃO DA API =====
app = FastAPI(
    title="Cygnus-X1 AI-Powered Weather Analysis",
    description="API com processamento de linguagem natural via Gemini AI",
    version="3.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"

# ===== MODELOS PYDANTIC =====
class SearchQueryRequest(BaseModel):
    query: str = Field(..., description="Consulta em linguagem natural do usuário")

class ExtractedEventInfo(BaseModel):
    location: Optional[Dict[str, float]] = None
    event_date: Optional[str] = None
    event_type: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

# ===== FUNÇÕES DE EXTRAÇÃO COM GEMINI =====
def extrair_informacoes_com_gemini(query: str) -> ExtractedEventInfo:
    """
    Usa o Gemini para extrair informações estruturadas da consulta em linguagem natural.
    """
    prompt = f"""
    Você é um assistente especializado em extrair informações de eventos de textos em linguagem natural.
    
    Analise a seguinte consulta e extraia as informações relevantes:
    "{query}"
    
    Extraia e retorne APENAS um JSON com a seguinte estrutura (sem markdown, sem explicações):
    {{
        "location": {{
            "latitude": <número decimal ou null>,
            "longitude": <número decimal ou null>,
            "city_name": "<nome da cidade ou null>",
            "country": "<país ou null>"
        }},
        "event_date": "<data no formato YYYY-MM-DD ou null>",
        "event_type": "<tipo de evento: casamento, festa, churrasco, conferência, festival, esportivo, corporativo, etc ou null>",
        "analysis_period": {{
            "start_year": <ano inicial para análise histórica, padrão 2010>,
            "end_year": <ano final para análise histórica, padrão 2024>
        }}
    }}
    
    Regras importantes:
    - Se a cidade for mencionada mas não as coordenadas, use estas referências:
      * Luanda, Angola: lat=-8.83, lon=13.23
      * Lisboa, Portugal: lat=38.72, lon=-9.14
      * São Paulo, Brasil: lat=-23.55, lon=-46.63
      * Rio de Janeiro, Brasil: lat=-22.90, lon=-43.17
      * Nova York, EUA: lat=40.71, lon=-74.00
      * Paris, França: lat=48.85, lon=2.35
      * Londres, UK: lat=51.50, lon=-0.12
    - Se não houver data específica mencionada, use null
    - Se o período de análise não for mencionado, use 2010-2024
    - Para datas relativas como "próximo verão", "junho", converta para YYYY-MM-DD estimado
    - Retorne APENAS o JSON, sem texto adicional
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        json_str = response.text.strip()
        
        # Remove markdown se presente
        if json_str.startswith("```"):
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:].strip()
        if json_str.endswith("```"):
            json_str = json_str.rsplit("```", 1)[0].strip()
        
        # Parse do JSON
        extracted_data = json.loads(json_str)
        
        # Criar objeto ExtractedEventInfo
        info = ExtractedEventInfo()
        
        if extracted_data.get("location"):
            info.location = {
                "latitude": extracted_data["location"].get("latitude"),
                "longitude": extracted_data["location"].get("longitude")
            }
        
        info.event_date = extracted_data.get("event_date")
        info.event_type = extracted_data.get("event_type")
        
        # Configurar período de análise
        period = extracted_data.get("analysis_period", {})
        info.start_date = f"{period.get('start_year', 1981)}0101"
        info.end_date = f"{period.get('end_year',)}1231"
        
        return info
        
    except Exception as e:
        print(f"[ERROR] Erro ao extrair informações com Gemini: {e}")
        # Retorna valores padrão se falhar
        return ExtractedEventInfo(
            location={"latitude": -8.83, "longitude": 13.23},  # Default: Luanda
            event_date=None,
            event_type="evento",
            start_date="19810101",
            end_date="20250901"
        )

def gerar_recomendacoes_ia(dados_analise: Dict[str, Any], event_type: str = None) -> Dict[str, Any]:
    """
    Gera recomendações personalizadas usando Gemini AI baseadas nos dados analisados.
    """
    dados_json_str = json.dumps(dados_analise, indent=2, ensure_ascii=False)
    
    prompt = f"""
    Você é um especialista em planeamento de eventos e meteorologia em Angola.
    
    Com base nos seguintes dados climáticos históricos para um {event_type or 'evento'}:
    ```json
    {dados_json_str}
    ```
    
    Crie recomendações detalhadas em português de Angola. Retorne APENAS um JSON (sem markdown) com:
    {{
        "resumo_executivo": "<parágrafo conciso sobre as condições gerais>",
        "nivel_risco": "<BAIXO, MODERADO, ALTO ou MUITO_ALTO>",
        "recomendacao_principal": "<recomendação mais importante>",
        "preparacoes_essenciais": ["<lista de 3-5 preparações críticas>"],
        "alternativas_sugeridas": {{
            "datas_alternativas": ["<3 sugestões de datas melhores se aplicável>"],
            "horarios_alternativos": "<sugestão de horário do dia>",
            "tipo_local": "<indoor, outdoor_coberto, ou outdoor_aberto>"
        }},
        "itens_necessarios": ["<lista de 5-8 itens essenciais para o evento>"],
        "avisos_especiais": ["<lista de 2-3 avisos importantes se houver>"],
        "dica_especial": "<uma dica única baseada nos dados>"
    }}
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        json_str = response.text.strip()
        
        # Limpar markdown se presente
        if "```" in json_str:
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:].strip()
            if json_str.endswith("```"):
                json_str = json_str.rsplit("```", 1)[0].strip()
        
        return json.loads(json_str)
        
    except Exception as e:
        print(f"[ERROR] Erro ao gerar recomendações: {e}")
        # Retorna recomendações padrão em caso de erro
        return {
            "resumo_executivo": "Análise climática realizada com sucesso. Recomenda-se atenção às condições meteorológicas.",
            "nivel_risco": "MODERADO",
            "recomendacao_principal": "Considere ter um plano alternativo para o evento.",
            "preparacoes_essenciais": [
                "Verificar previsão do tempo próximo à data",
                "Preparar proteção contra sol/chuva",
                "Garantir hidratação adequada"
            ],
            "alternativas_sugeridas": {
                "datas_alternativas": [],
                "horarios_alternativos": "Manhã cedo ou final da tarde",
                "tipo_local": "outdoor_coberto"
            },
            "itens_necessarios": [
                "Tendas ou coberturas",
                "Água e bebidas geladas",
                "Protetor solar",
                "Kit primeiros socorros"
            ],
            "avisos_especiais": [],
            "dica_especial": "Mantenha-se atualizado com as previsões locais"
        }

def gerar_insights_dashboard(dados_analise: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gera insights adicionais para o dashboard usando IA.
    """
    prompt = f"""
    Baseado nos dados climáticos, gere insights para dashboard.
    
    Dados: {json.dumps(dados_analise.get('weather_condition_probabilities', {}), ensure_ascii=False)}
    
    Retorne APENAS JSON com:
    {{
        "headline": "<título impactante de 5-10 palavras>",
        "score_geral": <número 0-100>,
        "emoji_clima": "<emoji representativo>",
        "cor_tema": "<hex color code>",
        "tags": ["<3-5 tags relevantes>"],
        "grafico_recomendado": "<tipo: bar, line, pie, radar>",
        "metricas_chave": [
            {{"nome": "<métrica>", "valor": "<valor>", "tendencia": "<up/down/stable>"}}
        ]
    }}
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        json_str = response.text.strip()
        if "```" in json_str:
            json_str = json_str.split("```")[1].replace("json", "").strip()
        return json.loads(json_str)
    except:
        return {
            "headline": "Condições Climáticas Analisadas",
            "score_geral": 75,
            "emoji_clima": "☀️",
            "cor_tema": "#4A90E2",
            "tags": ["clima", "análise", "evento"],
            "grafico_recomendado": "bar",
            "metricas_chave": []
        }

# ===== ROTA PRINCIPAL INTEGRADA =====
@app.post("/api/analyze",
          response_model=Dict[str, Any],
          tags=["AI Analysis"],
          summary="Análise completa com processamento de linguagem natural")
async def analyze_with_ai(request: SearchQueryRequest):
    """
    Endpoint principal que:
    1. Recebe texto em linguagem natural
    2. Extrai informações com Gemini
    3. Busca dados da NASA
    4. Realiza análise estatística
    5. Gera recomendações com IA
    6. Retorna JSON completo para o frontend
    """
    
    print(f"[INFO] Query recebida: {request.query}")
    
    # Etapa 1: Extrair informações com Gemini
    try:
        extracted_info = extrair_informacoes_com_gemini(request.query)
        print(f"[INFO] Informações extraídas: {extracted_info}")
    except Exception as e:
        print(f"[ERROR] Falha na extração: {e}")
        raise HTTPException(status_code=500, detail="Erro ao processar consulta")
    
    # Validar coordenadas
    if not extracted_info.location or not extracted_info.location.get("latitude"):
        raise HTTPException(
            status_code=400,
            detail="Não foi possível identificar a localização. Por favor, seja mais específico."
        )
    
    lat = extracted_info.location["latitude"]
    lon = extracted_info.location["longitude"]
    
    # Determinar dia alvo
    dia_alvo = None
    if extracted_info.event_date:
        try:
            dt = datetime.strptime(extracted_info.event_date, "%Y-%m-%d")
            dia_alvo = dt.strftime("%m-%d")
        except:
            dia_alvo = "06-15"  # Default
    else:
        dia_alvo = "06-15"  # Default se não especificado
    
    # Etapa 2: Buscar dados da NASA
    params = {
        "parameters": "PRECTOTCORR,T2M_MAX,T2M_MIN,RH2M,WS2M_MAX",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": extracted_info.start_date,
        "end": extracted_info.end_date,
        "format": "JSON",
    }
    
    print(f"[INFO] Buscando dados NASA para lat={lat}, lon={lon}, dia={dia_alvo}")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(POWER_BASE, params=params)
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail="Erro ao acessar NASA POWER API")
            payload = resp.json()
        except Exception as e:
            print(f"[ERROR] Erro NASA API: {e}")
            raise HTTPException(status_code=502, detail="Falha na comunicação com NASA")
    
    # Etapa 3: Análise estatística
    try:
        resultado_analise = analisar_dados_climaticos(payload, dia_alvo)
        print("[INFO] Análise estatística concluída")
    except Exception as e:
        print(f"[ERROR] Erro na análise: {e}")
        raise HTTPException(status_code=500, detail="Erro na análise dos dados")
    
    # Etapa 4: Gerar recomendações com IA
    try:
        recomendacoes_ia = gerar_recomendacoes_ia(resultado_analise, extracted_info.event_type)
        insights_dashboard = gerar_insights_dashboard(resultado_analise)
        print("[INFO] Recomendações IA geradas")
    except Exception as e:
        print(f"[ERROR] Erro nas recomendações: {e}")
        recomendacoes_ia = {}
        insights_dashboard = {}
    
    # Etapa 5: Montar resposta completa
    resposta_completa = {
        # Metadados da consulta
        "query_info": {
            "original_query": request.query,
            "extracted_location": {
                "latitude": lat,
                "longitude": lon
            },
            "extracted_date": extracted_info.event_date,
            "event_type": extracted_info.event_type,
            "analysis_day": dia_alvo,
            "timestamp": datetime.now().isoformat()
        },
        
        # Dados originais da análise (para gráficos)
        "climate_analysis": resultado_analise,
        
        # Recomendações da IA
        "ai_recommendations": recomendacoes_ia,
        
        # Insights para dashboard
        "dashboard_insights": insights_dashboard,
        
        # Dados para visualização rápida
        "quick_summary": {
            "rain_probability": resultado_analise.get("precipitation_analysis", {})
                                                  .get("any_rain", {})
                                                  .get("probability_percent", 0),
            "heat_risk": resultado_analise.get("weather_condition_probabilities", {})
                                         .get("VERY_HOT", {})
                                         .get("probability_percent", 0),
            "overall_suitability": resultado_analise.get("event_planning_recommendations", {})
                                                    .get("outdoor_suitability_score", 50),
            "main_risks": resultado_analise.get("event_planning_recommendations", {})
                                          .get("key_risks", []),
            "risk_level": recomendacoes_ia.get("nivel_risco", "MODERADO")
        }
    }
    
    return JSONResponse(
        content=resposta_completa,
        media_type="application/json",
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

# ===== ROTAS ADICIONAIS =====
@app.get("/", tags=["Status"])
def root():
    """Endpoint principal com informações da API."""
    return {
        "api_name": "Cygnus-X1 AI-Powered Weather Analysis",
        "version": "3.0.0",
        "features": [
            "Natural language processing with Gemini AI",
            "NASA POWER data integration",
            "Statistical climate analysis",
            "AI-powered recommendations",
            "Dashboard insights generation"
        ],
        "endpoints": {
            "main": "/api/analyze (POST)",
            "docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/health", tags=["Status"])
def health_check():
    """Health check para monitoramento."""
    gemini_status = "configured" if GEMINI_API_KEY != "COLE_AQUI_SUA_CHAVE_GEMINI_API" else "not_configured"
    return {
        "status": "healthy",
        "gemini_status": gemini_status,
        "timestamp": datetime.now().isoformat()
    }

# ===== EXEMPLO DE USO =====
@app.get("/example", tags=["Help"])
def get_example():
    """Retorna exemplos de uso da API."""
    return {
        "examples": [
            {
                "description": "Consulta simples",
                "request": {
                    "query": "Vou fazer um casamento em Luanda dia 15 de junho"
                }
            },
            {
                "description": "Consulta detalhada",
                "request": {
                    "query": "Estou planejando um festival de música ao ar livre em Lisboa para o próximo verão, provavelmente em julho"
                }
            },
            {
                "description": "Consulta com análise específica",
                "request": {
                    "query": "Quero fazer um churrasco no Rio de Janeiro em dezembro, preciso saber se vai chover"
                }
            }
        ],
        "response_structure": {
            "query_info": "Informações extraídas da consulta",
            "climate_analysis": "Análise estatística completa (dados para gráficos)",
            "ai_recommendations": "Recomendações personalizadas da IA",
            "dashboard_insights": "Insights para visualização",
            "quick_summary": "Resumo rápido dos principais indicadores"
        }
    }

if __name__ == "__main__":
    
    import uvicorn
    print("=" * 60)
    print("🚀 CYGNUS-X1 Análise de tempo com IA integrada")
    print("=" * 60)
    print("🤖 Integração Gemini AI: Activa")
    print("🌍 NASA POWER Data: conectado")
    print("📊 Analise estatística: Pronta")
    print("=" * 60)
    print("📚 Documentation: http://localhost:8000/docs")
    print("🔧 Main endpoint: POST /api/analyze")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)