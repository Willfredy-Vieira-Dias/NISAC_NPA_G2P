# x1tempo.py (VERSAO ENHANCED PARA DESAFIO NASA)
# -*- coding: utf-8 -*-

import os
import json
import traceback
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

import httpx
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Importa o modulo de analise
from analise_clima import analisar_dados_climaticos

# Modelos Pydantic para documentação da API
class LocationInput(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude em graus decimais")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude em graus decimais")

class WeatherCondition(str, Enum):
    VERY_HOT = "VERY_HOT"
    VERY_COLD = "VERY_COLD"
    VERY_WINDY = "VERY_WINDY"
    VERY_WET = "VERY_WET"
    VERY_UNCOMFORTABLE = "VERY_UNCOMFORTABLE"

class ChallengeInfo(BaseModel):
    challenge_name: str = "Will It Rain On My Parade?"
    team_name: str = "Cygnus-X1"
    location: str = "Luanda, Angola"
    event_dates: str = "October 3-5, 2025"
    nasa_data_source: str = "NASA POWER API"

app = FastAPI(
    title="Cygnus-X1 Weather Analysis API",
    description="""
    ## NASA International Space Apps Challenge 2025
    ### Challenge: Will It Rain On My Parade?
    
    Esta API fornece análise probabilística de condições climáticas para qualquer localização e dia do ano,
    utilizando dados históricos da NASA POWER. Desenvolvida para ajudar no planeamento de eventos ao ar livre
    com meses de antecedência.
    
    ### Funcionalidades Principais:
    - ✅ **Usa dados de observação da Terra da NASA** (NASA POWER API)
    - ✅ **Fornece probabilidades de condições climáticas** (very hot, very cold, very windy, very wet, very uncomfortable)
    - ✅ **Interface personalizável** por localização e data
    - ✅ **Análise de eventos extremos** com diferentes níveis de intensidade
    - ✅ **Indicadores de mudanças climáticas** através de análise de tendências
    - ✅ **Recomendações para planeamento** baseadas em dados históricos
    
    ### Endpoints Disponíveis:
    - `/clima` - Análise completa de probabilidades climáticas
    - `/challenge-info` - Informações sobre o desafio e conformidade
    - `/thresholds` - Configurações de limites utilizados na análise
    - `/help` - Guia de uso da API
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:5500",
    "*"  # Para desenvolvimento - remover em produção
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Configurações de limites padrão
DEFAULT_THRESHOLDS = {
    "temperature": {
        "very_hot_celsius": 32.0,
        "very_cold_celsius": 15.0,
        "uncomfortable_heat_celsius": 30.0
    },
    "precipitation": {
        "light_mm": 0.1,
        "moderate_mm": 5.0,
        "heavy_mm": 10.0,
        "extreme_mm": 25.0
    },
    "wind": {
        "very_windy_ms": 10.0
    },
    "humidity": {
        "very_humid_percent": 80.0,
        "uncomfortable_percent": 70.0
    }
}

def _normalize_dia_alvo(dia_alvo: Optional[str], fallback_end: Optional[str]) -> str:
    """
    Aceita dia_alvo em "MM-DD" ou "YYYY-MM-DD". 
    Se None, usa fallback_end (YYYYMMDD ou YYYY-MM-DD) e retorna MM-DD.
    """
    if dia_alvo:
        try:
            if len(dia_alvo) == 5 and dia_alvo[2] == "-":
                month = int(dia_alvo[:2])
                day = int(dia_alvo[3:5])
                if 1 <= month <= 12 and 1 <= day <= 31:
                    return dia_alvo
                else:
                    raise ValueError("Mês ou dia inválido")
            else:
                dt = datetime.strptime(dia_alvo, "%Y-%m-%d")
                return dt.strftime("%m-%d")
        except Exception:
            try:
                dt = datetime.strptime(dia_alvo, "%Y%m%d")
                return dt.strftime("%m-%d")
            except Exception:
                raise ValueError("dia_alvo deve ser 'MM-DD' ou 'YYYY-MM-DD' ou 'YYYYMMDD'")
    
    if fallback_end:
        try:
            if len(fallback_end) == 8 and fallback_end.isdigit():
                dt = datetime.strptime(fallback_end, "%Y%m%d")
            else:
                dt = datetime.strptime(fallback_end, "%Y-%m-%d")
            return dt.strftime("%m-%d")
        except Exception:
            raise ValueError("fallback end date tem formato inválido")
    
    raise ValueError("dia_alvo ou end deve ser fornecido")


@app.get("/", tags=["Status"])
def root():
    """
    Endpoint principal com informações sobre a API e links úteis.
    """
    return {
        "api_name": "Cygnus-X1 Weather Analysis API",
        "challenge": "NASA Space Apps Challenge 2025 - Will It Rain On My Parade?",
        "status": "OPERATIONAL",
        "version": "2.0.0",
        "team": "Cygnus-X1",
        "location": "Luanda, Angola",
        "useful_links": {
            "documentation": "/docs",
            "alternative_docs": "/redoc",
            "challenge_info": "/challenge-info",
            "example_query": "/clima?lat=-8.83&lon=13.23&start=20150101&end=20241231&dia_alvo=06-15",
            "help": "/help"
        }
    }


@app.get("/clima", 
         response_model=Dict[str, Any],
         tags=["Weather Analysis"],
         summary="Análise completa de probabilidades climáticas",
         description="Retorna probabilidades de condições climáticas extremas baseadas em dados históricos da NASA POWER")
async def obter_dados_climaticos(
    lat: float = Query(..., ge=-90, le=90, description="Latitude em graus decimais"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude em graus decimais"),
    start: str = Query(..., description="Data de início no formato YYYYMMDD", example="20150101"),
    end: str = Query(..., description="Data de fim no formato YYYYMMDD", example="20241231"),
    dia_alvo: Optional[str] = Query(None, description="Dia alvo no formato MM-DD ou YYYY-MM-DD", example="06-15"),
    pretty: bool = Query(False, description="Formatar JSON para melhor legibilidade")
):
    """
    Endpoint principal para análise de probabilidades climáticas.
    
    Este endpoint atende todos os requisitos do desafio NASA:
    - Usa dados de observação da Terra da NASA (POWER API)
    - Fornece probabilidades para condições "very hot", "very cold", "very windy", "very wet", "very uncomfortable"
    - Permite personalização por localização e data
    - Inclui análise de tendências de mudanças climáticas
    - Retorna dados estruturados para visualização
    
    ### Exemplo de uso:
    ```
    GET /clima?lat=-8.83&lon=13.23&start=20150101&end=20241231&dia_alvo=06-15
    ```
    """
    
    print(f"[INFO] Nova requisição - lat={lat}, lon={lon}, start={start}, end={end}, dia_alvo={dia_alvo}")
    
    # Validar e normalizar dia_alvo
    try:
        dia_mmdd = _normalize_dia_alvo(dia_alvo, end)
        print(f"[INFO] Dia normalizado: {dia_mmdd}")
    except ValueError as e:
        print(f"[ERROR] Erro na normalização: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    # Preparar parâmetros para NASA POWER API
    params = {
        "parameters": "PRECTOTCORR,T2M_MAX,T2M_MIN,RH2M,WS2M_MAX",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON",
    }
    
    print(f"[INFO] Chamando NASA POWER API...")

    # Fazer requisição à NASA POWER API
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(POWER_BASE, params=params)
            print(f"[INFO] Resposta recebida - Status: {resp.status_code}")
        except httpx.RequestError as exc:
            print(f"[ERROR] Erro na requisição: {exc}")
            raise HTTPException(
                status_code=502, 
                detail={
                    "error": "Failed to contact NASA POWER API",
                    "details": str(exc)
                }
            )

    if resp.status_code != 200:
        detail = None
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        print(f"[ERROR] NASA POWER retornou erro: {resp.status_code}")
        raise HTTPException(
            status_code=502, 
            detail={
                "error": "NASA POWER API error",
                "power_status": resp.status_code,
                "details": detail
            }
        )

    try:
        payload = resp.json()
        print(f"[INFO] JSON parseado com sucesso")
        
        # Debug: Verificar estrutura
        if 'properties' in payload and 'parameter' in payload['properties']:
            param_keys = list(payload['properties']['parameter'].keys())
            print(f"[INFO] Parâmetros recebidos: {param_keys}")
                
    except Exception as e:
        print(f"[ERROR] Erro ao parsear JSON: {e}")
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Failed to parse NASA POWER response",
                "details": str(e)
            }
        )

    # Analisar dados
    try:
        print(f"[INFO] Iniciando análise para dia {dia_mmdd}")
        resultado = analisar_dados_climaticos(payload, dia_mmdd)
        print(f"[INFO] Análise completa com sucesso")
        
        # Se houver erro na análise
        if 'erro' in resultado and resultado.get('status') == 'ERRO':
            print(f"[WARNING] Análise retornou erro: {resultado['erro']}")
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Analysis failed",
                    "message": resultado['erro'],
                    "suggestion": resultado.get('sugestao', 'Check your parameters')
                }
            )
        
        # Retornar com formatação se solicitado
        if pretty:
            return JSONResponse(
                content=resultado,
                media_type="application/json",
                headers={"Content-Type": "application/json; charset=utf-8"}
            )
        
        return resultado
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Erro inesperado na análise: {e}")
        print(f"[TRACEBACK] {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Unexpected analysis error",
                "message": str(e)
            }
        )


@app.get("/challenge-info", 
         response_model=Dict[str, Any],
         tags=["Challenge Information"])
def get_challenge_info():
    """
    Retorna informações sobre o desafio NASA e conformidade da API.
    """
    return {
        "challenge": {
            "name": "Will It Rain On My Parade?",
            "year": 2025,
            "event": "NASA International Space Apps Challenge",
            "objective": "Build an app that uses NASA Earth observation data to provide weather condition probabilities"
        },
        "team": {
            "name": "Cygnus-X1",
            "location": "Luanda, Angola",
            "members": 5,
            "event_dates": "October 3-5, 2025"
        },
        "compliance_checklist": {
            "uses_nasa_earth_observation_data": {
                "status": "✅ COMPLIANT",
                "implementation": "Uses NASA POWER API for historical weather data"
            },
            "provides_weather_probabilities": {
                "status": "✅ COMPLIANT",
                "implementation": "Calculates probabilities for all required conditions"
            },
            "customizable_query": {
                "status": "✅ COMPLIANT",
                "implementation": "Accepts latitude, longitude, and date parameters"
            },
            "personalized_dashboard": {
                "status": "✅ COMPLIANT",
                "implementation": "Returns structured data ready for dashboard visualization"
            },
            "extreme_weather_analysis": {
                "status": "✅ COMPLIANT",
                "implementation": "Analyzes multiple severity levels for each condition"
            },
            "climate_change_trends": {
                "status": "✅ COMPLIANT",
                "implementation": "Compares historical periods to identify trends"
            },
            "data_visualization_ready": {
                "status": "✅ COMPLIANT",
                "implementation": "Structured JSON output suitable for charts and maps"
            },
            "text_explanations": {
                "status": "✅ COMPLIANT",
                "implementation": "Includes descriptions and recommendations in response"
            }
        },
        "api_features": [
            "Real-time analysis of NASA POWER historical data",
            "Probability calculations for 5 main weather conditions",
            "Climate trend analysis",
            "Event planning recommendations",
            "Configurable thresholds",
            "Comprehensive error handling",
            "CORS support for web applications",
            "Interactive API documentation"
        ]
    }


@app.get("/thresholds", 
         response_model=Dict[str, Any],
         tags=["Configuration"])
def get_thresholds():
    """
    Retorna os limites (thresholds) utilizados para classificar condições extremas.
    """
    return {
        "description": "Threshold values used to classify extreme weather conditions",
        "thresholds": DEFAULT_THRESHOLDS,
        "units": {
            "temperature": "Celsius",
            "precipitation": "millimeters",
            "wind": "meters per second",
            "humidity": "percentage"
        },
        "notes": {
            "customization": "These thresholds can be adjusted based on local conditions and user preferences",
            "very_uncomfortable": "Combination of high temperature AND high humidity"
        }
    }


@app.get("/help", 
         response_model=Dict[str, Any],
         tags=["Help"])
def get_help():
    """
    Retorna guia de uso da API com exemplos.
    """
    return {
        "title": "API Usage Guide",
        "description": "How to use the Cygnus-X1 Weather Analysis API",
        "quick_start": {
            "step1": "Choose a location (latitude and longitude)",
            "step2": "Define the historical period for analysis (start and end dates)",
            "step3": "Specify the target day of year (MM-DD format)",
            "step4": "Call the /clima endpoint with these parameters",
            "step5": "Receive probability analysis for extreme weather conditions"
        },
        "examples": {
            "luanda_june": {
                "description": "Analysis for June 15th in Luanda, Angola",
                "url": "/clima?lat=-8.83&lon=13.23&start=20150101&end=20241231&dia_alvo=06-15"
            },
            "new_york_december": {
                "description": "Analysis for December 25th in New York, USA",
                "url": "/clima?lat=40.71&lon=-74.00&start=20100101&end=20231231&dia_alvo=12-25"
            },
            "paris_july": {
                "description": "Analysis for July 14th in Paris, France",
                "url": "/clima?lat=48.85&lon=2.35&start=20100101&end=20231231&dia_alvo=07-14"
            }
        },
        "parameters": {
            "lat": {
                "type": "float",
                "range": "[-90, 90]",
                "description": "Latitude in decimal degrees"
            },
            "lon": {
                "type": "float",
                "range": "[-180, 180]",
                "description": "Longitude in decimal degrees"
            },
            "start": {
                "type": "string",
                "format": "YYYYMMDD",
                "description": "Start date of historical period"
            },
            "end": {
                "type": "string",
                "format": "YYYYMMDD",
                "description": "End date of historical period"
            },
            "dia_alvo": {
                "type": "string",
                "format": "MM-DD or YYYY-MM-DD",
                "description": "Target day of year for analysis"
            },
            "pretty": {
                "type": "boolean",
                "default": "false",
                "description": "Format JSON output for readability"
            }
        },
        "response_structure": {
            "nasa_challenge_compliance": "Metadata about challenge requirements",
            "query_parameters": "Location and date information",
            "weather_condition_probabilities": "Main probabilities (VERY_HOT, VERY_COLD, etc.)",
            "precipitation_analysis": "Detailed rain probability analysis",
            "historical_statistics": "Average and percentile values",
            "climate_change_indicators": "Trend analysis",
            "event_planning_recommendations": "Practical suggestions",
            "threshold_configurations": "Limits used in analysis",
            "response_metadata": "Status and quality indicators"
        },
        "tips": [
            "Use at least 10 years of historical data for better accuracy",
            "Check multiple nearby locations for regional patterns",
            "Consider seasonal variations when planning events",
            "Use the 'pretty' parameter for human-readable output",
            "Consult /thresholds endpoint to understand classification criteria"
        ]
    }


@app.get("/test", 
         tags=["Testing"])
async def test_endpoint():
    """
    Endpoint de teste para verificar se a API está funcionando.
    """
    return {
        "message": "API is working correctly",
        "timestamp": datetime.now().isoformat(),
        "test_query": "http://127.0.0.1:8000/clima?lat=-8.83&lon=13.23&start=20150101&end=20241231&dia_alvo=06-15",
        "documentation": "http://127.0.0.1:8000/docs"
    }


@app.get("/health", 
         tags=["Status"])
def health_check():
    """
    Health check endpoint para monitoramento.
    """
    return {
        "status": "healthy",
        "service": "Cygnus-X1 Weather Analysis API",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }


# Tratamento de erros personalizado
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Tratamento personalizado de exceções HTTP.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "status_code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url)
            }
        }
    )


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 CYGNUS-X1 WEATHER ANALYSIS API")
    print("=" * 60)
    print("📍 NASA Space Apps Challenge 2025")
    print("🌍 Challenge: Will It Rain On My Parade?")
    print("👥 Team: Cygnus-X1 - Luanda, Angola")
    print("=" * 60)
    print("📚 Documentation: http://127.0.0.1:8000/docs")
    print("🔍 Alternative docs: http://127.0.0.1:8000/redoc")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)