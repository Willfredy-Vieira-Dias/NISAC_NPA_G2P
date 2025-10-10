# api_npa.py (Versão 2.1.0 - Refatorada com Boas Práticas)
# -*- coding: utf-8 -*-

import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Optional
import os
import json
import asyncio

import httpx
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

# <<< MELHORIA: Importar o módulo de análise com um fallback para o caso de não existir
try:
    from analise_clima import analisar_dados_climaticos
except ImportError:
    # Função mock para que a API possa iniciar mesmo sem o módulo de análise
    def analisar_dados_climaticos(payload: Dict, dia_alvo: str) -> Dict:
        logging.warning("Módulo 'analise_clima' não encontrado. Usando dados mock.")
        return {"erro": "Módulo de análise indisponível.", "status": "ERRO"}

# <<< MELHORIA: Configuração de logging profissional para substituir os prints
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# <<< MELHORIA: Gestão de Configuração com Pydantic Settings
class Settings(BaseSettings):
    """
    Centraliza as configurações da aplicação. Carrega de variáveis de ambiente ou de um ficheiro .env.
    """
    NASA_POWER_BASE_URL: str = "https://power.larc.nasa.gov/api/temporal/daily/point"
    DEFAULT_THRESHOLDS: Dict[str, Any] = {
        "temperature": {"very_hot_celsius": 32.0, "very_cold_celsius": 15.0},
        "precipitation": {"heavy_mm": 10.0},
        "wind": {"very_windy_ms": 10.0},
        "humidity": {"uncomfortable_percent": 75.0}
    }
    
    class Config:
        env_file = ".env" # Permite carregar de um ficheiro .env para desenvolvimento

# Instanciar as configurações
settings = Settings()


# <<< MELHORIA: Modelos Pydantic para validação robusta e documentação clara
class ClimateQueryParams(BaseModel):
    """Modelo para agrupar e validar os parâmetros de consulta do endpoint de clima."""
    lat: float = Field(..., ge=-90, le=90, description="Latitude em graus decimais", example=-8.83)
    lon: float = Field(..., ge=-180, le=180, description="Longitude em graus decimais", example=13.23)
    start: str = Field(..., description="Data de início no formato YYYYMMDD", pattern=r"^\d{8}$", example="20150101")
    end: str = Field(..., description="Data de fim no formato YYYYMMDD", pattern=r"^\d{8}$", example="20241231")
    dia_alvo: Optional[str] = Field(None, description="Dia alvo no formato MM-DD ou YYYY-MM-DD", example="06-15")
    pretty: bool = Field(False, description="Formatar JSON para melhor legibilidade")

    @field_validator('start', 'end')
    def validate_dates(cls, v):
        try:
            datetime.strptime(v, '%Y%m%d')
        except ValueError:
            raise ValueError("As datas devem estar no formato YYYYMMDD.")
        return v

class ApiResponseModel(BaseModel):
    """Modelo de resposta para garantir consistência e documentação."""
    nasa_challenge_compliance: Dict[str, Any]
    query_parameters: Dict[str, Any]
    weather_condition_probabilities: Dict[str, Any]
    event_planning_recommendations: Dict[str, Any]
    response_metadata: Dict[str, Any]


# --- Início da Aplicação FastAPI ---
app = FastAPI(
    title="Cygnus-X1 Weather Analysis API",
    description="""
    ## NASA International Space Apps Challenge 2025
    ### Challenge: Will It Rain On My Parade?
    Esta API fornece análise probabilística de condições climáticas para qualquer localização e dia do ano,
    utilizando dados históricos da NASA POWER.
    """,
    version="2.1.0",
    docs_url="/docs"
)

# <<< MELHORIA: Comentário sobre segurança em produção no CORS
app.add_middleware(
    CORSMiddleware,
    # ATENÇÃO: Em produção, restrinja as origens a domínios específicos.
    # Ex: allow_origins=["https://seu-frontend.com"]
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --- Funções Auxiliares (Lógica de Negócio) ---
def _normalize_dia_alvo(dia_alvo: Optional[str], fallback_date_str: str) -> str:
    """Valida e normaliza a data alvo para o formato MM-DD de forma mais limpa."""
    target_date_str = dia_alvo or fallback_date_str
    
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(target_date_str, fmt).strftime("%m-%d")
        except ValueError:
            continue
            
    if dia_alvo and len(dia_alvo) == 5 and dia_alvo[2] == '-':
        try:
            datetime.strptime(dia_alvo, "%m-%d")
            return dia_alvo
        except ValueError:
            pass

    raise ValueError("Formato de 'dia_alvo' inválido. Use 'MM-DD', 'YYYY-MM-DD' ou 'YYYYMMDD'.")

async def _fetch_nasa_power_data(params: ClimateQueryParams) -> Dict[str, Any]:
    """Busca e valida os dados da API NASA POWER com retries e cache em disco opcional.

    Para ativar o cache em disco defina a variável de ambiente `NASA_CACHE_DIR`.
    """
    api_params = {
        "parameters": "PRECTOTCORR,T2M_MAX,T2M_MIN,RH2M,WS2M_MAX",
        "community": "RE",
        "longitude": params.lon,
        "latitude": params.lat,
        "start": params.start,
        "end": params.end,
        "format": "JSON",
    }

    # Cache em disco (opcional)
    cache_dir = os.environ.get("NASA_CACHE_DIR")
    cache_path = None
    if cache_dir:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            key = f"nasa_{params.lat}_{params.lon}_{params.start}_{params.end}".replace('.', '_')
            cache_path = os.path.join(cache_dir, f"{key}.json")
            if os.path.exists(cache_path):
                logging.info(f"Carregando resposta da NASA a partir do cache: {cache_path}")
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logging.warning(f"Falha ao usar cache em disco: {e}")

    # Retries com backoff exponencial
    max_attempts = 3
    backoff_base = 1.0
    last_exc = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(1, max_attempts + 1):
            try:
                logging.info(f"A contactar a API da NASA POWER para lat={params.lat}, lon={params.lon} (tentativa {attempt})")
                resp = await client.get(settings.NASA_POWER_BASE_URL, params=api_params)
                resp.raise_for_status()
                data = resp.json()
                logging.info("Dados da NASA recebidos com sucesso.")
                if cache_path:
                    try:
                        with open(cache_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False)
                    except Exception as e:
                        logging.warning(f"Falha ao gravar cache em disco: {e}")
                return data
            except httpx.HTTPStatusError as exc:
                logging.error(f"Erro da API da NASA: {exc.response.status_code} - {exc.response.text}")
                last_exc = HTTPException(status_code=502, detail="Erro ao comunicar com a API da NASA.")
                break
            except httpx.RequestError as exc:
                logging.warning(f"Erro de rede ao contactar a NASA (tentativa {attempt}): {exc}")
                last_exc = HTTPException(status_code=504, detail="Falha de conexão com o servidor da NASA.")
                if attempt < max_attempts:
                    sleep_for = backoff_base * (2 ** (attempt - 1))
                    logging.info(f"A aguardar {sleep_for}s antes da próxima tentativa...")
                    await asyncio.sleep(sleep_for)
                    continue
                break

    # Se chegou aqui, houve uma falha irreparável
    if last_exc:
        raise last_exc
    raise HTTPException(status_code=500, detail="Falha desconhecida ao obter dados da NASA.")


# --- Endpoints da API ---
@app.get("/", tags=["Status"])
def root():
    """Endpoint principal com o estado da API e links úteis."""
    return {
        "api_name": app.title,
        "version": app.version,
        "status": "OPERATIONAL",
        "team": "Cygnus-X1",
        "location": "Luanda, Angola",
        "documentation": app.docs_url
    }

@app.get("/clima", 
         response_model=ApiResponseModel, # <<< MELHORIA: Usa o modelo de resposta
         tags=["Weather Analysis"],
         summary="Análise completa de probabilidades climáticas")
async def obter_dados_climaticos(params: ClimateQueryParams = Depends()): # <<< MELHORIA: Usa `Depends` para injetar e validar parâmetros
    """
    Endpoint principal para análise de probabilidades climáticas, utilizando dados históricos da NASA POWER.
    """
    logging.info(f"Nova requisição recebida: {params.model_dump()}")
    
    try:
        # 1. Normalizar a data alvo
        dia_mmdd = _normalize_dia_alvo(params.dia_alvo, params.end)
        logging.info(f"Dia alvo normalizado para análise: {dia_mmdd}")

        # 2. Buscar dados da NASA
        nasa_payload = await _fetch_nasa_power_data(params)

        # 3. Analisar os dados climáticos
        logging.info("A iniciar a análise estatística dos dados.")
        resultado_analise = analisar_dados_climaticos(nasa_payload, dia_mmdd)

        if 'erro' in resultado_analise and resultado_analise.get('status') == 'ERRO':
            logging.warning(f"Análise retornou um erro: {resultado_analise['erro']}")
            raise HTTPException(status_code=422, detail=resultado_analise['erro'])
        
        logging.info("Análise concluída com sucesso.")

        # 4. Formatar a resposta final
        if params.pretty:
            return JSONResponse(content=resultado_analise, headers={"Content-Type": "application/json; charset=utf-8"})
        
        return resultado_analise

    except ValueError as e:
        logging.error(f"Erro de validação: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Erro inesperado no endpoint /clima: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno inesperado.")

@app.get("/challenge-info", tags=["Challenge Information"])
def get_challenge_info():
    """Retorna informações sobre o desafio NASA e a conformidade da API."""
    return {
        "challenge_name": "Will It Rain On My Parade?",
        "team_name": "Cygnus-X1",
        "location": "Luanda, Angola",
        "event_dates": "October 3-5, 2025",
        "nasa_data_source": "NASA POWER API",
        "compliance_status": "Fully Compliant"
    }

@app.get("/thresholds", tags=["Configuration"])
def get_thresholds():
    """Retorna os limites (thresholds) utilizados para classificar as condições climáticas."""
    return {
        "description": "Valores limiar usados para classificar as condições climáticas.",
        "thresholds": settings.DEFAULT_THRESHOLDS,
        "units": {"temperature": "Celsius", "precipitation": "mm", "wind": "m/s", "humidity": "%"}
    }

@app.get("/health", tags=["Status"])
def health_check():
    """Endpoint de health check para monitorização de serviço."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print(f"🚀 {app.title.upper()} (v{app.version})")
    print("=" * 60)
    print("📍 NASA Space Apps Challenge 2025")
    print("🌍 Challenge: Will It Rain On My Parade?")
    print("👥 Team: Cygnus-X1 - Luanda, Angola")
    print("=" * 60)
    print(f"📚 Documentation: http://127.0.0.1:8000{app.docs_url}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)