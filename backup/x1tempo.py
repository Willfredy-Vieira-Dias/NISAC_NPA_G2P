# x1tempo.py (VERSAO CORRIGIDA COM DEBUG)
# -*- coding: utf-8 -*-

import os
import json
import traceback
from typing import Optional
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Importa o modulo de analise
from analise_clima import analisar_dados_climaticos

app = FastAPI(
    title="Cygnus-X1 WIROMP API",
    description="API interna para o desafio 'Will It Rain On My Parade?' que consome dados da NASA POWER e retorna probabilidades.",
    version="1.0.0",
)

# CORS (ajusta conforme necessario)
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "*"  # Para testes - remover em producao
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"


def _normalize_dia_alvo(dia_alvo: Optional[str], fallback_end: Optional[str]) -> str:
    """
    Aceita dia_alvo em "MM-DD" ou "YYYY-MM-DD". 
    Se None, usa fallback_end (YYYYMMDD ou YYYY-MM-DD) e retorna MM-DD.
    Retorna string "MM-DD".
    """
    if dia_alvo:
        # ja em MM-DD?
        try:
            if len(dia_alvo) == 5 and dia_alvo[2] == "-":
                # assume MM-DD
                month = int(dia_alvo[:2])
                day = int(dia_alvo[3:5])
                if 1 <= month <= 12 and 1 <= day <= 31:
                    return dia_alvo
                else:
                    raise ValueError("Mes ou dia invalido")
            else:
                # tenta YYYY-MM-DD
                dt = datetime.strptime(dia_alvo, "%Y-%m-%d")
                return dt.strftime("%m-%d")
        except Exception:
            # tenta tambem YYYYMMDD
            try:
                dt = datetime.strptime(dia_alvo, "%Y%m%d")
                return dt.strftime("%m-%d")
            except Exception:
                raise ValueError("dia_alvo deve ser 'MM-DD' ou 'YYYY-MM-DD' ou 'YYYYMMDD'")
    
    # fallback from end date
    if fallback_end:
        try:
            if len(fallback_end) == 8 and fallback_end.isdigit():
                dt = datetime.strptime(fallback_end, "%Y%m%d")
            else:
                dt = datetime.strptime(fallback_end, "%Y-%m-%d")
            return dt.strftime("%m-%d")
        except Exception:
            raise ValueError("fallback end date tem formato invalido")
    
    raise ValueError("dia_alvo ou end deve ser fornecido")


@app.get("/clima")
async def obter_dados_climaticos(
    lat: float,
    lon: float,
    start: str = Query(..., description="YYYYMMDD"),
    end: str = Query(..., description="YYYYMMDD"),
    dia_alvo: Optional[str] = Query(None, description="MM-DD ou YYYY-MM-DD (opcional). Se omitido usa 'end' para extrair MM-DD"),
):
    """
    Obtem dados historicos da NASA POWER para lat/lon no intervalo start..end,
    e devolve probabilidades para o dia (dia_alvo) usando analise_clima.analisar_dados_climaticos.

    - lat, lon: coordenadas
    - start, end: strings YYYYMMDD (POWER API espera esse formato)
    - dia_alvo: opcional; formato "MM-DD" ou "YYYY-MM-DD". Se omitido, usa o 'end' como referencia.
    """
    
    print(f"DEBUG: Recebido - lat={lat}, lon={lon}, start={start}, end={end}, dia_alvo={dia_alvo}")
    
    # Validar e normalizar dia_alvo
    try:
        dia_mmdd = _normalize_dia_alvo(dia_alvo, end)
        print(f"DEBUG: dia_mmdd normalizado = {dia_mmdd}")
    except ValueError as e:
        print(f"ERROR: Normalizacao falhou - {e}")
        raise HTTPException(status_code=400, detail=str(e))

    params = {
        "parameters": "PRECTOTCORR,T2M_MAX,T2M_MIN,RH2M,WS2M_MAX",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON",
    }
    
    print(f"DEBUG: Parametros para NASA POWER API: {params}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            print("DEBUG: Fazendo chamada para NASA POWER API...")
            resp = await client.get(POWER_BASE, params=params)
            print(f"DEBUG: Resposta recebida - Status Code: {resp.status_code}")
        except httpx.RequestError as exc:
            print(f"ERROR: Erro ao contactar POWER API - {exc}")
            raise HTTPException(status_code=502, detail=f"Erro ao contactar POWER API: {exc}") from exc

    if resp.status_code != 200:
        # tenta expor o erro da NASA POWER
        detail = None
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        print(f"ERROR: NASA POWER retornou erro - Status: {resp.status_code}, Detail: {detail}")
        raise HTTPException(status_code=502, detail={"power_status": resp.status_code, "detail": detail})

    try:
        payload = resp.json()
        print(f"DEBUG: JSON parseado com sucesso. Keys: {list(payload.keys()) if payload else 'None'}")
        
        # Debug: Verificar estrutura do payload
        if 'properties' in payload and 'parameter' in payload['properties']:
            param_keys = list(payload['properties']['parameter'].keys())
            print(f"DEBUG: Parametros recebidos: {param_keys}")
            
            # Verificar se temos dados
            for key in param_keys:
                data_points = len(payload['properties']['parameter'][key])
                print(f"DEBUG: {key} tem {data_points} pontos de dados")
                
    except Exception as e:
        print(f"ERROR: Nao foi possivel parsear JSON - {e}")
        raise HTTPException(status_code=502, detail=f"Nao foi possivel parsear JSON da POWER: {e}")

    # Chama a funcao de analise (sincrona) - ela usa pandas/numpy
    try:
        print(f"DEBUG: Chamando analisar_dados_climaticos com dia_mmdd={dia_mmdd}")
        resultado = analisar_dados_climaticos(payload, dia_mmdd)
        print(f"DEBUG: Analise completa. Keys do resultado: {list(resultado.keys()) if resultado else 'None'}")
        
        # Se houver erro no resultado
        if 'erro' in resultado:
            print(f"WARNING: Analise retornou erro: {resultado['erro']}")
            
    except Exception as e:
        print(f"ERROR: Erro na analise dos dados - {e}")
        print(f"TRACEBACK: {traceback.format_exc()}")
        # Se a analise falhar por dados insuficientes, devolve 422 (unprocessable)
        raise HTTPException(status_code=422, detail=f"Erro na analise dos dados: {e}")

    return resultado


@app.get("/")
def health_check():
    return {"status": "A API X1Tempo esta operacional!"}


@app.get("/test")
async def test_endpoint():
    """Endpoint de teste para verificar se a API esta funcionando"""
    return {
        "message": "Teste OK",
        "timestamp": datetime.now().isoformat(),
        "exemplo_url": "http://127.0.0.1:8000/clima?lat=-8.83&lon=13.23&start=20220101&end=20221231&dia_alvo=01-15"
    }