# analise_clima.py (Versão 2.1.0 - Refatorada com Boas Práticas)
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
import logging
from pydantic import BaseModel, ValidationError

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# <<< MELHORIA: Centralizar todos os thresholds num único dicionário de configuração
THRESHOLDS = {
    "temperature": {
        "very_hot_c": 32.0,
        "very_cold_c": 15.0,
        "uncomfortable_heat_c": 30.0
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
        "very_humid_perc": 80.0,
        "uncomfortable_perc": 70.0
    },
    "trends": {
        "temp_change_threshold": 0.5, # Variação em °C para considerar tendência
        "precip_change_threshold": 2.0 # Variação em mm para considerar tendência
    }
}

# --- Funções Auxiliares Modulares ---

def _safe_round(value: Optional[float], precision: int = 1) -> Optional[float]:
    """Arredonda um valor de forma segura, retornando None se for NaN."""
    return round(value, precision) if value is not None and not pd.isna(value) else None

def _criar_dataframe(json_data: Dict[str, Any]) -> pd.DataFrame:
    """Extrai dados do JSON da NASA e cria um DataFrame limpo."""
    try:
        parametros = json_data['properties']['parameter']
        dados = {
            'precipitacao_mm': parametros['PRECTOTCORR'],
            'temp_max_c': parametros['T2M_MAX'],
            'temp_min_c': parametros['T2M_MIN'],
            'umidade_perc': parametros['RH2M'],
            'vento_max_ms': parametros['WS2M_MAX']
        }
        
        df = pd.DataFrame(dados)
        df.index = pd.to_datetime(df.index, format='%Y%m%d')
        df.replace(-999.0, np.nan, inplace=True)
        return df

    except KeyError as e:
        logging.error(f"Estrutura de JSON inválida. Campo ausente: {e}")
        raise ValueError(f"Formato de dados da NASA inválido. Parâmetro esperado '{e}' não encontrado.")

def _calcular_probabilidade(series: pd.Series, threshold: float, operator: str = '>') -> float:
    """Calcula a probabilidade de uma condição ocorrer numa série de dados."""
    if series.empty or series.isnull().all():
        return 0.0
    
    total_validos = series.count()
    if total_validos == 0:
        return 0.0

    if operator == '>':
        condicao = series > threshold
    elif operator == '<':
        condicao = series < threshold
    else:
        raise ValueError("Operador inválido. Use '>' ou '<'.")
        
    return (condicao.sum() / total_validos) * 100


def _bootstrap_confidence_interval(series: pd.Series, threshold: float, operator: str = '>', n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Retorna (point_estimate, lower_pct, upper_pct) em percentuais usando bootstrap simples."""
    # Remover NaNs
    vals = series.dropna()
    if vals.empty:
        return 0.0, 0.0, 0.0

    import random
    estimates = []
    for _ in range(n_boot):
        sample = vals.sample(frac=1.0, replace=True)
        est = _calcular_probabilidade(sample, threshold, operator)
        estimates.append(est)

    estimates_sorted = sorted(estimates)
    lower = estimates_sorted[int((alpha/2) * len(estimates_sorted))]
    upper = estimates_sorted[int((1 - alpha/2) * len(estimates_sorted)) - 1]
    point = _calcular_probabilidade(vals, threshold, operator)
    return point, lower, upper


def _mann_kendall_trend(series: pd.Series) -> Dict[str, Any]:
    """Implementação simples do teste de Mann-Kendall e Sen's slope para detetar tendência.

    Retorna dicionário com 'trend' (AQUECENDO/ESFRIANDO/ESTÁVEL), p_value_approx e sens_slope.
    """
    x = series.dropna().values
    n = len(x)
    if n < 10:
        return {"trend": "INSUFICIENTE", "p_value_approx": None, "sens_slope": None}

    # Mann-Kendall S statistic
    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            if x[j] > x[k]: s += 1
            elif x[j] < x[k]: s -= 1

    # variance approximation (assume no ties)
    var_s = (n*(n-1)*(2*n+5)) / 18.0
    if var_s == 0:
        return {"trend": "INSUFICIENTE", "p_value_approx": None, "sens_slope": None}

    from math import erf, sqrt
    z = 0
    if s > 0:
        z = (s - 1) / sqrt(var_s)
    elif s < 0:
        z = (s + 1) / sqrt(var_s)
    else:
        z = 0

    # two-sided p-value approximation from z (normal)
    p_value = 1.0 - 0.5*(1 + erf(abs(z)/sqrt(2)))

    # Sen's slope median of pairwise slopes
    slopes = []
    for i in range(n - 1):
        for j in range(i+1, n):
            denom = (j - i)
            if denom != 0:
                slopes.append((x[j] - x[i]) / denom)
    sens_slope = float(np.median(slopes)) if slopes else None

    if sens_slope is None:
        trend = "ESTÁVEL"
    elif sens_slope > THRESHOLDS['trends']['temp_change_threshold']:
        trend = "AQUECENDO"
    elif sens_slope < -THRESHOLDS['trends']['temp_change_threshold']:
        trend = "ESFRIANDO"
    else:
        trend = "ESTÁVEL"

    return {"trend": trend, "p_value_approx": p_value, "sens_slope": sens_slope}

def _calcular_estatisticas(df: pd.DataFrame) -> Dict[str, Any]:
    """Calcula as principais estatísticas históricas a partir do DataFrame."""
    return {
        "temperature": {
            "max_average_celsius": _safe_round(df['temp_max_c'].mean()),
            "min_average_celsius": _safe_round(df['temp_min_c'].mean()),
            "max_90th_percentile": _safe_round(df['temp_max_c'].quantile(0.9)),
            "min_10th_percentile": _safe_round(df['temp_min_c'].quantile(0.1))
        },
        "precipitation": {
            "average_mm": _safe_round(df['precipitacao_mm'].mean()),
            "maximum_recorded_mm": _safe_round(df['precipitacao_mm'].max()),
            "75th_percentile_mm": _safe_round(df['precipitacao_mm'].quantile(0.75))
        },
        "humidity": {
            "average_percent": _safe_round(df['umidade_perc'].mean()),
            "maximum_recorded_percent": _safe_round(df['umidade_perc'].max())
        },
        "wind": {
            "average_max_ms": _safe_round(df['vento_max_ms'].mean()),
            "maximum_recorded_ms": _safe_round(df['vento_max_ms'].max())
        }
    }

def _analisar_tendencias(df: pd.DataFrame) -> Tuple[str, str]:
    """Analisa as tendências de temperatura e precipitação ao longo do tempo."""
    if len(df) < 10: # Análise de tendência requer um mínimo de dados
        return "INSUFICIENTE", "INSUFICIENTE"

    meio_ponto = len(df) // 2
    primeira_metade = df.iloc[:meio_ponto]
    segunda_metade = df.iloc[meio_ponto:]

    # Tendência de Temperatura
    dif_temp = segunda_metade['temp_max_c'].mean() - primeira_metade['temp_max_c'].mean()
    if dif_temp > THRESHOLDS["trends"]["temp_change_threshold"]:
        tendencia_temperatura = "AQUECENDO"
    elif dif_temp < -THRESHOLDS["trends"]["temp_change_threshold"]:
        tendencia_temperatura = "ESFRIANDO"
    else:
        tendencia_temperatura = "ESTÁVEL"
        
    # Tendência de Precipitação
    dif_precip = segunda_metade['precipitacao_mm'].mean() - primeira_metade['precipitacao_mm'].mean()
    if dif_precip > THRESHOLDS["trends"]["precip_change_threshold"]:
        tendencia_precipitacao = "AUMENTANDO"
    elif dif_precip < -THRESHOLDS["trends"]["precip_change_threshold"]:
        tendencia_precipitacao = "DIMINUINDO"
    else:
        tendencia_precipitacao = "ESTÁVEL"
        
    return tendencia_temperatura, tendencia_precipitacao

def _gerar_recomendacoes(probabilidades: Dict[str, float]) -> Dict[str, Any]:
    """Gera riscos e sugestões para planeamento de eventos com base nas probabilidades."""
    recomendacoes = {"key_risks": [], "preparation_suggestions": []}
    
    if probabilidades["prob_chuva_forte"] > 30:
        recomendacoes["key_risks"].append("Risco significativo de chuva forte")
        recomendacoes["preparation_suggestions"].append("Considere um plano B em local coberto")
    
    if probabilidades["prob_muito_quente"] > 50:
        recomendacoes["key_risks"].append("Alta probabilidade de calor extremo")
        recomendacoes["preparation_suggestions"].append("Providencie sombra e postos de hidratação")
        
    if probabilidades["prob_muito_frio"] > 50:
        recomendacoes["key_risks"].append("Alta probabilidade de frio")
        recomendacoes["preparation_suggestions"].append("Recomende agasalhos aos participantes")
        
    if probabilidades["prob_muito_ventoso"] > 40:
        recomendacoes["key_risks"].append("Risco moderado a alto de ventos fortes")
        recomendacoes["preparation_suggestions"].append("Proteja decorações e estruturas leves")
        
    if probabilidades["prob_desconforto"] > 40:
        recomendacoes["key_risks"].append("Provável combinação de calor e humidade desconfortáveis")
        recomendacoes["preparation_suggestions"].append("Planeie áreas de arrefecimento e pausas frequentes")
        
    if not recomendacoes["key_risks"]:
        recomendacoes["key_risks"].append("Baixo risco de condições meteorológicas adversas")
        recomendacoes["preparation_suggestions"].append("As condições parecem favoráveis para eventos ao ar livre")
        
    # Calcular score de adequação para eventos ao ar livre
    score = 100 - (
        (probabilidades["prob_chuva_forte"] + probabilidades["prob_muito_quente"] + probabilidades["prob_desconforto"]) / 3
    )
    recomendacoes["outdoor_suitability_score"] = _safe_round(score, 0)

    return recomendacoes

# --- Função Principal de Análise ---

def analisar_dados_climaticos(json_data: Dict[str, Any], dia_alvo: str) -> Dict[str, Any]:
    """
    Analisa dados históricos da NASA POWER para um dia específico e calcula probabilidades climáticas.
    Formatado para o desafio "Will It Rain On My Parade?" da NASA Space Apps Challenge.
    """
    try:
        # 1. Transformar e Limpar Dados
        df_completo = _criar_dataframe(json_data)
        
        # 2. Filtrar para o Dia de Interesse
        df_completo['mes_dia'] = df_completo.index.strftime('%m-%d')
        df_dia = df_completo[df_completo['mes_dia'] == dia_alvo]
        
        total_anos = len(df_dia)
        if total_anos == 0:
            return {"status": "ERRO", "erro": f"Nenhum dado encontrado para o dia {dia_alvo}."}

        # 3. Calcular Probabilidades
        probabilidades = {
            "prob_qualquer_chuva": _calcular_probabilidade(df_dia['precipitacao_mm'], THRESHOLDS["precipitation"]["light_mm"]),
            "prob_chuva_moderada": _calcular_probabilidade(df_dia['precipitacao_mm'], THRESHOLDS["precipitation"]["moderate_mm"]),
            "prob_chuva_forte": _calcular_probabilidade(df_dia['precipitacao_mm'], THRESHOLDS["precipitation"]["heavy_mm"]),
            "prob_chuva_extrema": _calcular_probabilidade(df_dia['precipitacao_mm'], THRESHOLDS["precipitation"]["extreme_mm"]),
            "prob_muito_quente": _calcular_probabilidade(df_dia['temp_max_c'], THRESHOLDS["temperature"]["very_hot_c"]),
            "prob_muito_frio": _calcular_probabilidade(df_dia['temp_min_c'], THRESHOLDS["temperature"]["very_cold_c"], operator='<'),
            "prob_muito_ventoso": _calcular_probabilidade(df_dia['vento_max_ms'], THRESHOLDS["wind"]["very_windy_ms"]),
        }
        dias_desconfortaveis = df_dia[
            (df_dia['temp_max_c'] > THRESHOLDS["temperature"]["uncomfortable_heat_c"]) & 
            (df_dia['umidade_perc'] > THRESHOLDS["humidity"]["uncomfortable_perc"])
        ]
        probabilidades["prob_desconforto"] = (len(dias_desconfortaveis) / total_anos) * 100 if total_anos > 0 else 0.0

        # Adicionar intervalos de confiança via bootstrap para algumas probabilidades-chave
        try:
            point_any, low_any, high_any = _bootstrap_confidence_interval(df_dia['precipitacao_mm'], THRESHOLDS["precipitation"]["light_mm"], operator='>')
            probabilidades["prob_qualquer_chuva_ci"] = {"point": _safe_round(point_any), "lower": _safe_round(low_any), "upper": _safe_round(high_any)}
        except Exception as e:
            logging.debug(f"Falha ao calcular bootstrap CI: {e}")

        try:
            point_hot, low_hot, high_hot = _bootstrap_confidence_interval(df_dia['temp_max_c'], THRESHOLDS["temperature"]["very_hot_c"], operator='>')
            probabilidades["prob_muito_quente_ci"] = {"point": _safe_round(point_hot), "lower": _safe_round(low_hot), "upper": _safe_round(high_hot)}
        except Exception as e:
            logging.debug(f"Falha ao calcular bootstrap CI (hot): {e}")

        # 4. Calcular Estatísticas e Tendências
        estatisticas = _calcular_estatisticas(df_dia)
        tendencia_temp, tendencia_precip = _analisar_tendencias(df_dia)
        # Mann-Kendall e Sen's slope para temperatura máxima histórica
        try:
            mk_temp = _mann_kendall_trend(df_dia['temp_max_c'])
            estatisticas['temperature']['mann_kendall'] = mk_temp
        except Exception as e:
            logging.debug(f"Falha ao calcular Mann-Kendall: {e}")
        
        # 5. Gerar Recomendações
        recomendacoes = _gerar_recomendacoes(probabilidades)

        # 6. Montar a Resposta Final
        resultado = {
            "nasa_challenge_compliance": {
                "challenge_name": "Will It Rain On My Parade?",
                "data_source": "NASA POWER API",
                "analysis_timestamp": datetime.now().isoformat()
            },
            "query_parameters": {
                "location": {
                    "latitude": json_data['geometry']['coordinates'][1],
                    "longitude": json_data['geometry']['coordinates'][0],
                },
                "target_date": {"day_of_year": dia_alvo},
                "historical_period": {
                    "start_year": int(df_dia.index.year.min()),
                    "end_year": int(df_dia.index.year.max()),
                    "total_years_analyzed": total_anos,
                }
            },
            "weather_condition_probabilities": {
                "VERY_HOT": {"probability_percent": _safe_round(probabilidades["prob_muito_quente"])},
                "VERY_COLD": {"probability_percent": _safe_round(probabilidades["prob_muito_frio"])},
                "VERY_WINDY": {"probability_percent": _safe_round(probabilidades["prob_muito_ventoso"])},
                "VERY_WET": {"probability_percent": _safe_round(probabilidades["prob_chuva_forte"])},
                "VERY_UNCOMFORTABLE": {"probability_percent": _safe_round(probabilidades["prob_desconforto"])}
            },
            "precipitation_analysis": {
                "any_rain": {"probability_percent": _safe_round(probabilidades["prob_qualquer_chuva"])},
                "moderate_rain": {"probability_percent": _safe_round(probabilidades["prob_chuva_moderada"])},
                "heavy_rain": {"probability_percent": _safe_round(probabilidades["prob_chuva_forte"])},
                "extreme_rain": {"probability_percent": _safe_round(probabilidades["prob_chuva_extrema"])}
            },
            "historical_statistics": estatisticas,
            "climate_change_indicators": {
                "temperature_trend": tendencia_temp,
                "precipitation_trend": tendencia_precip,
            },
            "event_planning_recommendations": recomendacoes,
            "threshold_configurations": THRESHOLDS,
            "response_metadata": {
                "status": "SUCCESS",
                "confidence_level": "HIGH" if total_anos >= 20 else "MODERATE" if total_anos >= 10 else "LOW",
            }
        }
        # Validar estrutura do resultado antes de retornar
        try:
            class AnalysisResultModel(BaseModel):
                nasa_challenge_compliance: Dict[str, Any]
                query_parameters: Dict[str, Any]
                weather_condition_probabilities: Dict[str, Any]
                precipitation_analysis: Dict[str, Any]
                historical_statistics: Dict[str, Any]
                climate_change_indicators: Dict[str, Any]
                event_planning_recommendations: Dict[str, Any]
                threshold_configurations: Dict[str, Any]
                response_metadata: Dict[str, Any]

            AnalysisResultModel(**resultado)
        except ValidationError as ve:
            logging.error(f"Validação do resultado falhou: {ve}")
            return {"status": "ERRO", "erro": f"Resultado da análise inválido: {ve}"}

        return resultado

    except (KeyError, ValueError) as e:
        logging.error(f"Erro ao processar dados da NASA: {e}")
        return {"status": "ERRO", "erro": f"Falha ao processar os dados recebidos: {e}"}
    except Exception as e:
        logging.error(f"Erro inesperado na análise: {e}", exc_info=True)
        return {"status": "ERRO", "erro": f"Ocorreu um erro inesperado: {e}"}