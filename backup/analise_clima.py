# analise_clima.py (VERSAO CORRIGIDA)
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from typing import Dict, Any

def analisar_dados_climaticos(json_data: Dict[str, Any], dia_alvo: str) -> Dict[str, Any]:
    """
    Analisa os dados historicos da NASA POWER para um dia especifico do ano
    e calcula as probabilidades de varias condicoes climaticas.

    Args:
        json_data: O JSON bruto recebido da API da NASA POWER.
        dia_alvo: O dia do ano de interesse, no formato "MM-DD" (ex: "06-15" para 15 de Junho).

    Returns:
        Um dicionario com as probabilidades calculadas e outras estatisticas.
    """
    
    try:
        # --- 1. TRANSFORMAR O JSON NUM DATAFRAME DO PANDAS ---
        
        # Extrai o dicionario de parametros do JSON
        parametros = json_data['properties']['parameter']
        
        # Monta um dicionario com todos os dados primeiro para garantir o alinhamento
        dados_completos = {
            'precipitacao_mm': parametros['PRECTOTCORR'],
            'temp_max_c': parametros['T2M_MAX'],
            'temp_min_c': parametros['T2M_MIN'],
            'umidade_perc': parametros['RH2M'],
            'vento_max_ms': parametros['WS2M_MAX']
        }
        
        # Cria o DataFrame de uma so vez a partir do dicionario completo
        df = pd.DataFrame(dados_completos)
        
        # Agora, com o DataFrame ja montado e alinhado, converte o indice para datetime
        df.index = pd.to_datetime(df.index, format='%Y%m%d')

        # --- 2. LIMPAR OS DADOS ---
        
        # A API da NASA usa -999.0 para valores em falta. Substituimos por NaN
        df.replace(-999.0, np.nan, inplace=True)
        
        # --- 3. FILTRAR PARA O DIA DE INTERESSE ---
        
        # Cria uma nova coluna 'mes_dia' no formato "MM-DD" para filtrar
        df['mes_dia'] = df.index.strftime('%m-%d')
        
        # Filtra o DataFrame para conter apenas os dados do dia que o utilizador pediu
        dados_do_dia = df[df['mes_dia'] == dia_alvo]
        
        total_anos_analisados = len(dados_do_dia)
        
        if total_anos_analisados == 0:
            return {"erro": f"Nenhum dado encontrado para o dia {dia_alvo}. Verifique o intervalo de datas."}

        # --- 4. DEFINIR OS LIMITES (THRESHOLDS) ---
        
        limite_chuva_forte_mm = 10.0
        limite_muito_quente_c = 32.0
        limite_muito_frio_c = 15.0
        limite_vento_forte_ms = 10.0
        
        # Condicao de "desconforto" (pode ser uma combinacao)
        limite_desconforto_temp = 30.0
        limite_desconforto_umid = 80.0

        # --- 5. CALCULAR AS PROBABILIDADES ---
        
        dias_com_chuva = dados_do_dia[dados_do_dia['precipitacao_mm'] > 0.1]
        prob_chuva = (len(dias_com_chuva) / total_anos_analisados) * 100 if total_anos_analisados > 0 else 0
        
        dias_chuva_forte = dados_do_dia[dados_do_dia['precipitacao_mm'] > limite_chuva_forte_mm]
        prob_chuva_forte = (len(dias_chuva_forte) / total_anos_analisados) * 100 if total_anos_analisados > 0 else 0
        
        dias_muito_quentes = dados_do_dia[dados_do_dia['temp_max_c'] > limite_muito_quente_c]
        prob_muito_quente = (len(dias_muito_quentes) / total_anos_analisados) * 100 if total_anos_analisados > 0 else 0
        
        dias_muito_frios = dados_do_dia[dados_do_dia['temp_min_c'] < limite_muito_frio_c]
        prob_muito_frio = (len(dias_muito_frios) / total_anos_analisados) * 100 if total_anos_analisados > 0 else 0
        
        dias_vento_forte = dados_do_dia[dados_do_dia['vento_max_ms'] > limite_vento_forte_ms]
        prob_vento_forte = (len(dias_vento_forte) / total_anos_analisados) * 100 if total_anos_analisados > 0 else 0
        
        dias_desconfortaveis = dados_do_dia[
            (dados_do_dia['temp_max_c'] > limite_desconforto_temp) & 
            (dados_do_dia['umidade_perc'] > limite_desconforto_umid)
        ]
        prob_desconforto = (len(dias_desconfortaveis) / total_anos_analisados) * 100 if total_anos_analisados > 0 else 0

        # --- 6. CALCULAR MEDIAS HISTORICAS ---
        
        # Calcular medias de forma mais simples, sem walrus operator
        temp_max_media = dados_do_dia['temp_max_c'].mean()
        temp_min_media = dados_do_dia['temp_min_c'].mean()
        precipitacao_media = dados_do_dia['precipitacao_mm'].mean()
        umidade_media = dados_do_dia['umidade_perc'].mean()
        
        # --- 7. MONTAR O RESULTADO FINAL ---
        
        resultado = {
            "localizacao": {
                "latitude": json_data['geometry']['coordinates'][1],
                "longitude": json_data['geometry']['coordinates'][0]
            },
            "analise_para_o_dia": dia_alvo,
            "periodo_historico_analisado": {
                "ano_inicio": int(dados_do_dia.index.year.min()) if not dados_do_dia.empty else None,
                "ano_fim": int(dados_do_dia.index.year.max()) if not dados_do_dia.empty else None,
                "total_registos": total_anos_analisados
            },
            "probabilidades_percentual": {
                "qualquer_chuva": round(prob_chuva, 2),
                "chuva_forte": round(prob_chuva_forte, 2),
                "muito_quente": round(prob_muito_quente, 2),
                "muito_frio": round(prob_muito_frio, 2),
                "vento_forte": round(prob_vento_forte, 2),
                "desconforto_quente_humido": round(prob_desconforto, 2)
            },
            "medias_historicas_para_este_dia": {
                "temperatura_maxima_c": round(temp_max_media, 2) if not pd.isna(temp_max_media) else None,
                "temperatura_minima_c": round(temp_min_media, 2) if not pd.isna(temp_min_media) else None,
                "precipitacao_mm": round(precipitacao_media, 2) if not pd.isna(precipitacao_media) else None,
                "umidade_percentual": round(umidade_media, 2) if not pd.isna(umidade_media) else None
            },
            "limites_considerados": {
                "chuva_forte_mm": f"> {limite_chuva_forte_mm}",
                "muito_quente_c": f"> {limite_muito_quente_c}",
                "muito_frio_c": f"< {limite_muito_frio_c}",
                "vento_forte_ms": f"> {limite_vento_forte_ms}",
                "desconforto": f"Temp > {limite_desconforto_temp}C e Umidade > {limite_desconforto_umid}%"
            }
        }
        
        return resultado
        
    except KeyError as e:
        return {"erro": f"Erro ao processar dados da NASA POWER: Campo ausente - {str(e)}"}
    except Exception as e:
        return {"erro": f"Erro inesperado na analise: {str(e)}"}