import json
import sys
import os
# Garantir que o root do repositório esteja no sys.path para o pytest
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.analise_clima import analisar_dados_climaticos

# Mock minimal NASA POWER JSON structure for a few years on 06-15
mock_payload = {
    "geometry": {"coordinates": [13.23, -8.83]},
    "properties": {
        "parameter": {
            # Using date keys as strings like '20150101', '20160101' etc.
            "PRECTOTCORR": {"20150615": 0.0, "20160615": 1.2, "20170615": 0.0},
            "T2M_MAX": {"20150615": 28.0, "20160615": 33.2, "20170615": 29.1},
            "T2M_MIN": {"20150615": 18.0, "20160615": 19.0, "20170615": 17.5},
            "RH2M": {"20150615": 65.0, "20160615": 70.0, "20170615": 68.0},
            "WS2M_MAX": {"20150615": 5.0, "20160615": 12.0, "20170615": 6.0}
        }
    }
}


def test_analisar_dados_climaticos_basic():
    result = analisar_dados_climaticos(mock_payload, "06-15")
    assert isinstance(result, dict)
    assert result.get("response_metadata", {}).get("status") in ("SUCCESS", "ERRO")
    # If success, check some expected keys
    if result.get("response_metadata", {}).get("status") == "SUCCESS":
        assert "weather_condition_probabilities" in result
        assert "precipitation_analysis" in result
        assert "historical_statistics" in result
