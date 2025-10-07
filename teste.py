import requests
r = requests.post(
  "https://nisac-npa-g2p.onrender.com/api/analyze",
  json={"query":"Vamos fazer uma caminhada no polo norte sem camisola e com os pés descalços no dia 25 de dezembro de 2026. Será que vamos sofrer queimaduras do frio?"},
  timeout=180
)
print(r.status_code, r.json())
