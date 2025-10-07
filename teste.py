import requests
r = requests.post(
  "https://nisac-npa-g2p.onrender.com/api/analyze",
  json={"query":"Vou fazer uma competição de ficar sem roupa no frio no dia 18 de Agosto em Manchester, UK."},
  timeout=180
)
print(r.status_code, r.json())
