import requests
r = requests.post(
  "https://nisac-npa-g2p.onrender.com/api/analyze",
  json={"query":"Vamos fazer uma natação de 45KM no Dubai com olhos vendados no dia 21 de fevereiro."},
  timeout=180
)
print(r.status_code, r.json())
