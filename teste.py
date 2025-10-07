import requests
r = requests.post(
  "https://nisac-npa-g2p.onrender.com/api/analyze",
  json={"query":"Vamos fazer uma natação de 45KM em hong kong com olhos vendados no dia 10 de janeiro."},
  timeout=180
)
print(r.status_code, r.json())
