import requests
r = requests.post(
  "https://nisac-npa-g2p.onrender.com/api/analyze",
  json={"query":"Vou fazer um casamento com casacos pesados no grand canyon no dia 17 de janeiro"},
  timeout=180
)
print(r.status_code, r.json())
